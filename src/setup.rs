//! `mangajanai setup`: download what this machine needs to upscale, next to the program.
//!
//! What's installed follows MangaJaNai GUI's updater (its RecommendedPacks):
//! - Windows: ONNX Runtime + DirectML (always: AMD/Intel GPUs, and Auto's fallback).
//! - An NVIDIA GPU: the TensorRT runtime, the builder resources for each GPU generation present (or
//!   the PTX fallback for one without its own), and on Windows TensorRT-RTX on the GPUs it's the
//!   default for (RTX 20 series and newer). A GPU older than TensorRT supports (compute capability
//!   below 7.5, `nvml::TRT_MIN_COMPUTE`) counts as no NVIDIA GPU.
//! - Models: MangaJaNai and IllustrationJaNai, plus the DAT2/HAT variant this GPU runs best (bf16 on
//!   NVIDIA compute capability 8.0+, fp32 otherwise).
//!
//! Sources, each checked against a SHA-256 compiled in here: the runtime packs from the shared
//! `runtime-<n>` release (animejanai-inference scripts/runtime_packs.py), TensorRT-RTX from
//! libaji's release, the models from the MangaJaNai model releases. libaji itself ships in the
//! program's archive.
//!
//! Runtime files go next to the program, models into `models/` beside it, and `setup.json` records
//! what was installed (by archive SHA-256), so a re-run fetches only what's missing or changed.
//!
//! **Store mode** (`setup --store --app-id ID --app-path EXE`, for a CLI an app installed, e.g.
//! yosh's add-on): TensorRT, its builder packs, TensorRT-RTX and the models go into the per-user
//! pack store instead, shared with the other apps (the-database/aji-runtime), from the shared
//! runtime release (animejanai-inference `runtime-<n>`). Matching files already in this folder are
//! taken over instead of downloaded, and removed from here once the store has them. libaji finds
//! the store through `aji-runtime.json`, the model search through the app's ref. ONNX Runtime +
//! DirectML stay private, next to the program. `setup.json` remembers the mode, so a plain
//! `setup` re-run stays in it; `setup --remove` drops the app's ref (its uninstaller runs it).
//! Without `--store` (a portable CLI) nothing changes.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::nvml;

/// The runtime packs, for both modes: the shared runtime release every libaji app pins
/// (animejanai-inference scripts/runtime_packs.py TAG; its runtime-packs.yml asks bump-pins.yml
/// here for the bump PR), and the SHA-256 of each packs-<platform>.json, which pins every pack.
/// (runtime-1, on this repo, was the CLI's own release before it.)
const RUNTIME_REPO: &str = "the-database/animejanai-inference";
const RUNTIME_TAG: &str = "runtime-2";
const RUNTIME_INDEX_SHA256_WINDOWS: &str = "5b6f0450ba4eed34d6e25dde072ce1298f083d91d142fd858495483e7711aea1";
const RUNTIME_INDEX_SHA256_LINUX: &str = "d441210f7747b255f17483f7c8b921817ad3f997abab5a1ba5d348b51511ab61";

/// TensorRT-RTX: aji_trtrtx.dll + the TensorRT for RTX runtime, from the libaji release this
/// program's archive bundles.
const AJI_REPO: &str = "the-database/animejanai-inference";
const AJI_VERSION: &str = "v0.28.0";
const TRTRTX_ASSET: &str = "aji-trtrtx-windows-x64.zip";
const TRTRTX_SHA256: &str = "4e35677dee83610b85764b87b1d178064d34f417f7ac9b1d73623855f5cf0d0b";
const TRTRTX_BYTES: u64 = 95_428_810;

/// The default models, the same archives (and SHA-256s) MangaJaNai GUI installs.
const MODELS_REPO: &str = "the-database/MangaJaNai";
const MODELS: &[(&str, &str, &str, &str, u64)] = &[
    ("models-mangajanai", "1.0.0", "MangaJaNai_V1_ONNX_fp16.7z", "2c3809d712e3e96d107e2a59802186c284fac49d91a211435c8c896432ba27ab", 421_712_283),
    ("models-illustrationjanai", "3.0.0", "IllustrationJaNai_V3_ONNX_fp16.7z", "bd7db0632b37b95b06410e99f95347980010b2ccc54730e2d68501f25bae898c", 112_554_037),
    ("models-illustrationjanai-fp32", "3.0.0", "IllustrationJaNai_V3_ONNX_fp32.7z", "8b38bfdc95f194fb18eaa51d7ca869839ebbb1c29bed1e7a943a65b372d090ac", 234_850_558),
    ("models-illustrationjanai-bf16", "3.0.0", "IllustrationJaNai_V3_ONNX_bf16.7z", "48e292c3f92aa2df79f9f83c6318a6c2052c29e6ea9fe2ed23b61cc526900247", 93_681_845),
];

const STATE_FILE: &str = "setup.json";
const DOWNLOAD_DIR: &str = ".setup-download";

pub struct Options {
    pub no_models: bool,
    pub check: bool,
    /// A folder holding the archives by asset name (and the runtime packs-<platform>.json), used
    /// instead of downloading; the index isn't pin-checked, the archives still are.
    pub mirror: Option<PathBuf>,
    /// `--progress lines`: also report on stdout in lines another program can read (yosh installs
    /// its AI add-on this way). See [`Lines`].
    pub lines: bool,
    /// `--store --app-id ID --app-path EXE`: install into the shared pack store as app ID.
    pub store: Option<StoreMode>,
    /// `--remove`: in store mode, drop this app's ref (before its folder is deleted).
    pub remove: bool,
}

/// Which installed app this CLI belongs to (its ref in the pack store).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StoreMode {
    pub app_id: String,
    /// The app's executable: the store keeps this app's packs while it exists.
    pub app_path: PathBuf,
}

impl Options {
    pub fn private(no_models: bool) -> Options {
        Options { no_models, check: false, mirror: None, lines: false, store: None, remove: false }
    }
}

/// The stdout line protocol of `setup --progress lines`, one event per line, fields separated by
/// single spaces (a component name never contains one):
///
/// - `SETUP_PLAN <name> <bytes> installed|download`: each component this machine needs.
/// - `SETUP_TOTAL <bytes>`: what will be downloaded (0: everything is installed).
/// - `SETUP_DOWNLOAD <name> <got> <total>`: download progress, at most once per percent.
/// - `SETUP_EXTRACT <name>`: the component's archive is being unpacked.
/// - `SETUP_DONE`: finished, nothing left to do (not printed after `--check` with work left).
///
/// (`mangajanai store move --progress lines` has its own lines, `STORE_*`: see `store.rs`.)
///
/// A failure ends the run without `SETUP_DONE`: the message is on stderr (`error: ...`) and the exit
/// code isn't 0. The human output on stderr is unchanged, apart from the download percentages.
struct Lines(bool);

impl Lines {
    fn say(&self, s: std::fmt::Arguments) {
        if self.0 {
            let mut out = std::io::stdout().lock();
            let _ = writeln!(out, "{s}");
            let _ = out.flush();
        }
    }
}

#[derive(Deserialize)]
struct Index {
    packs: Vec<Pack>,
}

#[derive(Deserialize, Clone)]
struct Pack {
    name: String,
    version: String,
    asset: String,
    sha256: String,
    bytes: u64,
    /// Where the archive is. A pack unchanged since an earlier runtime release keeps that release's
    /// archive, so this isn't always on `RUNTIME_TAG`; `None` in an index without urls.
    #[serde(default)]
    url: Option<String>,
    /// Each file's SHA-256 and size (the shared release's index has them).
    #[serde(default)]
    files: Vec<PackFile>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
struct PackFile {
    path: String,
    sha256: String,
    bytes: u64,
}

#[derive(Serialize, Deserialize, Default)]
struct State {
    /// What setup installed here (store mode: only what stays private, ONNX Runtime + DirectML).
    components: BTreeMap<String, Installed>,
    /// Set in store mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    store: Option<StoreMode>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Installed {
    version: String,
    sha256: String,
    /// Paths relative to the install folder.
    files: Vec<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    SevenZ,
    Zip,
}

#[derive(Clone)]
struct Component {
    name: String,
    version: String,
    url: String,
    asset: String,
    sha256: String,
    bytes: u64,
    format: Format,
    /// Extract into `models/` instead of next to the program.
    models: bool,
    /// The files the archive holds, with their SHA-256s (runtime packs; empty when unknown).
    files: Vec<PackFile>,
}

fn platform() -> &'static str {
    if cfg!(windows) { "windows-x64" } else { "linux-x64" }
}

/// Where setup installs: the program's own folder.
pub fn install_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_default()
}

/// Whether setup has installed the runtime here (libaji alone ships with the program, so this looks
/// for what setup adds).
pub fn runtime_installed(dir: &Path) -> bool {
    let probe: &[&str] = if cfg!(windows) { &["nvinfer_11.dll", "onnxruntime.dll"] } else { &["libnvinfer.so.11"] };
    // store mode: libaji loads TensorRT from the pack store through aji-runtime.json
    probe.iter().any(|f| dir.join(f).is_file()) || dir.join("aji-runtime.json").is_file()
}

/// The store mode setup recorded in `dir`, if any.
pub fn store_mode(dir: &Path) -> Option<StoreMode> {
    load_state(dir).store
}

/// In store mode: the model pack folders this app's ref lists, for the model search.
pub fn store_model_dirs(dir: &Path) -> Vec<PathBuf> {
    let Some(mode) = store_mode(dir) else { return Vec::new() };
    let Some(store) = aji_runtime::Store::default_root().ok().and_then(|r| aji_runtime::Store::open_existing(&r)) else {
        return Vec::new();
    };
    aji_runtime::app_packs(&store, &mode.app_id).into_iter().filter(|(m, _)| m.kind == "models").map(|(_, d)| d).collect()
}

/// TensorRT-RTX's default GPUs ([`nvml::trtrtx_default`]).
fn trtrtx_default(g: &nvml::Gpu) -> bool {
    nvml::trtrtx_default(g.major, g.minor)
}

fn mb(b: u64) -> String {
    format!("{:.0} MB", b as f64 / 1e6)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// Lowercase hex (sha2 0.11's digests don't implement LowerHex).
fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_bytes(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

/// Download `url` to `dest`, showing progress on stderr (and as `SETUP_DOWNLOAD` lines). Stalls time
/// out and transient failures are retried, resuming where they stopped (`download.rs`).
fn download(url: &str, dest: &Path, label: &str, expected: u64, lines: &Lines) -> Result<(), String> {
    let part = dest.with_extension("part");
    let mut shown = Shown::new();
    let mut said = (u64::MAX, u64::MAX);
    let r = crate::download::download(url, &part, expected, &crate::download::Policy::default(), &mut |got, total| {
        // the first report of each request, then once per percent
        let pct = (got * 100).checked_div(total).unwrap_or(0);
        if got == 0 || (pct, total) != said {
            said = (pct, total);
            lines.say(format_args!("SETUP_DOWNLOAD {label} {got} {total}"));
        }
        if total > 0 {
            shown.show(label, got, total);
        }
    });
    shown.end();
    r?;
    std::fs::rename(&part, dest).map_err(|e| format!("{}: {e}", dest.display()))
}

/// Download percentages on stderr: every percent in a terminal, else every 25%.
struct Shown {
    tty: bool,
    step: u64,
}

impl Shown {
    fn new() -> Shown {
        Shown { tty: std::io::stderr().is_terminal(), step: u64::MAX }
    }
    fn show(&mut self, label: &str, got: u64, total: u64) {
        let pct = (got * 100).checked_div(total).unwrap_or(0);
        let step = if self.tty { pct } else { pct / 25 * 25 };
        if step != self.step {
            self.step = step;
            if self.tty {
                eprint!("\r  {label}: {pct:>3}%  {} / {}   ", mb(got), mb(total));
            } else {
                eprintln!("  {label}: {pct}%");
            }
        }
    }
    fn end(&mut self) {
        if self.tty && self.step != u64::MAX {
            eprintln!();
        }
        self.step = u64::MAX;
    }
}

/// Extract an archive into `dest`, returning the files it wrote (relative to `dest`).
fn extract(archive: &Path, format: Format, dest: &Path) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut files = Vec::new();
    match format {
        Format::SevenZ => {
            sevenz_rust2::decompress_file_with_extract_fn(archive, dest, |entry, reader, path| {
                if !entry.is_directory() {
                    files.push(entry.name().replace('\\', "/"));
                }
                sevenz_rust2::default_entry_extract_fn(entry, reader, path)
            })
            .map_err(|e| format!("extracting {}: {e}", archive.display()))?;
        }
        Format::Zip => {
            // Flat: the libaji archives keep their files at the root.
            let f = std::fs::File::open(archive).map_err(|e| e.to_string())?;
            let mut zip = zip::ZipArchive::new(f).map_err(|e| format!("{}: {e}", archive.display()))?;
            for i in 0..zip.len() {
                let mut e = zip.by_index(i).map_err(|e| e.to_string())?;
                if e.is_dir() {
                    continue;
                }
                let entry_name = e.name().map_err(|err| format!("{}: {err}", archive.display()))?.into_owned();
                let Some(name) = Path::new(&entry_name).file_name().map(|n| n.to_string_lossy().into_owned()) else { continue };
                let mut out = std::fs::File::create(dest.join(&name)).map_err(|e| format!("{name}: {e}"))?;
                std::io::copy(&mut e, &mut out).map_err(|e| format!("{name}: {e}"))?;
                files.push(name);
            }
        }
    }
    #[cfg(unix)]
    for f in &files {
        use std::os::unix::fs::PermissionsExt;
        if f == "trtexec" {
            let _ = std::fs::set_permissions(dest.join(f), std::fs::Permissions::from_mode(0o755));
        }
    }
    Ok(files)
}

/// Fetch a small file (the runtime index) into memory, with `download.rs`'s timeouts and retries.
fn fetch_small(url: &str) -> Result<Vec<u8>, String> {
    crate::download::fetch_small(url, &crate::download::Policy::default())
}

fn runtime_index(opts: &Options) -> Result<Vec<Pack>, String> {
    let file = format!("packs-{}.json", platform());
    let bytes = match &opts.mirror {
        Some(dir) => std::fs::read(dir.join(&file)).map_err(|e| format!("{}: {e}", dir.join(&file).display()))?,
        None => {
            let pin = if cfg!(windows) { RUNTIME_INDEX_SHA256_WINDOWS } else { RUNTIME_INDEX_SHA256_LINUX };
            if pin.is_empty() {
                return Err(format!("this build of mangajanai has no runtime packs pinned ({RUNTIME_TAG} isn't published yet)"));
            }
            let b = fetch_small(&format!("https://github.com/{RUNTIME_REPO}/releases/download/{RUNTIME_TAG}/{file}"))?;
            if sha256_bytes(&b) != pin {
                return Err(format!("{file} from {RUNTIME_TAG} doesn't match the SHA-256 this program pins"));
            }
            b
        }
    };
    let index: Index = serde_json::from_slice(&bytes).map_err(|e| format!("{file}: {e}"))?;
    Ok(index.packs)
}

/// A runtime pack's download: the index's url, else (an index without urls) on `RUNTIME_TAG`.
fn pack_url(p: &Pack) -> String {
    match &p.url {
        Some(u) if !u.is_empty() => u.clone(),
        _ => format!("https://github.com/{RUNTIME_REPO}/releases/download/{RUNTIME_TAG}/{}", p.asset),
    }
}

/// What this machine needs, in install order.
fn plan(opts: &Options, gpus: &[nvml::Gpu]) -> Result<Vec<Component>, String> {
    let packs = runtime_index(opts)?;
    let (all_gpus, gpus) = (gpus, &nvml::trt_gpus(gpus));
    let pack = |name: &str| packs.iter().find(|p| p.name == name).cloned();
    let runtime = |p: Pack| Component {
        url: pack_url(&p),
        name: p.name,
        version: p.version,
        asset: p.asset,
        sha256: p.sha256,
        bytes: p.bytes,
        format: Format::SevenZ,
        models: false,
        files: p.files,
    };
    let mut out = Vec::new();
    if cfg!(windows) {
        out.push(runtime(pack("dml-runtime").ok_or("the runtime packs have no dml-runtime")?));
    }
    if !gpus.is_empty() {
        out.push(runtime(pack("trt-runtime").ok_or("the runtime packs have no trt-runtime")?));
        let sms: BTreeSet<String> = gpus.iter().map(|g| g.sm()).collect();
        let mut builders = BTreeSet::new();
        for sm in sms {
            let name = if pack(&format!("trt-{sm}")).is_some() { format!("trt-{sm}") } else { "trt-ptx".into() };
            builders.insert(name);
        }
        for b in builders {
            out.push(runtime(pack(&b).ok_or_else(|| format!("the runtime packs have no {b}"))?));
        }
        if cfg!(windows) && gpus.iter().any(trtrtx_default) {
            out.push(Component {
                name: "trt-rtx".into(),
                version: AJI_VERSION.into(),
                url: format!("https://github.com/{AJI_REPO}/releases/download/{AJI_VERSION}/{TRTRTX_ASSET}"),
                asset: TRTRTX_ASSET.into(),
                sha256: TRTRTX_SHA256.into(),
                bytes: TRTRTX_BYTES,
                format: Format::Zip,
                models: false,
                files: Vec::new(),
            });
        }
    } else if !cfg!(windows) {
        return Err(no_trt_gpu_on_linux(all_gpus));
    }
    if !opts.no_models {
        let bf16 = gpus.iter().any(|g| g.major >= 8);
        let variant = if bf16 { "models-illustrationjanai-bf16" } else { "models-illustrationjanai-fp32" };
        for (name, tag, asset, sha, bytes) in MODELS {
            // One DAT2/HAT variant per machine.
            if (name.ends_with("-bf16") || name.ends_with("-fp32")) && *name != variant {
                continue;
            }
            out.push(Component {
                name: name.to_string(),
                version: tag.to_string(),
                url: format!("https://github.com/{MODELS_REPO}/releases/download/{tag}/{asset}"),
                asset: asset.to_string(),
                sha256: sha.to_string(),
                bytes: *bytes,
                format: Format::SevenZ,
                models: true,
                files: Vec::new(),
            });
        }
    }
    Ok(out)
}

fn load_state(dir: &Path) -> State {
    std::fs::read(dir.join(STATE_FILE)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
}

fn save_state(dir: &Path, s: &State) -> Result<(), String> {
    let p = dir.join(STATE_FILE);
    std::fs::write(&p, serde_json::to_vec_pretty(s).unwrap_or_default()).map_err(|e| format!("{}: {e}", p.display()))
}

fn rel_path(c: &Component, f: &str) -> String {
    if c.models { format!("models/{f}") } else { f.to_string() }
}

/// Components installed from an older archive whose files are exactly the new archive's (a new
/// runtime release that repacked the same files: runtime-1 to runtime-2) are recorded as the new
/// one instead of downloaded again.
fn refresh_unchanged(dir: &Path, wanted: &[Component], state: &mut State) -> Result<(), String> {
    let mut changed = false;
    for c in wanted.iter().filter(|c| !c.files.is_empty()) {
        let Some(i) = state.components.get(&c.name) else { continue };
        if i.sha256 == c.sha256 {
            continue;
        }
        let mut have: Vec<&str> = i.files.iter().map(String::as_str).collect();
        let mut want: Vec<String> = c.files.iter().map(|f| rel_path(c, &f.path)).collect();
        have.sort_unstable();
        want.sort();
        if have != want {
            continue;
        }
        let same = c.files.iter().all(|f| {
            let p = dir.join(rel_path(c, &f.path));
            std::fs::metadata(&p).is_ok_and(|m| m.len() == f.bytes) && sha256_file(&p).is_ok_and(|h| h == f.sha256)
        });
        if same {
            let i = state.components.get_mut(&c.name).expect("checked above");
            i.sha256 = c.sha256.clone();
            i.version = c.version.clone();
            changed = true;
        }
    }
    if changed {
        save_state(dir, state)?;
    }
    Ok(())
}

fn needed(dir: &Path, c: &Component, state: &State) -> bool {
    match state.components.get(&c.name) {
        Some(i) => i.sha256 != c.sha256 || i.files.iter().any(|f| !dir.join(f).exists()),
        None => true,
    }
}

fn print_gpus(gpus: &[nvml::Gpu]) {
    if gpus.is_empty() {
        eprintln!("GPU: no NVIDIA GPU found{}", if cfg!(windows) { "; using DirectML (AMD/Intel/other GPUs)" } else { "" });
    } else {
        for g in gpus {
            let old = if g.trt_capable() { "" } else { "; too old for TensorRT" };
            eprintln!("GPU: {} (compute {}.{}{old})", g.name, g.major, g.minor);
        }
        if cfg!(windows) && !gpus.iter().any(nvml::Gpu::trt_capable) {
            eprintln!("{}; using DirectML", nvml::trt_needs());
        }
    }
}

/// Linux, where TensorRT is the only backend, without a GPU it runs on.
fn no_trt_gpu_on_linux(gpus: &[nvml::Gpu]) -> String {
    if gpus.is_empty() {
        "no NVIDIA GPU found. On Linux mangajanai runs on TensorRT, which needs an NVIDIA GPU and driver.".into()
    } else {
        let found: Vec<String> = gpus.iter().map(|g| format!("{} is compute {}.{}", g.name, g.major, g.minor)).collect();
        format!("{}; {}. On Linux mangajanai runs on TensorRT only, so it can't upscale on this PC.", nvml::trt_needs(), found.join(", "))
    }
}

/// Run setup. Returns the exit code.
pub fn run(opts: &Options) -> Result<i32, String> {
    let dir = install_dir();
    // a store install stays one when setup runs again without the flags
    let store = opts.store.clone().or_else(|| load_state(&dir).store);
    if store.is_some() {
        // a moved store whose folder is missing (an unplugged drive): say so, rather than fail to
        // open it or install into it
        crate::store::check_available()?;
    }
    if opts.remove {
        return remove(&dir, store, &Lines(opts.lines));
    }
    if let Some(mode) = store {
        return run_store(opts, &dir, &mode);
    }
    let lines = Lines(opts.lines);
    let gpus = nvml::gpus();
    print_gpus(&gpus);
    let wanted = plan(opts, &gpus)?;
    let mut state = load_state(&dir);
    refresh_unchanged(&dir, &wanted, &mut state)?;
    let needed = |c: &Component, state: &State| needed(&dir, c, state);

    eprintln!("\nInstall folder: {}", dir.display());
    let mut total = 0u64;
    for c in &wanted {
        let todo = needed(c, &state);
        if todo {
            total += c.bytes;
        }
        eprintln!("  {:<32} {:<26} {:>8}  {}", c.name, c.version, mb(c.bytes), if todo { "to download" } else { "installed" });
        lines.say(format_args!("SETUP_PLAN {} {} {}", c.name, c.bytes, if todo { "download" } else { "installed" }));
    }
    lines.say(format_args!("SETUP_TOTAL {total}"));
    let todo: Vec<&Component> = wanted.iter().filter(|c| needed(c, &state)).collect();
    if todo.is_empty() {
        eprintln!("\nEverything is installed.");
        lines.say(format_args!("SETUP_DONE"));
        return Ok(0);
    }
    eprintln!("\nTo download: {}", mb(total));
    if opts.check {
        return Ok(0);
    }
    install_components(&dir, &todo, &mut state, opts, &lines)?;
    eprintln!("\nSetup complete. Try: mangajanai devices, then mangajanai init");
    lines.say(format_args!("SETUP_DONE"));
    Ok(0)
}

/// Download, check and extract private components next to the program, recording them in
/// setup.json.
fn install_components(dir: &Path, todo: &[&Component], state: &mut State, opts: &Options, lines: &Lines) -> Result<(), String> {
    if todo.is_empty() {
        return Ok(());
    }
    let dir = dir.to_path_buf();
    let dl = dir.join(DOWNLOAD_DIR);
    std::fs::create_dir_all(&dl).map_err(|e| {
        format!("can't write next to the program ({}): {e}. Move mangajanai to a folder you can write to and run setup again.", dir.display())
    })?;
    for &c in todo {
        let archive = dl.join(&c.asset);
        let have = archive.is_file() && sha256_file(&archive)? == c.sha256;
        if !have {
            match &opts.mirror {
                Some(m) => {
                    eprintln!("  {}: copying from {}", c.name, m.display());
                    std::fs::copy(m.join(&c.asset), &archive).map_err(|e| format!("{}: {e}", m.join(&c.asset).display()))?;
                    lines.say(format_args!("SETUP_DOWNLOAD {} {} {}", c.name, c.bytes, c.bytes));
                }
                None => download(&c.url, &archive, &c.name, c.bytes, lines)?,
            }
            let got = sha256_file(&archive)?;
            if got != c.sha256 {
                let _ = std::fs::remove_file(&archive);
                return Err(format!("{} failed its SHA-256 check (got {got}); run setup again to retry", c.asset));
            }
        }
        eprintln!("  {}: extracting", c.name);
        lines.say(format_args!("SETUP_EXTRACT {}", c.name));
        let dest = if c.models { dir.join("models") } else { dir.clone() };
        let files: Vec<String> = extract(&archive, c.format, &dest)?.iter().map(|f| rel_path(c, f)).collect();
        // Files the previous version of this component had and the new one doesn't.
        if let Some(old) = state.components.get(&c.name) {
            for f in old.files.iter().filter(|f| !files.contains(f)) {
                let _ = std::fs::remove_file(dir.join(f));
            }
        }
        state.components.insert(c.name.clone(), Installed { version: c.version.clone(), sha256: c.sha256.clone(), files });
        save_state(&dir, state)?;
        let _ = std::fs::remove_file(&archive);
    }
    let _ = std::fs::remove_dir(&dl);
    Ok(())
}

fn store_open() -> Result<aji_runtime::Store, String> {
    let root = aji_runtime::Store::default_root().map_err(|e| e.to_string())?;
    aji_runtime::Store::open(&root).map_err(|e| e.to_string())
}

/// Store mode's index: the shared runtime release's, checked against its pin (from `--mirror`:
/// unchecked, as for the private packs).
fn store_index(opts: &Options) -> Result<aji_runtime::Index, String> {
    let file = format!("packs-{}.json", platform());
    match &opts.mirror {
        Some(dir) => {
            let b = std::fs::read(dir.join(&file)).map_err(|e| format!("{}: {e}", dir.join(&file).display()))?;
            aji_runtime::Index::parse(&b, None).map_err(|e| e.to_string())
        }
        None => {
            let pin = if cfg!(windows) { RUNTIME_INDEX_SHA256_WINDOWS } else { RUNTIME_INDEX_SHA256_LINUX };
            if pin.is_empty() {
                return Err(format!("this build of mangajanai has no shared runtime pinned ({RUNTIME_TAG} isn't published yet)"));
            }
            let url = format!("https://github.com/{RUNTIME_REPO}/releases/download/{RUNTIME_TAG}/{file}");
            aji_runtime::Index::load(&url, Some(pin)).map_err(|e| e.to_string())
        }
    }
}

/// What store mode installs, as `aji_runtime::Index::select` wants: TensorRT with this machine's
/// builder packs, TensorRT-RTX where it's the default (if this folder has its libaji backend),
/// and the models, chosen as in [`plan`].
fn store_wants(dir: &Path, gpus: &[nvml::Gpu], no_models: bool) -> Result<Vec<String>, String> {
    let (all_gpus, gpus) = (gpus, &nvml::trt_gpus(gpus));
    let mut w = Vec::new();
    if !gpus.is_empty() {
        w.push("trt".to_string());
        if cfg!(windows) && gpus.iter().any(trtrtx_default) && dir.join("aji_trtrtx.dll").is_file() {
            w.push("trtrtx".to_string());
        }
    } else if !cfg!(windows) {
        return Err(no_trt_gpu_on_linux(all_gpus));
    }
    if !no_models {
        let bf16 = gpus.iter().any(|g| g.major >= 8);
        w.push("models-mangajanai".into());
        w.push("models-illustrationjanai".into());
        w.push(if bf16 { "models-illustrationjanai-bf16" } else { "models-illustrationjanai-fp32" }.into());
    }
    Ok(w)
}

fn is_libaji(f: &str) -> bool {
    let n = f.rsplit('/').next().unwrap_or(f).to_ascii_lowercase();
    (n.starts_with("aji") && n.ends_with(".dll")) || n.starts_with("libaji")
}

/// After the store has an app's packs: remove this folder's private copies of them, i.e. the
/// components setup installed here that the store now provides, and untracked files identical to
/// a store file (an install from before setup.json). libaji's own files always stay.
fn remove_private_copies(dir: &Path, state: &mut State, packs: &[(aji_runtime::IndexPack, PathBuf)]) -> Vec<String> {
    let has = |name: &str| packs.iter().any(|(p, _)| p.name == name);
    let has_kind = |kind: &str| packs.iter().any(|(p, _)| p.kind == kind);
    let covered = |c: &str| match c {
        "trt-rtx" => has_kind("trtrtx"),
        c if c.starts_with("trt-") => has_kind("trt"),
        c => has(c),
    };
    let mut removed = Vec::new();
    for name in state.components.keys().filter(|k| covered(k)).cloned().collect::<Vec<_>>() {
        if let Some(c) = state.components.remove(&name) {
            for f in c.files.iter().filter(|f| !is_libaji(f)) {
                if std::fs::remove_file(dir.join(f)).is_ok() {
                    removed.push(f.clone());
                }
            }
        }
    }
    for (p, _) in packs {
        for f in &p.files {
            for (rel, cand) in [(f.path.clone(), dir.join(&f.path)), (format!("models/{}", f.path), dir.join("models").join(&f.path))] {
                let same = std::fs::metadata(&cand).is_ok_and(|m| m.is_file() && m.len() == f.bytes)
                    && sha256_file(&cand).is_ok_and(|h| h == f.sha256);
                if same && !is_libaji(&rel) && std::fs::remove_file(&cand).is_ok() {
                    removed.push(rel);
                }
            }
        }
    }
    let _ = std::fs::remove_dir(dir.join("models"));
    removed
}

/// `setup` in store mode. The SETUP_* lines are the same as in private mode; a pack taken over from
/// this folder counts as installed.
fn run_store(opts: &Options, dir: &Path, mode: &StoreMode) -> Result<i32, String> {
    let lines = Lines(opts.lines);
    let gpus = nvml::gpus();
    print_gpus(&gpus);
    let index = store_index(opts)?;
    let wants = store_wants(dir, &gpus, opts.no_models)?;
    // only the GPUs TensorRT runs on get builder packs
    let rgpus: Vec<aji_runtime::nvml::Gpu> = nvml::trt_gpus(&gpus)
        .into_iter()
        .map(|g| aji_runtime::nvml::Gpu { name: g.name, major: g.major, minor: g.minor })
        .collect();
    let packs = index.select(&wants, &rgpus).map_err(|e| e.to_string())?;
    // ONNX Runtime + DirectML: private, as before
    let mut private = Vec::new();
    if cfg!(windows) {
        let p = index.pack("dml-runtime").ok_or("the shared runtime has no dml-runtime")?;
        private.push(Component {
            name: p.name.clone(),
            version: p.version.clone(),
            url: p.url.clone(),
            asset: p.asset.clone(),
            sha256: p.sha256.clone(),
            bytes: p.bytes,
            format: Format::SevenZ,
            models: false,
            files: p.files.iter().map(|f| PackFile { path: f.path.clone(), sha256: f.sha256.clone(), bytes: f.bytes }).collect(),
        });
    }
    let mut state = load_state(dir);
    refresh_unchanged(dir, &private, &mut state)?;
    let store = store_open()?;
    eprintln!("\nInstall folder: {}", dir.display());
    eprintln!("Shared pack store: {} (as {})", store.root().display(), mode.app_id);
    let private_todo: Vec<&Component> = private.iter().filter(|c| needed(dir, c, &state)).collect();
    for c in &private {
        let todo = private_todo.iter().any(|t| t.name == c.name);
        eprintln!("  {:<32} {:<26} {:>8}  {}", c.name, c.version, mb(c.bytes), if todo { "to download" } else { "installed" });
        lines.say(format_args!("SETUP_PLAN {} {} {}", c.name, c.bytes, if todo { "download" } else { "installed" }));
    }
    let private_bytes: u64 = private_todo.iter().map(|c| c.bytes).sum();
    let ensure = aji_runtime::EnsureOptions {
        app_id: mode.app_id.clone(),
        app_version: env!("CARGO_PKG_VERSION").into(),
        app_path: mode.app_path.clone(),
        packs,
        mirror: opts.mirror.clone(),
        adopt_from: vec![dir.to_path_buf()],
        aji_dir: Some(dir.to_path_buf()),
        check: opts.check,
    };
    let mut total = 0u64;
    let mut shown = Shown::new();
    let r = aji_runtime::ensure(&store, &ensure, &mut |ev| match ev {
        aji_runtime::Event::Waiting => eprintln!("  (waiting for another installer to finish with the pack store)"),
        aji_runtime::Event::Plan { pack, state } => {
            let (human, line) = match state {
                aji_runtime::PlanState::Installed => ("in the pack store", "installed"),
                aji_runtime::PlanState::Adopted => ("moved to the pack store", "installed"),
                aji_runtime::PlanState::Download => ("to download", "download"),
            };
            eprintln!("  {:<32} {:<26} {:>8}  {human}", pack.name, pack.version, mb(pack.bytes));
            lines.say(format_args!("SETUP_PLAN {} {} {line}", pack.name, pack.bytes));
        }
        aji_runtime::Event::Total(t) => {
            total = t + private_bytes;
            lines.say(format_args!("SETUP_TOTAL {total}"));
        }
        aji_runtime::Event::Download { pack, got, total } => {
            lines.say(format_args!("SETUP_DOWNLOAD {} {got} {total}", pack.name));
            shown.show(&pack.name, got, total);
        }
        aji_runtime::Event::Extract { pack } => {
            shown.end();
            eprintln!("  {}: extracting", pack.name);
            lines.say(format_args!("SETUP_EXTRACT {}", pack.name));
        }
        aji_runtime::Event::Done => {}
    })
    .map_err(|e| e.to_string())?;
    if opts.check {
        if r.complete && private_todo.is_empty() {
            eprintln!("\nEverything is installed.");
            lines.say(format_args!("SETUP_DONE"));
        } else {
            eprintln!("\nTo download: {}", mb(total));
        }
        return Ok(0);
    }
    install_components(dir, &private_todo, &mut state, opts, &lines)?;
    let removed = remove_private_copies(dir, &mut state, &r.packs);
    if !removed.is_empty() {
        eprintln!("  removed {} file(s) from this folder that the pack store now holds", removed.len());
    }
    state.store = Some(mode.clone());
    save_state(dir, &state)?;
    for k in &r.gc.deleted {
        eprintln!("  pack store: removed {k} (no installed app uses it)");
    }
    eprintln!("\nSetup complete. Try: mangajanai devices, then mangajanai init");
    lines.say(format_args!("SETUP_DONE"));
    Ok(0)
}

/// `setup --remove`: drop this app's ref from the pack store (its uninstaller runs this before
/// deleting the folder), freeing the packs no other app uses. Nothing to do for a private install.
fn remove(dir: &Path, store: Option<StoreMode>, lines: &Lines) -> Result<i32, String> {
    let Some(mode) = store else {
        eprintln!("Not a pack store install ({}): nothing to remove from the store.", dir.display());
        lines.say(format_args!("SETUP_DONE"));
        return Ok(0);
    };
    let s = store_open()?;
    let (_, gc) = aji_runtime::unref(&s, &mode.app_id, &mut || eprintln!("  (waiting for another installer to finish with the pack store)"))
        .map_err(|e| e.to_string())?;
    for k in &gc.deleted {
        eprintln!("  pack store: removed {k} (no installed app uses it)");
    }
    for k in &gc.in_use {
        eprintln!("  pack store: kept {k} for now, it's in use");
    }
    let _ = std::fs::remove_file(dir.join("aji-runtime.json"));
    let mut state = load_state(dir);
    state.store = None;
    save_state(dir, &state)?;
    eprintln!("Removed {} from the pack store.", mode.app_id);
    lines.say(format_args!("SETUP_DONE"));
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_packs_download_from_the_index_url() {
        // a pack unchanged since an earlier runtime release keeps that release's archive
        let p: Pack = serde_json::from_str(
            r#"{"name":"dml-runtime","version":"1","asset":"dml-runtime-windows.7z","sha256":"x","bytes":1,
                "url":"https://github.com/the-database/animejanai-inference/releases/download/runtime-2/dml-runtime-windows.7z"}"#,
        )
        .unwrap();
        assert_eq!(pack_url(&p), "https://github.com/the-database/animejanai-inference/releases/download/runtime-2/dml-runtime-windows.7z");
        // an index without urls: the pinned release
        let p: Pack = serde_json::from_str(r#"{"name":"a","version":"1","asset":"a.7z","sha256":"x","bytes":1}"#).unwrap();
        assert_eq!(pack_url(&p), format!("https://github.com/{RUNTIME_REPO}/releases/download/{RUNTIME_TAG}/a.7z"));
    }

    fn pack(name: &str, kind: &str, files: &[(&str, &[u8])]) -> (aji_runtime::IndexPack, PathBuf) {
        let p = aji_runtime::IndexPack {
            name: name.into(),
            version: "1".into(),
            kind: kind.into(),
            sm: None,
            asset: format!("{name}.7z"),
            url: String::new(),
            sha256: String::new(),
            bytes: 0,
            installed_bytes: 0,
            files: files
                .iter()
                .map(|(n, d)| aji_runtime::PackFile { path: n.to_string(), sha256: sha256_bytes(d), bytes: d.len() as u64 })
                .collect(),
        };
        (p, PathBuf::new())
    }

    fn installed(files: &[&str]) -> Installed {
        Installed { version: "1".into(), sha256: "x".into(), files: files.iter().map(|f| f.to_string()).collect() }
    }

    #[test]
    fn setup_json_without_store_still_reads_and_round_trips() {
        let old: State = serde_json::from_str(r#"{"components":{"trt-runtime":{"version":"v","sha256":"s","files":["a.dll"]}}}"#).unwrap();
        assert!(old.store.is_none());
        assert!(!serde_json::to_string(&old).unwrap().contains("store"), "private installs write what they always did");
        let mut s = old;
        s.store = Some(StoreMode { app_id: "yosh-addon".into(), app_path: "C:/yosh/yosh.exe".into() });
        let back: State = serde_json::from_slice(&serde_json::to_vec(&s).unwrap()).unwrap();
        assert_eq!(back.store, s.store);
    }

    #[test]
    fn libaji_files_are_recognised() {
        for f in ["aji.dll", "aji_trtrtx.dll", "AJI_TRT.DLL", "libaji.so", "libaji_trt.so", "x/aji_dml.dll"] {
            assert!(is_libaji(f), "{f}");
        }
        for f in ["nvinfer_11.dll", "tensorrt_rtx_1_7.dll", "models/aji.onnx", "cudart64_13.dll"] {
            assert!(!is_libaji(f), "{f}");
        }
    }

    #[test]
    fn private_copies_go_once_the_store_has_them() {
        let dir = std::env::temp_dir().join(format!("mangajanai-store-cleanup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models")).unwrap();
        let w = |f: &str, d: &[u8]| std::fs::write(dir.join(f), d).unwrap();
        w("nvinfer_11.dll", b"nv");
        w("cudart64_13.dll", b"cu");
        w("nvinfer_builder_resource_sm86_11.dll", b"sm86");
        w("aji_trtrtx.dll", b"backend");
        w("tensorrt_rtx_1_7.dll", b"rtx");
        w("onnxruntime.dll", b"ort");
        w("models/a.onnx", b"A");
        w("models/b.onnx", b"B"); // untracked, identical to the store's
        w("models/c.onnx", b"my own c"); // untracked, same name as a store file but different
        w("models/mine.onnx", b"user model");
        let mut state = State::default();
        state.components.insert("trt-runtime".into(), installed(&["nvinfer_11.dll", "cudart64_13.dll"]));
        state.components.insert("trt-sm86".into(), installed(&["nvinfer_builder_resource_sm86_11.dll"]));
        state.components.insert("trt-rtx".into(), installed(&["aji_trtrtx.dll", "tensorrt_rtx_1_7.dll"]));
        state.components.insert("dml-runtime".into(), installed(&["onnxruntime.dll"]));
        state.components.insert("models-mangajanai".into(), installed(&["models/a.onnx"]));
        let packs = vec![
            pack("trt-runtime", "trt", &[("nvinfer_11.dll", b"nv")]),
            pack("trtrtx-runtime", "trtrtx", &[("tensorrt_rtx_1_7.dll", b"rtx")]),
            pack("models-mangajanai", "models", &[("a.onnx", b"A"), ("b.onnx", b"B"), ("c.onnx", b"C")]),
        ];
        let mut removed = remove_private_copies(&dir, &mut state, &packs);
        removed.sort();
        assert_eq!(
            removed,
            ["cudart64_13.dll", "models/a.onnx", "models/b.onnx", "nvinfer_11.dll", "nvinfer_builder_resource_sm86_11.dll", "tensorrt_rtx_1_7.dll"]
        );
        for kept in ["aji_trtrtx.dll", "onnxruntime.dll", "models/c.onnx", "models/mine.onnx"] {
            assert!(dir.join(kept).is_file(), "{kept} must stay");
        }
        assert_eq!(state.components.keys().collect::<Vec<_>>(), ["dml-runtime"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn components_the_store_does_not_cover_stay() {
        let dir = std::env::temp_dir().join(format!("mangajanai-store-keep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("tensorrt_rtx_1_7.dll"), b"rtx").unwrap();
        let mut state = State::default();
        state.components.insert("trt-rtx".into(), installed(&["aji_trtrtx.dll", "tensorrt_rtx_1_7.dll"]));
        state.components.insert("models-illustrationjanai-fp32".into(), installed(&["models/x.onnx"]));
        // no TensorRT-RTX and no fp32 models in the store packs
        let packs = vec![pack("trt-runtime", "trt", &[("nvinfer_11.dll", b"nv")])];
        assert!(remove_private_copies(&dir, &mut state, &packs).is_empty());
        assert!(dir.join("tensorrt_rtx_1_7.dll").is_file());
        assert_eq!(state.components.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_repacked_archive_with_the_same_files_is_not_downloaded_again() {
        let dir = std::env::temp_dir().join(format!("mangajanai-repacked-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("nvinfer_11.dll"), b"nv").unwrap();
        std::fs::write(dir.join("trtexec.exe"), b"te").unwrap();
        let comp = |sha: &str, files: &[(&str, &[u8])]| Component {
            name: "trt-runtime".into(),
            version: "11.4.0.106-cuda13.4".into(),
            url: String::new(),
            asset: "trt-runtime-windows-x64.7z".into(),
            sha256: sha.into(),
            bytes: 1,
            format: Format::SevenZ,
            models: false,
            files: files.iter().map(|(p, d)| PackFile { path: p.to_string(), sha256: sha256_bytes(d), bytes: d.len() as u64 }).collect(),
        };
        let mut state = State::default();
        state.components.insert("trt-runtime".into(), installed(&["nvinfer_11.dll", "trtexec.exe"]));
        // same files in a new archive: recorded as the new one
        let same = comp("new", &[("nvinfer_11.dll", b"nv"), ("trtexec.exe", b"te")]);
        refresh_unchanged(&dir, std::slice::from_ref(&same), &mut state).unwrap();
        assert!(!needed(&dir, &same, &state));
        assert_eq!(state.components["trt-runtime"].version, "11.4.0.106-cuda13.4");
        // a file that changed: downloaded
        let changed = comp("newer", &[("nvinfer_11.dll", b"nv 2"), ("trtexec.exe", b"te")]);
        refresh_unchanged(&dir, std::slice::from_ref(&changed), &mut state).unwrap();
        assert!(needed(&dir, &changed, &state));
        // a file more or less: downloaded
        let more = comp("more", &[("nvinfer_11.dll", b"nv"), ("trtexec.exe", b"te"), ("x.dll", b"x")]);
        refresh_unchanged(&dir, std::slice::from_ref(&more), &mut state).unwrap();
        assert!(needed(&dir, &more, &state));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_wants_follow_the_gpus() {
        let dir = std::env::temp_dir();
        let g = |major, minor| nvml::Gpu { name: "g".into(), major, minor };
        let w = store_wants(&dir, &[g(7, 5)], false).unwrap();
        assert_eq!(w[0], "trt");
        assert_eq!(w[w.len() - 3..], ["models-mangajanai", "models-illustrationjanai", "models-illustrationjanai-fp32"]);
        let w = store_wants(&dir, &[g(12, 0)], true).unwrap();
        assert_eq!(w[0], "trt");
        assert!(!w.iter().any(|x| x.starts_with("models-")));
        if cfg!(windows) {
            assert_eq!(store_wants(&dir, &[], true).unwrap(), Vec::<String>::new(), "DirectML only");
        } else {
            assert!(store_wants(&dir, &[], true).is_err());
        }
    }

    // ---- GPUs too old for TensorRT (compute capability below 7.5) ----------------------------

    fn gpu(name: &str, major: i32, minor: i32) -> nvml::Gpu {
        nvml::Gpu { name: name.into(), major, minor }
    }

    #[test]
    fn trt_threshold_is_turing() {
        assert_eq!(nvml::TRT_MIN_COMPUTE, (7, 5));
        for (major, minor) in [(5, 2), (6, 1), (7, 0), (7, 2)] {
            assert!(!nvml::trt_capable(major, minor), "{major}.{minor}");
        }
        for (major, minor) in [(7, 5), (8, 6), (12, 0), (13, 0)] {
            assert!(nvml::trt_capable(major, minor), "{major}.{minor}");
        }
        let all = [gpu("GTX 1080", 6, 1), gpu("RTX 3080", 8, 6), gpu("Titan V", 7, 0)];
        assert_eq!(nvml::trt_gpus(&all).iter().map(|g| g.name.as_str()).collect::<Vec<_>>(), ["RTX 3080"]);
    }

    #[test]
    fn plan_treats_pre_turing_gpus_as_no_nvidia_gpu() {
        let dir = std::env::temp_dir().join(format!("mangajanai-setup-preturing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let packs: Vec<_> = ["dml-runtime", "trt-runtime", "trt-sm75", "trt-sm86", "trt-ptx"]
            .iter()
            .map(|n| serde_json::json!({"name": n, "version": "1", "asset": format!("{n}.7z"), "sha256": "x", "bytes": 1}))
            .collect();
        std::fs::write(dir.join(format!("packs-{}.json", platform())), serde_json::json!({ "packs": packs }).to_string()).unwrap();
        let opts = Options { no_models: true, check: true, mirror: Some(dir.clone()), lines: false, store: None, remove: false };
        let names = |gpus: &[nvml::Gpu]| plan(&opts, gpus).map(|v| v.into_iter().map(|c| c.name).filter(|n| n != "trt-rtx").collect::<Vec<_>>());
        let dml: &[&str] = if cfg!(windows) { &["dml-runtime"] } else { &[] };

        // a GTX 1080 next to an RTX 3080: only the 3080 gets TensorRT's builder pack, no PTX
        let mut want = dml.to_vec();
        want.extend(["trt-runtime", "trt-sm86"]);
        assert_eq!(names(&[gpu("GTX 1080", 6, 1), gpu("RTX 3080", 8, 6)]).unwrap(), want);
        // a future generation without its own pack still gets PTX
        let mut want = dml.to_vec();
        want.extend(["trt-runtime", "trt-ptx"]);
        assert_eq!(names(&[gpu("next", 13, 0)]).unwrap(), want);
        // only old GPUs: no TensorRT at all (DirectML on Windows; on Linux, an error saying why)
        for old in [vec![gpu("GTX 1080", 6, 1)], vec![gpu("Titan V", 7, 0), gpu("GTX 1060", 6, 1)]] {
            if cfg!(windows) {
                assert_eq!(names(&old).unwrap(), dml);
            } else {
                let e = names(&old).unwrap_err();
                assert!(e.starts_with("TensorRT needs an NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability 7.5+)"), "{e}");
                assert!(e.contains(&format!("{} is compute {}.{}", old[0].name, old[0].major, old[0].minor)), "{e}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_wants_treat_pre_turing_gpus_as_no_nvidia_gpu() {
        let dir = std::env::temp_dir();
        // mixed: TensorRT for the new GPU (aji-runtime then picks its builder pack)
        let w = store_wants(&dir, &[gpu("GTX 1080", 6, 1), gpu("RTX 3080", 8, 6)], true).unwrap();
        assert_eq!(w[0], "trt");
        // only old: as without an NVIDIA GPU, with the fp32 model variant
        let old = [gpu("GTX 1080", 6, 1)];
        if cfg!(windows) {
            assert_eq!(store_wants(&dir, &old, true).unwrap(), Vec::<String>::new());
            assert_eq!(store_wants(&dir, &old, false).unwrap().last().map(String::as_str), Some("models-illustrationjanai-fp32"));
        } else {
            let e = store_wants(&dir, &old, true).unwrap_err();
            assert!(e.contains("compute capability 7.5+") && e.contains("on TensorRT only"), "{e}");
        }
    }
}
