//! `mangajanai setup`: download what this machine needs to upscale, next to the program.
//!
//! What's installed follows MangaJaNai GUI's updater (its RecommendedPacks):
//! - Windows: ONNX Runtime + DirectML (always: AMD/Intel GPUs, and Auto's fallback).
//! - An NVIDIA GPU: the TensorRT runtime, the builder resources for each GPU generation present (or
//!   the PTX fallback for one without its own), and on Windows TensorRT-RTX on the GPUs it's the
//!   default for (RTX 20 series and newer).
//! - Models: MangaJaNai and IllustrationJaNai, plus the DAT2/HAT variant this GPU runs best (bf16 on
//!   NVIDIA compute capability 8.0+, fp32 otherwise).
//!
//! Sources, each checked against a SHA-256 compiled in here: the runtime packs from this repo's
//! `runtime-<n>` release (scripts/runtime_packs.py), TensorRT-RTX from libaji's release, the models
//! from the MangaJaNai model releases. libaji itself ships in the program's archive.
//!
//! Runtime files go next to the program, models into `models/` beside it, and `setup.json` records
//! what was installed (by archive SHA-256), so a re-run fetches only what's missing or changed.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::nvml;

const RUNTIME_REPO: &str = "the-database/MangaJaNaiCLI";
/// The runtime packs' release (scripts/runtime_packs.py TAG) and the SHA-256 of its
/// packs-<platform>.json, which pins every pack. Empty: this build has no published packs to use
/// (development builds use `--mirror`).
const RUNTIME_TAG: &str = "runtime-1";
const RUNTIME_INDEX_SHA256_WINDOWS: &str = "25c350b9b507a69ed05a300f1d894d8c78e39160a23ff1b45b6a2348bdf60e26";
const RUNTIME_INDEX_SHA256_LINUX: &str = "a00a6aa8b9ce794d2d6917bb4cea5bf6759583aa959442fa032d2ecda4b76e60";

/// TensorRT-RTX: aji_trtrtx.dll + the TensorRT for RTX runtime, from the libaji release this
/// program's archive bundles.
const AJI_REPO: &str = "the-database/animejanai-inference";
const AJI_VERSION: &str = "v0.21.0";
const TRTRTX_ASSET: &str = "aji-trtrtx-windows-x64.zip";
const TRTRTX_SHA256: &str = "213204243f947affebbe32db99379a4fea74cc1ab50657370dfff26fb72df0de";
const TRTRTX_BYTES: u64 = 95_346_023;

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
}

#[derive(Serialize, Deserialize, Default)]
struct State {
    components: BTreeMap<String, Installed>,
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
    probe.iter().any(|f| dir.join(f).is_file())
}

/// TensorRT-RTX's default GPUs, as MangaJaNai GUI and yosh decide it: Turing (7.5), consumer
/// Ampere (8.6), Ada (8.9) and Blackwell (12.x).
fn trtrtx_default(g: &nvml::Gpu) -> bool {
    matches!((g.major, g.minor), (7, 5) | (8, 6) | (8, 9)) || g.major >= 12
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
    Ok(format!("{:x}", h.finalize()))
}

fn sha256_bytes(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}

/// Download `url` to `dest`, showing progress on stderr (and as `SETUP_DOWNLOAD` lines).
fn download(url: &str, dest: &Path, label: &str, expected: u64, lines: &Lines) -> Result<(), String> {
    let resp = ureq::get(url).call().map_err(|e| format!("downloading {url}: {e}"))?;
    let total = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(expected);
    let mut reader = resp.into_body().into_reader();
    let part = dest.with_extension("part");
    let mut out = std::io::BufWriter::new(std::fs::File::create(&part).map_err(|e| format!("{}: {e}", part.display()))?);
    let tty = std::io::stderr().is_terminal();
    let (mut got, mut shown, mut said) = (0u64, u64::MAX, u64::MAX);
    lines.say(format_args!("SETUP_DOWNLOAD {label} 0 {total}"));
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("downloading {url}: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        got += n as u64;
        if total > 0 {
            let t = total;
            let pct = got * 100 / t;
            if pct != said {
                said = pct;
                lines.say(format_args!("SETUP_DOWNLOAD {label} {got} {t}"));
            }
            let step = if tty { pct } else { pct / 25 * 25 };
            if step != shown {
                shown = step;
                if tty {
                    eprint!("\r  {label}: {pct:>3}%  {} / {}   ", mb(got), mb(t));
                } else {
                    eprintln!("  {label}: {pct}%");
                }
            }
        }
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    if tty {
        eprintln!();
    }
    std::fs::rename(&part, dest).map_err(|e| format!("{}: {e}", dest.display()))
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

/// Fetch a small file (the runtime index) into memory.
fn fetch_small(url: &str) -> Result<Vec<u8>, String> {
    let mut resp = ureq::get(url).call().map_err(|e| format!("downloading {url}: {e}"))?;
    resp.body_mut().read_to_vec().map_err(|e| format!("downloading {url}: {e}"))
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

/// What this machine needs, in install order.
fn plan(opts: &Options, gpus: &[nvml::Gpu]) -> Result<Vec<Component>, String> {
    let packs = runtime_index(opts)?;
    let pack = |name: &str| packs.iter().find(|p| p.name == name).cloned();
    let runtime = |p: Pack| Component {
        url: format!("https://github.com/{RUNTIME_REPO}/releases/download/{RUNTIME_TAG}/{}", p.asset),
        name: p.name,
        version: p.version,
        asset: p.asset,
        sha256: p.sha256,
        bytes: p.bytes,
        format: Format::SevenZ,
        models: false,
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
            });
        }
    } else if !cfg!(windows) {
        return Err("no NVIDIA GPU found. On Linux mangajanai runs on TensorRT, which needs an NVIDIA GPU and driver.".into());
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

/// Run setup. Returns the exit code.
pub fn run(opts: &Options) -> Result<i32, String> {
    let dir = install_dir();
    let lines = Lines(opts.lines);
    let gpus = nvml::gpus();
    if gpus.is_empty() {
        eprintln!("GPU: no NVIDIA GPU found{}", if cfg!(windows) { "; using DirectML (AMD/Intel/other GPUs)" } else { "" });
    } else {
        for g in &gpus {
            eprintln!("GPU: {} (compute {}.{})", g.name, g.major, g.minor);
        }
    }
    let wanted = plan(opts, &gpus)?;
    let mut state = load_state(&dir);
    let needed = |c: &Component, state: &State| match state.components.get(&c.name) {
        Some(i) => i.sha256 != c.sha256 || i.files.iter().any(|f| !dir.join(f).exists()),
        None => true,
    };

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

    let dl = dir.join(DOWNLOAD_DIR);
    std::fs::create_dir_all(&dl).map_err(|e| {
        format!("can't write next to the program ({}): {e}. Move mangajanai to a folder you can write to and run setup again.", dir.display())
    })?;
    for c in todo {
        let archive = dl.join(&c.asset);
        let have = archive.is_file() && sha256_file(&archive)? == c.sha256;
        if !have {
            match &opts.mirror {
                Some(m) => {
                    eprintln!("  {}: copying from {}", c.name, m.display());
                    std::fs::copy(m.join(&c.asset), &archive).map_err(|e| format!("{}: {e}", m.join(&c.asset).display()))?;
                    lines.say(format_args!("SETUP_DOWNLOAD {} {} {}", c.name, c.bytes, c.bytes));
                }
                None => download(&c.url, &archive, &c.name, c.bytes, &lines)?,
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
        save_state(&dir, &state)?;
        let _ = std::fs::remove_file(&archive);
    }
    let _ = std::fs::remove_dir(&dl);
    eprintln!("\nSetup complete. Try: mangajanai devices, then mangajanai init");
    lines.say(format_args!("SETUP_DONE"));
    Ok(0)
}
