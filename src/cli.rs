//! The program: `mangajanai` (the command line) and `mangajanai_upscale` (the same program under
//! the name MangaJaNai GUI runs). `--settings`, `--list-devices` and `--model-info` are the GUI's
//! interface and behave exactly as they always have; everything else is the command line.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::config::{self, File};
use crate::settings::{self, Chain, Format, Target, Workflow};
use crate::upscaler::{self, BackendChoice, Upscaler};
use crate::{process, progress, run};

const CONFIG_NAME: &str = "mangajanai.ini";

fn help(prog: &str) -> String {
    format!(
        r#"{prog} {version}: batch image and manga upscaling with TensorRT (also DirectML and TensorRT-RTX
on Windows), through libaji.

Usage:
  {prog} [inputs...] [options]          upscale (settings from {CONFIG_NAME} and/or flags)
  {prog} init [file]                    write a commented {CONFIG_NAME} to start from
  {prog} devices [--backend b]          list the GPUs (the numbers --device takes)
  {prog} bench <model.onnx> [options]   time a model on a synthetic image
  {prog} setup [--no-models] [--check]  download what this PC needs to upscale (the GPU runtime and
                                        the models), next to the program; run it again to update

With no inputs on the command line, `input` from the settings file is used. The settings file is
-c <file>, else {CONFIG_NAME} in the current folder, else next to the program; flags override it.

Inputs and output:
  <inputs...>               images, archives (zip cbz rar cbr 7z cb7; written as .cbz) or folders
                            (searched recursively; the output keeps their subfolders)
  -o, --output <dir>        where results go (default: out)
  -c, --config <file>       settings file to use        --no-config   ignore any settings file
  --no-images, --no-archives   in folders, skip loose images / archives

Model (one of; default: --preset manga):
  --model <file.onnx>       run every image through this model
  --preset manga            MangaJaNai's default rules (gray pages by height, color pages)
  --workflow <file.mwf>     a workflow exported from MangaJaNai GUI
  -m, --models-dir <dir>    where model files are looked up by name (repeatable)
  --auto-levels, --no-auto-levels   stretch gray pages to full black/white before the model
  --gray-threshold <n>      how colorful a page may be and still count as gray (default 12)

Size (default: the model's own scale):
  --scale <x>  |  --width <px>  |  --height <px>  |  --fit <WxH>  |  --native
  -u, --upscale-factor <x>  same as --scale

Saving:
  --format webp|png|jpg|avif   (default webp)    -q, --quality <0-100>   (default 80)
  --lossless                lossless webp
  --suffix <text>           appended to file names    --filename <template>   %filename% = input name
  --skip-existing           keep outputs that exist (default)    --overwrite   redo them

GPU:
  --backend <b>             auto (default) | tensorrt | directml | tensorrt-rtx
  --device <d>              first (default) | all | 0,1 (numbers from `{prog} devices`)
  --streams <n>             inference threads per GPU (default 1; with libaji's GPU tiling,
                            2 rarely helps, for about twice the VRAM)
  --tile <px>               tile size (default 512; one engine per model and tile)
  --overlap <px>            minimum tile overlap (default: libaji's, 128)
  --even-tiles              TensorRT: split each page into even tiles of at most --tile from
                            one engine for all tile shapes: fewer pixels inferred, often
                            faster (slower per pixel on TensorRT-RTX)
  --cache-dir <dir>         TensorRT engine cache (default: next to each model)
  --aji-dir <dir>           where aji.dll / libaji.so is (default: next to the program, else an
                            installed MangaJaNai GUI's inference folder)

Output:
  -v, --verbose             every file and chain decision    --progress lines   the GUI's line protocol
  -h, --help                -V, --version

MangaJaNai GUI's interface (unchanged):
  --settings <appstate2.json>   run the GUI's selected workflow, reporting in its line protocol
  --list-devices [--backend b]  devices as JSON: [{{"id", "name", "vram", "compute_capability"}}]
  --model-info <file.onnx> [--backend b] [--device <id>]   the model's precision, as JSON
"#,
        version = env!("CARGO_PKG_VERSION")
    )
}

fn exe_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf())).unwrap_or_default()
}

/// Where the command line looks for libaji without `--aji-dir`: next to the program, else (Windows)
/// an installed MangaJaNai GUI's `inference` folder, which holds libaji with the TensorRT,
/// TensorRT-RTX and DirectML runtimes.
fn default_aji_dir() -> PathBuf {
    let lib = if cfg!(windows) { "aji.dll" } else { "libaji.so" };
    let exe = exe_dir();
    // libaji ships next to the program; it's usable there once `mangajanai setup` added the runtime
    // (or a runtime was copied in by hand), else an installed MangaJaNai GUI's is the better pick.
    if exe.join(lib).is_file() && crate::setup::runtime_installed(&exe) {
        return exe;
    }
    if cfg!(windows)
        && let Some(local) = std::env::var_os("LOCALAPPDATA")
    {
        let gui = PathBuf::from(local).join(r"Programs\MangaJaNaiConverterGui\inference");
        if gui.join(lib).is_file() {
            return gui;
        }
    }
    exe
}

fn prog_name() -> String {
    std::env::args_os()
        .next()
        .and_then(|a| Path::new(&a).file_stem().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "mangajanai".into())
}

/// Everything the flags can say; `None` = not given.
#[derive(Default)]
struct Flags {
    // the GUI's interface
    settings: Option<PathBuf>,
    list_devices: bool,
    model_info: Option<PathBuf>,
    // the command line
    config: Option<PathBuf>,
    no_config: bool,
    inputs: Vec<PathBuf>,
    output: Option<PathBuf>,
    model: Option<String>,
    preset: Option<String>,
    workflow: Option<PathBuf>,
    models_dirs: Vec<PathBuf>,
    target: Option<Target>,
    format: Option<Format>,
    quality: Option<u8>,
    lossless: Option<bool>,
    filename: Option<String>,
    suffix: Option<String>,
    skip_existing: Option<bool>,
    images: Option<bool>,
    archives: Option<bool>,
    gray_threshold: Option<i32>,
    auto_levels: Option<bool>,
    backend: Option<BackendChoice>,
    device: Option<String>,
    streams: Option<u32>,
    tile: Option<u32>,
    overlap: Option<u32>,
    even_tiles: Option<bool>,
    cache_dir: Option<PathBuf>,
    aji_dir: Option<PathBuf>,
    verbose: bool,
    lines: bool,
    // bench
    size: Option<(u32, u32)>,
    iters: Option<u32>,
    gray: bool,
    force: bool,
    // setup
    no_models: bool,
    check: bool,
    mirror: Option<PathBuf>,
}

fn num<T: std::str::FromStr>(name: &str, v: &str) -> Result<T, String> {
    v.trim().parse().map_err(|_| format!("{name}: '{v}' is not a valid number"))
}

fn parse_flags(args: Vec<String>) -> Result<Flags, String> {
    let mut f = Flags::default();
    let mut it = args.into_iter();
    while let Some(arg) = it.next() {
        // --key=value is the same as --key value
        let (key, inline) = match arg.split_once('=') {
            Some((k, v)) if k.starts_with("--") => (k.to_string(), Some(v.to_string())),
            _ => (arg.clone(), None),
        };
        let mut inline = inline;
        let mut value = |name: &str| inline.take().or_else(|| it.next()).ok_or_else(|| format!("{name} needs a value"));
        match key.as_str() {
            "--settings" => f.settings = Some(value(&key)?.into()),
            "--list-devices" => f.list_devices = true,
            "--model-info" => f.model_info = Some(value(&key)?.into()),
            "-c" | "--config" => f.config = Some(value(&key)?.into()),
            "--no-config" => f.no_config = true,
            "-f" | "--file-path" | "-d" | "--folder-path" | "-i" | "--input" => f.inputs.push(value(&key)?.into()),
            "-o" | "--output" | "--output-folder-path" => f.output = Some(value(&key)?.into()),
            "--model" => f.model = Some(value(&key)?),
            "--preset" => f.preset = Some(value(&key)?),
            "--workflow" => f.workflow = Some(value(&key)?.into()),
            "-m" | "--models-dir" | "--models-directory" => f.models_dirs.push(value(&key)?.into()),
            "-u" | "--upscale-factor" | "--scale" => {
                let s: f64 = num(&key, &value(&key)?)?;
                if s.is_nan() || s <= 0.0 {
                    return Err(format!("{key} must be above 0"));
                }
                f.target = Some(Target::Scale(s));
            }
            "--width" => f.target = Some(Target::Width(num(&key, &value(&key)?)?)),
            "--height" => f.target = Some(Target::Height(num(&key, &value(&key)?)?)),
            "--fit" => {
                let v = value(&key)?;
                let (w, h) = config::parse_size(&v).ok_or_else(|| format!("--fit must be WIDTHxHEIGHT, not '{v}'"))?;
                f.target = Some(Target::Fit(w, h));
            }
            "--native" => f.target = Some(Target::Native),
            "--format" => f.format = Some(config::parse_format(&value(&key)?)?),
            "-q" | "--quality" => f.quality = Some(num::<u8>(&key, &value(&key)?)?.min(100)),
            "--lossless" => f.lossless = Some(true),
            "--suffix" => f.suffix = Some(value(&key)?),
            "--filename" => f.filename = Some(value(&key)?),
            "--skip-existing" => f.skip_existing = Some(true),
            "--overwrite" => f.skip_existing = Some(false),
            "--no-images" => f.images = Some(false),
            "--no-archives" => f.archives = Some(false),
            "--gray-threshold" => f.gray_threshold = Some(num(&key, &value(&key)?)?),
            "--auto-levels" => f.auto_levels = Some(true),
            "--no-auto-levels" => f.auto_levels = Some(false),
            "--backend" => f.backend = Some(BackendChoice::parse(&value(&key)?)?),
            "--device" | "--devices" => f.device = Some(value(&key)?),
            "--streams" => f.streams = Some(num::<u32>(&key, &value(&key)?)?.clamp(1, 8)),
            "--tile" => f.tile = Some(num(&key, &value(&key)?)?),
            "--overlap" => f.overlap = Some(num(&key, &value(&key)?)?),
            "--even-tiles" => f.even_tiles = Some(true),
            "--fixed-tiles" => f.even_tiles = Some(false),
            "--cache-dir" => f.cache_dir = Some(value(&key)?.into()),
            "--aji-dir" => f.aji_dir = Some(value(&key)?.into()),
            "-v" | "--verbose" => f.verbose = true,
            "--progress" => match value(&key)?.as_str() {
                "lines" => f.lines = true,
                "human" | "bar" => f.lines = false,
                other => return Err(format!("--progress: '{other}' (lines or human)")),
            },
            "--size" => {
                let v = value(&key)?;
                f.size = Some(config::parse_size(&v).ok_or_else(|| format!("--size must be WIDTHxHEIGHT, not '{v}'"))?);
            }
            "--iters" => f.iters = Some(num(&key, &value(&key)?)?),
            "--gray" => f.gray = true,
            "--force" => f.force = true,
            "--no-models" => f.no_models = true,
            "--check" => f.check = true,
            "--mirror" => f.mirror = Some(value(&key)?.into()),
            "-V" | "--version" => {
                println!("{} {}", prog_name(), env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            "-h" | "--help" => {
                print!("{}", help(&prog_name()));
                std::process::exit(0);
            }
            other if other.starts_with('-') && other.len() > 1 => return Err(format!("unknown option '{other}' (see --help)")),
            _ => f.inputs.push(arg.into()),
        }
        if inline.is_some() {
            return Err(format!("{key} takes no value"));
        }
    }
    Ok(f)
}

pub fn main() {
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    let code = match dispatch(args) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            2
        }
    };
    std::process::exit(code);
}

fn dispatch(mut args: Vec<String>) -> Result<i32, String> {
    let sub = args.first().filter(|a| matches!(a.as_str(), "init" | "devices" | "bench" | "setup") && !Path::new(a.as_str()).exists()).cloned();
    if sub.is_some() {
        args.remove(0);
    }
    let flags = parse_flags(args)?;
    match sub.as_deref() {
        Some("init") => return init(&flags),
        Some("setup") => {
            return crate::setup::run(&crate::setup::Options { no_models: flags.no_models, check: flags.check, mirror: flags.mirror.clone() });
        }
        Some("devices") => return devices(&flags),
        Some("bench") => return bench(flags),
        _ => {}
    }
    if flags.list_devices {
        return Ok(gui_list_devices(&flags));
    }
    if let Some(model) = &flags.model_info {
        return Ok(gui_model_info(&flags, model));
    }
    if flags.settings.is_some() {
        return Ok(gui_run(flags));
    }
    upscale(flags)
}

// ---- the GUI's interface (unchanged behavior) ------------------------------------------------

fn gui_device_id(flags: &Flags) -> Result<u64, String> {
    flags.device.as_deref().map_or(Ok(0), |d| d.trim().parse().map_err(|_| "--device must be a device id".to_string()))
}

fn gui_list_devices(flags: &Flags) -> i32 {
    use yosh_image::aji::Backend;
    let backend = match flags.backend {
        Some(BackendChoice::DirectMl) => Backend::DirectMl,
        Some(BackendChoice::TensorRtRtx) => Backend::TensorRtRtx,
        _ => Backend::TensorRt,
    };
    match upscaler::devices_json(&flags.aji_dir.clone().unwrap_or_else(exe_dir), backend) {
        Ok(json) => {
            println!("{json}");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

fn gui_model_info(flags: &Flags, model: &Path) -> i32 {
    let device = match gui_device_id(flags) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    match upscaler::model_info_json(model, &flags.aji_dir.clone().unwrap_or_else(exe_dir), flags.backend, device) {
        Ok(json) => {
            println!("{json}");
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}

/// `--settings appstate2.json`: the GUI's selected workflow, in the GUI's line protocol.
fn gui_run(flags: Flags) -> i32 {
    let path = flags.settings.clone().unwrap();
    let device = match gui_device_id(&flags) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let state = match std::fs::read_to_string(&path).map_err(|e| e.to_string()).and_then(|s| settings::AppState::parse(&s)) {
        Ok(s) => s,
        Err(e) => {
            progress::error(&path.display().to_string(), &e);
            return 1;
        }
    };
    let mut models_dirs = flags.models_dirs.clone();
    if let Some(dir) = state.models_directory.as_deref().filter(|d| !d.is_empty()) {
        models_dirs.push(dir.into());
    }
    let wf = match state.current_workflow() {
        Ok(w) => w.clone(),
        Err(e) => {
            progress::error("settings", &e);
            return 1;
        }
    };
    let input = match wf.selected_tab_index {
        0 => run::Input::File(wf.input_file_path.clone().into()),
        _ => run::Input::Folder { dir: wf.input_folder_path.clone().into(), images: wf.upscale_images, archives: wf.upscale_archives },
    };
    models_dirs.push(exe_dir().join("..").join("models"));

    let output = run::Output {
        dir: wf.output_folder_path.clone().into(),
        filename: if wf.output_filename.is_empty() { "%filename%".into() } else { wf.output_filename.clone() },
        overwrite: wf.overwrite_existing_files,
    };
    let job = process::Job {
        chains: wf.chains.values.clone(),
        threshold: wf.grayscale_detection_threshold,
        target: wf.target(),
        format: wf.format(),
        quality: wf.quality(),
        lossless: wf.use_lossless_compression,
    };

    let start = Instant::now();
    let backend = flags.backend.unwrap_or(BackendChoice::Auto);
    let (aji_dir, cache_dir) = (flags.aji_dir.clone().unwrap_or_else(exe_dir), flags.cache_dir.clone());
    let open: run::OpenUpscaler = Box::new(move || Upscaler::open(&aji_dir, backend, device, cache_dir, models_dirs, None, false));
    run::run(vec![input], output, job, vec![open]);
    progress::finish(start.elapsed());
    0
}

// ---- the command line ------------------------------------------------------------------------

/// The settings file to use, if any: `-c`, else the default name in the current folder, else next
/// to the program.
fn find_config(flags: &Flags) -> Result<Option<(PathBuf, File)>, String> {
    if flags.no_config {
        return Ok(None);
    }
    let path = match &flags.config {
        Some(p) => Some(p.clone()),
        None => [PathBuf::from(CONFIG_NAME), exe_dir().join(CONFIG_NAME)].into_iter().find(|p| p.is_file()),
    };
    match path {
        Some(p) => {
            let file = File::load(&p)?;
            let p = absolute(&p);
            Ok(Some((p, file)))
        }
        None => Ok(None),
    }
}

/// `p` as an absolute path, without Windows' `\\?\` prefix that `canonicalize` adds (it would show
/// in every message and isn't needed for paths of normal length).
fn absolute(p: &Path) -> PathBuf {
    let c = std::fs::canonicalize(p).unwrap_or_else(|_| std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()));
    let s = c.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC\\") && rest.len() < 250 => PathBuf::from(rest),
        _ => c,
    }
}

/// Where model files are looked up, first match wins: the flags', the settings file's, the settings
/// file's folder, next to the program and to libaji (`models/` beside them or beside their folder,
/// as MangaJaNai GUI lays out `inference/` and `models/`), then MangaJaNai GUI's own model folders,
/// so an exported workflow's model names resolve as they do in the GUI.
fn models_dirs(flags: &Flags, file: Option<(&Path, &File)>, aji_dir: &Path) -> Vec<PathBuf> {
    let mut dirs = flags.models_dirs.clone();
    if let Some((base, f)) = file {
        dirs.extend(f.models_dir.iter().map(|d| config::rel(base, d)));
        dirs.push(base.to_path_buf());
    }
    for d in [exe_dir(), aji_dir.to_path_buf()] {
        dirs.push(d.join("models"));
        dirs.push(d.join("..").join("models"));
    }
    if cfg!(windows) {
        for (var, sub) in [("APPDATA", r"MangaJaNaiConverterGui\models"), ("LOCALAPPDATA", r"Programs\MangaJaNaiConverterGui\models")] {
            if let Some(root) = std::env::var_os(var) {
                dirs.push(PathBuf::from(root).join(sub));
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|d| d.is_dir() && seen.insert(std::fs::canonicalize(d).unwrap_or_else(|_| d.clone())));
    dirs
}

/// A model named on the command line or in the settings file: a path that exists (relative to
/// `base`) becomes absolute; anything else stays a name, looked up in the models folders.
fn model_ref(base: &Path, m: &str) -> String {
    let p = config::rel(base, m);
    if p.is_file() {
        absolute(&p).to_string_lossy().into_owned()
    } else {
        m.to_string()
    }
}

/// MangaJaNai GUI's "Upscale Manga (Default)" chains: IllustrationJaNai for color pages; for gray
/// pages the MangaJaNai model trained nearest the page height, 2x up to a 2x target and 4x above.
pub fn default_chains() -> Vec<Chain> {
    let chain = |min: &str, max: &str, gray: bool, min_s: i32, max_s: i32, model: String| Chain {
        min_resolution: min.into(),
        max_resolution: max.into(),
        is_grayscale: gray,
        is_color: !gray,
        min_scale_factor: Some(min_s),
        max_scale_factor: Some(max_s),
        model_file_path: model,
        auto_adjust_levels: gray,
        ..Chain::default()
    };
    let mut chains = vec![
        chain("0x0", "0x0", false, 0, 2, "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx".into()),
        chain("0x0", "0x0", false, 2, 0, "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx".into()),
    ];
    for (lo, hi, bucket, k2, k4) in [
        ("0x0", "0x1250", "1200p", "70k", "70k"),
        ("0x1251", "0x1350", "1300p", "75k", "75k"),
        ("0x1351", "0x1450", "1400p", "70k", "105k"),
        ("0x1451", "0x1550", "1500p", "90k", "105k"),
        ("0x1551", "0x1760", "1600p", "90k", "70k"),
        ("0x1761", "0x1984", "1920p", "70k", "105k"),
        ("0x1985", "0x0", "2048p", "95k", "70k"),
    ] {
        chains.push(chain(lo, hi, true, 0, 2, format!("2x_MangaJaNai_{bucket}_V1_ESRGAN_{k2}_fp16.onnx")));
        chains.push(chain(lo, hi, true, 2, 0, format!("4x_MangaJaNai_{bucket}_V1_ESRGAN_{k4}_fp16.onnx")));
    }
    for (i, c) in chains.iter_mut().enumerate() {
        c.chain_number = (i + 1).to_string();
    }
    chains
}

/// One chain that sends every image, gray or color, through `model`.
fn single_model_chain(model: String) -> Chain {
    Chain { chain_number: "1".into(), is_grayscale: true, is_color: true, model_file_path: model, ..Chain::default() }
}

fn load_workflow(path: &Path) -> Result<Workflow, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(text.trim_start_matches('\u{feff}')).map_err(|e| format!("{}: not a MangaJaNai workflow: {e}", path.display()))
}

/// A resolved command-line run.
struct Plan {
    inputs: Vec<run::Input>,
    output: run::Output,
    job: process::Job,
    backend: BackendChoice,
    device: Option<String>,
    streams: u32,
    overlap: Option<u32>,
    even_tiles: bool,
    cache_dir: Option<PathBuf>,
    aji_dir: PathBuf,
    models_dirs: Vec<PathBuf>,
    /// The model setting, for the run's heading.
    describe: String,
}

/// Layer the settings: built-in defaults < an imported workflow < the settings file < flags.
fn resolve(flags: &Flags) -> Result<Plan, String> {
    let cfg = find_config(flags)?;
    let (base, file) = match &cfg {
        Some((p, f)) => (p.parent().map(Path::to_path_buf).unwrap_or_default(), Some(f)),
        None => (PathBuf::new(), None),
    };
    let cwd = PathBuf::new();
    let fv = |get: fn(&File) -> Option<String>| file.and_then(get);

    // the model source: flags first (one of), else the file's (one of), else the manga preset
    let flag_sources = [flags.model.is_some(), flags.preset.is_some(), flags.workflow.is_some()];
    if flag_sources.iter().filter(|&&s| s).count() > 1 {
        return Err("give only one of --model, --preset or --workflow".into());
    }
    enum Source {
        Model(String),
        Preset(String),
        Workflow(PathBuf),
        Rules(Vec<config::Rule>),
    }
    let source = if let Some(m) = &flags.model {
        Source::Model(model_ref(&cwd, m))
    } else if let Some(p) = &flags.preset {
        Source::Preset(p.clone())
    } else if let Some(w) = &flags.workflow {
        Source::Workflow(w.clone())
    } else if let Some(m) = fv(|f| f.model.clone()) {
        Source::Model(model_ref(&base, &m))
    } else if let Some(p) = fv(|f| f.preset.clone()) {
        Source::Preset(p)
    } else if let Some(w) = fv(|f| f.workflow.clone()) {
        Source::Workflow(config::rel(&base, &w))
    } else if let Some(f) = file.filter(|f| !f.rule.is_empty()) {
        Source::Rules(f.rule.clone())
    } else {
        Source::Preset("manga".into())
    };

    let wf = match &source {
        Source::Workflow(p) => Some(load_workflow(p)?),
        _ => None,
    };
    let (mut chains, describe) = match &source {
        Source::Model(m) => (vec![single_model_chain(m.clone())], format!("model {m}")),
        Source::Preset(p) if p.eq_ignore_ascii_case("manga") => (default_chains(), "preset manga".into()),
        Source::Preset(p) => return Err(format!("unknown preset '{p}' (manga)")),
        Source::Workflow(p) => (wf.as_ref().unwrap().chains.values.clone(), format!("workflow {}", p.display())),
        Source::Rules(r) => (r.iter().enumerate().map(|(i, r)| r.to_chain(i + 1, |m| model_ref(&base, m))).collect(), format!("{} rules", r.len())),
    };
    if let Some(t) = flags.tile.or(file.and_then(|f| f.gpu.tile)) {
        for c in &mut chains {
            c.model_tile_size = t.to_string();
        }
    }
    if let Some(a) = flags.auto_levels.or(file.and_then(|f| f.auto_levels)) {
        for c in &mut chains {
            c.auto_adjust_levels = a;
        }
    }

    // inputs
    let images = flags.images.or(file.and_then(|f| f.images)).or(wf.as_ref().map(|w| w.upscale_images)).unwrap_or(true);
    let archives = flags.archives.or(file.and_then(|f| f.archives)).or(wf.as_ref().map(|w| w.upscale_archives)).unwrap_or(true);
    let mut paths: Vec<PathBuf> = flags.inputs.clone();
    if paths.is_empty()
        && let Some(f) = file
    {
        paths = f.input.iter().map(|p| config::rel(&base, p)).collect();
    }
    if paths.is_empty()
        && let Some(w) = &wf
    {
        let p = if w.selected_tab_index == 0 { &w.input_file_path } else { &w.input_folder_path };
        if !p.is_empty() {
            paths.push(p.into());
        }
    }
    if paths.is_empty() {
        return Err(format!("nothing to upscale: give inputs, or a path under [input] in {CONFIG_NAME} (see --help)"));
    }
    let mut inputs = Vec::new();
    for p in paths {
        if p.is_dir() {
            inputs.push(run::Input::Folder { dir: p, images, archives });
        } else if p.is_file() {
            inputs.push(run::Input::File(p));
        } else {
            return Err(format!("input not found: {}", p.display()));
        }
    }

    // output
    let dir = flags
        .output
        .clone()
        .or(fv(|f| f.output.clone()).map(|o| config::rel(&base, &o)))
        .or(wf.as_ref().map(|w| w.output_folder_path.clone()).filter(|o| !o.is_empty()).map(PathBuf::from))
        .unwrap_or_else(|| "out".into());
    let filename = match (&flags.filename, &flags.suffix) {
        (Some(f), _) => f.clone(),
        (None, Some(s)) => format!("%filename%{s}"),
        (None, None) => match file.map(|f| (&f.save.filename, &f.save.suffix)) {
            Some((Some(f), _)) => f.clone(),
            Some((None, Some(s))) => format!("%filename%{s}"),
            _ => wf.as_ref().map(|w| w.output_filename.clone()).filter(|n| !n.is_empty()).unwrap_or_else(|| "%filename%".into()),
        },
    };
    let skip = flags
        .skip_existing
        .or(file.and_then(|f| f.save.skip_existing))
        .or(wf.as_ref().map(|w| !w.overwrite_existing_files))
        .unwrap_or(true);
    let output = run::Output { dir, filename, overwrite: !skip };

    // the job
    let file_target = match file {
        Some(f) => f.resize.target()?,
        None => None,
    };
    let target = flags.target.or(file_target).or(wf.as_ref().map(|w| w.target())).unwrap_or(Target::Native);
    let file_format = match file.and_then(|f| f.save.format.as_deref()) {
        Some(s) => Some(config::parse_format(s)?),
        None => None,
    };
    let job = process::Job {
        chains,
        threshold: flags.gray_threshold.or(file.and_then(|f| f.gray_threshold)).or(wf.as_ref().map(|w| w.grayscale_detection_threshold)).unwrap_or(12),
        target,
        format: flags.format.or(file_format).or(wf.as_ref().map(|w| w.format())).unwrap_or(Format::Webp),
        quality: flags.quality.or(file.and_then(|f| f.save.quality)).or(wf.as_ref().map(|w| w.quality())).unwrap_or(80).min(100),
        lossless: flags.lossless.or(file.and_then(|f| f.save.lossless)).or(wf.as_ref().map(|w| w.use_lossless_compression)).unwrap_or(false),
    };

    // the GPU
    let backend = match (flags.backend, file.and_then(|f| f.gpu.backend.as_deref())) {
        (Some(b), _) => b,
        (None, Some(s)) => BackendChoice::parse(s)?,
        (None, None) => BackendChoice::Auto,
    };
    let aji_dir = flags
        .aji_dir
        .clone()
        .or(file.and_then(|f| f.gpu.aji_dir.as_deref().map(|a| config::rel(&base, a))))
        .unwrap_or_else(default_aji_dir);
    Ok(Plan {
        inputs,
        output,
        job,
        backend,
        device: flags.device.clone().or(file.and_then(|f| f.gpu.devices.clone())),
        streams: flags.streams.or(file.and_then(|f| f.gpu.streams)).unwrap_or(1).clamp(1, 8),
        overlap: flags.overlap.or(file.and_then(|f| f.gpu.overlap)),
        even_tiles: flags.even_tiles.or(file.and_then(|f| f.gpu.even_tiles)).unwrap_or(false),
        cache_dir: flags.cache_dir.clone().or(file.and_then(|f| f.gpu.cache_dir.as_deref().map(|c| config::rel(&base, c)))),
        models_dirs: models_dirs(flags, file.map(|f| (base.as_path(), f)), &aji_dir),
        aji_dir,
        describe,
    })
}

/// The device ids for a `--device` spec: `None`/"first" = the backend's first device (id 0, no
/// device list needed); "all"; or numbers, each an index into the device list (as `devices` prints
/// it) or, if larger, a device id from `--list-devices`.
fn device_ids(aji_dir: &Path, backend: BackendChoice, spec: Option<&str>) -> Result<Vec<u64>, String> {
    let spec = spec.map(str::trim).unwrap_or("first");
    if spec.is_empty() || spec.eq_ignore_ascii_case("first") {
        return Ok(vec![0]);
    }
    let aji = upscaler::load_aji(aji_dir).map_err(|e| format!("loading libaji from {}: {e}", aji_dir.display()))?;
    let devices = backend
        .candidates()
        .iter()
        .find_map(|&b| aji.devices(b).ok().filter(|d| !d.is_empty()))
        .ok_or("no compatible GPU found")?;
    if spec.eq_ignore_ascii_case("all") {
        return Ok(devices.iter().map(|d| d.id).collect());
    }
    let mut ids = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let n: u64 = part.parse().map_err(|_| format!("--device: '{part}' (first, all, or numbers from `devices`)"))?;
        let id = if (n as usize) < devices.len() {
            devices[n as usize].id
        } else if devices.iter().any(|d| d.id == n) {
            n
        } else {
            return Err(format!("--device: no GPU {n} (there are {}; see `devices`)", devices.len()));
        };
        // A GPU listed twice gets two inference threads (two contexts): the CPU side of one
        // image's tiles then overlaps the other's inference, which can help light models.
        ids.push(id);
    }
    if ids.is_empty() {
        return Err("--device: no GPUs given".into());
    }
    Ok(ids)
}

/// Before a run that would find no GPU runtime: say how to get one, and in a terminal offer to run
/// `mangajanai setup` now. Ok(false) = don't run.
fn ensure_runtime(aji_dir: &Path) -> Result<bool, String> {
    let lib = if cfg!(windows) { "aji.dll" } else { "libaji.so" };
    if crate::setup::runtime_installed(aji_dir) || (aji_dir != exe_dir() && aji_dir.join(lib).is_file()) {
        return Ok(true);
    }
    eprintln!("No GPU runtime found next to mangajanai ({}).", aji_dir.display());
    eprintln!("`mangajanai setup` downloads what this PC needs, once: the TensorRT or DirectML runtime and the models.");
    use std::io::{BufRead, IsTerminal, Write};
    if !std::io::stdin().is_terminal() || !std::io::stderr().is_terminal() {
        return Ok(false);
    }
    eprint!("Run setup now? [Y/n] ");
    let _ = std::io::stderr().flush();
    let mut answer = String::new();
    let _ = std::io::stdin().lock().read_line(&mut answer);
    if matches!(answer.trim().to_ascii_lowercase().as_str(), "" | "y" | "yes") {
        Ok(crate::setup::run(&crate::setup::Options { no_models: false, check: false, mirror: None })? == 0)
    } else {
        Ok(false)
    }
}

fn upscale(flags: Flags) -> Result<i32, String> {
    let plan = resolve(&flags)?;
    if flags.aji_dir.is_none() && !ensure_runtime(&plan.aji_dir)? {
        return Ok(1);
    }
    let ids: Vec<u64> = device_ids(&plan.aji_dir, plan.backend, plan.device.as_deref())?
        .into_iter()
        .flat_map(|id| std::iter::repeat_n(id, plan.streams as usize))
        .collect();
    let total = run::total(&plan.inputs, &plan.output, &plan.job);
    if !flags.lines {
        progress::set_human(flags.verbose, Some(total));
        let gpus = if ids.len() > 1 { format!(", {} GPUs", ids.len()) } else { String::new() };
        progress::info(&format!("{} item(s) to write, {}{gpus} -> {}", total, plan.describe, plan.output.dir.display()));
    }
    let openers: Vec<run::OpenUpscaler> = ids
        .into_iter()
        .map(|id| {
            let (aji_dir, cache_dir, dirs, backend, overlap, even) =
                (plan.aji_dir.clone(), plan.cache_dir.clone(), plan.models_dirs.clone(), plan.backend, plan.overlap, plan.even_tiles);
            Box::new(move || Upscaler::open(&aji_dir, backend, id, cache_dir, dirs, overlap, even)) as run::OpenUpscaler
        })
        .collect();
    let start = Instant::now();
    run::run(plan.inputs, plan.output, plan.job, openers);
    let failed = progress::finish(start.elapsed());
    Ok(if failed > 0 { 1 } else { 0 })
}

// ---- subcommands -----------------------------------------------------------------------------

fn init(flags: &Flags) -> Result<i32, String> {
    let path = flags.inputs.first().cloned().unwrap_or_else(|| CONFIG_NAME.into());
    if path.exists() && !flags.force {
        return Err(format!("{} exists (--force to replace it)", path.display()));
    }
    std::fs::write(&path, config::TEMPLATE).map_err(|e| format!("{}: {e}", path.display()))?;
    println!("wrote {}: edit it, then run `{}`", path.display(), prog_name());
    Ok(0)
}

fn devices(flags: &Flags) -> Result<i32, String> {
    use yosh_image::aji::Backend;
    let aji_dir = flags.aji_dir.clone().unwrap_or_else(default_aji_dir);
    let aji = upscaler::load_aji(&aji_dir).map_err(|e| format!("loading libaji from {}: {e}", aji_dir.display()))?;
    let backends: Vec<Backend> = match flags.backend {
        Some(b) => b.candidates().to_vec(),
        None if cfg!(windows) => vec![Backend::TensorRt, Backend::TensorRtRtx, Backend::DirectMl],
        None => vec![Backend::TensorRt],
    };
    for b in backends {
        match aji.devices(b) {
            Ok(list) if !list.is_empty() => {
                println!("{}:", b.label());
                for (i, d) in list.iter().enumerate() {
                    let cc = d.compute_capability.map_or(String::new(), |(a, b)| format!(", compute {a}.{b}"));
                    println!("  {i}  {}  ({:.1} GB{cc}; id {})", d.name, d.vram_bytes as f64 / (1u64 << 30) as f64, d.id);
                }
            }
            Ok(_) => println!("{}: no devices", b.label()),
            Err(e) => println!("{}: not available ({e})", b.label()),
        }
    }
    Ok(0)
}

/// Time a model: one warm-up upscale (which includes a first-use engine build), then `--iters`
/// timed ones of a synthetic image.
fn bench(flags: Flags) -> Result<i32, String> {
    let model = flags.model.clone().or_else(|| flags.inputs.first().map(|p| p.to_string_lossy().into_owned())).ok_or("bench needs a model: bench <model.onnx>")?;
    let model = model_ref(Path::new(""), &model);
    let aji_dir = flags.aji_dir.clone().unwrap_or_else(default_aji_dir);
    let backend = flags.backend.unwrap_or(BackendChoice::Auto);
    let id = device_ids(&aji_dir, backend, flags.device.as_deref())?[0];
    let tile = flags.tile.unwrap_or(512);
    let (w, h) = flags.size.unwrap_or((tile * 2, tile * 3));
    let channels: u8 = if flags.gray { 1 } else { 3 };
    let iters = flags.iters.unwrap_or(10).max(1);

    progress::set_human(true, None);
    let mut up = Upscaler::open(&aji_dir, backend, id, flags.cache_dir.clone(), models_dirs(&flags, None, &aji_dir), flags.overlap, flags.even_tiles.unwrap_or(false))?;
    let path = up.resolve(&model).ok_or_else(|| format!("model not found: {model}"))?;
    progress::info(&format!("{} on {}: {w}x{h} {}, tile {tile}", path.file_name().unwrap_or_default().to_string_lossy(), up.description, if flags.gray { "gray" } else { "RGB" }));

    // deterministic texture: smooth gradients plus noise, so the model does real work
    let mut seed = 0x2545_f491u32;
    let pixels: Vec<u8> = (0..w * h * channels as u32)
        .map(|i| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            let (x, y) = ((i / channels as u32) % w, (i / channels as u32) / w);
            (((x ^ y) & 0xff) as u8 / 2).wrapping_add((seed & 0x3f) as u8)
        })
        .collect();

    let t0 = Instant::now();
    let (_, scale) = up.upscale(&path, tile, w, h, channels, &pixels)?;
    let first = t0.elapsed();
    let mut times = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        let t = Instant::now();
        up.upscale(&path, tile, w, h, channels, &pixels)?;
        times.push(t.elapsed());
    }
    times.sort();
    let mean = times.iter().sum::<Duration>() / iters;
    let median = times[times.len() / 2];
    let mpx = (w as f64 * h as f64) / 1e6;
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    println!("scale         {scale}x -> {}x{}", w * scale, h * scale);
    println!("first run     {:.0} ms (model load, engine build if needed, warm-up)", ms(first));
    println!("per image     {:.1} ms mean, {:.1} ms median, {:.1} ms best  ({iters} runs)", ms(mean), ms(median), ms(times[0]));
    println!("throughput    {:.2} images/s, {:.2} input MP/s", 1.0 / mean.as_secs_f64(), mpx / mean.as_secs_f64());
    Ok(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flags(args: &[&str]) -> Flags {
        parse_flags(args.iter().map(|s| s.to_string()).collect()).unwrap()
    }

    #[test]
    fn flags_parse() {
        let f = flags(&["a", "b.cbz", "-o", "out2", "--model=m.onnx", "--height", "2400", "--format", "png", "-q", "95", "--device", "all", "--suffix", "_x"]);
        assert_eq!(f.inputs, [PathBuf::from("a"), PathBuf::from("b.cbz")]);
        assert_eq!(f.output, Some("out2".into()));
        assert_eq!(f.model.as_deref(), Some("m.onnx"));
        assert_eq!(f.target, Some(Target::Height(2400)));
        assert_eq!(f.format, Some(Format::Png));
        assert_eq!(f.quality, Some(95));
        assert_eq!(f.device.as_deref(), Some("all"));
        // the legacy forms still work
        let f = flags(&["-f", "x.png", "-u", "4", "--models-dir", "m"]);
        assert_eq!(f.inputs, [PathBuf::from("x.png")]);
        assert_eq!(f.target, Some(Target::Scale(4.0)));
        assert_eq!(f.models_dirs, [PathBuf::from("m")]);
        assert!(parse_flags(vec!["--bogus".into()]).is_err());
        assert!(parse_flags(vec!["--lossless=yes".into()]).is_err(), "a switch takes no value");
    }

    #[test]
    fn file_then_flags() {
        let dir = std::env::temp_dir().join(format!("mangajanai-cli-resolve-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("in")).unwrap();
        let cfg = dir.join(CONFIG_NAME);
        std::fs::write(
            &cfg,
            "[input]\npath = in\n[output]\nfolder = o\nformat = png\nsuffix = _up\n[model]\nfile = m.onnx\n[resize]\nheight = 100\n[gpu]\ntile = 384\n",
        )
        .unwrap();
        let mut f = flags(&["--format", "jpg"]);
        f.config = Some(cfg.clone());
        let p = resolve(&f).unwrap();
        assert!(matches!(&p.inputs[..], [run::Input::Folder { .. }]));
        assert!(p.output.dir.ends_with("o"), "relative to the file");
        assert_eq!(p.output.filename, "%filename%_up");
        assert_eq!(p.job.format, Format::Jpeg, "the flag wins");
        assert_eq!(p.job.target, Target::Height(100));
        assert_eq!(p.job.chains.len(), 1);
        assert_eq!(p.job.chains[0].model_tile_size, "384");
        assert!(p.job.chains[0].is_grayscale && p.job.chains[0].is_color);
        assert!(!p.output.overwrite, "skip existing by default");

        // a flag's model replaces the file's
        let mut f = flags(&["--preset", "manga"]);
        f.config = Some(cfg.clone());
        assert_eq!(resolve(&f).unwrap().job.chains.len(), default_chains().len());

        // --no-config: no inputs left
        let f = flags(&["--no-config"]);
        assert!(resolve(&f).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn workflow_import() {
        let dir = std::env::temp_dir().join(format!("mangajanai-cli-mwf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mwf = dir.join("w.mwf");
        std::fs::write(
            &mwf,
            r#"{"$type":"W","WorkflowName":"x","SelectedTabIndex":1,"InputFolderPath":"","OutputFolderPath":"wo",
               "PngSelected":true,"WebpSelected":false,"ModeHeightSelected":true,"ModeScaleSelected":false,
               "ResizeHeightAfterUpscale":2160,"Chains":{"$type":"L","$values":[{"$type":"C","IsGrayscale":true,"ModelFilePath":"a.onnx"}]}}"#,
        )
        .unwrap();
        let mut f = flags(&["--no-config"]);
        f.workflow = Some(mwf);
        f.inputs = vec![dir.clone()];
        let p = resolve(&f).unwrap();
        assert_eq!(p.job.format, Format::Png);
        assert_eq!(p.job.target, Target::Height(2160));
        assert_eq!(p.output.dir, PathBuf::from("wo"));
        assert_eq!(p.job.chains[0].model_file_path, "a.onnx");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
