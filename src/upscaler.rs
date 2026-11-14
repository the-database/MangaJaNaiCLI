//! The inference side: one libaji context on the chosen backend/device, the models loaded so far
//! (by file and tile), auto mode's tiles, and the upscale itself — including recovering alpha for
//! transparent images.
//!
//! Auto mode (`--tile auto`, the default) sizes tiles to the GPU's free memory (yosh-image's
//! `tiles::MemoryUse`, from what libaji reports about the models it loaded) and
//!
//! - on TensorRT runs every page on one even-tiles engine per model: tiles of at most 1024 (less
//!   where memory is short), each page split into the fewest even tiles
//!   ([`Upscaler::even_tile_for`]);
//! - on TensorRT-RTX and DirectML, where even tiles are slow or missing, runs pages on the 512
//!   square and builds engines fitted to page sizes that come up often enough ([`Fit`]).
//!
//! An engine auto mode chose that fails at inference (out of memory, say) is dropped for the run
//! and its pages go to the 512 square.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use yosh_image::aji::{Aji, Backend, Context, ContextParams, Device, LoadParams, Model, ModelState, Pixels, PixelsMut, Upscaled};
use yosh_image::tiles::Budget;

use crate::tiles::{self, FALLBACK, MemoryUse, Shape, Tile};
use crate::{nvml, precision, progress};

/// Auto mode's state for one model and fitted shape (TensorRT-RTX and DirectML).
///
/// An engine fitted to a page size is loaded at once when it's been built before (in the engine
/// record, yosh-image's `tiles::fitted_known`, which yosh's upscaler shares). Otherwise pages of
/// that size first run on the square tile, adding up the time the fitted shape would have saved
/// (`lost`), and its build starts in the background (pages keep going on the square) when either
///
/// - `lost` reaches what a build costs ([`BUILD_COST`]): the ski-rental rule, which never loses
///   more than one build's worth per size whatever comes next, or
/// - the run's pages still to come, at the share of this size seen so far, would save that much:
///   a volume's pages are all one size, so its engine starts building on its second or third page.
///
/// The fitted shape is the largest that fits the GPU's free memory (measured with the square's
/// engine loaded). Once ready, the engine has to beat the square on its first [`CHECK_PAGES`]
/// pages, or it's dropped. So a volume gets its engine within a few pages, and a folder of mixed
/// sizes builds nothing.
enum Fit {
    /// `seen`: pages of this size so far; `ran`: of those, the ones timed on the square, which took
    /// `square` seconds in all, of which the fitted shape would have saved `lost`.
    Renting { lost: f64, square: f64, seen: u32, ran: u32 },
    /// `bytes`: the memory it's predicted to take, held in [`RESERVED`] until it's loaded.
    /// `square`: seconds per page on the square, for the check.
    Building { model: Model, bytes: u64, square: Option<f64> },
    Ready { model: Model, last_use: u64, check: Option<Check> },
    Failed,
}

/// A ready fitted engine's first pages (`runs`, which took `took` seconds), against the square's
/// `square` seconds per page.
struct Check {
    square: f64,
    runs: u32,
    took: f64,
}

/// A ready fitted engine is timed on this many pages, and kept if it's at least [`CHECK_GAIN`]
/// faster per page than the square was.
const CHECK_PAGES: u32 = 3;
const CHECK_GAIN: f64 = 0.05;

/// Fitted engines kept loaded per model; past it, the least recently used is unloaded.
const MAX_FITTED: usize = 2;

/// What building a fitted engine costs, in seconds of square-tile time to lose before buying it.
/// TensorRT-RTX and DirectML build in seconds; the build runs in the background while pages
/// continue, and its engine is cached for later runs, so this is below the build's own duration.
const BUILD_COST: f64 = 3.0;

/// Fitted engines some upscaler is building, so threads on several GPUs don't build one twice.
static FITTING: Mutex<Vec<(PathBuf, Shape)>> = Mutex::new(Vec::new());

fn fitting_done(key: &(PathBuf, Shape)) {
    FITTING.lock().unwrap_or_else(|p| p.into_inner()).retain(|k| k != key);
}

/// Memory, by device id, of engines decided on but not loaded yet: streams on one GPU would each
/// see it as free.
static RESERVED: Mutex<Vec<(u64, u64)>> = Mutex::new(Vec::new());

fn reserved(device: u64) -> u64 {
    RESERVED.lock().unwrap_or_else(|p| p.into_inner()).iter().filter(|r| r.0 == device).map(|r| r.1).sum()
}

fn reserve(device: u64, bytes: u64) {
    RESERVED.lock().unwrap_or_else(|p| p.into_inner()).push((device, bytes));
}

fn unreserve(device: u64, bytes: u64) {
    let mut r = RESERVED.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(i) = r.iter().position(|&x| x == (device, bytes)) {
        r.remove(i);
    }
}

/// A loaded model's key: file, tile (with even tiles, the largest), even tiles.
type Key = (PathBuf, Shape, bool);

/// The engine a page runs on.
#[derive(Clone, Copy)]
struct Pick {
    shape: Shape,
    /// An even-tiles engine: tiles of at most `shape`.
    even: bool,
    /// A fitted engine (in `Upscaler::fitted`), not one of `Upscaler::models`.
    fitted: bool,
    /// Auto mode chose it over the square: if it fails, the page is retried on the square.
    auto: bool,
    /// Running on the square while renting this fitted shape, which would save this share.
    rent: Option<(Shape, f64)>,
}

impl Pick {
    fn tile(shape: Shape) -> Self {
        Pick { shape, even: false, fitted: false, auto: false, rent: None }
    }
}

/// Load libaji from `dir`. On Linux, first load the CUDA and TensorRT libraries `mangajanai setup`
/// put beside it, by full path and globally: libaji_trt.so asks for them by soname, which then
/// resolves to these without LD_LIBRARY_PATH or a system install. (Its engine builds run trtexec,
/// which gets LD_LIBRARY_PATH from `trtexec()`.) A system TensorRT works as before.
pub fn load_aji(dir: &Path) -> Result<Aji, String> {
    #[cfg(unix)]
    for lib in ["libcudart.so.13", "libnvinfer.so.11", "libnvinfer_plugin.so.11", "libnvonnxparser.so.11"] {
        use libloading::os::unix::{Library, RTLD_GLOBAL, RTLD_NOW};
        let path = dir.join(lib);
        if path.is_file() {
            // SAFETY: NVIDIA's libraries; their initializers only set up their own state. Kept
            // loaded for the process's life (libaji uses them until exit).
            if let Ok(l) = unsafe { Library::open(Some(&path), RTLD_NOW | RTLD_GLOBAL) } {
                std::mem::forget(l);
            }
        }
    }
    Aji::load(dir)
}

/// Which backend to run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendChoice {
    /// On Windows, TensorRT-RTX on a GPU where it's the default (`nvml::trtrtx_default`) if it's
    /// installed, then TensorRT if an NVIDIA GPU it runs on (`nvml::TRT_MIN_COMPUTE`) and the
    /// TensorRT runtime are there, else DirectML: the order MangaJaNai GUI's and yosh's Auto use.
    /// Off Windows, TensorRT.
    Auto,
    TensorRt,
    DirectMl,
    /// TensorRT for RTX (opt-in, experimental; needs aji_trtrtx.dll beside aji.dll), falling
    /// back to TensorRT when it can't start, and per model when one fails to load or build,
    /// except for lack of GPU memory.
    TensorRtRtx,
}

impl BackendChoice {
    /// The backends `Upscaler::open` tries, in order. Off Windows there is only TensorRT, so Auto
    /// reports TensorRT's own error there instead of a DirectML one. Auto's TensorRT-RTX is tried
    /// only on a GPU it suits ([`BackendChoice::runs_on`]).
    pub fn candidates(self) -> &'static [Backend] {
        match self {
            BackendChoice::Auto if cfg!(windows) => &[Backend::TensorRtRtx, Backend::TensorRt, Backend::DirectMl],
            BackendChoice::Auto => &[Backend::TensorRt],
            BackendChoice::TensorRt => &[Backend::TensorRt],
            BackendChoice::DirectMl => &[Backend::DirectMl],
            BackendChoice::TensorRtRtx => &[Backend::TensorRtRtx, Backend::TensorRt],
        }
    }

    /// Whether this choice runs `backend` on `device`: Auto takes TensorRT-RTX only where it's the
    /// default ([`nvml::trtrtx_default`]), and goes on to TensorRT elsewhere without a word.
    pub fn runs_on(self, backend: Backend, device: &Device) -> bool {
        self != BackendChoice::Auto
            || backend != Backend::TensorRtRtx
            || device.compute_capability.is_some_and(|(major, minor)| nvml::trtrtx_default(major, minor))
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let choice = match s.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "trt" | "tensorrt" => Self::TensorRt,
            "dml" | "directml" => Self::DirectMl,
            "tensorrt-rtx" | "trt-rtx" | "trtrtx" => Self::TensorRtRtx,
            _ => return Err(format!("unknown backend '{s}' (auto, tensorrt, directml, tensorrt-rtx)")),
        };
        if !cfg!(windows) && matches!(choice, Self::DirectMl | Self::TensorRtRtx) {
            return Err(format!("the {s} backend is Windows only; use tensorrt (or auto)"));
        }
        Ok(choice)
    }
}

pub fn backend_name(b: Backend) -> &'static str {
    b.label()
}

/// A backend's devices split into the ones it can upscale on and the ones too old for it:
/// TensorRT and TensorRT-RTX don't run below `nvml::TRT_MIN_COMPUTE`, so such a GPU counts as no
/// NVIDIA GPU there (Auto goes on to DirectML). DirectML runs on every device it lists.
pub fn split_devices(backend: Backend, devices: Vec<Device>) -> (Vec<Device>, Vec<Device>) {
    if backend == Backend::DirectMl {
        return (devices, Vec::new());
    }
    devices.into_iter().partition(|d| d.compute_capability.is_none_or(|(major, minor)| nvml::trt_capable(major, minor)))
}

/// Why TensorRT (or TensorRT-RTX) can't run on the GPUs in `old`, which are all too old for it.
pub fn too_old_message(old: &[Device]) -> String {
    let found: Vec<String> = old
        .iter()
        .map(|d| match d.compute_capability {
            Some((major, minor)) => format!("{} is compute {major}.{minor}", d.name),
            None => d.name.clone(),
        })
        .collect();
    let instead = if cfg!(windows) { "Use DirectML instead." } else { "On Linux mangajanai runs on TensorRT only." };
    format!("{}; {}. {instead}", nvml::trt_needs(), found.join(", "))
}

/// A chain's model file: as given if it's a path that exists, else looked up by name in the models
/// folders (the user's first, then the bundled ones). Exactly that file: no variant is swapped in.
fn resolve_in(model: &str, dirs: &[PathBuf]) -> Option<PathBuf> {
    let p = Path::new(model);
    if p.is_absolute() && p.is_file() {
        return Some(p.to_path_buf());
    }
    dirs.iter().map(|d| d.join(model)).find(|p| p.is_file())
}

fn file_name(path: &Path) -> String {
    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

pub struct Upscaler {
    ctx: Context,
    /// For [`Upscaler::close`]'s device release.
    aji: Aji,
    backend: Backend,
    /// The device `ctx` runs on, for the precision hints.
    device: Device,
    /// The model files whose precision hint was already given (once per file, not per tile).
    hinted: HashSet<PathBuf>,
    /// On TensorRT-RTX: where to open the TensorRT context a failed model falls back to.
    fallback: Option<Fallback>,
    model_dirs: Vec<PathBuf>,
    models: HashMap<Key, Model>,
    /// Auto mode's fitted engines, by model file and shape.
    fitted: HashMap<(PathBuf, Shape), Fit>,
    /// Auto mode's fitted shape (and the share of the square's work it saves) by model file and
    /// page size, decided on the first page of that size whose model's memory use is known.
    shapes: HashMap<(PathBuf, Shape), Option<(Shape, f64)>>,
    /// Auto mode on TensorRT: each model's even-tiles size, decided on its first page; 0 once its
    /// engine has failed (the square instead).
    even: HashMap<PathBuf, u32>,
    /// What [`Upscaler::even_tile_for`] reserved for an even-tiles engine not loaded yet.
    even_reserved: HashMap<PathBuf, u64>,
    /// Each model's memory use, measured from its engines (see yosh-image's `tiles::Budget`).
    memory: HashMap<PathBuf, MemoryUse>,
    /// Pages picked so far, for the fitted engines' least-recently-used order.
    clock: u64,
    /// The largest fitted tile area by the GPU's size, for a libaji that can't report memory.
    area_budget: u64,
    /// Where engines are cached (`None`: next to each model), for the engine records.
    cache_dir: Option<PathBuf>,
    /// Pages the run still has for this upscaler ([`Upscaler::set_pages_left`]), and the pages
    /// auto mode has seen, for guessing how many more of a size will come.
    pages_left: u64,
    pages_seen: u64,
    /// Input-pixel tile overlap; `None` = libaji's default (128 since its seam fix).
    overlap: Option<u32>,
    /// With a fixed tile, split images into even tiles from a shape-range engine
    /// (`LoadParams::even_tiles`; auto mode does on TensorRT anyway).
    even_tiles: bool,
    pub description: String,
}

/// Held while a model loads, so inference threads on several GPUs never build the same engine (same
/// model, tile and GPU kind share one cache file) at once: the second waits, then loads the engine
/// the first one built. Uncontended with one GPU.
static LOAD_LOCK: Mutex<()> = Mutex::new(());

impl Upscaler {
    /// Open the backend: the requested one, or for Auto the first that has a usable device. A
    /// device id of 0 means the backend's first device.
    pub fn open(
        aji_dir: &Path,
        choice: BackendChoice,
        device_id: u64,
        cache_dir: Option<PathBuf>,
        model_dirs: Vec<PathBuf>,
        overlap: Option<u32>,
        even_tiles: bool,
    ) -> Result<Self, String> {
        let aji = load_aji(aji_dir)?;
        let trtexec_env = trtexec(aji_dir);
        let mut err = String::from("no compatible GPU found");
        let mut too_old = false;
        // Auto: why TensorRT-RTX, tried first, didn't start (reported once another backend has)
        let mut rtx_failed: Option<String> = None;
        for &backend in choice.candidates() {
            if choice == BackendChoice::TensorRtRtx && backend == Backend::TensorRt {
                if too_old {
                    break; // TensorRT doesn't run on these GPUs either
                }
                progress::backend_fallback(None, &err, "TensorRT");
            }
            let (devices, old) = match aji.devices(backend) {
                Ok(d) => split_devices(backend, d),
                Err(e) => {
                    err = format!("{}: {e}", backend_name(backend));
                    continue;
                }
            };
            // The GPU asked for, or every GPU there, is too old for this backend: Auto goes on to
            // DirectML, and TensorRT on its own says why it can't run.
            let chosen_old = old.iter().find(|d| device_id != 0 && d.id == device_id);
            if chosen_old.is_some() || (devices.is_empty() && !old.is_empty()) {
                err = too_old_message(chosen_old.map_or(&old[..], std::slice::from_ref));
                too_old = true;
                continue;
            }
            if devices.is_empty() {
                continue;
            }
            let device = devices.iter().find(|d| d.id == device_id).unwrap_or(&devices[0]);
            if !choice.runs_on(backend, device) {
                continue;
            }
            match open_context(&aji, backend, device.id, &cache_dir, &trtexec_env) {
                Ok(ctx) => {
                    if let Some(why) = &rtx_failed {
                        progress::backend_fallback(None, why, backend_name(backend));
                    }
                    let description = format!("{} on {}", backend_name(backend), device.name);
                    let fallback = (backend == Backend::TensorRtRtx).then(|| Fallback {
                        aji: aji.clone(),
                        device_id: device.id,
                        cache_dir: cache_dir.clone(),
                        trtexec_env: trtexec_env.clone(),
                        ctx: None,
                    });
                    return Ok(Self {
                        ctx,
                        aji: aji.clone(),
                        backend,
                        device: device.clone(),
                        hinted: HashSet::new(),
                        fallback,
                        model_dirs,
                        models: HashMap::new(),
                        fitted: HashMap::new(),
                        shapes: HashMap::new(),
                        even: HashMap::new(),
                        even_reserved: HashMap::new(),
                        memory: HashMap::new(),
                        clock: 0,
                        area_budget: tiles::area_budget(device.vram_bytes),
                        cache_dir: cache_dir.clone(),
                        pages_left: 0,
                        pages_seen: 0,
                        overlap,
                        even_tiles,
                        description,
                    });
                }
                Err(e) => {
                    err = format!("{}: {e}", backend_name(backend));
                    if choice == BackendChoice::Auto && backend == Backend::TensorRtRtx {
                        rtx_failed = Some(err.clone());
                    }
                }
            }
        }
        Err(err)
    }

    /// Closes the models and contexts, then has libaji release the device: on the CUDA backends
    /// that frees the CUDA context TensorRT's runtime would otherwise keep until the process exits
    /// (a few hundred MB), so a pause gives back all of its GPU memory. Nothing else in this
    /// process uses CUDA. The release waits for the last context on the GPU (another stream's),
    /// and an older libaji without it just keeps that memory, as before.
    pub fn close(self) {
        let (aji, id) = (self.aji.clone(), self.device.id);
        let mut backends = vec![self.backend];
        if self.fallback.as_ref().is_some_and(|f| matches!(f.ctx, Some(Ok(_)))) {
            backends.push(Backend::TensorRt);
        }
        drop(self);
        for backend in backends {
            let _ = aji.release_device(backend, id);
        }
    }

    /// A chain's model file (see [`resolve_in`]).
    pub fn resolve(&self, model: &str) -> Option<PathBuf> {
        resolve_in(model, &self.model_dirs)
    }

    /// The loaded model for `path` at `tile` (`even`: an even-tiles engine of at most `tile`),
    /// loading it (and waiting out a first-use TensorRT engine build, reported on stdout) if
    /// needed. On TensorRT-RTX, a model that fails to load or build is retried on TensorRT (that
    /// context opens the first time it's needed), unless it ran out of GPU memory: a second CUDA
    /// context and another copy of the model would need more, and libaji's message (what it
    /// needed, how much is free) is the useful error. The first load of a file prints its
    /// precision hint, if it has one, before anything else about it.
    fn model(&mut self, path: &Path, tile: Shape, even: bool) -> Result<&Model, String> {
        let key = (path.to_path_buf(), tile, even);
        if !self.models.contains_key(&key) {
            let name = file_name(path);
            if self.hinted.insert(path.to_path_buf()) {
                // A file that can't be read is reported by the load below.
                let p = precision::model_weights(path).map_or(precision::Precision::Unknown, |w| w.precision());
                let d = &self.device;
                if let Some(h) = precision::hint(self.backend, d.compute_capability, &d.name, p) {
                    progress::hint(&name, &h);
                }
            }
            let _loading = LOAD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let loaded = load_model(&self.ctx, self.backend, path, tile, self.overlap, even, &name);
            if even && let Some(bytes) = self.even_reserved.remove(path) {
                unreserve(self.device.id, bytes);
            }
            let model = match loaded {
                Ok(m) => {
                    self.measure(path, &m, tile);
                    m
                }
                Err(e) => {
                    let Some(fb) = self.fallback.as_mut().filter(|_| !yosh_image::aji::is_out_of_memory(&e)) else {
                        return Err(format!("{name}: {e}"));
                    };
                    progress::backend_fallback(Some(&name), &e, "TensorRT");
                    let ctx = fb.context()?;
                    load_model(&ctx, Backend::TensorRt, path, tile, self.overlap, even, &name).map_err(|e| format!("{name}: {e}"))?
                }
            };
            if !even {
                self.record_engine(&(key.0.clone(), tile));
            }
            self.models.insert(key.clone(), model);
        }
        Ok(&self.models[&key])
    }

    /// How many pages the run still has for this upscaler (its share, with several GPUs), for
    /// auto mode's guess at how many more pages of a size will come (0, the default: unknown).
    pub fn set_pages_left(&mut self, pages: u64) {
        self.pages_left = pages;
    }

    /// Note `path`'s memory use from its loaded engine with tiles of `tile`, here and in the
    /// record beside the engines (yosh-image's `tiles::record_memory`).
    fn measure(&mut self, path: &Path, model: &Model, tile: Shape) {
        let Ok(info) = model.info() else { return };
        let Some(m) = model.memory().and_then(|mem| MemoryUse::measured(&mem, tile, info.scale)) else { return };
        if let (Some(dir), Some(file)) = (self.record_dir(path), path.file_name()) {
            tiles::tiles::record_memory(&dir, &file.to_string_lossy(), self.backend.label(), &self.device.name, &m);
        }
        self.memory.insert(path.to_path_buf(), m);
    }

    /// `path`'s memory use as measured this run or recorded by an earlier one.
    fn memory_use(&self, path: &Path) -> Option<MemoryUse> {
        self.memory.get(path).copied().or_else(|| {
            let (dir, file) = (self.record_dir(path)?, path.file_name()?);
            tiles::tiles::known_memory(&dir, &file.to_string_lossy(), self.backend.label(), &self.device.name)
        })
    }

    /// The GPU's free memory now, to check `path`'s engines against (its memory use guessed if
    /// nothing is known yet).
    fn budget(&self, path: &Path) -> Budget {
        Budget {
            mem: self.ctx.memory(),
            reserved: reserved(self.device.id),
            area_budget: self.area_budget,
            use_: self.memory_use(path).unwrap_or_else(MemoryUse::guess),
        }
    }

    /// Auto mode on TensorRT: the even-tiles size for `path`, decided on its first page (`page`)
    /// as the largest of yosh-image's `tiles::EVEN_TILES` that fits the GPU's free memory, by its
    /// recorded memory use or else a generous guess. The engine is cached per size, so a size
    /// once built is kept while it fits. 0 once its engine has failed.
    fn even_tile_for(&mut self, path: &Path, page: Shape) -> u32 {
        if let Some(&t) = self.even.get(path) {
            return t;
        }
        let _deciding = LOAD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let budget = self.budget(path);
        let t = tiles::tiles::even_tile(|t| budget.fits((t, t), page).0);
        let bytes = budget.fits((t, t), page).1;
        reserve(self.device.id, bytes);
        self.even_reserved.insert(path.to_path_buf(), bytes);
        self.even.insert(path.to_path_buf(), t);
        t
    }

    /// The engine for a `w`x`h` page: a fixed tile as is; in auto mode on TensorRT the model's
    /// even-tiles engine, elsewhere a ready fitted engine for the page's size, else the square
    /// (see [`Fit`]), starting the fitted engine's load once it's worth it.
    fn pick(&mut self, path: &Path, tile: Tile, w: u32, h: u32) -> Pick {
        let square = Pick::tile((FALLBACK, FALLBACK));
        if let Tile::Fixed(tw, th) = tile {
            return Pick { even: self.even_tiles, ..Pick::tile((tw, th)) };
        }
        if self.backend == Backend::TensorRt {
            let t = self.even_tile_for(path, (w, h));
            return if t > 0 { Pick { even: true, auto: true, ..Pick::tile((t, t)) } } else { square };
        }
        let page = (path.to_path_buf(), (w, h));
        let fitted = match self.shapes.get(&page) {
            Some(&f) => f,
            // decided once the model's memory use is known: from the square's engine after its
            // first page, or from an earlier run
            None if self.memory_use(path).is_none() && self.ctx.memory().is_some() => return square,
            None => {
                let overlap = self.overlap.unwrap_or(tiles::DEFAULT_OVERLAP);
                let budget = self.budget(path);
                let f = tiles::fit(w, h, overlap, |s| budget.fits(s, (w, h)).0, FALLBACK);
                self.shapes.insert(page, f);
                f
            }
        };
        let Some((shape, gain)) = fitted else { return square };
        self.clock += 1;
        self.pages_seen += 1;
        let key = (path.to_path_buf(), shape);
        let name = file_name(path);
        let state = match self.fitted.remove(&key) {
            Some(state) => state,
            None if self.engine_known(&key) => self.start_fitted(&key, &name, (w, h), Fit::Renting { lost: 0.0, square: 0.0, seen: 0, ran: 0 }),
            None => Fit::Renting { lost: 0.0, square: 0.0, seen: 0, ran: 0 },
        };
        let state = match state {
            Fit::Building { model, bytes, square } => match model.state() {
                ModelState::Building => Fit::Building { model, bytes, square },
                ModelState::Ready => {
                    fitting_done(&key);
                    unreserve(self.device.id, bytes);
                    self.record_engine(&key);
                    self.measure(path, &model, shape);
                    progress::fitted_engine(&name, &tiles::shape_label(shape), self.backend.label(), true);
                    Fit::Ready { model, last_use: 0, check: square.map(|square| Check { square, runs: 0, took: 0.0 }) }
                }
                _ => {
                    fitting_done(&key);
                    unreserve(self.device.id, bytes);
                    let e = self.ctx.last_error();
                    progress::log(&format!("{name}: no engine fitted to tile {} ({e}); staying on tile {FALLBACK}", tiles::shape_label(shape)));
                    Fit::Failed
                }
            },
            Fit::Renting { lost, square, seen, ran } => {
                let seen = seen + 1;
                let renting = Fit::Renting { lost, square, seen, ran };
                // pages of this size still to come, at its share of the pages so far
                let coming = self.pages_left.saturating_sub(1) as f64 * seen as f64 / self.pages_seen as f64;
                let per_page = if ran > 0 { lost / ran as f64 } else { 0.0 };
                if lost >= BUILD_COST || (seen >= 2 && coming * per_page >= BUILD_COST) {
                    self.start_fitted(&key, &name, (w, h), renting)
                } else {
                    renting
                }
            }
            other => other,
        };
        let (state, pick) = match state {
            Fit::Ready { model, check, .. } => (Fit::Ready { model, last_use: self.clock, check }, Pick { fitted: true, auto: true, ..Pick::tile(shape) }),
            Fit::Renting { lost, square: sq, seen, ran } => (Fit::Renting { lost, square: sq, seen, ran }, Pick { rent: Some((shape, gain)), ..square }),
            other => (other, square),
        };
        self.fitted.insert(key, state);
        pick
    }

    /// Start loading the fitted engine `key` for `page`-size pages: at once if it's cached, else
    /// built in the background. Unloads the least recently used fitted engines of the model past
    /// [`MAX_FITTED`], or while it wouldn't fit in the GPU's free memory. `renting`: the state to
    /// stay in while another GPU's upscaler builds it.
    fn start_fitted(&mut self, key: &(PathBuf, Shape), name: &str, page: Shape, renting: Fit) -> Fit {
        {
            let mut building = FITTING.lock().unwrap_or_else(|p| p.into_inner());
            if building.contains(key) {
                // another GPU's upscaler is building it: load it from the cache once that's done
                return renting;
            }
            building.push(key.clone());
        }
        let square = match renting {
            Fit::Renting { square, ran, .. } if ran > 0 => Some(square / ran as f64),
            _ => None,
        };
        let label = tiles::shape_label(key.1);
        let mut room = self.budget(&key.0).fits(key.1, page);
        loop {
            let loaded: Vec<_> = self
                .fitted
                .iter()
                .filter_map(|(k, f)| match f {
                    Fit::Ready { last_use, .. } if k.0 == key.0 => Some((*last_use, k.clone())),
                    _ => None,
                })
                .collect();
            if loaded.is_empty() || (room.0 && loaded.len() < MAX_FITTED) {
                break;
            }
            let (_, oldest) = loaded.into_iter().min_by_key(|(t, _)| *t).expect("not empty");
            // in the engine record, so its size coming back loads it at once
            self.fitted.remove(&oldest);
            room = self.budget(&key.0).fits(key.1, page);
        }
        let (fits, bytes) = room;
        if !fits {
            fitting_done(key);
            progress::log(&format!("{name}: no room in GPU memory for tile {label} (~{} MB); staying on tile {FALLBACK}", bytes >> 20));
            return Fit::Failed;
        }
        let params = LoadParams { onnx: &key.0, tile: Some(key.1), overlap: self.overlap, async_build: true, even_tiles: false };
        let _loading = LOAD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        match self.ctx.load_model(&params) {
            Ok(model) => match model.state() {
                ModelState::Building => {
                    progress::fitted_engine(name, &label, self.backend.label(), false);
                    reserve(self.device.id, bytes);
                    return Fit::Building { model, bytes, square };
                }
                ModelState::Ready => {
                    fitting_done(key);
                    self.record_engine(key);
                    self.measure(&key.0, &model, key.1);
                    return Fit::Ready { model, last_use: 0, check: square.map(|square| Check { square, runs: 0, took: 0.0 }) };
                }
                _ => progress::log(&format!("{name}: no engine fitted to tile {label} ({}); staying on tile {FALLBACK}", self.ctx.last_error())),
            },
            Err(e) => progress::log(&format!("{name}: no engine fitted to tile {label} ({e}); staying on tile {FALLBACK}")),
        }
        fitting_done(key);
        Fit::Failed
    }

    /// Upscale `w`x`h` pixels with `channels` 1 (gray) or 3 (RGB) through the model, on `tile`
    /// (auto: see [`Upscaler::pick`]). Returns the output and its scale.
    pub fn upscale(&mut self, path: &Path, tile: Tile, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
        let pick = self.pick(path, tile, w, h);
        let (out, took) = match self.upscale_on(path, &pick, w, h, channels, pixels) {
            Ok(done) => done,
            Err(e) if pick.auto && e != "cancelled" => {
                self.drop_auto(path, &pick, &e);
                self.upscale_on(path, &Pick::tile((FALLBACK, FALLBACK)), w, h, channels, pixels)?
            }
            Err(e) => return Err(e),
        };
        let secs = took.as_secs_f64();
        if let Some((shape, gain)) = pick.rent
            && let Some(Fit::Renting { lost, square, ran, .. }) = self.fitted.get_mut(&(path.to_path_buf(), shape))
        {
            *lost += secs * gain;
            *square += secs;
            *ran += 1;
        }
        if pick.fitted {
            self.check_fitted(path, pick.shape, secs);
        }
        Ok(out)
    }

    /// Time a ready fitted engine's first pages against the square's, and drop it if it isn't
    /// faster ([`CHECK_PAGES`]).
    fn check_fitted(&mut self, path: &Path, shape: Shape, secs: f64) {
        let key = (path.to_path_buf(), shape);
        let Some(Fit::Ready { check: Some(c), .. }) = self.fitted.get_mut(&key) else { return };
        c.runs += 1;
        c.took += secs;
        if c.runs < CHECK_PAGES {
            return;
        }
        let (mean, square) = (c.took / c.runs as f64, c.square);
        if mean > square * (1.0 - CHECK_GAIN) {
            progress::log(&format!(
                "{}: tile {} ran a page in {:.0} ms against {:.0} on tile {FALLBACK}; staying on tile {FALLBACK}",
                file_name(path),
                tiles::shape_label(shape),
                mean * 1e3,
                square * 1e3
            ));
            self.fitted.insert(key, Fit::Failed);
        } else if let Some(Fit::Ready { check, .. }) = self.fitted.get_mut(&key) {
            *check = None;
        }
    }

    /// An engine auto mode chose failed on a page (`e`): drop it for the run; its pages go to the
    /// square from now on.
    fn drop_auto(&mut self, path: &Path, pick: &Pick, e: &str) {
        progress::log(&format!("{}: tile {} failed ({e}); using tile {FALLBACK} instead", file_name(path), tiles::shape_label(pick.shape)));
        if pick.fitted {
            self.fitted.insert((path.to_path_buf(), pick.shape), Fit::Failed);
        } else if pick.even {
            self.models.remove(&(path.to_path_buf(), pick.shape, true));
            self.even.insert(path.to_path_buf(), 0);
        }
    }

    /// Auto mode's engine for `w`x`h` pages of `path` once it has settled: `Some` of its tile
    /// (`"512"`, `"1216x992"`, `"1024 (even)"`), `None` while a fitted engine may still come
    /// (renting or building).
    pub fn settled(&self, path: &Path, w: u32, h: u32) -> Option<String> {
        if self.backend == Backend::TensorRt {
            return Some(match self.even.get(path) {
                Some(&t) if t > 0 => format!("{t} (even)"),
                _ => FALLBACK.to_string(),
            });
        }
        match self.shapes.get(&(path.to_path_buf(), (w, h)))? {
            None => Some(FALLBACK.to_string()),
            Some((shape, _)) => match self.fitted.get(&(path.to_path_buf(), *shape))? {
                Fit::Ready { check: None, .. } => Some(tiles::shape_label(*shape)),
                Fit::Failed => Some(FALLBACK.to_string()),
                _ => None,
            },
        }
    }

    /// The engine record's folder for a model: where its engines are cached.
    fn record_dir(&self, model: &Path) -> Option<PathBuf> {
        self.cache_dir.clone().or_else(|| model.parent().map(Path::to_path_buf))
    }

    /// Whether `key`'s engine (model, tile shape) was built before on this backend and GPU.
    fn engine_known(&self, (model, shape): &(PathBuf, Shape)) -> bool {
        let (Some(dir), Some(file)) = (self.record_dir(model), model.file_name()) else { return false };
        tiles::tiles::fitted_known(&dir, &file.to_string_lossy(), self.backend.label(), &self.device.name, *shape)
    }

    /// Note `key`'s engine in the record (best effort: a record that can't be written only costs a
    /// later run its head start).
    fn record_engine(&self, (model, shape): &(PathBuf, Shape)) {
        let (Some(dir), Some(file)) = (self.record_dir(model), model.file_name()) else { return };
        tiles::tiles::record_fitted(&dir, &file.to_string_lossy(), self.backend.label(), &self.device.name, *shape);
    }

    /// Upscale on the picked engine; with the time the inference itself took (not a model load).
    fn upscale_on(&mut self, path: &Path, pick: &Pick, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<((Vec<u8>, u32), Duration), String> {
        let key = (path.to_path_buf(), pick.shape);
        let model = if pick.fitted
            && let Some(Fit::Ready { model, .. }) = self.fitted.get(&key)
        {
            model
        } else {
            self.model(path, pick.shape, pick.even)?
        };
        let scale = model.info()?.scale;
        let (ow, oh) = (w * scale, h * scale);
        let mut out = vec![0u8; ow as usize * oh as usize * channels as usize];
        let start = Instant::now();
        match model.upscale(
            Pixels::packed(w, h, channels as u32, pixels),
            PixelsMut::packed(ow, oh, channels as u32, &mut out),
            None,
        )? {
            Upscaled::Done => Ok(((out, scale), start.elapsed())),
            Upscaled::Cancelled => Err("cancelled".into()),
        }
    }

    /// Upscale an image with alpha (`channels` 2 = gray + alpha, 4 = RGBA, straight alpha).
    ///
    /// Uniform alpha (every pixel equally transparent) skips the trick below: the color is upscaled
    /// once and the alpha rebuilt at the output size. Otherwise the color is composited once onto
    /// black and once onto white, both are upscaled, and alpha is recovered per channel from how far
    /// apart they come out (1 − (white − black)); on color the channels' estimates are blended toward
    /// their extremes (`max·mean + min·(1 − mean)`, chaiNNer's denoise, which suppresses model noise
    /// in semi-transparent areas). The color is the black pass un-premultiplied.
    pub fn upscale_with_alpha(&mut self, path: &Path, tile: Tile, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
        let cc = (channels - 1) as usize;
        let n = channels as usize;
        if let Some(a) = uniform_alpha(pixels, n) {
            let color: Vec<u8> = pixels.chunks_exact(n).flat_map(|px| px[..cc].iter().copied()).collect();
            let (up, scale) = self.upscale(path, tile, w, h, cc as u8, &color)?;
            let out = up.chunks_exact(cc).flat_map(|px| px.iter().copied().chain([a])).collect();
            return Ok((out, scale));
        }
        let (on_black, on_white) = on_black_and_white(pixels, n);
        let (black, scale) = self.upscale(path, tile, w, h, cc as u8, &on_black)?;
        let (white, _) = self.upscale(path, tile, w, h, cc as u8, &on_white)?;
        Ok((recover_alpha(&black, &white, cc), scale))
    }
}

/// The alpha value every pixel has, if they all have the same one (`n` channels, alpha last).
fn uniform_alpha(pixels: &[u8], n: usize) -> Option<u8> {
    let mut it = pixels.chunks_exact(n).map(|px| px[n - 1]);
    let first = it.next()?;
    it.all(|a| a == first).then_some(first)
}

/// The color channels of straight-alpha `pixels` (`n` channels, alpha last) composited onto black
/// and onto white.
fn on_black_and_white(pixels: &[u8], n: usize) -> (Vec<u8>, Vec<u8>) {
    let cc = n - 1;
    let mut on_black = Vec::with_capacity(pixels.len() / n * cc);
    let mut on_white = Vec::with_capacity(on_black.capacity());
    for px in pixels.chunks_exact(n) {
        let a = px[cc] as u32;
        for &v in &px[..cc] {
            on_black.push(((v as u32 * a + 127) / 255) as u8);
            on_white.push(((v as u32 * a + 255 * (255 - a) + 127) / 255) as u8);
        }
    }
    (on_black, on_white)
}

/// Straight-alpha pixels (`cc` color channels + alpha) from the upscaled black and white passes.
fn recover_alpha(black: &[u8], white: &[u8], cc: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(black.len() / cc * (cc + 1));
    for (b, wh) in black.chunks_exact(cc).zip(white.chunks_exact(cc)) {
        let est = |i: usize| (255 - (wh[i] as i32 - b[i] as i32)).clamp(0, 255) as f32 / 255.0;
        let a = if cc == 1 {
            est(0)
        } else {
            let (mut lo, mut hi, mut sum) = (1.0f32, 0.0f32, 0.0f32);
            for i in 0..cc {
                let e = est(i);
                lo = lo.min(e);
                hi = hi.max(e);
                sum += e;
            }
            let mean = sum / cc as f32;
            hi * mean + lo * (1.0 - mean)
        };
        let a = (a * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
        for &v in b {
            out.push((v as u32 * 255 + a / 2).checked_div(a).map_or(0, |c| c.min(255) as u8));
        }
        out.push(a as u8);
    }
    out
}

/// The `trtexec` that builds TensorRT engines, and an environment prefix for it: `(None, None)`
/// lets libaji use the one beside its backend library (always so on Windows, where the TensorRT
/// runtime ships next to it).
type Trtexec = (Option<PathBuf>, Option<String>);

/// On Linux, a `trtexec` bundled beside libaji finds its TensorRT libraries there (the engine
/// builds it runs get `LD_LIBRARY_PATH` pointing at that folder); without one, the system
/// TensorRT's is used: NVIDIA's apt package (libnvinfer-bin) installs it in
/// /usr/src/tensorrt/bin, which isn't on PATH, else whatever `trtexec` is on PATH.
fn trtexec(aji_dir: &Path) -> Trtexec {
    if cfg!(windows) {
        return (None, None);
    }
    if aji_dir.join("trtexec").is_file() {
        let dir = std::fs::canonicalize(aji_dir).unwrap_or_else(|_| aji_dir.to_path_buf());
        // single-quoted for the shell; a ' inside becomes '\''
        let quoted = dir.to_string_lossy().replace('\'', r"'\''");
        return (None, Some(format!("LD_LIBRARY_PATH='{quoted}'${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}")));
    }
    let on_path = std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("trtexec")).find(|t| t.is_file()));
    let system = [PathBuf::from("/usr/src/tensorrt/bin/trtexec")].into_iter().find(|t| t.is_file()).or(on_path);
    (system, None)
}

/// The TensorRT context a TensorRT-RTX upscaler falls back to, opened on first need.
struct Fallback {
    aji: Aji,
    device_id: u64,
    cache_dir: Option<PathBuf>,
    trtexec_env: Trtexec,
    ctx: Option<Result<Context, String>>,
}

impl Fallback {
    fn context(&mut self) -> Result<Context, String> {
        let (aji, id, cache, env) = (&self.aji, self.device_id, &self.cache_dir, &self.trtexec_env);
        self.ctx
            .get_or_insert_with(|| open_context(aji, Backend::TensorRt, id, cache, env))
            .clone()
            .map_err(|e| format!("TensorRT fallback: {e}"))
    }
}

fn open_context(aji: &Aji, backend: Backend, device_id: u64, cache_dir: &Option<PathBuf>, trtexec: &Trtexec) -> Result<Context, String> {
    aji.context(&ContextParams {
        backend,
        device_id,
        cache_dir: cache_dir.clone(),
        trtexec: trtexec.0.clone(),
        trtexec_env: trtexec.1.clone(),
        log: Some(std::sync::Arc::new(|level, msg| {
            if level <= 2 {
                progress::backend_log(msg);
            }
        })),
    })
}

/// Load `path` at `tile` on `ctx`, waiting out a background engine build (reported on stdout).
fn load_model(ctx: &Context, backend: Backend, path: &Path, tile: Shape, overlap: Option<u32>, even_tiles: bool, name: &str) -> Result<Model, String> {
    let model = ctx.load_model(&LoadParams { onnx: path, tile: Some(tile), overlap, async_build: true, even_tiles })?;
    if model.state() == ModelState::Building {
        progress::engine_build(name, &tiles::shape_label(tile), backend.label());
        while model.state() == ModelState::Building {
            std::thread::sleep(Duration::from_millis(250));
        }
        if model.state() == ModelState::Ready {
            progress::engine_ready(name);
        }
    }
    if model.state() == ModelState::Failed {
        return Err(ctx.last_error());
    }
    Ok(model)
}

/// The devices of one backend, as JSON for the GUI's device list:
/// `[{"id", "name", "vram", "compute_capability"}]` (`id` as a decimal string — it's a full 64-bit
/// value; `compute_capability` as "major.minor", null on DirectML). An unloadable backend is an
/// empty list, and GPUs too old for TensorRT aren't on its lists ([`split_devices`]).
pub fn devices_json(aji_dir: &Path, backend: Backend) -> Result<String, String> {
    let aji = load_aji(aji_dir)?;
    let (devices, _) = split_devices(backend, aji.devices(backend).unwrap_or_default());
    let list: Vec<_> = devices
        .iter()
        .map(|d| {
            serde_json::json!({
                "id": d.id.to_string(),
                "name": d.name,
                "vram": d.vram_bytes,
                "compute_capability": d.compute_capability.map(|(major, minor)| format!("{major}.{minor}")),
            })
        })
        .collect();
    serde_json::to_string(&list).map_err(|e| e.to_string())
}

/// `--model-info`'s JSON object, fields in the documented order.
#[derive(serde::Serialize)]
struct ModelInfo {
    precision: &'static str,
    weights: precision::Weights,
    hint: Option<String>,
}

/// The precision of the ONNX model at `path`, as JSON:
/// `{"precision", "weights": {"fp32", "fp16", "bf16"}, "hint"}`. With a `backend`, `hint` is
/// [`precision::hint`] for the device an upscale would run on (see [`hint_device`]); without one,
/// or when no device is found, it is null.
pub fn model_info_json(path: &Path, aji_dir: &Path, backend: Option<BackendChoice>, device_id: u64) -> Result<String, String> {
    let weights = precision::model_weights(path)?;
    let p = weights.precision();
    let hint = backend.and_then(|choice| {
        let (backend, device) = hint_device(aji_dir, choice, device_id)?;
        precision::hint(backend, device.compute_capability, &device.name, p)
    });
    serde_json::to_string(&ModelInfo { precision: p.as_str(), weights, hint }).map_err(|e| e.to_string())
}

/// The backend and device an upscale with `choice` and `device_id` would run on, as far as the
/// device lists tell (no context is opened): the first of [`BackendChoice::candidates`] that lists
/// a device, and on it the device with that id, else its first.
fn hint_device(aji_dir: &Path, choice: BackendChoice, device_id: u64) -> Option<(Backend, Device)> {
    let aji = match load_aji(aji_dir) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("no hint: {e}");
            return None;
        }
    };
    choice.candidates().iter().find_map(|&b| {
        let (devices, old) = split_devices(b, aji.devices(b).ok()?);
        if old.iter().any(|d| device_id != 0 && d.id == device_id) {
            return None; // as in Upscaler::open: on to the next backend
        }
        let device = devices.iter().find(|d| d.id == device_id).or(devices.first())?;
        choice.runs_on(b, device).then(|| (b, device.clone()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_alpha_is_detected() {
        assert_eq!(uniform_alpha(&[1, 2, 3, 128, 4, 5, 6, 128], 4), Some(128));
        assert_eq!(uniform_alpha(&[1, 2, 3, 128, 4, 5, 6, 127], 4), None);
        assert_eq!(uniform_alpha(&[9, 0, 8, 0], 2), Some(0));
    }

    #[test]
    fn alpha_survives_an_identity_model() {
        // With a model that changes nothing, the two passes give the alpha back (to rounding) and
        // the color un-premultiplied (exact where alpha is high, coarser where it's low).
        let px = [200u8, 100, 50, 255, 200, 100, 50, 128, 10, 20, 30, 0, 255, 255, 255, 64];
        let (b, w) = on_black_and_white(&px, 4);
        let out = recover_alpha(&b, &w, 3);
        for (o, i) in out.as_chunks::<4>().0.iter().zip(px.as_chunks::<4>().0) {
            assert!((o[3] as i32 - i[3] as i32).abs() <= 1, "alpha {} vs {}", o[3], i[3]);
            if i[3] >= 64 {
                for c in 0..3 {
                    assert!((o[c] as i32 - i[c] as i32).abs() <= 4, "color {} vs {}", o[c], i[c]);
                }
            }
        }
        let gray = [100u8, 200, 30, 0];
        let (b, w) = on_black_and_white(&gray, 2);
        let out = recover_alpha(&b, &w, 1);
        assert!((out[1] as i32 - 200).abs() <= 1 && out[3] == 0);
    }

    #[test]
    fn resolve_takes_the_first_models_folder_that_has_the_file() {
        let root = std::env::temp_dir().join(format!("mangajanai-cli-resolve-test-{}", std::process::id()));
        let (user, bundled) = (root.join("user"), root.join("bundled"));
        for d in [&user, &bundled] {
            std::fs::create_dir_all(d).unwrap();
        }
        let touch = |p: PathBuf| {
            std::fs::write(&p, b"").unwrap();
            p
        };
        let a = touch(bundled.join("a.onnx"));
        let b = touch(user.join("b.onnx"));
        touch(bundled.join("b.onnx"));
        let dirs = [user.clone(), bundled.clone()];

        assert_eq!(resolve_in("a.onnx", &dirs), Some(a.clone()));
        // The user's folder comes first.
        assert_eq!(resolve_in("b.onnx", &dirs), Some(b));
        // A path the user picked is used as is.
        assert_eq!(resolve_in(a.to_str().unwrap(), &dirs), Some(a));
        assert_eq!(resolve_in("missing.onnx", &dirs), None);
        let _ = std::fs::remove_dir_all(&root);
    }

    // ---- GPUs too old for TensorRT (compute capability below 7.5) ----------------------------

    fn device(id: u64, name: &str, cc: Option<(i32, i32)>) -> Device {
        Device { id, name: name.into(), vram_bytes: 8 << 30, compute_capability: cc }
    }

    #[test]
    fn pre_turing_gpus_are_off_the_tensorrt_lists() {
        let list = || vec![device(1, "GTX 1080", Some((6, 1))), device(2, "RTX 3080", Some((8, 6))), device(3, "Titan V", Some((7, 0)))];
        let ids = |v: &[Device]| v.iter().map(|d| d.id).collect::<Vec<_>>();
        for b in [Backend::TensorRt, Backend::TensorRtRtx] {
            let (ok, old) = split_devices(b, list());
            assert_eq!((ids(&ok), ids(&old)), (vec![2], vec![1, 3]), "{b:?}");
        }
        // DirectML runs on all of them (it lists no compute capability, but don't rely on that)
        let (ok, old) = split_devices(Backend::DirectMl, list());
        assert_eq!((ids(&ok), ids(&old)), (vec![1, 2, 3], vec![]));
        // a device without a compute capability isn't judged
        let (ok, _) = split_devices(Backend::TensorRt, vec![device(4, "?", None)]);
        assert_eq!(ids(&ok), [4]);
    }

    #[test]
    fn auto_tries_tensorrt_rtx_first_on_windows_where_it_is_the_default() {
        let auto = BackendChoice::Auto.candidates();
        if cfg!(windows) {
            assert_eq!(auto, [Backend::TensorRtRtx, Backend::TensorRt, Backend::DirectMl]);
        } else {
            assert_eq!(auto, [Backend::TensorRt]);
        }
        // the GPUs MangaJaNai GUI and yosh default to TensorRT-RTX on
        for cc in [(7, 5), (8, 6), (8, 9), (12, 0), (12, 1), (13, 0)] {
            assert!(BackendChoice::Auto.runs_on(Backend::TensorRtRtx, &device(1, "", Some(cc))), "{cc:?}");
        }
        // pre-Turing, the data-center parts TensorRT-RTX doesn't support, and a device that
        // reports no compute capability
        for cc in [Some((6, 1)), Some((7, 0)), Some((8, 0)), Some((8, 7)), Some((9, 0)), Some((10, 0)), Some((11, 0)), None] {
            assert!(!BackendChoice::Auto.runs_on(Backend::TensorRtRtx, &device(1, "", cc)), "{cc:?}");
        }
        // Auto's other backends, and every explicit choice, run on whatever they list
        for cc in [Some((9, 0)), None] {
            let d = device(1, "", cc);
            assert!(BackendChoice::Auto.runs_on(Backend::TensorRt, &d) && BackendChoice::Auto.runs_on(Backend::DirectMl, &d));
            assert!(BackendChoice::TensorRtRtx.runs_on(Backend::TensorRtRtx, &d));
            assert!(BackendChoice::TensorRt.runs_on(Backend::TensorRt, &d));
        }
    }

    #[test]
    fn too_old_message_says_why() {
        let m = too_old_message(&[device(1, "NVIDIA GeForce GTX 1080", Some((6, 1)))]);
        assert!(m.starts_with("TensorRT needs an NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability 7.5+); NVIDIA GeForce GTX 1080 is compute 6.1. "), "{m}");
        assert!(m.ends_with(if cfg!(windows) { "Use DirectML instead." } else { "On Linux mangajanai runs on TensorRT only." }), "{m}");
    }
}
