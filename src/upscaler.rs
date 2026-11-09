//! The inference side: one libaji context on the chosen backend/device, the models loaded so far
//! (by file and tile), and the upscale itself — including recovering alpha for transparent images.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

use yosh_image::aji::{Aji, Backend, Context, ContextParams, Device, LoadParams, Model, ModelState, Pixels, PixelsMut, Upscaled};

use crate::{nvml, precision, progress};

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
    /// TensorRT if an NVIDIA GPU it runs on (`nvml::TRT_MIN_COMPUTE`) and the TensorRT runtime
    /// are there, else DirectML.
    Auto,
    TensorRt,
    DirectMl,
    /// TensorRT for RTX (opt-in, experimental; needs aji_trtrtx.dll beside aji.dll), falling
    /// back to TensorRT when it can't start, and per model when one fails to build.
    TensorRtRtx,
}

impl BackendChoice {
    /// The backends `Upscaler::open` tries, in order. Off Windows there is only TensorRT, so Auto
    /// reports TensorRT's own error there instead of a DirectML one.
    pub fn candidates(self) -> &'static [Backend] {
        match self {
            BackendChoice::Auto if cfg!(windows) => &[Backend::TensorRt, Backend::DirectMl],
            BackendChoice::Auto => &[Backend::TensorRt],
            BackendChoice::TensorRt => &[Backend::TensorRt],
            BackendChoice::DirectMl => &[Backend::DirectMl],
            BackendChoice::TensorRtRtx => &[Backend::TensorRtRtx, Backend::TensorRt],
        }
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
    models: HashMap<(PathBuf, u32), Model>,
    /// Input-pixel tile overlap; `None` = libaji's default (128 since its seam fix).
    overlap: Option<u32>,
    /// Split images into even tiles from a shape-range engine (`LoadParams::even_tiles`).
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
        for &backend in choice.candidates() {
            if choice == BackendChoice::TensorRtRtx && backend == Backend::TensorRt {
                if too_old {
                    break; // TensorRT doesn't run on these GPUs either
                }
                progress::backend_fallback(None, &err);
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
            match open_context(&aji, backend, device.id, &cache_dir, &trtexec_env) {
                Ok(ctx) => {
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
                        overlap,
                        even_tiles,
                        description,
                    });
                }
                Err(e) => err = format!("{}: {e}", backend_name(backend)),
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

    /// The loaded model for `path` at `tile`, loading it (and waiting out a first-use TensorRT
    /// engine build, reported on stdout) if needed. On TensorRT-RTX, a model that fails to load
    /// or build is retried on TensorRT (that context opens the first time it's needed). The first
    /// load of a file prints its precision hint, if it has one, before anything else about it.
    fn model(&mut self, path: &Path, tile: u32) -> Result<&Model, String> {
        let key = (path.to_path_buf(), tile);
        if !self.models.contains_key(&key) {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            if self.hinted.insert(path.to_path_buf()) {
                // A file that can't be read is reported by the load below.
                let p = precision::model_weights(path).map_or(precision::Precision::Unknown, |w| w.precision());
                let d = &self.device;
                if let Some(h) = precision::hint(self.backend, d.compute_capability, &d.name, p) {
                    progress::hint(&name, &h);
                }
            }
            let _loading = LOAD_LOCK.lock().unwrap_or_else(|p| p.into_inner());
            let model = match load_model(&self.ctx, self.backend, path, tile, self.overlap, self.even_tiles, &name) {
                Ok(m) => m,
                Err(e) => {
                    let Some(fb) = self.fallback.as_mut() else {
                        return Err(format!("{name}: {e}"));
                    };
                    progress::backend_fallback(Some(&name), &e);
                    let ctx = fb.context()?;
                    load_model(&ctx, Backend::TensorRt, path, tile, self.overlap, self.even_tiles, &name).map_err(|e| format!("{name}: {e}"))?
                }
            };
            self.models.insert(key.clone(), model);
        }
        Ok(&self.models[&key])
    }

    /// Upscale `w`x`h` pixels with `channels` 1 (gray) or 3 (RGB) through the model. Returns the
    /// output and its scale.
    pub fn upscale(&mut self, path: &Path, tile: u32, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
        let model = self.model(path, tile)?;
        let scale = model.info()?.scale;
        let (ow, oh) = (w * scale, h * scale);
        let mut out = vec![0u8; ow as usize * oh as usize * channels as usize];
        match model.upscale(
            Pixels::packed(w, h, channels as u32, pixels),
            PixelsMut::packed(ow, oh, channels as u32, &mut out),
            None,
        )? {
            Upscaled::Done => Ok((out, scale)),
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
    pub fn upscale_with_alpha(&mut self, path: &Path, tile: u32, w: u32, h: u32, channels: u8, pixels: &[u8]) -> Result<(Vec<u8>, u32), String> {
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
fn load_model(ctx: &Context, backend: Backend, path: &Path, tile: u32, overlap: Option<u32>, even_tiles: bool, name: &str) -> Result<Model, String> {
    let model = ctx.load_model(&LoadParams { onnx: path, tile: Some((tile, tile)), overlap, async_build: true, even_tiles })?;
    if model.state() == ModelState::Building {
        progress::engine_build(name, tile, backend.label());
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
        Some((b, device.clone()))
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
    fn too_old_message_says_why() {
        let m = too_old_message(&[device(1, "NVIDIA GeForce GTX 1080", Some((6, 1)))]);
        assert!(m.starts_with("TensorRT needs an NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability 7.5+); NVIDIA GeForce GTX 1080 is compute 6.1. "), "{m}");
        assert!(m.ends_with(if cfg!(windows) { "Use DirectML instead." } else { "On Linux mangajanai runs on TensorRT only." }), "{m}");
    }
}
