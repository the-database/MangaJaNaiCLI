//! NVIDIA GPU detection through NVML (`nvml.dll` / `libnvidia-ml.so.1`, installed with the driver),
//! loaded at run time like MangaJaNai GUI's updater does: no driver, no NVIDIA GPU.

use std::ffi::{c_char, c_int, c_uint, c_void};

#[derive(Clone, Debug)]
pub struct Gpu {
    pub name: String,
    pub major: i32,
    pub minor: i32,
}

/// The oldest compute capability TensorRT and TensorRT-RTX run on: Turing (GeForce RTX 20 / GTX 16
/// series). The TensorRT 11 on CUDA 13 that setup installs supports 7.5 and up, and libaji's CUDA
/// kernels start there; PTX only compiles forward, so the `trt-ptx` pack can't help an older GPU.
/// Such a GPU counts as no NVIDIA GPU: DirectML on Windows, nothing on Linux.
pub const TRT_MIN_COMPUTE: (i32, i32) = (7, 5);

/// Whether TensorRT can run on a GPU of this compute capability ([`TRT_MIN_COMPUTE`]).
pub fn trt_capable(major: i32, minor: i32) -> bool {
    (major, minor) >= TRT_MIN_COMPUTE
}

/// "TensorRT needs an NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability 7.5+)"
pub fn trt_needs() -> String {
    let (major, minor) = TRT_MIN_COMPUTE;
    format!("TensorRT needs an NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability {major}.{minor}+)")
}

impl Gpu {
    /// "sm86", the name TensorRT's builder resources and the runtime packs use.
    pub fn sm(&self) -> String {
        format!("sm{}{}", self.major, self.minor)
    }

    pub fn trt_capable(&self) -> bool {
        trt_capable(self.major, self.minor)
    }
}

/// The GPUs TensorRT can run on, from [`gpus`]: the others count as no NVIDIA GPU.
pub fn trt_gpus(gpus: &[Gpu]) -> Vec<Gpu> {
    gpus.iter().filter(|g| g.trt_capable()).cloned().collect()
}

type Device = *mut c_void;

/// The NVIDIA GPUs, or an empty list when there is no NVIDIA driver (or NVML fails).
pub fn gpus() -> Vec<Gpu> {
    detect().unwrap_or_default()
}

fn detect() -> Option<Vec<Gpu>> {
    let name = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };
    // SAFETY: NVML's initializers are the driver's own; nothing else runs at load.
    let lib = unsafe { libloading::Library::new(name) }.ok()?;
    // SAFETY: the symbols have these C signatures in every NVML since the _v2 entry points (2014).
    unsafe {
        let init: libloading::Symbol<unsafe extern "C" fn() -> c_int> = lib.get(b"nvmlInit_v2\0").ok()?;
        let shutdown: libloading::Symbol<unsafe extern "C" fn() -> c_int> = lib.get(b"nvmlShutdown\0").ok()?;
        let count: libloading::Symbol<unsafe extern "C" fn(*mut c_uint) -> c_int> = lib.get(b"nvmlDeviceGetCount_v2\0").ok()?;
        let handle: libloading::Symbol<unsafe extern "C" fn(c_uint, *mut Device) -> c_int> =
            lib.get(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
        let cc: libloading::Symbol<unsafe extern "C" fn(Device, *mut c_int, *mut c_int) -> c_int> =
            lib.get(b"nvmlDeviceGetCudaComputeCapability\0").ok()?;
        let dev_name: libloading::Symbol<unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int> =
            lib.get(b"nvmlDeviceGetName\0").ok()?;
        if init() != 0 {
            return None;
        }
        let mut out = Vec::new();
        let mut n: c_uint = 0;
        if count(&mut n) == 0 {
            for i in 0..n {
                let mut d: Device = std::ptr::null_mut();
                if handle(i, &mut d) != 0 {
                    continue;
                }
                let (mut major, mut minor) = (0, 0);
                if cc(d, &mut major, &mut minor) != 0 {
                    continue;
                }
                let mut buf = [0 as c_char; 96];
                let name = if dev_name(d, buf.as_mut_ptr(), buf.len() as c_uint) == 0 {
                    std::ffi::CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
                } else {
                    "NVIDIA GPU".into()
                };
                out.push(Gpu { name, major, minor });
            }
        }
        shutdown();
        Some(out)
    }
}
