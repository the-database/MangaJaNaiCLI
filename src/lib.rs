//! Batch manga/image upscaling on yosh-image + libaji: images and archives are decoded, matched to a
//! chain (which model, if any), upscaled through libaji (TensorRT, DirectML or TensorRT-RTX),
//! resized to the target and encoded. Two programs run it (`cli::main`): `mangajanai_upscale`, the
//! name MangaJaNai GUI runs, and `mangajanai`, the command line.

pub mod chain;
pub mod cli;
pub mod config;
pub mod download;
pub mod encode;
pub mod levels;
pub mod nvml;
pub mod precision;
pub mod process;
pub mod progress;
pub mod run;
pub mod settings;
pub mod setup;
pub mod store;
pub mod upscaler;
