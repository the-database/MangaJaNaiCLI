# MangaJaNaiCLI

Batch image and manga upscaling from the command line, with ONNX models on TensorRT (also DirectML
and TensorRT-RTX on Windows). The command-line counterpart of
[MangaJaNai GUI](https://github.com/the-database/MangaJaNaiGUI) and the successor to the
Python [MangaJaNaiTRT](https://github.com/the-database/MangaJaNaiTRT). It's built from the same
parts as the GUI: [yosh](https://github.com/the-database/yosh)'s imaging core for decoding,
archives, grayscale detection and screentone-safe resizing, and
[libaji](https://github.com/the-database/animejanai-inference) for inference. Models:
[MangaJaNai](https://github.com/the-database/MangaJaNai).

The command is `mangajanai`. The same program is MangaJaNai GUI's upscaling backend,
`mangajanai_upscale`, which the GUI runs with `--settings`.

- Images (png jpg webp bmp avif tif jxl), archives (zip cbz rar cbr 7z cb7, written back as .cbz in page
  order) and folders (recursive; the output keeps the subfolders).
- Animations (WebP, APNG, JPEG XL or AVIF, and GIF inside archives, with more than one frame) are
  copied unchanged, keeping their format, instead of being upscaled as one frame. Images with an EXIF orientation (JPEG, PNG,
  WebP, TIFF) are upscaled upright.
- One model for everything, MangaJaNai's per-page rules (gray pages by height, color pages), your own
  rules, or a workflow exported from MangaJaNai GUI.
- Resize after the model to a scale, width, height or a box; webp or jxl (lossy/lossless), png, jpg,
  avif.
- Transparency: uniform alpha is kept as is; varying alpha is recovered with the black/white two-pass
  trick, denoised and un-premultiplied.
- Engines are built once per model, tile size and GPU, and cached. Several GPUs, or several streams on
  one, share the work.

## Install

1. Download `mangajanai-windows-x64.zip` or `mangajanai-linux-x64.tar.zst` from
   [Releases](https://github.com/the-database/MangaJaNaiCLI/releases) and unpack it anywhere.
2. Run `mangajanai setup`. It detects the GPU and downloads what this PC needs, into the same folder:
   - the TensorRT runtime and the builder files for your GPU's generation (NVIDIA GeForce RTX 20 /
     GTX 16 series and newer; older NVIDIA GPUs, like the GTX 10 series, run on DirectML);
   - TensorRT-RTX (Windows, RTX 20 series and newer);
   - DirectML (Windows, for any GPU);
   - the MangaJaNai and IllustrationJaNai models (skip them with `--no-models`).

   `mangajanai setup --check` lists what it would download without downloading. Run `mangajanai setup`
   again after an update to fetch what changed. Every download is checked against a pinned SHA-256.
3. Run `mangajanai <input> -o <output>`.

All you need beforehand is a GPU driver. On Linux, mangajanai runs on TensorRT only, so it needs an
NVIDIA GeForce RTX 20 / GTX 16 series or newer GPU (compute capability 7.5+) and driver. The runtime comes from setup; no CUDA or TensorRT install is needed.

On Windows, if you skip setup and MangaJaNai GUI is installed, mangajanai uses the GUI's inference
runtime and models. To use a runtime from elsewhere instead (e.g. an mpv-AnimeJaNai
`animejanai\inference` folder), pass `--aji-dir <folder>`. On Linux, a system TensorRT 11 from NVIDIA's
CUDA apt repository works too: `trtexec` is then found on `PATH`.

### The shared runtime folder

An app's install of mangajanai (yosh's AI add-on) can share the TensorRT runtime and the models
with your other JaNai apps, in one per-user folder (`%LOCALAPPDATA%\the-database\aji`, or
`~/.local/share/the-database/aji`). To put those few GB on another drive:

```sh
mangajanai store location            # where it is, and whether its folder is there
mangajanai store move D:\JaNai       # move it to a new or empty folder; nothing is downloaded again
mangajanai store move --default      # move it back
mangajanai store reset               # the moved folder is gone for good: start an empty one
```

Close the other JaNai apps first. The default path becomes a link to the new folder, so every app
keeps working. While that folder is missing (an unplugged drive), mangajanai stops with a message
saying so.

## Use

Keep your settings in a file so a run is just `mangajanai`:

```sh
mangajanai init          # writes a commented mangajanai.ini
# edit it in any text editor
mangajanai               # runs it
mangajanai --format png  # the same, one setting changed for this run
```

```ini
[input]
path = D:\manga\My Series
path = D:\manga\one more chapter.cbz

[output]
folder = D:\manga\upscaled
format = webp
quality = 90
suffix = _x2

[model]
file = 2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx

[resize]
height = 2400
```

The file is `-c <file>`, else `mangajanai.ini` in the current folder, else next to the program.
Flags override it, and it overrides a GUI workflow it imports. Values are written as you'd say them:
paths as they are (quotes from "Copy as path" are fine), `yes`/`no`, `2x`, `90%`; a list is the same
line repeated; an empty value means the default. Mistakes stop the run with the line number and, for
a misspelled setting, the right name. `mangajanai init` documents every setting.

Or use flags only:

```sh
mangajanai chapter.cbz -o out --model 4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx
mangajanai manga/ -o out --preset manga --height 2400 --format webp -q 90
mangajanai manga/ -o out --workflow "Upscale Manga.mwf" --device all
mangajanai devices                       # GPUs, numbered for --device
mangajanai bench 2x_model.onnx --tile 512 --iters 20
```

`mangajanai --help` lists every flag.

### Speed

- **`--even-tiles`** (`[gpu] even_tiles = yes`), TensorRT: fixed tiles are pushed flush against a
  page's edges and often re-infer most of their area. With even tiles, each page is split into
  tiles of at most `--tile` from one engine that takes any tile shape. With MangaJaNai ESRGAN 4x
  on 24 pages (RTX 5090) the time in inference went from 19.3 s to 15.0 s at `--tile 512`, and
  to 11.8 s at `--tile 1024`, which needs ~2.5x the VRAM. It's off by default: it's only been
  measured on one GPU, and on TensorRT-RTX the shape-range engine runs ~30% slower per pixel.
- **TensorRT-RTX** (`--backend tensorrt-rtx`, Windows, RTX 20 series and newer) builds engines in
  seconds instead of minutes. Its speed relative to TensorRT depends on the model: on an RTX 5090
  it was ~25% faster with MangaJaNai ESRGAN and about the same with FDAT_M. A model that fails on
  it falls back to TensorRT. Auto tries it first on the GPUs `setup` installs it for (RTX 20 /
  GTX 16 series and newer), then TensorRT, then DirectML, as MangaJaNai GUI and yosh do.
- **Tile size** is one engine per size. 512 is a good default; larger tiles help light models.
  Tiles overlap by 128 px and blend only at the centre of the overlap, which keeps generated
  screentone seamless; a smaller `--overlap` is faster but can show seams.
- **`--workers <n>`** (`[output] workers`) sets how many threads decode and encode images. The
  default, half the CPU's threads up to 8 and at least 4 per GPU, keeps one GPU busy with any
  format. Several GPUs can outrun the encoders: on 4x RTX 5090 (48 threads, 1050x1500 pages at
  2x), lossless JPEG XL ran at 4.3 pages/s with 8 workers, 8.3 with 16, 11.2 with 24 and 14.3
  with 48 (WebP and lossy JPEG XL reach the GPUs' 17 pages/s at 16). Each worker holds an image,
  so more workers use more memory (about 180 MB each for 3360x4800 lossless JPEG XL pages).
- **`--streams 2`** runs two inference threads per GPU. It doubles VRAM. On TensorRT and
  TensorRT-RTX, current libaji tiles on the GPU, so this rarely helps any more. DirectML still
  tiles on the CPU, where one image's tiling can overlap the other's inference, so it may help
  there; that hasn't been measured.

## From the Python MangaJaNaiTRT

| MangaJaNaiTRT config.ini | mangajanai.ini / flag |
|---|---|
| `[paths] input_path` | `[input] path` / positional inputs |
| `[paths] input_onnx` | `[model] file` / `--model` |
| `[paths] output_dir` | `[output] folder` / `-o` |
| `[paths] engine_cache_dir` | `[gpu] engine_folder` / `--cache-dir` |
| `[output] format`, `quality`, `webp_lossless` | `[output] format`, `quality`, `lossless` |
| `[output] note` | `[output] suffix = _note` / `--suffix` |
| `[output] skip_existing` | `[output] skip_existing` / `--skip-existing`, `--overwrite` |
| `[tiling] dynamic_shape_opt` / `max` | `[gpu] tile` / `--tile` (libaji builds one static engine per tile) |
| `[tiling] tile_overlap` | `[gpu] overlap` / `--overlap` |
| `[trt] device_id = 0,1` | `[gpu] devices = 0,1` / `--device 0,1` |
| `[tiling] dynamic_shape_min`, `tile_align`; `[trt] batch_size`, `workspace_gb`, `opt_level`; `[runtime]` | not needed: libaji handles engine building, padding and pipelining |
| `--clean-engines` | not needed: stale engines are removed automatically |

Not carried over: URL inputs, DDS inputs, and the VRAM monitor.

## Build

```sh
cargo build --release                  # target/release/mangajanai(.exe) and mangajanai_upscale(.exe)
cargo build --release --features avif  # with AVIF input decoding (needs nasm + dav1d, as CI builds it)
cargo build --release --no-default-features  # without JPEG XL output (no CMake or clang-cl needed)
cargo test
```

The C/C++ dependencies are libwebp, UnRAR and libjxl, built from source by their crates (a C/C++
compiler is needed). libjxl (JPEG XL output, the default `jxl` feature) also needs CMake and, on
Windows, clang-cl: the Visual Studio component "C++ Clang tools for Windows". libaji is loaded at run
time, so building needs no CUDA or TensorRT.

## Related projects
- [mpv-AnimeJaNai](https://github.com/the-database/mpv-AnimeJaNai): Real-time anime upscaling in mpv
- [VideoJaNai](https://github.com/the-database/VideoJaNai): Upscale video files
- [MangaJaNai GUI](https://github.com/the-database/MangaJaNaiGUI): Upscale manga and images
- [yosh](https://github.com/the-database/yosh): Fast manga, comic and image reader
- [traiNNer-redux](https://github.com/the-database/traiNNer-redux): Train your own upscaling models

## License

GPL-3.0, like MangaJaNai GUI. See [LICENSE](LICENSE).
