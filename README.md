# MangaJaNaiCLI

Batch image and manga upscaling from the command line, with ONNX models on TensorRT (also DirectML
and TensorRT-RTX on Windows). The command-line counterpart of
[MangaJaNai GUI](https://github.com/the-database/MangaJaNaiConverterGui) and the successor to the
Python [MangaJaNaiTRT](https://github.com/the-database/MangaJaNaiTRT). It's built from the same
parts as the GUI: [yosh](https://github.com/the-database/yosh)'s imaging core for decoding,
archives, grayscale detection and screentone-safe resizing, and
[libaji](https://github.com/the-database/animejanai-inference) for inference. Models:
[MangaJaNai](https://github.com/the-database/mangajanai).

The command is `mangajanai`. The same program is MangaJaNai GUI's upscaling backend,
`mangajanai_upscale`, which the GUI runs with `--settings`.

- Images (png jpg webp bmp avif tif jxl), archives (zip cbz rar cbr 7z cb7, written back as .cbz in page
  order) and folders (recursive; the output keeps the subfolders).
- One model for everything, MangaJaNai's per-page rules (gray pages by height, color pages), your own
  rules, or a workflow exported from MangaJaNai GUI.
- Resize after the model to a scale, width, height or a box; webp (lossy/lossless), png, jpg, avif.
- Transparency: uniform alpha is kept as is; varying alpha is recovered with the black/white two-pass
  trick, denoised and un-premultiplied.
- Engines are built once per model, tile size and GPU, and cached. Several GPUs, or several streams on
  one, share the work.

## Install

Download from [Releases](https://github.com/the-database/MangaJaNaiCLI/releases).

**Windows**: unzip `mangajanai-windows-x64.zip` anywhere. With MangaJaNai GUI installed nothing
else is needed: mangajanai uses the GUI's inference runtime and finds its models. Otherwise extract
libaji (`aji-windows-x64.zip` from animejanai-inference's releases) and the TensorRT runtime next to
`mangajanai.exe`, or point `--aji-dir` at a folder that has them (e.g. an mpv-AnimeJaNai
`animejanai\inference` folder).

**Linux** (TensorRT only): an NVIDIA driver plus TensorRT 11 from NVIDIA's CUDA apt repository:

```sh
sudo apt-get install libnvinfer11 libnvinfer-plugin11 libnvonnxparsers11 libnvinfer-bin
tar --zstd -xf mangajanai-linux-x64.tar.zst -C ~/mangajanai   # mangajanai + libaji
```

`trtexec` (which builds engines) is found in `/usr/src/tensorrt/bin`, where `libnvinfer-bin` puts it,
on `PATH`, or beside `libaji.so`.

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

- **`--streams 2`** runs two inference threads per GPU, so one image's tile copies and blending overlap
  the other's inference. On an RTX 5090 that was ~45% faster with a light model (SPAN_S) and ~20%
  with FDAT_M, for about twice the VRAM.
- **TensorRT-RTX** (`--backend tensorrt-rtx`, Windows, RTX 20 series and newer) builds engines in
  seconds instead of minutes and ran ~15% faster than TensorRT on an RTX 5090. A model that fails on
  it falls back to TensorRT. Auto is TensorRT, then DirectML.
- **Tile size** is one engine per size. 512 is a good default; larger tiles help light models.

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
cargo test
```

The only C dependencies are libwebp and UnRAR, built from source by their crates (a C/C++ compiler is
needed). libaji is loaded at run time, so building needs no CUDA or TensorRT.

## License

GPL-3.0, like MangaJaNai GUI. See [LICENSE](LICENSE).
