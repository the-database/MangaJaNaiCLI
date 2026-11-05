# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MangaJaNaiCLI: one Rust program under two names (`src/cli.rs::main`):

- `mangajanai_upscale` is MangaJaNai GUI's upscaling backend
  ([MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui)). The GUI runs it
  as `mangajanai_upscale.exe` with `--settings appstate2.json --aji-dir --backend --cache-dir
  --models-dir x2 [--device <id>]`, plus `--list-devices` and `--model-info` for queries. It reads
  the stdout line protocol in `progress.rs` (`PROGRESS=`, `TOTALZIP=`, `ENGINE_BUILD`, `ENGINE_READY`,
  `BACKEND_FALLBACK`, `HINT`, `ERROR`).
  **This interface is a contract: don't change the exe name, the flags the GUI passes, or a line of
  the protocol.** Check a change by running the same `appstate2.json` through the old and new
  binaries and comparing stdout and output digests (cbz entries compare by name/CRC; zip timestamps
  differ).
- `mangajanai` is the command line: flags plus `mangajanai.ini` (`config.rs`), with human progress
  output on stderr (`progress.rs`'s Human mode). The format is deliberately forgiving plain INI, for
  non-technical users: values are taken as typed, switches accept yes/no, a list is a repeated line,
  and errors name the line. Don't move it to a typed format.

Pipeline (`run.rs`): walk inputs → parallel prepare (decode, gray detection, chain choice, pre-resize,
levels; `process.rs`) → one inference thread per GPU stream (`upscaler.rs`, libaji's image ABI) →
parallel finish (final resize, encode; `encode.rs`) → writes in input order.

## Dependencies

- **yosh-image** is a git dependency pinned to a commit of
  [the-database/yosh](https://github.com/the-database/yosh). It provides decoders, archives,
  screentone-safe resizing and the runtime-loaded libaji bindings (`yosh_image::aji`). A change there
  is a yosh PR first, then a bump of `rev` in `Cargo.toml`: `bump-pins.yml` (`yosh-image`) opens
  that PR after a yosh release, or by hand for any yosh ref.
- **libaji** ([animejanai-inference](https://github.com/the-database/animejanai-inference),
  `include/aji_image.h`) is loaded at run time from `--aji-dir`. The default is next to the
  program, else (command line, Windows) an installed MangaJaNai GUI's `inference` folder.
  `AJI_VERSION` in `src/setup.rs` pins the libaji release: both archives bundle it (the release
  workflow reads it from there) and `setup` fetches TensorRT-RTX from it.

## Commands

```sh
cargo build --release
cargo test
cargo clippy --all-targets
```

## Releases

`.github/workflows/release.yml` builds and publishes on a `v*` tag, which must equal `Cargo.toml`'s
version. Assets:

- `mangajanai-cli-windows-x64.zip` holds `mangajanai_upscale.exe` alone. MangaJaNaiConverterGui's
  `BuildMangaJaNai/Program.cs` downloads it by tag (`CliVersion`). Keep its name and contents.
- `mangajanai-windows-x64.zip` holds `mangajanai.exe` + libaji + the VC++ runtime DLLs.
- `mangajanai-linux-x64.tar.zst` holds `mangajanai` + libaji.

`mangajanai setup` (`src/setup.rs`) installs the rest next to the program, tracked in `setup.json`.
It picks the components from the GPUs it finds through NVML (`src/nvml.rs`):

- The runtime packs: TensorRT's runtime, plus one builder-resource pack per GPU generation, and
  ONNX Runtime + DirectML. They live on their own `runtime-<n>` release, published by
  `.github/workflows/runtime-packs.yml` (`scripts/runtime_packs.py`), not on CLI releases.
  `setup.rs` pins `RUNTIME_TAG` and the SHA-256 of each `packs-<platform>.json` index.
- TensorRT-RTX, from the libaji release.
- The models: the same archives MangaJaNai GUI installs.

To change a TensorRT/ORT/DirectML version:

1. Bump `TAG` and the pins in `runtime_packs.py`.
2. Run the workflow.
3. Merge the PR it asks `bump-pins.yml` for, which updates `RUNTIME_TAG` and the index hashes.

`--mirror <folder>` installs from local archives, for testing before anything is published.

**yosh installs its AI add-on with this program**: it downloads `mangajanai-windows-x64.zip` /
`mangajanai-linux-x64.tar.zst` from a release it pins (by SHA-256) into its add-on folder and runs
`mangajanai setup --progress lines` there, reading the `SETUP_*` stdout lines (`setup.rs`, `Lines`).
Those archives' layout (libaji at the root) and that protocol are a contract too: extend it, don't
change a line.

Releases before the move (0.1.x) are on the-database/yosh under `mangajanai-cli-v*` tags.

### Bump PRs

A `v*` release asks yosh and MangaJaNai GUI to open PRs that pin it (`notify-downstream.yml`).
Here, `bump-pins.yml` runs `scripts/bump_pins.py <source> <tag>` and opens a PR on `bump/<source>`
for a libaji release (`libaji`, sent by animejanai-inference), the runtime packs
(`runtime-packs`) and yosh-image (`yosh-image`, sent by a yosh release). The script downloads each
pinned asset for its SHA-256; run it locally to check a bump. Both workflows need the
`JANAI_BUMP_TOKEN` secret.

## Conventions

- Commit author is `the-database <25811902+the-database@users.noreply.github.com>`, with a short
  imperative subject and no Co-Authored-By trailer.
- Be sparing with CI runs: validate locally, and don't tag just to test.
