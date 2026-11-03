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
  is a yosh PR first, then a bump of `rev` in `Cargo.toml`.
- **libaji** ([animejanai-inference](https://github.com/the-database/animejanai-inference),
  `include/aji_image.h`) is loaded at run time from `--aji-dir`. The default is next to the
  program, else (command line, Windows) an installed MangaJaNai GUI's `inference` folder.
  `AJI_VERSION` in the release workflow pins the libaji that the Linux bundle carries.

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
- `mangajanai-windows-x64.zip` holds `mangajanai.exe`.
- `mangajanai-linux-x64.tar.zst` holds `mangajanai` + libaji.

Releases before the move (0.1.x) are on the-database/yosh under `mangajanai-cli-v*` tags.

## Conventions

- Commit author is `the-database <25811902+the-database@users.noreply.github.com>`, with a short
  imperative subject and no Co-Authored-By trailer.
- Be sparing with CI runs: validate locally, and don't tag just to test.
