//! What a run reports, in one of two modes:
//!
//! - **Lines** (the default, and always with `--settings`): the stdout protocol MangaJaNaiConverterGui
//!   reads. The `PROGRESS=` and `TOTALZIP=` lines are the ones the 1.x Python backend printed, so the
//!   GUI's progress bars work unchanged; everything else is a log line for its console.
//! - **Human** (the command line): a progress bar with rate and ETA on stderr, notices for engine
//!   builds, fallbacks and errors, per-file detail only with `--verbose`, and a summary at the end.

use std::io::{IsTerminal, Write};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static HUMAN: OnceLock<Mutex<Human>> = OnceLock::new();

fn line(s: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{s}");
    let _ = out.flush();
}

/// Switch to human output for the rest of the process. `total` is the number of items (images and
/// archive entries) the run will write, when known up front.
pub fn set_human(verbose: bool, total: Option<u64>) {
    let tty = std::io::stderr().is_terminal();
    let _ = HUMAN.set(Mutex::new(Human {
        verbose,
        tty,
        total,
        done: 0,
        skipped: 0,
        animated: 0,
        errors: 0,
        start: Instant::now(),
        first_done: None,
        last_draw: None,
        bar_len: 0,
    }));
}

fn human() -> Option<std::sync::MutexGuard<'static, Human>> {
    HUMAN.get().map(|h| h.lock().unwrap_or_else(|p| p.into_inner()))
}

struct Human {
    verbose: bool,
    tty: bool,
    total: Option<u64>,
    done: u64,
    skipped: u64,
    /// Animations copied unchanged (counted in `done` too).
    animated: u64,
    errors: u64,
    start: Instant,
    /// When the first item finished: the rate is measured from there, so a first-use engine build
    /// doesn't drag the ETA for the rest of the run.
    first_done: Option<(Instant, u64)>,
    last_draw: Option<Instant>,
    /// The width of the bar on screen (0 = none), to blank it with spaces (no VT escapes needed).
    bar_len: usize,
}

fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs();
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m{:02}s", s / 60, s % 60),
        _ => format!("{}h{:02}m", s / 3600, s / 60 % 60),
    }
}

impl Human {
    fn rate(&self) -> Option<f64> {
        let (t0, n0) = self.first_done?;
        let secs = t0.elapsed().as_secs_f64();
        (self.done > n0 && secs >= 0.5).then(|| (self.done - n0) as f64 / secs)
    }

    fn status(&self) -> String {
        let elapsed = fmt_duration(self.start.elapsed());
        let rate = self.rate();
        let rate_s = rate.map_or(String::new(), |r| format!("  {r:.2} img/s"));
        let errs = if self.errors > 0 { format!("  {} failed", self.errors) } else { String::new() };
        match self.total {
            Some(total) if total > 0 => {
                let frac = (self.done as f64 / total as f64).min(1.0);
                const W: usize = 24;
                let filled = (frac * W as f64).round() as usize;
                let eta = rate
                    .filter(|&r| r > 0.0)
                    .map_or(String::new(), |r| format!("  ETA {}", fmt_duration(Duration::from_secs_f64(total.saturating_sub(self.done) as f64 / r))));
                format!(
                    "{:>3}% [{}{}] {}/{}{rate_s}{eta}  {elapsed}{errs}",
                    (frac * 100.0) as u32,
                    "#".repeat(filled),
                    "-".repeat(W - filled),
                    self.done,
                    total
                )
            }
            _ => format!("{} done{rate_s}  {elapsed}{errs}", self.done),
        }
    }

    fn clear(&mut self) {
        if self.bar_len > 0 {
            let _ = write!(std::io::stderr(), "\r{}\r", " ".repeat(self.bar_len));
            self.bar_len = 0;
        }
    }

    /// Redraw the bar (a terminal: in place, at most 10 times a second) or print a status line
    /// (not a terminal: every 10 seconds).
    fn draw(&mut self, force: bool) {
        let every = if self.tty { Duration::from_millis(100) } else { Duration::from_secs(10) };
        if !force && self.last_draw.is_some_and(|t| t.elapsed() < every) {
            return;
        }
        self.last_draw = Some(Instant::now());
        let mut err = std::io::stderr().lock();
        if self.tty {
            let s = self.status();
            let n = s.chars().count();
            let _ = write!(err, "\r{s}{}", " ".repeat(self.bar_len.saturating_sub(n)));
            self.bar_len = n.max(self.bar_len);
        } else {
            let _ = writeln!(err, "{}", self.status());
        }
        let _ = err.flush();
    }

    /// A message above the bar.
    fn say(&mut self, msg: &str) {
        self.clear();
        let _ = writeln!(std::io::stderr(), "{msg}");
        if self.tty && self.last_draw.is_some() {
            self.draw(true);
        }
    }

    fn tick(&mut self) {
        self.done += 1;
        if self.first_done.is_none() {
            self.first_done = Some((Instant::now(), self.done));
        }
        let last = self.total.is_some_and(|t| self.done >= t);
        self.draw(last);
    }
}

/// A single-file upscale finished.
pub fn image_done() {
    match human() {
        Some(mut h) => h.tick(),
        None => line("PROGRESS=postprocess_worker_image"),
    }
}

/// One loose image of a folder upscale finished.
pub fn folder_image_done() {
    match human() {
        Some(mut h) => h.tick(),
        None => line("PROGRESS=postprocess_worker_folder"),
    }
}

/// An archive is starting: how many entries it has.
pub fn archive_start(entries: usize) {
    if HUMAN.get().is_none() {
        line(&format!("TOTALZIP={entries}"));
    }
}

/// One archive entry was written (upscaled or copied).
pub fn archive_entry_done() {
    match human() {
        Some(mut h) => h.tick(),
        None => line("PROGRESS=postprocess_worker_zip_image"),
    }
}

/// An archive finished.
pub fn archive_done() {
    if HUMAN.get().is_none() {
        line("PROGRESS=postprocess_worker_zip_archive");
    }
}

/// A model's engine is being built (first use of this model and tile on this GPU). The line names
/// the backend ("TensorRT" or "TensorRT-RTX"); the GUI tells them apart by it.
pub fn engine_build(model: &str, tile: u32, backend: &str) {
    let takes = if backend == "TensorRT" { "a few minutes" } else { "a few seconds" };
    match human() {
        Some(mut h) => h.say(&format!("Building the {backend} engine for {model} (tile {tile}) on this GPU; this takes {takes}, once")),
        None => line(&format!("ENGINE_BUILD {model} (tile {tile}): building the {backend} engine for this GPU; this takes {takes}, once")),
    }
}

pub fn engine_ready(model: &str) {
    match human() {
        Some(mut h) => h.say(&format!("Engine ready: {model}")),
        None => line(&format!("ENGINE_READY {model}")),
    }
}

/// TensorRT-RTX failed and `instead` runs instead: at start (`model` None, for the whole run) or
/// for one model (the others stay on TensorRT-RTX). `instead` is TensorRT, except under Auto at
/// start, where it can be DirectML. The GUI shows it as a warning.
pub fn backend_fallback(model: Option<&str>, why: &str, instead: &str) {
    let msg = match model {
        None => format!("TensorRT-RTX failed ({why}); using {instead} instead"),
        Some(m) => format!("TensorRT-RTX failed for {m} ({why}); using {instead} for it"),
    };
    match human() {
        Some(mut h) => h.say(&format!("warning: {msg}")),
        None => line(&format!("BACKEND_FALLBACK {msg}")),
    }
}

/// A model's precision is suboptimal for the backend and GPU it runs on (`precision::hint`):
/// printed once per model file, before it loads. The GUI shows it as a non-blocking notice.
/// The run let go of the GPU after a pause request (`--control stdin`'s `PAUSE`).
pub fn paused() {
    match human() {
        Some(mut h) => h.say("Paused: the models are unloaded"),
        None => line("PAUSED"),
    }
}

pub fn resumed() {
    match human() {
        Some(mut h) => h.say("Resumed"),
        None => line("RESUMED"),
    }
}

pub fn hint(model: &str, text: &str) {
    match human() {
        Some(mut h) => h.say(&format!("note: {model}: {text}")),
        None => line(&format!("HINT {model}: {text}")),
    }
}

/// A file (or archive entry) that failed; the run continues with the next one.
pub fn error(what: &str, msg: &str) {
    match human() {
        Some(mut h) => {
            h.errors += 1;
            h.say(&format!("error: {what}: {msg}"));
        }
        None => line(&format!("ERROR {what}: {msg}")),
    }
}

/// An output that exists already and is kept (overwrite off).
pub fn skipped(path: &std::path::Path) {
    match human() {
        Some(mut h) => {
            h.skipped += 1;
            if h.verbose {
                h.say(&format!("exists, skipped: {}", path.display()));
            }
        }
        None => line(&format!("file exists, skip: {}", path.display())),
    }
}

/// An animation (`kind`: "GIF", "WebP", ...) copied unchanged instead of upscaled; `to` is the
/// copy's path for a file, `None` for an archive entry (kept under its name). A log line for the GUI;
/// on the command line a count in the summary, and the line with `--verbose`.
pub fn animated_copied(what: &str, kind: &str, to: Option<&std::path::Path>) {
    let msg = match to {
        Some(to) => format!("{what}: animated {kind}, copied unchanged to {}", to.display()),
        None => format!("{what}: animated {kind}, copied unchanged"),
    };
    match human() {
        Some(mut h) => {
            h.animated += 1;
            if h.verbose {
                h.say(&msg);
            }
        }
        None => line(&msg),
    }
}

/// Detail: always a line for the GUI, on the command line only with `--verbose`.
pub fn log(msg: &str) {
    match human() {
        Some(mut h) => {
            if h.verbose {
                h.say(msg);
            }
        }
        None => line(msg),
    }
}

/// libaji's own warnings and errors: stderr for the GUI (as before), above the bar on the command line.
pub fn backend_log(msg: &str) {
    match human() {
        Some(mut h) => h.say(&format!("[aji] {msg}")),
        None => eprintln!("[aji] {msg}"),
    }
}

/// Worth seeing in either mode (which backend and GPU runs, the elapsed time).
pub fn info(msg: &str) {
    match human() {
        Some(mut h) => h.say(msg),
        None => line(msg),
    }
}

/// End of the run: the GUI's elapsed-time line, or the command line's summary. Returns how many
/// files failed.
pub fn finish(elapsed: Duration) -> u64 {
    match human() {
        Some(mut h) => {
            h.clear();
            let secs = elapsed.as_secs_f64();
            let rate = if secs > 0.0 && h.done > 0 { format!(" ({:.2} img/s)", h.done as f64 / secs) } else { String::new() };
            let mut msg = format!("Done: {} written in {}{rate}", h.done.saturating_sub(h.errors), fmt_duration(elapsed));
            if h.animated > 0 {
                msg += &format!(", {} of them animations copied unchanged", h.animated);
            }
            if h.skipped > 0 {
                msg += &format!(", {} skipped (already there)", h.skipped);
            }
            if h.errors > 0 {
                msg += &format!(", {} failed", h.errors);
            }
            let _ = writeln!(std::io::stderr(), "{msg}");
            h.errors
        }
        None => {
            line(&format!("Elapsed time: {:.2} seconds", elapsed.as_secs_f64()));
            0
        }
    }
}
