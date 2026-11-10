//! The batch: walk the inputs into an ordered stream of items, push them through a pipeline —
//! parallel decode/prepare, one inference thread per GPU (the models live there; each takes the next
//! prepared image when it's free), parallel resize/encode — and write the results back in input
//! order, so archive entries keep their order.
//!
//! Nothing a single file does can stop the run: a file that fails is reported (`ERROR …`) and
//! counted as done, an archive entry that can't be decoded is copied as-is, and every output is
//! written under a temporary name and renamed into place only when complete.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use yosh_image::archive;
use yosh_image::resize::Resizer;

use crate::process::{self, Job, Work};
use crate::progress;
use crate::upscaler::Upscaler;

/// Inputs counted as images and archives (matched case-insensitively).
pub const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "avif", "tif", "tiff", "jxl"];
pub const ARCHIVE_EXTENSIONS: &[&str] = &["zip", "cbz", "rar", "cbr", "7z", "cb7"];
/// Inside an archive, also try these as images (anything that fails to decode is copied as-is).
const ENTRY_IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "avif", "gif", "jxl", "tif", "tiff"];

fn ext_in(path: &Path, list: &[&str]) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| list.iter().any(|x| x.eq_ignore_ascii_case(e)))
}

/// Where a run's input and output are.
pub enum Input {
    File(PathBuf),
    Folder { dir: PathBuf, images: bool, archives: bool },
}

pub struct Output {
    pub dir: PathBuf,
    /// The file name template; `%filename%` is the input's name without its extension.
    pub filename: String,
    pub overwrite: bool,
    /// Outputs written at or after this are kept even when overwriting: a resumed run keeps what
    /// the interrupted one already saved, and still redoes files that were there before it.
    pub keep_since: Option<std::time::SystemTime>,
}

impl Output {
    /// Whether the output at `path` is kept (not redone).
    fn keeps(&self, path: &Path) -> bool {
        let Ok(meta) = std::fs::metadata(path) else { return false };
        !self.overwrite || self.keep_since.is_some_and(|since| meta.modified().is_ok_and(|m| m >= since))
    }
}

impl Output {
    fn name_for(&self, input: &Path, ext: &str) -> String {
        render(&self.filename, input, ext)
    }
}

fn render(template: &str, input: &Path, ext: &str) -> String {
    let stem = input.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    format!("{}.{ext}", template.replace("%filename%", &stem))
}

/// One input and the file it's written to.
struct Planned {
    src: PathBuf,
    dest: PathBuf,
    is_archive: bool,
    /// Came from a folder walk (picks the progress line).
    folder: bool,
}

/// Paths compare without case on Windows and macOS, whose file systems ignore it.
fn path_key(p: &Path) -> String {
    let s = p.to_string_lossy();
    if cfg!(any(windows, target_os = "macos")) { s.to_lowercase() } else { s.into_owned() }
}

/// Which of `dests` share their path with another.
fn colliding(dests: &[PathBuf]) -> Vec<bool> {
    let mut n: HashMap<String, usize> = HashMap::new();
    for d in dests {
        *n.entry(path_key(d)).or_default() += 1;
    }
    dests.iter().map(|d| n[&path_key(d)] > 1).collect()
}

/// Each input's output file. Inputs that would be written to the same file (a name without
/// `%filename%` in a folder, or `a.cbz` and `a.zip`) are told apart, and only those: first by
/// adding `-%filename%` to a name without it, then by adding the input's extension (`a-zip.cbz`),
/// then by a number.
fn assign(raw: Vec<(PathBuf, PathBuf, bool, bool)>, out: &Output, format_ext: &str, report: bool) -> Vec<Planned> {
    let ext = |is_archive: bool| if is_archive { "cbz" } else { format_ext };
    let first: Vec<PathBuf> = raw.iter().map(|(src, dir, a, _)| dir.join(out.name_for(src, ext(*a)))).collect();
    let mut dests = first.clone();

    let template = if out.filename.contains("%filename%") { out.filename.clone() } else { format!("{}-%filename%", out.filename) };
    for (i, c) in colliding(&dests).into_iter().enumerate() {
        if c {
            let (src, dir, a, _) = &raw[i];
            dests[i] = dir.join(render(&template, src, ext(*a)));
        }
    }
    for (i, c) in colliding(&dests).into_iter().enumerate() {
        if c {
            let (src, dir, a, _) = &raw[i];
            let src_ext = src.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
            dests[i] = dir.join(render(&format!("{template}-{src_ext}"), src, ext(*a)));
        }
    }
    let mut used = HashSet::new();
    for d in dests.iter_mut() {
        let base = d.clone();
        let stem = base.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let e = base.extension().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let mut n = 2;
        while !used.insert(path_key(d)) {
            *d = base.with_file_name(format!("{stem}-{n}.{e}"));
            n += 1;
        }
    }

    let renamed: Vec<&PathBuf> = first.iter().zip(&dests).filter(|(a, b)| a != b).map(|(_, b)| b).collect();
    if report && let Some(example) = renamed.first() {
        progress::log(&format!(
            "{} inputs would have been saved over each other, so their output names were made unique (for example {})",
            renamed.len(),
            example.display()
        ));
    }
    raw.into_iter()
        .zip(dests)
        .map(|((src, _, is_archive, folder), dest)| Planned { src, dest, is_archive, folder })
        .collect()
}

// ---- items through the pipeline ------------------------------------------------------------

#[derive(Clone)]
enum Dest {
    /// A file; `folder` picks the progress line (folder image vs. single file).
    File { path: PathBuf, folder: bool },
    /// An entry of the archive being written; `source` is its original name (for a raw copy).
    Entry { name: String, source: String },
}

enum Item {
    Image { seq: u64, label: String, src: Src, dest: Dest },
    Raw { seq: u64, name: String, bytes: Vec<u8>, is_dir: bool },
    BeginArchive { seq: u64, out: PathBuf, entries: usize },
    EndArchive { seq: u64 },
}

enum Src {
    Path(PathBuf),
    Bytes(Vec<u8>),
}

enum Stage {
    Work { seq: u64, label: String, dest: Dest, work: Work, original: Option<Vec<u8>> },
    Done { seq: u64, done: Done },
}

enum Done {
    Write { dest: Dest, bytes: Vec<u8> },
    Raw { name: String, bytes: Vec<u8>, is_dir: bool },
    BeginArchive { out: PathBuf, entries: usize },
    EndArchive,
    Failed { label: String, dest: Dest, err: String, original: Option<Vec<u8>> },
}

fn catch<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
        Err(format!("internal error: {}", msg.unwrap_or_else(|| "panic".into())))
    })
}

// ---- walking -------------------------------------------------------------------------------

struct Walker<'a> {
    tx: SyncSender<Item>,
    seq: u64,
    out: &'a Output,
}

impl Walker<'_> {
    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    fn send(&self, item: Item) -> bool {
        self.tx.send(item).is_ok()
    }

    fn skip(&self, path: &Path) -> bool {
        if self.out.keeps(path) {
            progress::skipped(path);
            return true;
        }
        false
    }

    fn image(&mut self, src: &Path, dest: &Path, folder: bool) -> bool {
        let dest = dest.to_path_buf();
        if self.skip(&dest) {
            return true;
        }
        let seq = self.next_seq();
        self.send(Item::Image { seq, label: src.display().to_string(), src: Src::Path(src.to_path_buf()), dest: Dest::File { path: dest, folder } })
    }

    fn archive(&mut self, src: &Path, dest: &Path, format_ext: &str) -> bool {
        let dest = dest.to_path_buf();
        if self.skip(&dest) {
            return true;
        }
        let entries = match archive::list(src) {
            Ok(e) => e.len(),
            Err(e) => {
                progress::error(&src.display().to_string(), &e.to_string());
                return true;
            }
        };
        let seq = self.next_seq();
        if !self.send(Item::BeginArchive { seq, out: dest, entries }) {
            return false;
        }
        let mut alive = true;
        let res = archive::for_each(src, |entry, bytes| {
            let seq = self.next_seq();
            let label = format!("{} :: {}", src.display(), entry.name);
            let item = if !entry.is_dir && ext_in(Path::new(&entry.name), ENTRY_IMAGE_EXTENSIONS) {
                let name = Path::new(&entry.name).with_extension(format_ext).to_string_lossy().replace('\\', "/");
                Item::Image { seq, label, src: Src::Bytes(bytes.to_vec()), dest: Dest::Entry { name, source: entry.name.clone() } }
            } else {
                Item::Raw { seq, name: entry.name.clone(), bytes: bytes.to_vec(), is_dir: entry.is_dir }
            };
            alive = self.tx.send(item).is_ok();
            Ok(alive)
        });
        if let Err(e) = res {
            progress::error(&src.display().to_string(), &format!("reading the archive stopped early: {e}"));
        }
        let seq = self.next_seq();
        alive && self.send(Item::EndArchive { seq })
    }
}

/// What one input holds: (source, output folder, is an archive, came from a folder walk), in order.
/// A path that is neither an image nor an archive is left out (and reported, with `report`).
fn plan(input: &Input, out: &Output, report: bool) -> Vec<(PathBuf, PathBuf, bool, bool)> {
    let error = |what: &Path, msg: &str| {
        if report {
            progress::error(&what.display().to_string(), msg);
        }
    };
    match input {
        Input::File(path) => {
            if ext_in(path, ARCHIVE_EXTENSIONS) {
                vec![(path.clone(), out.dir.clone(), true, false)]
            } else if ext_in(path, IMAGE_EXTENSIONS) {
                vec![(path.clone(), out.dir.clone(), false, false)]
            } else {
                error(path, "not an image or archive");
                Vec::new()
            }
        }
        Input::Folder { dir, images, archives } => {
            let files = match archive::walk_folder(dir, true) {
                Ok(f) => f,
                Err(e) => {
                    error(dir, &e.to_string());
                    return Vec::new();
                }
            };
            files
                .into_iter()
                .filter_map(|f| {
                    let rel_dir = f.parent().and_then(|p| p.strip_prefix(dir).ok()).unwrap_or(Path::new(""));
                    let out_dir = out.dir.join(rel_dir);
                    if *images && ext_in(&f, IMAGE_EXTENSIONS) {
                        Some((f, out_dir, false, true))
                    } else if *archives && ext_in(&f, ARCHIVE_EXTENSIONS) {
                        Some((f, out_dir, true, true))
                    } else {
                        None
                    }
                })
                .collect()
        }
    }
}

fn walk(inputs: &[Planned], out: &Output, format_ext: &str, tx: SyncSender<Item>) {
    let mut w = Walker { tx, seq: 0, out };
    for p in inputs {
        let alive = if p.is_archive { w.archive(&p.src, &p.dest, format_ext) } else { w.image(&p.src, &p.dest, p.folder) };
        if !alive {
            return;
        }
    }
}

/// How many items (images and archive entries) the run will write, for the progress bar: what
/// `walk` sends, without reading any image (archives only list their entries).
fn count(inputs: &[Planned], out: &Output) -> u64 {
    inputs
        .iter()
        .map(|p| {
            if out.keeps(&p.dest) {
                0
            } else if p.is_archive {
                archive::list(&p.src).map_or(0, |e| e.len() as u64)
            } else {
                1
            }
        })
        .sum()
}

// ---- stages --------------------------------------------------------------------------------

fn prepare_stage(item: Item, job: &Job, resizer: &mut Resizer) -> Stage {
    match item {
        Item::Image { seq, label, src, dest } => {
            let (bytes, keep) = match src {
                Src::Path(p) => match std::fs::read(&p) {
                    Ok(b) => (b, false),
                    Err(e) => return Stage::Done { seq, done: Done::Failed { label, dest, err: e.to_string(), original: None } },
                },
                Src::Bytes(b) => (b, true),
            };
            match catch(|| process::prepare(&bytes, job, &label, resizer)) {
                Ok(work) => Stage::Work { seq, label, dest, work, original: keep.then_some(bytes) },
                Err(err) => Stage::Done { seq, done: Done::Failed { label, dest, err, original: keep.then_some(bytes) } },
            }
        }
        Item::Raw { seq, name, bytes, is_dir } => Stage::Done { seq, done: Done::Raw { name, bytes, is_dir } },
        Item::BeginArchive { seq, out, entries } => Stage::Done { seq, done: Done::BeginArchive { out, entries } },
        Item::EndArchive { seq } => Stage::Done { seq, done: Done::EndArchive },
    }
}

fn infer_stage(stage: Stage, up: &mut Option<Upscaler>, up_err: &Option<String>) -> Stage {
    match stage {
        Stage::Work { seq, label, dest, work, original } => {
            let res = match (needs_model(&work), up.as_mut()) {
                (false, _) => Ok(work),
                (true, Some(up)) => catch(|| process::infer(work, up)),
                (true, None) => Err(up_err.clone().unwrap_or_else(|| "no upscaling backend".into())),
            };
            match res {
                Ok(work) => Stage::Work { seq, label, dest, work, original },
                Err(err) => Stage::Done { seq, done: Done::Failed { label, dest, err, original } },
            }
        }
        done => done,
    }
}

fn finish_stage(stage: Stage, job: &Job, resizer: &mut Resizer) -> (u64, Done) {
    match stage {
        Stage::Work { seq, label, dest, work, original } => match catch(|| process::finish(work, job, &label, resizer)) {
            Ok(bytes) => (seq, Done::Write { dest, bytes }),
            Err(err) => (seq, Done::Failed { label, dest, err, original }),
        },
        Stage::Done { seq, done } => (seq, done),
    }
}

/// `n` threads taking items from `rx`, each mapping them with `f` (given its own `Resizer`).
fn workers<I: Send + 'static, O: Send + 'static>(
    n: usize,
    rx: Receiver<I>,
    tx: SyncSender<O>,
    f: impl Fn(I, &mut Resizer) -> O + Send + Sync + 'static,
) -> Vec<thread::JoinHandle<()>> {
    let rx = Arc::new(Mutex::new(rx));
    let f = Arc::new(f);
    (0..n)
        .map(|_| {
            let (rx, tx, f) = (rx.clone(), tx.clone(), f.clone());
            thread::spawn(move || {
                let mut resizer = Resizer::new();
                loop {
                    let next = rx.lock().unwrap().recv();
                    let Ok(item) = next else { break };
                    if tx.send(f(item, &mut resizer)).is_err() {
                        break;
                    }
                }
            })
        })
        .collect()
}

// ---- writing -------------------------------------------------------------------------------

fn part_path(path: &Path) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(".part");
    PathBuf::from(p)
}

/// Written to `<name>.part`, flushed to disk, then renamed: an output under its real name is
/// complete even after a power loss, so a rerun that keeps existing outputs never keeps a broken one
/// (without the flush, the rename can reach the disk before the data).
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    let part = part_path(path);
    let mut f = std::fs::File::create(&part).map_err(|e| e.to_string())?;
    f.write_all(bytes).and_then(|_| f.sync_all()).map_err(|e| e.to_string())?;
    drop(f);
    std::fs::rename(&part, path).map_err(|e| e.to_string())
}

struct ArchiveOut {
    out: PathBuf,
    zip: zip::ZipWriter<std::fs::File>,
    names: HashSet<String>,
}

impl ArchiveOut {
    fn start(out: &Path) -> Result<Self, String> {
        if let Some(dir) = out.parent() {
            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
        }
        let file = std::fs::File::create(part_path(out)).map_err(|e| e.to_string())?;
        Ok(Self { out: out.to_path_buf(), zip: zip::ZipWriter::new(file), names: HashSet::new() })
    }

    /// Add a file; images are stored (already compressed), anything else deflated. A name already
    /// written (two pages that differed only by extension) gets the old extension kept in it.
    fn add(&mut self, name: &str, bytes: &[u8], compress: bool) -> Result<(), String> {
        use zip::write::SimpleFileOptions;
        let mut name = name.to_string();
        if !self.names.insert(name.clone()) {
            let p = Path::new(&name);
            let ext = p.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
            name = format!("{}-{}.{ext}", p.with_extension("").to_string_lossy(), self.names.len());
            self.names.insert(name.clone());
        }
        let method = if compress { zip::CompressionMethod::Deflated } else { zip::CompressionMethod::Stored };
        self.zip
            .start_file(name, SimpleFileOptions::default().compression_method(method).large_file(bytes.len() >= u32::MAX as usize))
            .map_err(|e| e.to_string())?;
        self.zip.write_all(bytes).map_err(|e| e.to_string())
    }

    fn add_dir(&mut self, name: &str) -> Result<(), String> {
        if self.names.insert(format!("{name}/")) {
            self.zip.add_directory(name, zip::write::SimpleFileOptions::default()).map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Flushed to disk before the rename, as [`write_file`] does.
    fn finish(self) -> Result<(), String> {
        let file = self.zip.finish().map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        drop(file);
        std::fs::rename(part_path(&self.out), &self.out).map_err(|e| e.to_string())
    }
}

struct Sink {
    archive: Option<ArchiveOut>,
}

impl Sink {
    fn file_done(folder: bool) {
        if folder {
            progress::folder_image_done();
        } else {
            progress::image_done();
        }
    }

    fn handle(&mut self, done: Done) {
        match done {
            Done::Write { dest: Dest::File { path, folder }, bytes } => {
                match write_file(&path, &bytes) {
                    Ok(()) => progress::log(&format!("save image: {}", path.display())),
                    Err(e) => progress::error(&path.display().to_string(), &e),
                }
                Self::file_done(folder);
            }
            Done::Write { dest: Dest::Entry { name, .. }, bytes } => {
                if let Some(a) = &mut self.archive
                    && let Err(e) = a.add(&name, &bytes, false)
                {
                    progress::error(&name, &e);
                }
                progress::archive_entry_done();
            }
            Done::Raw { name, bytes, is_dir } => {
                if let Some(a) = &mut self.archive {
                    let r = if is_dir { a.add_dir(&name) } else { a.add(&name, &bytes, true) };
                    if let Err(e) = r {
                        progress::error(&name, &e);
                    }
                }
                progress::archive_entry_done();
            }
            Done::BeginArchive { out, entries } => {
                match ArchiveOut::start(&out) {
                    Ok(a) => self.archive = Some(a),
                    Err(e) => progress::error(&out.display().to_string(), &e),
                }
                progress::archive_start(entries);
            }
            Done::EndArchive => {
                if let Some(a) = self.archive.take() {
                    let out = a.out.clone();
                    match a.finish() {
                        Ok(()) => progress::log(&format!("save archive: {}", out.display())),
                        Err(e) => progress::error(&out.display().to_string(), &e),
                    }
                }
                progress::archive_done();
            }
            Done::Failed { label, dest, err, original } => {
                progress::error(&label, &err);
                match dest {
                    Dest::File { folder, .. } => Self::file_done(folder),
                    Dest::Entry { source, .. } => {
                        // keep the page: copy the original bytes under their original name
                        if let (Some(a), Some(bytes)) = (&mut self.archive, original)
                            && let Err(e) = a.add(&source, &bytes, false)
                        {
                            progress::error(&source, &e);
                        }
                        progress::archive_entry_done();
                    }
                }
            }
        }
    }
}

// ---- the run -------------------------------------------------------------------------------

/// Opens one inference thread's upscaler (one GPU), or says why it couldn't.
/// Opens one inference thread's upscaler (one GPU), or says why it couldn't. Called again after a
/// pause, which closes it.
pub type OpenUpscaler = Box<dyn Fn() -> Result<Upscaler, String> + Send>;

/// Pausing a run (MangaJaNai GUI's Pause): each inference thread closes its upscaler (its models,
/// the libaji context and, with libaji 0.24+, the device's CUDA context, so the GPU memory is
/// freed) once it's done with the image it's on, and waits. Resuming opens it again when the next image needs it: its engines load from the cache,
/// nothing is rebuilt. Decoding and encoding stop on their own once the queues between the stages
/// are full. `PAUSED` is printed when every GPU has let go, `RESUMED` on resume.
pub struct Pause {
    state: Mutex<PauseState>,
    cv: Condvar,
}

#[derive(Default)]
struct PauseState {
    paused: bool,
    /// Inference threads holding nothing, waiting to resume.
    released: usize,
    threads: usize,
}

impl Pause {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { state: Mutex::new(PauseState::default()), cv: Condvar::new() })
    }

    pub fn pause(&self) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !s.paused {
            s.paused = true;
            if s.threads == 0 {
                progress::paused(); // nothing on a GPU yet
            }
        }
    }

    pub fn resume(&self) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if s.paused {
            s.paused = false;
            s.released = 0;
            self.cv.notify_all();
            progress::resumed();
        }
    }

    fn set_threads(&self, n: usize) {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).threads = n;
    }

    /// While paused: close `up`, then wait for the resume.
    fn hold(&self, up: &mut Option<Upscaler>) {
        let mut s = self.state.lock().unwrap_or_else(|p| p.into_inner());
        if !s.paused {
            return;
        }
        if let Some(u) = up.take() {
            u.close();
        }
        s.released += 1;
        if s.released == s.threads {
            progress::paused();
        }
        while s.paused {
            s = self.cv.wait(s).unwrap_or_else(|p| p.into_inner());
        }
    }
}

/// How many items `run` will write for these inputs (see [`count`]), for a progress bar.
pub fn total(inputs: &[Input], output: &Output, job: &Job) -> u64 {
    let planned = assign(inputs.iter().flat_map(|i| plan(i, output, false)).collect(), output, job.format.ext(), false);
    count(&planned, output)
}

/// Threads in each of the two CPU pools (decode/prepare and resize/encode), unless `--workers`
/// sets it: half the CPU's threads up to 8, and at least four per inference thread (at most all
/// the CPU's threads). One GPU doesn't outrun 8 encoders, but several do: on 4x RTX 5090 (48
/// threads, 1050x1500 pages at 2x) 8 workers held WebP to 14.9 pages/s, lossy JPEG XL to 9.5 and
/// lossless JPEG XL to 4.3; 16 gave 17.0, 16.8 (both GPU-bound) and 8.3. On one GPU, 8 and 32 ran
/// within 2% of each other. Each worker holds an image, so more workers use more memory.
pub fn default_workers(threads: usize, inference_threads: usize) -> usize {
    (threads / 2).clamp(2, 8).max(4 * inference_threads).min(threads.max(2))
}

/// Run the batch over `inputs`, in order. Each of `upscalers` (one per GPU) gets an inference
/// thread that opens it lazily (so a run that needs no model never loads the runtime) and then takes
/// prepared images as it frees up. `workers` sizes the CPU pools (default: [`default_workers`]).
pub fn run(inputs: Vec<Input>, output: Output, job: Job, upscalers: Vec<OpenUpscaler>, pause: Arc<Pause>, workers: Option<usize>) {
    let threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let n = workers.unwrap_or_else(|| default_workers(threads, upscalers.len().max(1))).max(1);
    let job = Arc::new(job);
    let format_ext = job.format.ext().to_string();

    let (item_tx, item_rx) = sync_channel::<Item>(n * 2);
    let (prep_tx, prep_rx) = sync_channel::<Stage>(n);
    let (infer_tx, infer_rx) = sync_channel::<Stage>(n);
    let (done_tx, done_rx) = sync_channel::<(u64, Done)>(n * 2);

    let j = job.clone();
    let mut handles = workers(n, item_rx, prep_tx, move |item, r| prepare_stage(item, &j, r));

    let prep_rx = Arc::new(Mutex::new(prep_rx));
    pause.set_threads(upscalers.len());
    for open in upscalers {
        let (prep_rx, infer_tx, pause) = (prep_rx.clone(), infer_tx.clone(), pause.clone());
        handles.push(thread::spawn(move || {
            let mut up: Option<Upscaler> = None;
            let mut up_err: Option<String> = None;
            loop {
                pause.hold(&mut up);
                // waits in short steps, so a pause while idle still lets go of the GPU
                let next = prep_rx.lock().unwrap_or_else(|p| p.into_inner()).recv_timeout(std::time::Duration::from_millis(200));
                let stage = match next {
                    Ok(stage) => stage,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
                };
                if let Stage::Work { work, .. } = &stage
                    && needs_model(work)
                    && up.is_none()
                    && up_err.is_none()
                {
                    match open() {
                        Ok(u) => {
                            progress::info(&format!("Upscaling with {}", u.description));
                            up = Some(u);
                        }
                        Err(e) => {
                            progress::error("upscaling backend", &e);
                            up_err = Some(e);
                        }
                    }
                }
                if infer_tx.send(infer_stage(stage, &mut up, &up_err)).is_err() {
                    break;
                }
            }
        }));
    }
    drop(infer_tx);

    let j = job.clone();
    handles.extend(workers(n, infer_rx, done_tx, move |stage, r| finish_stage(stage, &j, r)));

    let planned = assign(inputs.iter().flat_map(|i| plan(i, &output, true)).collect(), &output, &format_ext, true);
    let walker = thread::spawn(move || walk(&planned, &output, &format_ext, item_tx));

    // Write results in input order.
    let mut sink = Sink { archive: None };
    let mut pending: BTreeMap<u64, Done> = BTreeMap::new();
    let mut next = 1u64;
    for (seq, done) in done_rx {
        pending.insert(seq, done);
        while let Some(done) = pending.remove(&next) {
            sink.handle(done);
            next += 1;
        }
    }
    let _ = walker.join();
    for h in handles {
        let _ = h.join();
    }
}

fn needs_model(work: &Work) -> bool {
    work.chain.as_ref().is_some_and(|c| {
        let m = c.model_file_path.trim();
        !m.is_empty() && m != "No Model"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workers_are_half_the_threads_up_to_8_and_four_per_inference_thread() {
        assert_eq!(default_workers(32, 1), 8, "one GPU: half the threads, at most 8");
        assert_eq!(default_workers(8, 1), 4);
        assert_eq!(default_workers(2, 1), 2);
        assert_eq!(default_workers(1, 1), 2);
        assert_eq!(default_workers(48, 2), 8);
        assert_eq!(default_workers(48, 4), 16, "four per inference thread");
        assert_eq!(default_workers(12, 4), 12, "but no more than the CPU's threads");
    }

    #[test]
    fn a_pause_waits_until_resumed_and_lets_go_of_the_upscaler() {
        let pause = Pause::new();
        pause.set_threads(1);
        pause.hold(&mut None); // not paused: returns at once
        pause.pause();
        let p = pause.clone();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let d = done.clone();
        let t = thread::spawn(move || {
            p.hold(&mut None);
            d.store(true, std::sync::atomic::Ordering::SeqCst);
        });
        // the thread lets go and waits
        for _ in 0..100 {
            if pause.state.lock().unwrap().released == 1 {
                break;
            }
            thread::sleep(std::time::Duration::from_millis(10));
        }
        assert_eq!(pause.state.lock().unwrap().released, 1);
        assert!(!done.load(std::sync::atomic::Ordering::SeqCst), "still paused");
        pause.resume();
        t.join().unwrap();
        assert!(done.load(std::sync::atomic::Ordering::SeqCst));
        assert_eq!(pause.state.lock().unwrap().released, 0);
        pause.resume(); // not paused: nothing
    }

    fn names(filename: &str, inputs: &[(&str, &str, bool)]) -> Vec<String> {
        let out = Output { dir: "out".into(), filename: filename.into(), overwrite: true, keep_since: None };
        let raw = inputs.iter().map(|&(src, dir, a)| (PathBuf::from(src), PathBuf::from("out").join(dir), a, true)).collect();
        assign(raw, &out, "webp", false)
            .into_iter()
            .map(|p| p.dest.strip_prefix("out").unwrap().to_string_lossy().replace('\\', "/"))
            .collect()
    }

    #[test]
    fn keep_since_keeps_only_outputs_written_since() {
        let dir = std::env::temp_dir().join(format!("mangajanai-keep-since-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let old = dir.join("old.webp");
        std::fs::write(&old, b"x").unwrap();
        let since = std::fs::metadata(&old).unwrap().modified().unwrap() + std::time::Duration::from_secs(1);
        let new = dir.join("new.webp");
        std::fs::write(&new, b"x").unwrap();
        std::fs::File::options().write(true).open(&new).unwrap().set_modified(since).unwrap();
        let out = |overwrite, keep_since| Output { dir: dir.clone(), filename: "%filename%".into(), overwrite, keep_since };

        assert!(!out(true, None).keeps(&new), "overwriting redoes everything");
        assert!(out(true, Some(since)).keeps(&new), "written since: kept");
        assert!(!out(true, Some(since)).keeps(&old), "there before the run: redone");
        assert!(out(false, None).keeps(&old) && out(false, Some(since)).keeps(&old), "not overwriting keeps all");
        assert!(!out(false, Some(since)).keeps(&dir.join("missing.webp")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn unique_names_stay_as_they_are() {
        let n = names("%filename%-mangajanai", &[("in/Vol 1.cbz", "", true), ("in/Vol 2.cbz", "", true), ("in/001.png", "", false)]);
        assert_eq!(n, ["Vol 1-mangajanai.cbz", "Vol 2-mangajanai.cbz", "001-mangajanai.webp"]);
        assert_eq!(names("upscaled", &[("in/Vol 1.cbz", "", true)]), ["upscaled.cbz"], "one input: no change");
        assert_eq!(names("upscaled", &[("in/Vol 1.cbz", "", true), ("in/001.png", "", false)]), ["upscaled.cbz", "upscaled.webp"]);
        assert_eq!(names("upscaled", &[("in/a/x.cbz", "a", true), ("in/b/y.cbz", "b", true)]), ["a/upscaled.cbz", "b/upscaled.cbz"], "different folders");
    }

    #[test]
    fn a_name_without_filename_gets_it_where_outputs_collide() {
        let n = names("upscaled", &[("in/Vol 1.cbz", "", true), ("in/Vol 2.cbz", "", true), ("in/001.png", "", false)]);
        assert_eq!(n, ["upscaled-Vol 1.cbz", "upscaled-Vol 2.cbz", "upscaled.webp"], "only the colliding ones");
    }

    #[test]
    fn same_names_with_different_extensions_get_the_extension() {
        let n = names("%filename%-mangajanai", &[("in/Vol 1.cbz", "", true), ("in/Vol 1.zip", "", true), ("in/001.png", "", false), ("in/001.JPG", "", false)]);
        assert_eq!(n, ["Vol 1-mangajanai-cbz.cbz", "Vol 1-mangajanai-zip.cbz", "001-mangajanai-png.webp", "001-mangajanai-jpg.webp"]);
        let n = names("upscaled", &[("in/Vol 1.cbz", "", true), ("in/Vol 1.zip", "", true)]);
        assert_eq!(n, ["upscaled-Vol 1-cbz.cbz", "upscaled-Vol 1-zip.cbz"]);
    }

    #[test]
    fn anything_left_gets_a_number() {
        // a.png and a.PNG: both "-png" after the extension step
        let n = names("x", &[("in/a.png", "", false), ("in/a.PNG", "", false)]);
        assert_eq!(n, ["x-a-png.webp", "x-a-png-2.webp"]);
        // a renamed output that lands on another input's name
        let n = names("%filename%", &[("in/a-png.png", "", false), ("in/a.png", "", false), ("in/a.jpg", "", false)]);
        assert_eq!(n, ["a-png.webp", "a-png-2.webp", "a-jpg.webp"]);
    }
}
