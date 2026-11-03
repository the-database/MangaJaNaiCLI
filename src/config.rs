//! The command line's saved settings: `mangajanaitrt.ini`. Every setting is optional and mirrors a
//! flag; flags override the file, and the file overrides an imported GUI workflow (`workflow =`).
//!
//! The format is plain INI, made forgiving for people editing it in Notepad: values are taken as
//! typed (no quotes or escapes, so Windows paths work as is; quotes around a value, as Explorer's
//! "Copy as path" adds, are dropped), switches accept yes/no, true/false, on/off or 1/0, a list is
//! the same line repeated, and an empty value means "not set". Comments are lines starting with
//! `;` or `#` (only whole lines: `#` and `;` are common in file names). Section and setting names
//! ignore case. Anything unknown or malformed is an error naming the line, so a typo is caught.

use std::path::{Path, PathBuf};

use crate::settings::{Chain, Format, Target};

/// The file `mangajanaitrt init` writes, and the reference for every setting.
pub const TEMPLATE: &str = r#"; mangajanaitrt settings
;
; Run `mangajanaitrt` with no arguments to use this file. It is found in the current folder, else
; next to the program; -c <file> picks another. Anything given on the command line overrides it.
; Every setting is optional: delete a line, or leave it empty, to use the default.
; Relative paths are relative to this file's folder. Lines starting with ; are comments.

[input]
; What to upscale: an image, an archive (zip cbz rar cbr 7z cb7) or a folder (searched with its
; subfolders). Add more "path =" lines for more.
path = in
; In folders, also upscale loose images / archives (yes or no).
images = yes
archives = yes

[output]
; Where the results go. Archives come out as .cbz, folders keep their subfolders.
folder = out
; webp, png, jpg or avif
format = webp
; Lossy quality, 0-100 (webp, jpg, avif).
quality = 90
; Lossless webp (yes or no).
lossless = no
; Added to each file name: page01 -> page01_x4.webp. Leave empty for none.
suffix =
; Or the whole file name, where %filename% is the original name (overrides suffix).
filename =
; Keep results that already exist instead of redoing them (yes or no).
skip_existing = yes

[model]
; Which model runs. Pick one of these three, or write [rule] sections (at the end of this file):
;   file     = an ONNX model: every image goes through it (a path, or a file name found in the
;              model folders)
;   preset   = manga: MangaJaNai's own choice per page (the MangaJaNai model for the page height on
;              gray pages, IllustrationJaNai on color pages)
;   workflow = a .mwf workflow exported from MangaJaNai GUI (its models, size and format; the
;              settings in this file still override it)
file = 4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx
preset =
workflow =
; Folders where model files are looked up by name, first match first (add more lines for more).
; After these come this file's folder, the program's models folder and MangaJaNai GUI's models.
folder =
; How colorful a page may be and still count as gray, 0-255.
gray_threshold = 12
; Stretch gray pages to full black and white before the model (yes or no; empty = the rule's own).
auto_levels =

[resize]
; The final size after the model. Set at most one; with none, the output is at the model's scale.
; scale = times the original size; height / width = pixels, keeping the shape; fit = a box like 1920x1080
scale =
height =
width =
fit =

[gpu]
; auto, tensorrt, directml or tensorrt-rtx (directml and tensorrt-rtx: Windows only)
backend = auto
; first, all, or GPU numbers from `mangajanaitrt devices`, like 0,1
devices = first
; Inference threads per GPU. 2 keeps the GPU busier (about 20-45% faster) but needs about twice the VRAM.
streams = 1
; Tile size in pixels (one engine is built per model and tile size) and the overlap between tiles.
tile = 512
overlap = 32
; Where built engines are kept (empty = next to each model).
engine_folder =
; Where aji.dll / libaji.so is (empty = next to the program, else an installed MangaJaNai GUI's).
aji_folder =

; Your own rules instead of [model] file/preset/workflow: each [rule] section is one rule, and the
; first that matches an image is used. Only "model" is required; sizes are the original image's,
; limits include the number itself.
;
; [rule]
; ; gray pages only (gray = no: color pages only; leave out: both)
; gray = yes
; max_height = 1250
; ; the [resize] scale this rule is for (also min_scale)
; max_scale = 2
; model = 2x_MangaJaNai_1200p_V1_ESRGAN_70k_fp16.onnx
; auto_levels = yes
; ; also: min_width, min_height, max_width, tile, pre_resize_width, pre_resize_height, pre_resize_percent
"#;

#[derive(Debug, Default)]
pub struct File {
    pub input: Vec<String>,
    pub output: Option<String>,
    pub model: Option<String>,
    pub preset: Option<String>,
    pub workflow: Option<String>,
    pub models_dir: Vec<String>,
    pub images: Option<bool>,
    pub archives: Option<bool>,
    pub gray_threshold: Option<i32>,
    pub auto_levels: Option<bool>,
    pub save: Save,
    pub resize: Resize,
    pub gpu: Gpu,
    pub rule: Vec<Rule>,
}

#[derive(Debug, Default)]
pub struct Save {
    pub format: Option<String>,
    pub quality: Option<u8>,
    pub lossless: Option<bool>,
    pub suffix: Option<String>,
    pub filename: Option<String>,
    pub skip_existing: Option<bool>,
}

#[derive(Debug, Default)]
pub struct Resize {
    pub scale: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub fit: Option<String>,
}

#[derive(Debug, Default)]
pub struct Gpu {
    pub backend: Option<String>,
    /// As the `--device` flag takes it: "first", "all" or "0,1".
    pub devices: Option<String>,
    pub streams: Option<u32>,
    pub tile: Option<u32>,
    pub overlap: Option<u32>,
    pub cache_dir: Option<String>,
    pub aji_dir: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct Rule {
    pub model: String,
    pub gray: Option<bool>,
    pub color: Option<bool>,
    pub min_width: Option<u32>,
    pub min_height: Option<u32>,
    pub max_width: Option<u32>,
    pub max_height: Option<u32>,
    pub min_scale: Option<i32>,
    pub max_scale: Option<i32>,
    pub auto_levels: Option<bool>,
    pub tile: Option<u32>,
    pub pre_resize_width: Option<i32>,
    pub pre_resize_height: Option<i32>,
    pub pre_resize_percent: Option<f64>,
}

// ---- reading -------------------------------------------------------------------------------

/// One `[section]` and its `key = value` lines (line numbers kept for messages).
struct Section {
    name: String,
    line: usize,
    entries: Vec<(usize, String, String)>,
}

/// The text of a file Notepad may have saved as UTF-8 (with or without a BOM) or as "Unicode"
/// (UTF-16 with a BOM).
fn decode_text(bytes: &[u8]) -> Result<String, String> {
    if let Some(rest) = bytes.strip_prefix(&[0xFF, 0xFE]) {
        let units: Vec<u16> = rest.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect();
        return String::from_utf16(&units).map_err(|_| "the file is not valid UTF-16 text".into());
    }
    if let Some(rest) = bytes.strip_prefix(&[0xFE, 0xFF]) {
        let units: Vec<u16> = rest.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes(*c)).collect();
        return String::from_utf16(&units).map_err(|_| "the file is not valid UTF-16 text".into());
    }
    let bytes = bytes.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(bytes);
    String::from_utf8(bytes.to_vec()).map_err(|_| "the file is not UTF-8 text (save it as UTF-8)".into())
}

/// A value as typed, minus surrounding whitespace and one pair of matching quotes.
fn clean(v: &str) -> String {
    let v = v.trim();
    for q in ['"', '\''] {
        if v.len() >= 2 && v.starts_with(q) && v.ends_with(q) {
            return v[1..v.len() - 1].trim().to_string();
        }
    }
    v.to_string()
}

fn sections(text: &str) -> Result<Vec<Section>, String> {
    let mut out: Vec<Section> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let n = i + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[') {
            let name = rest.strip_suffix(']').ok_or_else(|| format!("line {n}: a section name needs a closing ]: {line}"))?;
            out.push(Section { name: name.trim().to_ascii_lowercase(), line: n, entries: Vec::new() });
            continue;
        }
        let (k, v) = line.split_once('=').ok_or_else(|| format!("line {n}: expected setting = value, found: {line}"))?;
        let Some(sec) = out.last_mut() else {
            return Err(format!("line {n}: '{}' comes before any [section]; put it under one, like [output]", k.trim()));
        };
        sec.entries.push((n, k.trim().to_ascii_lowercase(), clean(v)));
    }
    Ok(out)
}

/// Levenshtein distance, for "did you mean" on a misspelled name.
fn distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.chars().enumerate() {
        let mut cur = vec![i + 1];
        for (j, &cb) in b.iter().enumerate() {
            cur.push((prev[j] + (ca != cb) as usize).min(prev[j + 1] + 1).min(cur[j] + 1));
        }
        prev = cur;
    }
    prev[b.len()]
}

fn suggest(name: &str, known: &[&str]) -> String {
    known
        .iter()
        .map(|k| (distance(name, k), *k))
        .min()
        .filter(|&(d, _)| d <= 2.max(name.len() / 3))
        .map_or_else(
            || format!(" (known here: {})", known.join(", ")),
            |(_, k)| format!(" (did you mean {k}?)"),
        )
}

/// A section's settings, read one by one; whatever is left at the end is unknown.
struct Reader<'a> {
    sec: &'a Section,
    used: Vec<bool>,
}

impl<'a> Reader<'a> {
    fn new(sec: &'a Section) -> Self {
        Self { used: vec![false; sec.entries.len()], sec }
    }

    /// Every non-empty value of `key` (a list: the line repeated).
    fn all(&mut self, key: &str) -> Vec<(usize, String)> {
        let mut out = Vec::new();
        for (i, (n, k, v)) in self.sec.entries.iter().enumerate() {
            if k == key {
                self.used[i] = true;
                if !v.is_empty() {
                    out.push((*n, v.clone()));
                }
            }
        }
        out
    }

    /// The value of `key`, if set (empty counts as not set). Set twice is an error.
    fn one(&mut self, key: &str) -> Result<Option<(usize, String)>, String> {
        let mut vals = self.all(key);
        if vals.len() > 1 {
            return Err(format!(
                "line {}: {key} is set twice in [{}] (also on line {}); keep one",
                vals[1].0, self.sec.name, vals[0].0
            ));
        }
        Ok(vals.pop())
    }

    fn string(&mut self, key: &str) -> Result<Option<String>, String> {
        Ok(self.one(key)?.map(|(_, v)| v))
    }

    fn switch(&mut self, key: &str) -> Result<Option<bool>, String> {
        let Some((n, v)) = self.one(key)? else { return Ok(None) };
        match v.to_ascii_lowercase().as_str() {
            "yes" | "y" | "true" | "on" | "1" => Ok(Some(true)),
            "no" | "n" | "false" | "off" | "0" => Ok(Some(false)),
            _ => Err(format!("line {n}: {key} should be yes or no, not '{v}'")),
        }
    }

    /// A number; a trailing x, % or px (as people write "2x", "90%", "512px") is allowed.
    fn number<T: std::str::FromStr>(&mut self, key: &str, what: &str) -> Result<Option<T>, String> {
        let Some((n, v)) = self.one(key)? else { return Ok(None) };
        let t = v.trim();
        let t = t.strip_suffix("px").or_else(|| t.strip_suffix("PX")).unwrap_or(t).trim_end_matches(['x', 'X', '%']).trim();
        t.parse().map(Some).map_err(|_| format!("line {n}: {key} should be {what}, not '{v}'"))
    }

    fn finish(self, known: &[&str]) -> Result<(), String> {
        for (i, (n, k, _)) in self.sec.entries.iter().enumerate() {
            if !self.used[i] {
                return Err(format!("line {n}: unknown setting '{k}' in [{}]{}", self.sec.name, suggest(k, known)));
            }
        }
        Ok(())
    }
}

const SECTIONS: &[&str] = &["input", "output", "model", "resize", "gpu", "rule"];

impl File {
    /// Read and check a settings file. Errors name the file and line.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        decode_text(&bytes).and_then(|t| Self::parse(&t)).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let mut f = File::default();
        let mut seen: Vec<&str> = Vec::new();
        for sec in sections(text)?.iter() {
            let name = sec.name.as_str();
            if name != "rule" {
                if seen.contains(&name) {
                    return Err(format!("line {}: [{name}] appears twice; put its settings in one place", sec.line));
                }
                seen.push(name);
            }
            let mut r = Reader::new(sec);
            match name {
                "input" => {
                    f.input = r.all("path").into_iter().map(|(_, v)| v).collect();
                    f.images = r.switch("images")?;
                    f.archives = r.switch("archives")?;
                    r.finish(&["path", "images", "archives"])?;
                }
                "output" => {
                    f.output = r.string("folder")?;
                    if let Some((n, v)) = r.one("format")? {
                        parse_format(&v).map_err(|e| format!("line {n}: {e}"))?;
                        f.save.format = Some(v);
                    }
                    f.save.quality = r.number::<u8>("quality", "a number from 0 to 100")?;
                    if f.save.quality.is_some_and(|q| q > 100) {
                        return Err("[output] quality should be a number from 0 to 100".into());
                    }
                    f.save.lossless = r.switch("lossless")?;
                    f.save.suffix = r.string("suffix")?;
                    f.save.filename = r.string("filename")?;
                    f.save.skip_existing = r.switch("skip_existing")?;
                    r.finish(&["folder", "format", "quality", "lossless", "suffix", "filename", "skip_existing"])?;
                }
                "model" => {
                    f.model = r.string("file")?;
                    f.preset = r.string("preset")?;
                    if let Some(p) = &f.preset
                        && !p.eq_ignore_ascii_case("manga")
                    {
                        return Err(format!("[model] preset '{p}' is unknown; the preset is: manga"));
                    }
                    f.workflow = r.string("workflow")?;
                    f.models_dir = r.all("folder").into_iter().map(|(_, v)| v).collect();
                    f.gray_threshold = r.number("gray_threshold", "a number from 0 to 255")?;
                    f.auto_levels = r.switch("auto_levels")?;
                    r.finish(&["file", "preset", "workflow", "folder", "gray_threshold", "auto_levels"])?;
                }
                "resize" => {
                    f.resize.scale = r.number("scale", "a number like 2 or 1.5")?;
                    f.resize.height = r.number("height", "a number of pixels")?;
                    f.resize.width = r.number("width", "a number of pixels")?;
                    f.resize.fit = r.string("fit")?;
                    r.finish(&["scale", "height", "width", "fit"])?;
                }
                "gpu" => {
                    if let Some((n, v)) = r.one("backend")? {
                        crate::upscaler::BackendChoice::parse(&v).map_err(|e| format!("line {n}: {e}"))?;
                        f.gpu.backend = Some(v);
                    }
                    f.gpu.devices = r.string("devices")?.map(|d| d.split([',', ' ', ';']).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(","));
                    f.gpu.streams = r.number("streams", "a number like 1 or 2")?;
                    f.gpu.tile = r.number("tile", "a number of pixels like 512")?;
                    f.gpu.overlap = r.number("overlap", "a number of pixels like 32")?;
                    f.gpu.cache_dir = r.string("engine_folder")?;
                    f.gpu.aji_dir = r.string("aji_folder")?;
                    r.finish(&["backend", "devices", "streams", "tile", "overlap", "engine_folder", "aji_folder"])?;
                }
                "rule" => {
                    let model = r.string("model")?.ok_or_else(|| format!("line {}: this [rule] needs a model = line", sec.line))?;
                    f.rule.push(Rule {
                        model,
                        gray: r.switch("gray")?,
                        color: r.switch("color")?,
                        min_width: r.number("min_width", "a number of pixels")?,
                        min_height: r.number("min_height", "a number of pixels")?,
                        max_width: r.number("max_width", "a number of pixels")?,
                        max_height: r.number("max_height", "a number of pixels")?,
                        min_scale: r.number("min_scale", "a whole number like 2")?,
                        max_scale: r.number("max_scale", "a whole number like 2")?,
                        auto_levels: r.switch("auto_levels")?,
                        tile: r.number("tile", "a number of pixels like 512")?,
                        pre_resize_width: r.number("pre_resize_width", "a number of pixels")?,
                        pre_resize_height: r.number("pre_resize_height", "a number of pixels")?,
                        pre_resize_percent: r.number("pre_resize_percent", "a percentage like 50")?,
                    });
                    r.finish(&[
                        "model", "gray", "color", "min_width", "min_height", "max_width", "max_height", "min_scale", "max_scale",
                        "auto_levels", "tile", "pre_resize_width", "pre_resize_height", "pre_resize_percent",
                    ])?;
                }
                other => return Err(format!("line {}: unknown section [{other}]{}", sec.line, suggest(other, SECTIONS))),
            }
        }
        let sources = [f.model.is_some(), f.preset.is_some(), f.workflow.is_some(), !f.rule.is_empty()];
        if sources.iter().filter(|&&s| s).count() > 1 {
            return Err("pick one way to choose the model: [model] file, preset or workflow, or [rule] sections".into());
        }
        if f.save.suffix.is_some() && f.save.filename.is_some() {
            return Err("[output]: set suffix or filename, not both".into());
        }
        f.resize.target()?;
        Ok(f)
    }
}

impl Resize {
    /// The final size, if one is set (at most one may be).
    pub fn target(&self) -> Result<Option<Target>, String> {
        let mut set = Vec::new();
        if let Some(s) = self.scale {
            if s.is_nan() || s <= 0.0 {
                return Err("[resize] scale must be above 0".into());
            }
            set.push(Target::Scale(s));
        }
        if let Some(w) = self.width {
            set.push(Target::Width(w));
        }
        if let Some(h) = self.height {
            set.push(Target::Height(h));
        }
        if let Some(f) = &self.fit {
            let (w, h) = parse_size(f).ok_or_else(|| format!("[resize] fit should be a size like 1920x1080, not '{f}'"))?;
            set.push(Target::Fit(w, h));
        }
        match set.len() {
            0 => Ok(None),
            1 => Ok(Some(set[0])),
            _ => Err("[resize]: set only one of scale, height, width or fit".into()),
        }
    }
}

/// `"WxH"` with both positive.
pub fn parse_size(s: &str) -> Option<(u32, u32)> {
    let (w, h) = s.trim().split_once(['x', 'X', '*'])?;
    let (w, h) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    (w > 0 && h > 0).then_some((w, h))
}

pub fn parse_format(s: &str) -> Result<Format, String> {
    match s.trim().trim_start_matches('.').to_ascii_lowercase().as_str() {
        "webp" => Ok(Format::Webp),
        "png" => Ok(Format::Png),
        "jpg" | "jpeg" => Ok(Format::Jpeg),
        "avif" => Ok(Format::Avif),
        _ => Err(format!("format should be webp, png, jpg or avif, not '{s}'")),
    }
}

/// `path` as given if absolute, else relative to `base`.
pub fn rel(base: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
}

impl Rule {
    /// The chain this rule is. `model` resolves like `[model] file` (`resolve_model`).
    pub fn to_chain(&self, number: usize, resolve_model: impl Fn(&str) -> String) -> Chain {
        let (gray, color) = match (self.gray, self.color) {
            (None, None) => (true, true),
            (Some(g), None) => (g, !g),
            (None, Some(c)) => (!c, c),
            (Some(g), Some(c)) => (g, c),
        };
        let res = |w: Option<u32>, h: Option<u32>| format!("{}x{}", w.unwrap_or(0), h.unwrap_or(0));
        Chain {
            chain_number: number.to_string(),
            min_resolution: res(self.min_width, self.min_height),
            max_resolution: res(self.max_width, self.max_height),
            is_grayscale: gray,
            is_color: color,
            min_scale_factor: Some(self.min_scale.unwrap_or(0)),
            max_scale_factor: Some(self.max_scale.unwrap_or(0)),
            model_file_path: resolve_model(&self.model),
            model_tile_size: self.tile.map_or_else(|| "Auto (Estimate)".into(), |t| t.to_string()),
            auto_adjust_levels: self.auto_levels.unwrap_or(false),
            resize_height_before_upscale: Some(self.pre_resize_height.unwrap_or(0)),
            resize_width_before_upscale: Some(self.pre_resize_width.unwrap_or(0)),
            resize_factor_before_upscale: Some(self.pre_resize_percent.unwrap_or(100.0)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_template_parses() {
        let f = File::parse(TEMPLATE).unwrap();
        assert_eq!(f.input, ["in"]);
        assert_eq!(f.output.as_deref(), Some("out"));
        assert_eq!(f.save.quality, Some(90));
        assert_eq!(f.save.lossless, Some(false));
        assert_eq!(f.save.suffix, None, "empty = not set");
        assert_eq!(f.preset, None);
        assert_eq!(f.gpu.tile, Some(512));
        assert_eq!(f.gpu.devices.as_deref(), Some("first"));
        assert_eq!(f.resize.target().unwrap(), None);
        assert!(f.rule.is_empty(), "the example rule is commented out");
    }

    #[test]
    fn values_are_taken_as_typed() {
        let f = File::parse(
            "[input]\npath = D:\\manga\\Vol. 1, Part #2\npath = \"C:\\Users\\me\\My Comics\\x.cbz\"\n\
             [output]\nquality = 85%\nlossless = YES\nformat = .PNG\n[resize]\nscale = 2x\n[gpu]\ndevices = 0, 1\n",
        )
        .unwrap();
        assert_eq!(f.input, [r"D:\manga\Vol. 1, Part #2", r"C:\Users\me\My Comics\x.cbz"]);
        assert_eq!(f.save.quality, Some(85));
        assert_eq!(f.save.lossless, Some(true));
        assert_eq!(parse_format(f.save.format.as_deref().unwrap()), Ok(Format::Png));
        assert_eq!(f.resize.target().unwrap(), Some(Target::Scale(2.0)));
        assert_eq!(f.gpu.devices.as_deref(), Some("0,1"));
    }

    #[test]
    fn mistakes_name_the_line() {
        let err = |t: &str| File::parse(t).unwrap_err();
        let e = err("[output]\n\nqualty = 3");
        assert!(e.contains("line 3") && e.contains("did you mean quality?"), "{e}");
        assert!(err("[ouptut]\nformat = png").contains("did you mean output?"));
        assert!(err("quality = 3").contains("before any [section]"));
        assert!(err("[output]\nlossless = maybe").contains("yes or no"));
        assert!(err("[output]\nquality = high").contains("line 2"));
        assert!(err("[output]\nformat = gif").contains("webp, png, jpg or avif"));
        assert!(err("[output]\nformat = png\nformat = jpg").contains("set twice"));
        assert!(err("[output]\n[output]").contains("appears twice"));
        assert!(err("[model]\nfile = a.onnx\npreset = manga").contains("pick one"));
        assert!(err("[resize]\nscale = 2\nheight = 100").contains("only one"));
        assert!(err("[resize]\nfit = 100").contains("1920x1080"));
        assert!(err("[rule]\ngray = yes").contains("needs a model"));
        assert!(err("[output\nformat = png").contains("closing ]"));
    }

    #[test]
    fn rules_in_order() {
        let f = File::parse("[rule]\ngray = yes\nmax_height = 1250\nmax_scale = 2\nmodel = m2.onnx\n\n[rule]\nmodel = any.onnx\ntile = 384px\n").unwrap();
        assert_eq!(f.rule.len(), 2);
        let c = f.rule[0].to_chain(1, |m| m.to_string());
        assert!(c.is_grayscale && !c.is_color);
        assert_eq!((c.min_resolution.as_str(), c.max_resolution.as_str()), ("0x0", "0x1250"));
        assert_eq!(c.max_scale_factor, Some(2));
        let c = f.rule[1].to_chain(2, |m| m.to_string());
        assert!(c.is_grayscale && c.is_color);
        assert_eq!(c.model_tile_size, "384");
    }

    #[test]
    fn notepad_encodings() {
        let utf8_bom = [&[0xEF, 0xBB, 0xBF][..], b"[output]\nformat = png"].concat();
        assert_eq!(File::parse(&decode_text(&utf8_bom).unwrap()).unwrap().save.format.as_deref(), Some("png"));
        let utf16: Vec<u8> = [0xFF, 0xFE].into_iter().chain("[output]\r\nformat = png\r\n".encode_utf16().flat_map(|u| u.to_le_bytes())).collect();
        assert_eq!(File::parse(&decode_text(&utf16).unwrap()).unwrap().save.format.as_deref(), Some("png"));
    }

    #[test]
    fn sizes() {
        assert_eq!(parse_size("1920x1080"), Some((1920, 1080)));
        assert_eq!(parse_size(" 10 X 20 "), Some((10, 20)));
        assert_eq!(parse_size("0x5"), None);
        assert_eq!(parse_size("abc"), None);
    }
}
