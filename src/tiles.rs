//! The "tile" setting. The shape arithmetic and the record of fitted engines are yosh-image's
//! `tiles`, shared with yosh's upscaler (see `Upscaler` for auto mode's policy).


use std::fmt;

pub use yosh_image::tiles::{self, DEFAULT_OVERLAP, MemoryUse, Shape, area_budget, fit};

/// Auto mode's square tile: every page can run on it while (or instead of) a fitted engine builds.
pub const FALLBACK: u32 = 512;

/// A chain's tile setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tile {
    /// The square [`FALLBACK`], plus engines fitted to page sizes that come up often enough.
    Auto,
    /// Exactly this tile (input pixels, width x height) for every page, as before auto mode.
    Fixed(u32, u32),
}

impl Tile {
    /// `auto` (also MangaJaNai GUI's "Auto (Estimate)", or empty), a size in pixels (`512`,
    /// `512px`) for a square, or `WxH` (`1216x704`). Sides are rounded down to a multiple of 16
    /// within 64..4096. GUI 1.x's "Maximum" and "No Tiling" are 1024.
    pub fn parse(s: &str) -> Result<Tile, String> {
        let t = s.trim();
        let lower = t.to_ascii_lowercase();
        let lower = lower.strip_suffix("px").unwrap_or(&lower).trim();
        let side = |v: &str| v.trim().parse::<u32>().ok().map(|n| (n / 16 * 16).clamp(64, 4096));
        match lower {
            "" | "auto" | "auto (estimate)" => Ok(Tile::Auto),
            "maximum" | "no tiling" => Ok(Tile::Fixed(1024, 1024)),
            _ => {
                let parsed = match lower.split_once('x') {
                    Some((w, h)) => side(w).zip(side(h)),
                    None => side(lower).map(|n| (n, n)),
                };
                parsed.map(|(w, h)| Tile::Fixed(w, h)).ok_or_else(|| format!("'{t}' is not a tile size (auto, 512, or 1216x704)"))
            }
        }
    }

    /// A chain's "Model Tile Size", leniently: anything unrecognized is auto (it was 512, the
    /// square auto mode falls back to).
    pub fn from_setting(s: &str) -> Tile {
        Tile::parse(s).unwrap_or(Tile::Auto)
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Tile::Auto => f.write_str("auto"),
            Tile::Fixed(w, h) if w == h => write!(f, "{w}"),
            Tile::Fixed(w, h) => write!(f, "{w}x{h}"),
        }
    }
}

/// `1216x704`, or `512` for a square.
pub fn shape_label(shape: (u32, u32)) -> String {
    tiles::label(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_parse() {
        assert_eq!(Tile::parse("auto"), Ok(Tile::Auto));
        assert_eq!(Tile::parse("Auto (Estimate)"), Ok(Tile::Auto));
        assert_eq!(Tile::parse(""), Ok(Tile::Auto));
        assert_eq!(Tile::parse("512"), Ok(Tile::Fixed(512, 512)));
        assert_eq!(Tile::parse("384px"), Ok(Tile::Fixed(384, 384)));
        assert_eq!(Tile::parse("1216x704"), Ok(Tile::Fixed(1216, 704)));
        assert_eq!(Tile::parse("1200X964"), Ok(Tile::Fixed(1200, 960)));
        assert_eq!(Tile::parse("Maximum"), Ok(Tile::Fixed(1024, 1024)));
        assert_eq!(Tile::parse("No Tiling"), Ok(Tile::Fixed(1024, 1024)));
        assert_eq!(Tile::parse("8192"), Ok(Tile::Fixed(4096, 4096)));
        assert!(Tile::parse("big").is_err());
        assert_eq!(Tile::from_setting("garbage"), Tile::Auto);
        assert_eq!(Tile::Fixed(512, 512).to_string(), "512");
        assert_eq!(Tile::Fixed(1216, 704).to_string(), "1216x704");
        assert_eq!(Tile::Auto.to_string(), "auto");
    }
}
