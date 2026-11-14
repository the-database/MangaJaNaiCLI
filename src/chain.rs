//! Which chain handles an image, and the sizes around the model: ports of run_upscale.py's
//! `get_chain_for_image`, `should_chain_activate_for_image` and `final_target_resize`.

use crate::settings::{Chain, Target};

/// The scale the image must reach, from its original size. In fit-to-display mode it's the scale
/// that fits both dimensions (1.x used the height ratio alone to pick the chain). `Native` has no
/// required scale: 0, which only chains without a minimum scale accept.
pub fn required_scale(target: Target, (ow, oh): (u32, u32)) -> f64 {
    match target {
        Target::Scale(s) => s,
        Target::Width(tw) => tw as f64 / ow as f64,
        Target::Height(th) => th as f64 / oh as f64,
        Target::Fit(tw, th) => (th as f64 / oh as f64).min(tw as f64 / ow as f64),
        Target::Native => 0.0,
    }
}

/// The most a width, height or fit target may enlarge an image. Pages need at most about 6x (a
/// low-resolution scan going to a 4K display); an 800x59 banner needs 28x to be 1680 tall, which
/// makes a 22780 px wide file (MangaJaNaiGUI#90). Past the models' own 4x this is the screentone-safe
/// final resize, which is still better than a reader's own resize up to the display.
pub const MAX_TARGET_SCALE: f64 = 8.0;

/// `target` for an image of size `base`, with width, height and fit targets that would enlarge it
/// more than [`MAX_TARGET_SCALE`] limited to that scale. Scale targets are the user's own factor.
pub fn limit(target: Target, base: (u32, u32)) -> Target {
    match target {
        Target::Width(_) | Target::Height(_) | Target::Fit(..) if required_scale(target, base) > MAX_TARGET_SCALE => {
            Target::Scale(MAX_TARGET_SCALE)
        }
        t => t,
    }
}

/// `"WxH"` → (w, h); 0 means unbounded. Unparseable parts count as 0.
fn parse_resolution(s: &str) -> (u32, u32) {
    let mut it = s.split('x').map(|p| p.trim().parse::<u32>().unwrap_or(0));
    (it.next().unwrap_or(0), it.next().unwrap_or(0))
}

/// Whether `chain` handles an image of the original size `(ow, oh)`, gray or not, needing `scale`.
/// All bounds are inclusive; 0 means unbounded.
pub fn activates(chain: &Chain, (ow, oh): (u32, u32), gray: bool, scale: f64) -> bool {
    let (min_w, min_h) = parse_resolution(&chain.min_resolution);
    let (max_w, max_h) = parse_resolution(&chain.max_resolution);
    if (min_w != 0 && min_w > ow) || (min_h != 0 && min_h > oh) {
        return false;
    }
    if (max_w != 0 && max_w < ow) || (max_h != 0 && max_h < oh) {
        return false;
    }
    if (gray && !chain.is_grayscale) || (!gray && !chain.is_color) {
        return false;
    }
    let max = chain.max_scale_factor.unwrap_or(0);
    let min = chain.min_scale_factor.unwrap_or(0);
    if (max != 0 && scale > max as f64) || (min != 0 && scale < min as f64) {
        return false;
    }
    true
}

/// The first chain, in list order, that handles the image.
pub fn select(chains: &[Chain], orig: (u32, u32), gray: bool, scale: f64) -> Option<&Chain> {
    chains.iter().find(|c| activates(c, orig, gray, scale))
}

/// Python's `round` (half to even), which 1.x's sizes were computed with.
fn round(v: f64) -> u32 {
    v.round_ties_even().max(1.0) as u32
}

/// The size to resize the image to before the model, if any: height and width both set → exactly
/// that; height only or width only → keep the aspect ratio; else a factor other than 100%.
pub fn pre_resize(chain: &Chain, (w, h): (u32, u32)) -> Option<(u32, u32)> {
    let rh = chain.resize_height_before_upscale.unwrap_or(0).max(0) as u32;
    let rw = chain.resize_width_before_upscale.unwrap_or(0).max(0) as u32;
    let factor = chain.resize_factor_before_upscale.unwrap_or(100.0);
    let dims = if rw != 0 && rh != 0 {
        (rw, rh)
    } else if rh != 0 {
        (round(w as f64 * rh as f64 / h as f64), rh)
    } else if rw != 0 {
        (rw, round(h as f64 * rw as f64 / w as f64))
    } else if factor > 0.0 && factor != 100.0 {
        (round(w as f64 * factor / 100.0), round(h as f64 * factor / 100.0))
    } else {
        return None;
    };
    (dims != (w, h)).then_some(dims)
}

/// The final size for an image now `(w, h)`, measured from `base` (the size before the model: the
/// file's, or the chain's pre-resized size), or `None` if it's already there. Fit mode fits
/// whichever dimension is tighter; scale mode reaches `round(base_h * scale)` tall, keeping the
/// current aspect ratio.
pub fn final_size(target: Target, base: (u32, u32), (w, h): (u32, u32)) -> Option<(u32, u32)> {
    let (ow, oh) = (base.0 as f64, base.1 as f64);
    let target = match target {
        Target::Native => return None,
        Target::Fit(tw, th) => {
            if th as f64 / oh < tw as f64 / ow {
                Target::Height(th)
            } else {
                Target::Width(tw)
            }
        }
        t => t,
    };
    let dims = match target {
        Target::Height(th) => (round(w as f64 * th as f64 / h as f64), th),
        Target::Width(tw) => (tw, round(h as f64 * tw as f64 / w as f64)),
        Target::Scale(s) => {
            let nh = round(oh * s);
            (round(w as f64 * nh as f64 / h as f64), nh)
        }
        Target::Fit(..) | Target::Native => unreachable!(),
    };
    let unchanged = match target {
        Target::Width(tw) => w == tw,
        _ => h == dims.1,
    };
    (!unchanged).then_some(dims)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain(min: &str, max: &str, gray: bool, min_s: i32, max_s: i32, model: &str) -> Chain {
        Chain {
            min_resolution: min.into(),
            max_resolution: max.into(),
            is_grayscale: gray,
            is_color: !gray,
            min_scale_factor: Some(min_s),
            max_scale_factor: Some(max_s),
            model_file_path: model.into(),
            ..Chain::default()
        }
    }

    /// MangaJaNaiConverterGui's default workflow (two color chains, then a 2x/4x pair per
    /// grayscale height band).
    fn defaults() -> Vec<Chain> {
        let mut v = vec![
            chain("0x0", "0x0", false, 0, 2, "2x_color"),
            chain("0x0", "0x0", false, 2, 0, "4x_color"),
        ];
        for (lo, hi, name) in [
            ("0x0", "0x1250", "1200p"),
            ("0x1251", "0x1350", "1300p"),
            ("0x1351", "0x1450", "1400p"),
            ("0x1451", "0x1550", "1500p"),
            ("0x1551", "0x1760", "1600p"),
            ("0x1761", "0x1984", "1920p"),
            ("0x1985", "0x0", "2048p"),
        ] {
            v.push(chain(lo, hi, true, 0, 2, &format!("2x_{name}")));
            v.push(chain(lo, hi, true, 2, 0, &format!("4x_{name}")));
        }
        v
    }

    fn pick(orig: (u32, u32), gray: bool, scale: f64) -> &'static str {
        let chains = Box::leak(Box::new(defaults()));
        select(chains, orig, gray, scale).map(|c| c.model_file_path.as_str()).unwrap_or("none")
    }

    #[test]
    fn default_workflow_picks_like_1x() {
        assert_eq!(pick((800, 1200), true, 2.0), "2x_1200p", "exactly 2x matches the 2x chain first");
        assert_eq!(pick((800, 1250), true, 3.0), "4x_1200p");
        assert_eq!(pick((800, 1251), true, 1.5), "2x_1300p");
        assert_eq!(pick((1200, 1700), true, 1.27), "2x_1600p");
        assert_eq!(pick((1400, 1984), true, 2.5), "4x_1920p");
        assert_eq!(pick((2000, 3000), true, 1.1), "2x_2048p");
        assert_eq!(pick((800, 800), false, 2.7), "4x_color");
        assert_eq!(pick((2400, 2400), false, 1.2), "2x_color");
    }

    #[test]
    fn width_bounds_and_scale_ranges_apply() {
        let c = chain("1000x0", "2000x0", true, 2, 3, "m");
        assert!(activates(&c, (1000, 50), true, 2.0));
        assert!(!activates(&c, (999, 50), true, 2.0));
        assert!(!activates(&c, (2001, 50), true, 2.0));
        assert!(!activates(&c, (1500, 50), true, 1.9));
        assert!(!activates(&c, (1500, 50), true, 3.1));
        assert!(!activates(&c, (1500, 50), false, 2.0), "a gray-only chain skips color");
    }

    #[test]
    fn fit_mode_uses_the_tighter_dimension() {
        // 1000x2000 into 1200x1600: height allows 0.8x, width 1.2x -> 0.8x, fit height
        assert_eq!(required_scale(Target::Fit(1200, 1600), (1000, 2000)), 0.8);
        assert_eq!(final_size(Target::Fit(1200, 1600), (1000, 2000), (4000, 8000)), Some((800, 1600)));
        // a wide page fits the width
        assert_eq!(final_size(Target::Fit(1200, 1600), (2000, 1000), (8000, 4000)), Some((1200, 600)));
    }

    #[test]
    fn final_sizes() {
        assert_eq!(final_size(Target::Scale(2.0), (800, 1200), (1600, 2400)), None, "already there");
        assert_eq!(final_size(Target::Scale(3.0), (800, 1200), (3200, 4800)), Some((2400, 3600)));
        assert_eq!(final_size(Target::Height(2160), (800, 1200), (1600, 2400)), Some((1440, 2160)));
        assert_eq!(final_size(Target::Width(1000), (800, 1200), (1600, 2400)), Some((1000, 1500)));
        assert_eq!(final_size(Target::Width(1600), (800, 1200), (1600, 2400)), None);
        // half-to-even rounding, like Python: 5 * 5 / 2 = 12.5 -> 12
        assert_eq!(final_size(Target::Height(5), (5, 2), (5, 2)), Some((12, 5)));
    }

    #[test]
    fn size_targets_enlarge_at_most_8x() {
        let h = Target::Height(1680);
        assert_eq!(limit(h, (800, 59)), Target::Scale(8.0), "a banner");
        assert_eq!(final_size(limit(h, (800, 59)), (800, 59), (3200, 236)), Some((6400, 472)));
        assert_eq!(limit(h, (976, 1417)), h, "a page");
        assert_eq!(limit(Target::Height(3840), (533, 800)), Target::Height(3840), "4.8x: a low-res scan to 4K");
        assert_eq!(limit(Target::Height(1680), (100, 210)), Target::Height(1680), "exactly 8x");
        assert_eq!(limit(Target::Width(1600), (800, 20000)), Target::Width(1600), "a webtoon strip: 2x");
        assert_eq!(limit(Target::Fit(1200, 1600), (20, 20)), Target::Scale(8.0), "an icon in fit mode");
        assert_eq!(limit(Target::Fit(3000, 1680), (800, 59)), Target::Fit(3000, 1680), "fit bounds the banner by width: 3.75x");
        assert_eq!(limit(Target::Scale(10.0), (800, 59)), Target::Scale(10.0), "a scale is the user's own");
        assert_eq!(limit(Target::Native, (800, 59)), Target::Native);
        assert_eq!(required_scale(limit(h, (800, 59)), (800, 59)), 8.0, "and the chain is picked for 8x");
    }

    #[test]
    fn pre_resize_precedence() {
        let mut c = Chain::default();
        assert_eq!(pre_resize(&c, (1000, 2000)), None);
        c.resize_factor_before_upscale = Some(50.0);
        assert_eq!(pre_resize(&c, (1000, 2000)), Some((500, 1000)));
        c.resize_width_before_upscale = Some(800);
        assert_eq!(pre_resize(&c, (1000, 2000)), Some((800, 1600)), "width beats the factor");
        c.resize_height_before_upscale = Some(1000);
        assert_eq!(pre_resize(&c, (1000, 2000)), Some((800, 1000)), "both set: exact size");
        c.resize_width_before_upscale = Some(0);
        assert_eq!(pre_resize(&c, (1000, 2000)), Some((500, 1000)));
    }

    #[test]
    fn tile_sizes() {
        use crate::tiles::Tile;
        assert_eq!(Tile::from_setting("Auto (Estimate)"), Tile::Auto);
        assert_eq!(Tile::from_setting("Maximum"), Tile::Fixed(1024, 1024));
        assert_eq!(Tile::from_setting("No Tiling"), Tile::Fixed(1024, 1024));
        assert_eq!(Tile::from_setting("384"), Tile::Fixed(384, 384));
        assert_eq!(Tile::from_setting("200"), Tile::Fixed(192, 192), "snapped to a multiple of 16");
        assert_eq!(Tile::from_setting("16"), Tile::Fixed(64, 64));
        assert_eq!(Tile::from_setting("garbage"), Tile::Auto);
    }
}
