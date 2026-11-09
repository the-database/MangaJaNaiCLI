//! `mangajanai store`: where the shared pack store is, and moving it to another drive, for an app's
//! install of the CLI (yosh's add-on) to offer what MangaJaNai GUI's App Settings > Storage does.
//! It runs aji-runtime's `location`, `move` and `reset` through the library, with the same output:
//!
//! - `store location [--json]`: `default <root>`, or `moved <root> -> <target> (available|missing)`;
//!   with `--json`, `{"root", "moved", "target", "available"}`. Changes nothing.
//! - `store move FOLDER [--progress lines]`: move the store to a new or empty folder on a local
//!   drive. The default path becomes a link to it (a directory junction on Windows, a symlink on
//!   Linux), so every app keeps reaching it there. Nothing is downloaded again.
//! - `store move --default [--progress lines]`: move it back to the default path.
//! - `store reset`: drop a moved store whose folder is gone for good. Only the link goes; the next
//!   setup starts an empty store at the default path. Refused while the folder is there.
//!
//! The stdout lines of `--progress lines`, one event per line, fields separated by single spaces:
//!
//! - `STORE_WAIT`: another installer holds the store; the move waits for it.
//! - `STORE_MOVE <copied> <total>`: bytes copied so far, while `move` copies.
//! - `STORE_DONE`: the move finished.
//!
//! A failure prints `error: ...` on stderr and exits 1 without `STORE_DONE`, like setup. Nothing is
//! changed then: aji-runtime puts everything back (on Windows a store with a file loaded by any app
//! doesn't move: "close the apps that use it... Nothing was changed.").

use std::cell::RefCell;
use std::io::Write;
use std::path::{Path, PathBuf};

use aji_runtime::location::{self, Location};

/// What `store` was asked to do.
#[derive(Debug, PartialEq)]
pub enum Command {
    Location { json: bool },
    Move(PathBuf),
    MoveBack,
    Reset,
}

const USAGE: &str = "store location [--json] | store move FOLDER | store move --default | store reset";

impl Command {
    /// `args` are the words after `store`; `json` and `to_default` the `--json` and `--default`
    /// flags.
    pub fn parse(args: &[String], json: bool, to_default: bool) -> Result<Command, String> {
        let words: Vec<&str> = args.iter().map(String::as_str).collect();
        let c = match (words.as_slice(), to_default) {
            (["location"], false) => Command::Location { json },
            (["move", folder], false) => Command::Move(PathBuf::from(folder)),
            (["move"], true) => Command::MoveBack,
            (["reset"], false) => Command::Reset,
            (["move"], false) => return Err(format!("store move needs a folder, or --default (usage: {USAGE})")),
            _ => return Err(format!("usage: mangajanai {USAGE}")),
        };
        if json && !matches!(c, Command::Location { .. }) {
            return Err("--json goes with store location".into());
        }
        Ok(c)
    }
}

/// The store's default path, where the apps reach it (`AJI_RUNTIME_STORE` for tests).
pub fn root() -> Result<PathBuf, String> {
    aji_runtime::Store::default_root().map_err(|e| e.to_string())
}

/// When the store at `root` was moved to a folder that's missing now (an unplugged drive): the
/// message to stop with, instead of the missing-runtime error that would follow.
pub fn missing(root: &Path) -> Option<String> {
    match location::location(root) {
        Location::Moved { target, available: false } => Some(format!(
            "the shared runtime's folder {} isn't available. Is its drive connected? Reconnect it, or reset the location with `mangajanai store reset`.",
            target.display()
        )),
        _ => None,
    }
}

/// [`missing`] for the per-user store, as an error.
pub fn check_available() -> Result<(), String> {
    match missing(&root()?) {
        Some(m) => Err(m),
        None => Ok(()),
    }
}

/// `store location`'s output for the store at `root`.
pub fn location_text(root: &Path, json: bool) -> String {
    let loc = location::location(root);
    if json {
        let (moved, target, available) = match &loc {
            Location::Default => (false, None, true),
            Location::Moved { target, available } => (true, Some(target.to_string_lossy().into_owned()), *available),
        };
        let out = serde_json::json!({"root": root.to_string_lossy(), "moved": moved, "target": target, "available": available});
        serde_json::to_string_pretty(&out).unwrap()
    } else {
        match loc {
            Location::Default => format!("default {}", root.display()),
            Location::Moved { target, available } => {
                format!("moved {} -> {} ({})", root.display(), target.display(), if available { "available" } else { "missing" })
            }
        }
    }
}

fn mb(b: u64) -> String {
    format!("{:.0} MB", b as f64 / 1e6)
}

/// Runs `c` on the store at `root`, writing `location`'s output and the `--progress lines` (when
/// `lines`) to `out`, human output to stderr. Err is aji-runtime's message.
pub fn run_at(root: &Path, c: &Command, lines: bool, out: &mut dyn Write) -> Result<(), String> {
    // `waiting` and `progress` both write lines, so they share `out`.
    let out = RefCell::new(out);
    let say = |s: std::fmt::Arguments| {
        if lines {
            let mut o = out.borrow_mut();
            let _ = writeln!(o, "{s}");
            let _ = o.flush();
        }
    };
    let moved = match c {
        Command::Location { json } => {
            let _ = writeln!(out.borrow_mut(), "{}", location_text(root, *json));
            return Ok(());
        }
        Command::Reset => {
            // aji-runtime's refusal names its own command
            location::reset(root).map_err(|e| e.to_string().replace("aji-runtime move --default", "mangajanai store move --default"))?;
            eprintln!("The store is at the default path {} (empty until an app installs its packs again).", root.display());
            return Ok(());
        }
        Command::Move(dest) => Some(dest),
        Command::MoveBack => None,
    };
    let mut waiting = || {
        say(format_args!("STORE_WAIT"));
        eprintln!("Waiting for another installer to finish with the store...");
    };
    let mut last = 0u64;
    let mut progress = |done: u64, total: u64| {
        say(format_args!("STORE_MOVE {done} {total}"));
        if done == 0 || done == total || done - last >= 100_000_000 {
            last = done;
            eprintln!("Copying the store: {} of {}", mb(done), mb(total));
        }
    };
    match moved {
        Some(dest) => location::move_to(root, dest, &mut waiting, &mut progress),
        None => location::move_back(root, &mut waiting, &mut progress),
    }
    .map_err(|e| e.to_string())?;
    match location::location(root) {
        Location::Moved { target, .. } => eprintln!("The store is now in {} ({} links there).", target.display(), root.display()),
        Location::Default => eprintln!("The store is in {}.", root.display()),
    }
    say(format_args!("STORE_DONE"));
    Ok(())
}

/// `mangajanai store ...`: 0, or 1 after printing `error: ...` (a usage error is Err, exit 2).
pub fn run(args: &[String], json: bool, to_default: bool, lines: bool) -> Result<i32, String> {
    let c = Command::parse(args, json, to_default)?;
    let root = root()?;
    match run_at(&root, &c, lines, &mut std::io::stdout().lock()) {
        Ok(()) => Ok(0),
        Err(e) => {
            eprintln!("error: {e}");
            Ok(1)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mangajanai-store-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    fn run_lines(root: &Path, c: &Command) -> (Result<(), String>, Vec<String>) {
        let mut out = Vec::new();
        let r = run_at(root, c, true, &mut out);
        (r, String::from_utf8(out).unwrap().lines().map(str::to_string).collect())
    }

    /// A store with a pack in it, at `root`.
    fn store_at(root: &Path) {
        let s = aji_runtime::Store::open(root).unwrap();
        let p = s.pack_dir("trt", "1");
        std::fs::create_dir_all(&p).unwrap();
        std::fs::write(p.join("nvinfer.dll"), vec![7u8; 1000]).unwrap();
    }

    #[test]
    fn commands_parse() {
        assert_eq!(Command::parse(&args(&["location"]), false, false), Ok(Command::Location { json: false }));
        assert_eq!(Command::parse(&args(&["location"]), true, false), Ok(Command::Location { json: true }));
        assert_eq!(Command::parse(&args(&["move", "D:/x"]), false, false), Ok(Command::Move("D:/x".into())));
        assert_eq!(Command::parse(&args(&["move"]), false, true), Ok(Command::MoveBack));
        assert_eq!(Command::parse(&args(&["reset"]), false, false), Ok(Command::Reset));
        for (a, json, def) in [
            (&["move"][..], false, false),
            (&["move", "D:/x"][..], false, true),
            (&["reset"][..], true, false),
            (&["location", "x"][..], false, false),
            (&[][..], false, false),
            (&["bogus"][..], false, false),
        ] {
            assert!(Command::parse(&args(a), json, def).is_err(), "{a:?} {json} {def}");
        }
    }

    #[test]
    fn location_reads_like_aji_runtime() {
        let dir = temp("location");
        let root = dir.join("aji");
        assert_eq!(location_text(&root, false), format!("default {}", root.display()));
        let j: serde_json::Value = serde_json::from_str(&location_text(&root, true)).unwrap();
        assert_eq!(j["moved"], false);
        assert_eq!(j["available"], true);
        assert!(j["target"].is_null());
        assert_eq!(j["root"], root.to_string_lossy().as_ref());
        assert!(missing(&root).is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn move_away_and_back_with_progress_lines() {
        let dir = temp("move");
        let root = dir.join("aji");
        let dest = dir.join("elsewhere");
        store_at(&root);

        let (r, lines) = run_lines(&root, &Command::Move(dest.clone()));
        r.unwrap();
        // STORE_MOVE 0 <total>, ..., STORE_MOVE <total> <total>, STORE_DONE
        let moves: Vec<(u64, u64)> = lines
            .iter()
            .filter_map(|l| l.strip_prefix("STORE_MOVE "))
            .map(|l| l.split_once(' ').map(|(a, b)| (a.parse().unwrap(), b.parse().unwrap())).unwrap())
            .collect();
        let total = moves[0].1;
        assert!(total >= 1000);
        assert_eq!(moves.first(), Some(&(0, total)));
        assert_eq!(moves.last(), Some(&(total, total)));
        assert_eq!(lines.last().map(String::as_str), Some("STORE_DONE"));
        assert_eq!(lines.len(), moves.len() + 1, "{lines:?}");
        // reached through the default path
        assert_eq!(std::fs::read(root.join("packs/trt/1/nvinfer.dll")).unwrap(), vec![7u8; 1000]);
        let j: serde_json::Value = serde_json::from_str(&location_text(&root, true)).unwrap();
        assert_eq!(j["moved"], true);
        assert_eq!(j["available"], true);
        assert!(location_text(&root, false).ends_with("(available)"));

        // reset is refused while the folder is there, with nothing changed
        let (r, lines) = run_lines(&root, &Command::Reset);
        let e = r.unwrap_err();
        assert!(e.contains("move it back instead (mangajanai store move --default)"), "{e}");
        assert!(lines.is_empty());

        let (r, lines) = run_lines(&root, &Command::MoveBack);
        r.unwrap();
        assert_eq!(lines.last().map(String::as_str), Some("STORE_DONE"));
        assert_eq!(location_text(&root, false), format!("default {}", root.display()));
        assert!(!dest.exists());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refusals_change_nothing_and_print_no_done() {
        let dir = temp("refuse");
        let root = dir.join("aji");
        store_at(&root);
        let full = dir.join("full");
        std::fs::create_dir_all(&full).unwrap();
        std::fs::write(full.join("x"), b"x").unwrap();
        for dest in [full.clone(), PathBuf::from("relative"), root.join("packs")] {
            let (r, lines) = run_lines(&root, &Command::Move(dest.clone()));
            assert!(r.is_err(), "{}", dest.display());
            assert!(!lines.contains(&"STORE_DONE".to_string()));
            assert_eq!(location::location(&root), Location::Default);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_folder_says_so_until_reset() {
        let dir = temp("missing");
        let root = dir.join("aji");
        let dest = dir.join("drive");
        store_at(&root);
        run_lines(&root, &Command::Move(dest.clone())).0.unwrap();
        // the drive is unplugged
        std::fs::rename(&dest, dir.join("unplugged")).unwrap();
        assert!(location_text(&root, false).ends_with("(missing)"));
        let j: serde_json::Value = serde_json::from_str(&location_text(&root, true)).unwrap();
        assert_eq!((j["moved"].as_bool(), j["available"].as_bool()), (Some(true), Some(false)));
        let m = missing(&root).unwrap();
        assert!(m.contains("isn't available. Is its drive connected?") && m.contains("mangajanai store reset"), "{m}");
        // moving back can't, reset can
        assert!(run_lines(&root, &Command::MoveBack).0.is_err());
        run_lines(&root, &Command::Reset).0.unwrap();
        assert_eq!(location::location(&root), Location::Default);
        assert!(missing(&root).is_none());
        assert!(dir.join("unplugged").join("packs").is_dir(), "reset never touches the folder");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
