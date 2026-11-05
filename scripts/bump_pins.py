#!/usr/bin/env python3
"""Re-pin MangaJaNaiCLI's upstream releases: scripts/bump_pins.py <source> <tag>.

Sources:
  libaji         the-database/animejanai-inference release: AJI_VERSION and the TensorRT-RTX
                 asset's SHA-256/size in src/setup.rs (the release workflow reads AJI_VERSION too)
  runtime-packs  this repo's runtime-<n> release: RUNTIME_TAG and both packs-<platform>.json
                 SHA-256s in src/setup.rs
  yosh-image     a the-database/yosh ref (a release tag, a branch or a commit): the yosh-image
                 `rev` in Cargo.toml, and Cargo.lock (needs cargo). Skipped when nothing under
                 crates/yosh-image/ or the workspace Cargo.toml changed.

.github/workflows/bump-pins.yml runs this when an upstream release sends a repository_dispatch,
then opens a PR; it also runs by hand. Each pinned asset is downloaded to record its SHA-256.
--out DIR writes the PR title and body. Standard library only.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UA = "janai-bump-pins"
CHECKED = []  # assets confirmed on the release (downloaded or not)


class Fail(Exception):
    pass


def gh(path):
    """GET api.github.com/<path> (GH_TOKEN, when set, only raises the rate limit)."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": UA}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"https://api.github.com/{path}", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def release(repo, tag):
    rel = gh(f"repos/{repo}/releases/tags/{tag}")
    if rel is None:
        raise Fail(f"{repo} has no published release {tag}")
    if rel.get("draft"):
        raise Fail(f"{repo} {tag} is still a draft; its assets 404 for the builds")
    return rel


def asset(rel, name):
    for a in rel["assets"]:
        if a["name"] == name:
            CHECKED.append(name)
            return a
    have = ", ".join(a["name"] for a in rel["assets"]) or "none"
    raise Fail(f"{name} isn't on {rel['tag_name']} (it has: {have}). Upload it, then re-run.")


def digest(a):
    """Download a release asset; return (sha256, size). Cross-checks GitHub's recorded digest."""
    print(f"  downloading {a['name']} ({a['size']:,} bytes)", flush=True)
    req = urllib.request.Request(a["browser_download_url"], headers={"User-Agent": UA})
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(req, timeout=600) as r:
        while chunk := r.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    sha = h.hexdigest()
    if n != a["size"]:
        raise Fail(f"{a['name']}: downloaded {n} bytes, the release says {a['size']}")
    if a.get("digest") and a["digest"] != f"sha256:{sha}":
        raise Fail(f"{a['name']}: SHA-256 {sha} doesn't match GitHub's {a['digest']}")
    return sha, n


def version_key(tag):
    """Comparable key for v1.2.3 / 1.2.3 / runtime-4 / 2026-10-24-<hash> tags."""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})-", tag)
    if m:
        return tuple(int(x) for x in m.groups())
    return tuple(int(x) for x in re.findall(r"\d+", tag))


class File:
    """One file's text, edited in place by labelled one-match substitutions. Keeps the file's
    bytes otherwise (BOM, CRLF)."""

    def __init__(self, rel, changes):
        self.rel = rel
        self.path = ROOT / rel
        self.orig = self.text = self.path.read_bytes().decode("utf-8")
        self.changes = changes

    def _match(self, label, pattern):
        found = list(re.finditer(pattern, self.text, re.M))
        if len(found) != 1:
            raise Fail(f"{label}: expected one match of {pattern!r} in {self.rel}, found {len(found)}")
        return found[0]

    def get(self, label, pattern):
        return self._match(label, pattern).group(1)

    def set(self, label, pattern, value):
        """`pattern`'s group 1 is the pinned value."""
        m = self._match(label, pattern)
        old = m.group(1)
        if old != value:
            self.text = self.text[: m.start(1)] + value + self.text[m.end(1):]
        self.changes.append((self.rel, label, old, value))
        return old

    def save(self):
        if self.text != self.orig:
            self.path.write_bytes(self.text.encode("utf-8"))


def guard_downgrade(what, old, new, allow):
    if version_key(new) < version_key(old) and not allow:
        raise Fail(f"{what}: {new} is older than the pinned {old} (pass --allow-downgrade to pin it anyway)")


def rust_int(n):
    return f"{n:_}"


def run(sources, repo_name):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=sorted(sources))
    p.add_argument("tag", help="the upstream release tag (yosh-image: any yosh ref)")
    p.add_argument("--out", type=Path, help="write title.txt and body.md here when something changed")
    p.add_argument("--allow-downgrade", action="store_true")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", a.tag):
        sys.exit(f"error: not a tag: {a.tag!r}")
    changes = []
    try:
        files, title, notes = sources[a.source](a.tag, changes, a.allow_downgrade)
    except Fail as e:
        sys.exit(f"error: {e}")
    for f in files:
        f.save()
    edited = [c for c in changes if c[2] != c[3]]
    for rel, label, old, new in changes:
        print(f"{'changed' if old != new else 'same   '}  {rel}: {label}: {old} -> {new}")
    if not edited:
        print(f"{repo_name} already pins {a.source} {a.tag}; nothing to do")
        return
    if a.out:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "title.txt").write_text(title + "\n", encoding="utf-8")
        rows = "\n".join(f"| `{rel}` | {label} | `{old}` | `{new}` |" for rel, label, old, new in edited)
        checked = ""
        if CHECKED:
            checked = (" Assets confirmed on the release: " + ", ".join(f"`{n}`" for n in CHECKED)
                       + ". Every SHA-256 and size above was computed from the downloaded file.")
        body = (
            f"{notes}\n\n| File | Pin | Old | New |\n|---|---|---|---|\n{rows}\n\n"
            f"Opened by `.github/workflows/bump-pins.yml` (`scripts/bump_pins.py {a.source} {a.tag}`)."
            f"{checked} A newer release of the same source updates this PR instead of opening "
            "another. Review and merge by hand.\n"
        )
        (a.out / "body.md").write_text(body, encoding="utf-8")


AJI_REPO = "the-database/animejanai-inference"
AJI_VERSION = r'^const AJI_VERSION: &str = "([^"]+)";'
RUNTIME_TAG = r'^const RUNTIME_TAG: &str = "([^"]+)";'
YOSH_REV = r'^yosh-image = \{ git = "https://github\.com/the-database/yosh", rev = "([0-9a-f]{40})"'


def bump_libaji(tag, changes, allow_downgrade):
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag):
        raise Fail(f"{tag} isn't a libaji release tag (vX.Y.Z)")
    f = File("src/setup.rs", changes)
    guard_downgrade("AJI_VERSION", f.get("AJI_VERSION", AJI_VERSION), tag, allow_downgrade)
    rel = release(AJI_REPO, tag)
    # The release workflow bundles these two (by AJI_VERSION); setup downloads the third.
    for name in ("aji-windows-x64.zip", "aji-linux-x64.tar.zst"):
        asset(rel, name)
    sha, size = digest(asset(rel, "aji-trtrtx-windows-x64.zip"))
    f.set("AJI_VERSION", AJI_VERSION, tag)
    f.set("TRTRTX_SHA256", r'^const TRTRTX_SHA256: &str = "([0-9a-f]{64})";', sha)
    f.set("TRTRTX_BYTES", r"^const TRTRTX_BYTES: u64 = ([0-9_]+);", rust_int(size))
    notes = (
        f"Pins [libaji {tag}]({rel['html_url']}): the archives the release workflow bundles "
        "(`aji-windows-x64.zip`, `aji-linux-x64.tar.zst`) and the TensorRT-RTX pack `mangajanai setup` "
        "downloads. A change to `include/aji_image.h` (`AJI_IMAGE_API_VERSION`) needs code changes "
        "here, and maybe in yosh-image, before this can merge. To test it, run the `release` workflow "
        "by hand on this branch (it builds the archives without publishing)."
    )
    return [f], f"Bump libaji to {tag}", notes


def bump_runtime(tag, changes, allow_downgrade):
    if not re.fullmatch(r"runtime-\d+", tag):
        raise Fail(f"{tag} isn't a runtime-<n> tag")
    f = File("src/setup.rs", changes)
    guard_downgrade("RUNTIME_TAG", f.get("RUNTIME_TAG", RUNTIME_TAG), tag, allow_downgrade)
    rel = release("the-database/MangaJaNaiCLI", tag)
    f.set("RUNTIME_TAG", RUNTIME_TAG, tag)
    for plat, const in (("windows", "RUNTIME_INDEX_SHA256_WINDOWS"), ("linux", "RUNTIME_INDEX_SHA256_LINUX")):
        sha, _ = digest(asset(rel, f"packs-{plat}-x64.json"))
        f.set(const, rf'^const {const}: &str = "([0-9a-f]{{64}})";', sha)
    notes = (
        f"Pins the runtime packs on [{tag}]({rel['html_url']}) (`runtime-packs.yml`). `mangajanai "
        "setup` checks each `packs-<platform>.json` against these SHA-256s, and the index pins every pack."
    )
    return [f], f"Bump the runtime packs to {tag}", notes


def bump_yosh_image(ref, changes, allow_downgrade):
    f = File("Cargo.toml", changes)
    old = f.get("yosh-image rev", YOSH_REV)
    commit = gh(f"repos/the-database/yosh/commits/{ref}")
    if commit is None:
        raise Fail(f"the-database/yosh has no ref {ref}")
    new = commit["sha"]
    if new == old:
        f.set("yosh-image rev", YOSH_REV, new)
        return [f], "", ""
    cmp = gh(f"repos/the-database/yosh/compare/{old}...{new}")
    if cmp["status"] == "behind" and not allow_downgrade:
        raise Fail(f"yosh {ref} ({new[:10]}) is behind the pinned {old[:10]} (pass --allow-downgrade)")
    touched = [x["filename"] for x in cmp.get("files", [])
               if x["filename"].startswith("crates/yosh-image/") or x["filename"] == "Cargo.toml"]
    if not touched:
        print(f"yosh-image is unchanged between {old[:10]} and {new[:10]}; leaving the pin")
        return [f], "", ""
    f.set("yosh-image rev", YOSH_REV, new)
    f.save()
    subprocess.run(["cargo", "update", "-p", "yosh-image"], cwd=ROOT, check=True)
    shown = ref if ref == new else f"{ref} ({new[:10]})"
    files = "\n".join(f"- `{t}`" for t in touched[:30]) + ("\n- …" if len(touched) > 30 else "")
    notes = (
        f"Pins yosh-image to [yosh {shown}](https://github.com/the-database/yosh/commit/{new}) "
        f"([changes](https://github.com/the-database/yosh/compare/{old[:12]}...{new[:12]}), "
        f"{cmp.get('ahead_by', '?')} commits). Cargo.lock follows (`cargo update -p yosh-image`). "
        f"Files that changed:\n\n{files}"
    )
    return [f], f"Bump yosh-image to yosh {shown}", notes


SOURCES = {"libaji": bump_libaji, "runtime-packs": bump_runtime, "yosh-image": bump_yosh_image}

if __name__ == "__main__":
    run(SOURCES, "MangaJaNaiCLI")
