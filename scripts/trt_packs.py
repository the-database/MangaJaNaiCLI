#!/usr/bin/env python3
"""Build the TensorRT runtime packs that `mangajanai setup` installs.

NVIDIA ships TensorRT as one archive with builder resources for every GPU generation (several GB).
This splits it into packs so a machine downloads only what it needs:

    trt-runtime-<platform>.7z     nvinfer, the plugin and ONNX parser libraries, trtexec, cudart,
                                  NVIDIA's notices
    trt-<smNN|ptx>-<platform>.7z  the builder resources for one GPU generation (PTX: the fallback
                                  for generations without their own)

plus packs-<platform>.json: each pack's sha256, size, unpacked size and file list. Files sit at
each archive's root, so extracting into the runtime folder installs them.

Sources are pinned below and every download is checked against its SHA-256:
    windows  NVIDIA's redistributable TensorRT zip + the CUDA redistributable cudart zip
    linux    NVIDIA's CUDA apt-repo packages (per-file SHA-256 from its Packages index) + the CUDA
             redistributable cudart tarball. Needs dpkg-deb (Debian/Ubuntu).

usage: trt_packs.py --print-version
       trt_packs.py --platform windows|linux --out DIR [--work DIR] [--local-dir DIR] [--sevenzip PATH]
  --local-dir  take the files from a folder that already holds them (flat), instead of downloading
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

TRT_VERSION = "11.4.0.106"
TRT_CUDA = "13.4"
CUDART_VERSION = "13.4.49"

# NVIDIA publishes no checksum beside the Windows zip; recorded from a verified download (the same
# pin as MangaJaNai GUI's and mpv-AnimeJaNai's builders).
TRT_ZIP_SHA256 = "b6fd86301ad803492adf7a06de29ce45e3968d29637a1437d0fb0994f412c980"
# From developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json.
CUDART_SHA256 = {
    "windows": "e6663f3d3e8949eedc2d5ab92c7c5b9fa3f2a222086d91c42bfc5a38bf2b0225",
    "linux": "ec3df9f91560db23a064c6b395d4fde17b2893f5bf954c6bd193dd03a44de522",
}
# From the CUDA apt repo's Packages index (NVIDIA's own checksums).
CUDA_APT = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
DEB_SHA256 = {
    "libnvinfer11": "bd24db760728779e980c8852ba87e148a0bf9fb4e69d69bc14f3bdabc0423797",
    "libnvinfer-plugin11": "a99bd3de4fd45585d4a1fce8f6a8d6415e1a7946c86d907f64af0b9b487aee66",
    "libnvonnxparsers11": "d7b0e623e6b8d1c368a000b996938494137e2584b85e2eed8b89f7b34092436c",
    "libnvinfer-bin": "36b0a4f11c3a15bf9f2d4eb5a33ce02b7248582eb384696a1d80a154abf005f8",
}

WIN_KEEP = {"nvinfer_11.dll", "nvinfer_plugin_11.dll", "nvonnxparser_11.dll", "trtexec.exe"}
BUILDER = re.compile(r"builder_resource_([a-z0-9]+)[._]")


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest, want):
    if os.path.exists(dest) and sha256(dest) == want:
        log(f"cached {os.path.basename(dest)}")
        return dest
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "MangaJaNaiCLI-trt-packs"})
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(dest + ".part")
    if got != want:
        os.remove(dest + ".part")
        sys.exit(f"SHA-256 mismatch for {url}: got {got}, want {want}")
    os.replace(dest + ".part", dest)
    return dest


def stage_windows(work, stage):
    trt_dir = ".".join(TRT_VERSION.split(".")[:3])
    asset = f"TensorRT-Enterprise-{TRT_VERSION}-Windows-amd64-cuda-{TRT_CUDA}-Release-external.zip"
    z = fetch(f"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{trt_dir}/zip/{asset}",
              os.path.join(work, asset), TRT_ZIP_SHA256)
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if "/bin/" in info.filename and (name in WIN_KEEP or BUILDER.search(name)):
                out = name
            elif name.lower() == "acknowledgements.txt":
                out = "TensorRT_Acknowledgements.txt"
            else:
                continue
            with zf.open(info) as src, open(os.path.join(stage, out), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    stem = f"cuda_cudart-windows-x86_64-{CUDART_VERSION}-archive"
    c = fetch(f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/{stem}.zip",
              os.path.join(work, stem + ".zip"), CUDART_SHA256["windows"])
    with zipfile.ZipFile(c) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            out = "CUDA_LICENSE.txt" if name == "LICENSE" else name if re.fullmatch(r"cudart64_\d+\.dll", name) else None
            if out:
                with zf.open(info) as src, open(os.path.join(stage, out), "wb") as dst:
                    shutil.copyfileobj(src, dst)


def stage_linux(work, stage):
    root = os.path.join(work, "debroot")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for pkg, want in DEB_SHA256.items():
        deb = f"{pkg}_{TRT_VERSION}-1+cuda{TRT_CUDA}_amd64.deb"
        p = fetch(f"{CUDA_APT}/{deb}", os.path.join(work, deb), want)
        subprocess.run(["dpkg-deb", "-x", p, root], check=True)
    lib = os.path.join(root, "usr", "lib", "x86_64-linux-gnu")
    # Libraries are stored under their sonames (what libaji_trt.so and trtexec ask for); the builder
    # resources keep their file names, which is how libnvinfer opens them.
    for soname in ("libnvinfer.so.11", "libnvinfer_plugin.so.11", "libnvonnxparser.so.11"):
        shutil.copyfile(os.path.realpath(os.path.join(lib, soname)), os.path.join(stage, soname))
    for name in os.listdir(lib):
        if BUILDER.search(name) and not os.path.islink(os.path.join(lib, name)):
            shutil.copyfile(os.path.join(lib, name), os.path.join(stage, name))
    shutil.copyfile(os.path.join(root, "usr", "src", "tensorrt", "bin", "trtexec"), os.path.join(stage, "trtexec"))
    os.chmod(os.path.join(stage, "trtexec"), 0o755)
    copyright_ = os.path.join(root, "usr", "share", "doc", "libnvinfer11", "copyright")
    if os.path.exists(copyright_):
        shutil.copyfile(copyright_, os.path.join(stage, "TensorRT_Acknowledgements.txt"))
    stem = f"cuda_cudart-linux-x86_64-{CUDART_VERSION}-archive"
    c = fetch(f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/{stem}.tar.xz",
              os.path.join(work, stem + ".tar.xz"), CUDART_SHA256["linux"])
    with tarfile.open(c) as tf:
        members = {os.path.basename(m.name): m for m in tf.getmembers()}
        real = next(m for n, m in members.items() if re.fullmatch(r"libcudart\.so\.\d+\.\d+\.\d+", n))
        with tf.extractfile(real) as src, open(os.path.join(stage, "libcudart.so.13"), "wb") as dst:
            shutil.copyfileobj(src, dst)
        with tf.extractfile(members["LICENSE"]) as src, open(os.path.join(stage, "CUDA_LICENSE.txt"), "wb") as dst:
            shutil.copyfileobj(src, dst)


def pack_of(name):
    m = BUILDER.search(name)
    return f"trt-{m.group(1)}" if m else "trt-runtime"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-version", action="store_true", help="print the TensorRT version and exit")
    ap.add_argument("--platform", choices=["windows", "linux"])
    ap.add_argument("--out")
    ap.add_argument("--work", default="trt-work")
    ap.add_argument("--local-dir")
    ap.add_argument("--sevenzip", default=shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe")
    a = ap.parse_args()
    if a.print_version:
        print(TRT_VERSION)
        return
    if not a.platform or not a.out:
        ap.error("--platform and --out are required")
    plat = f"{a.platform}-x64"
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    stage = os.path.join(a.work, "stage")
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    if a.local_dir:
        for name in os.listdir(a.local_dir):
            keep = (name in WIN_KEEP or BUILDER.search(name) or re.fullmatch(r"cudart64_\d+\.dll", name)
                    or name in ("TensorRT_Acknowledgements.txt", "CUDA_LICENSE.txt", "trtexec")
                    or re.fullmatch(r"lib(nvinfer|nvinfer_plugin|nvonnxparser)\.so\.11|libcudart\.so\.13", name))
            if keep:
                shutil.copyfile(os.path.join(a.local_dir, name), os.path.join(stage, name))
    elif a.platform == "windows":
        stage_windows(a.work, stage)
    else:
        stage_linux(a.work, stage)

    groups = {}
    for name in sorted(os.listdir(stage)):
        groups.setdefault(pack_of(name), []).append(name)
    if "trt-runtime" not in groups or not any(g != "trt-runtime" for g in groups):
        sys.exit(f"incomplete runtime: {groups}")
    index = {"trt_version": TRT_VERSION, "cuda": TRT_CUDA, "platform": plat, "packs": []}
    for name, files in sorted(groups.items()):
        asset = f"{name}-{plat}.7z"
        dest = os.path.abspath(os.path.join(a.out, asset))
        if os.path.exists(dest):
            os.remove(dest)
        log(f"packing {asset}: {', '.join(files)}")
        # Plain LZMA2 without the BCJ/BCJ2 executable filters 7-Zip adds to DLLs by default: what
        # `mangajanai setup` decodes (sevenz-rust2) is LZMA2.
        subprocess.run([a.sevenzip, "a", "-t7z", "-m0=LZMA2", "-mx=7", "-mf=off", "-bso0", "-bsp0", dest] + files,
                       cwd=stage, check=True)
        index["packs"].append({
            "name": name,
            "asset": asset,
            "sha256": sha256(dest),
            "bytes": os.path.getsize(dest),
            "installed_bytes": sum(os.path.getsize(os.path.join(stage, f)) for f in files),
            "files": files,
        })
    with open(os.path.join(a.out, f"packs-{plat}.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    for p in index["packs"]:
        log(f"{p['asset']}: {p['bytes'] / 1e6:.0f} MB ({p['installed_bytes'] / 1e6:.0f} MB unpacked)")


if __name__ == "__main__":
    main()
