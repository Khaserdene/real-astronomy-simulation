"""Download the h5py wheel(s) the extension bundles, matching Blender's Python.

The extension reads ``.h5`` snapshots directly, which needs ``h5py`` inside
Blender's bundled Python.  Blender already ships ``numpy``; only ``h5py`` (and its
HDF5 binary, which the wheel includes) is missing.

Blender's Python is a specific CPython version (e.g. Blender 4.2 -> CPython 3.11).
Pass that version so pip fetches a compatible wheel:

    python fetch_wheels.py --pyver 311 --platform win_amd64

Then uncomment / edit the ``wheels = [...]`` line in ``blender_manifest.toml`` to
list the downloaded file(s), and build the extension (see README).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "wheels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyver", default="311",
                    help="Blender's CPython version tag, e.g. 311 (Blender 4.2)")
    ap.add_argument("--platform", default="win_amd64",
                    help="wheel platform tag, e.g. win_amd64 / macosx_11_0_arm64 "
                         "/ manylinux2014_x86_64")
    ap.add_argument("--package", default="h5py", help="package to fetch")
    args = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)
    cmd = [sys.executable, "-m", "pip", "download", args.package,
           "--only-binary=:all:", "--no-deps",
           "--python-version", args.pyver,
           "--platform", args.platform,
           "--implementation", "cp",
           "--dest", DEST]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    wheels = [f for f in os.listdir(DEST) if f.endswith(".whl")]
    print(f"\nDownloaded into {DEST}:")
    for w in wheels:
        print(f"  ./wheels/{w}")
    print("\nNow set in blender_manifest.toml:")
    print("  wheels = [" + ", ".join(f'"./wheels/{w}"' for w in wheels) + "]")


if __name__ == "__main__":
    main()
