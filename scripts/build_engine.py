"""
build_engine.py
---------------
Compiles the C++ groundwater simulation engine into a Python-importable
shared library using pybind11.

Works on Linux, macOS, and Windows (requires a C++17-capable compiler).

Usage:
    python scripts/build_engine.py

Requirements:
    - g++ / clang++ / MSVC (C++17)
    - pybind11  (pip install pybind11)
    - numpy     (pip install numpy)
"""
import sys
import os
import subprocess
import sysconfig
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
CPP_SRC = ROOT / "src" / "cpp" / "groundwater_engine.cpp"
OUT_DIR = ROOT / "src" / "cpp"


def get_extension_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def get_include_dirs() -> list[str]:
    import pybind11
    import numpy
    return [
        sysconfig.get_path("include"),
        pybind11.get_include(),
        numpy.get_include(),
    ]


def build_unix(includes: list[str], out_path: Path) -> int:
    cmd = [
        "g++", "-O3", "-march=native",
        "-shared", "-fPIC", "-std=c++17",
        *[f"-I{d}" for d in includes],
        str(CPP_SRC),
        "-o", str(out_path),
    ]
    print("Building (Unix):", " ".join(cmd))
    return subprocess.call(cmd)


def build_windows(includes: list[str], out_path: Path) -> int:
    py_lib = Path(sys.prefix) / "libs"
    cmd = [
        "cl.exe", "/O2", "/std:c++17",
        "/LD",
        *[f"/I{d}" for d in includes],
        str(CPP_SRC),
        f"/Fe:{out_path}",
        f"/LIBPATH:{py_lib}",
    ]
    print("Building (Windows):", " ".join(cmd))
    return subprocess.call(cmd)


def main():
    suffix   = get_extension_suffix()
    out_path = OUT_DIR / f"groundwater_engine{suffix}"
    includes = get_include_dirs()

    print(f"Source : {CPP_SRC}")
    print(f"Output : {out_path}")
    print(f"Suffix : {suffix}")

    if sys.platform == "win32":
        ret = build_windows(includes, out_path)
    else:
        ret = build_unix(includes, out_path)

    if ret == 0:
        print(f"\n✅  Build successful → {out_path}")
        # Quick import test
        sys.path.insert(0, str(OUT_DIR))
        import groundwater_engine as gwe
        dt = gwe.compute_stable_dt(K=5.0, S=0.001, b=50.0, dx=10000.0)
        print(f"✅  Import OK — stability dt = {dt:.1f} days")
    else:
        print(f"\n❌  Build failed (exit code {ret})")
        sys.exit(ret)


if __name__ == "__main__":
    main()
