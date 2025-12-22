#!/usr/bin/env python3
"""Compatibility wrapper for moved smoke test."""

from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(Path(__file__).resolve().parent / "dev" / "verify_benchmarks.py", run_name="__main__")
