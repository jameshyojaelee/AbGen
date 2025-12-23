#!/usr/bin/env python3
"""Legacy wrapper for scripts/dev/bench_mamba.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).parent / "dev" / "bench_mamba.py"), run_name="__main__")
