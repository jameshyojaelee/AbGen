#!/usr/bin/env python3
"""Run regression + smoke guardrails for transformer, Mamba, and DPO models."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transformer-new", type=Path)
    parser.add_argument("--transformer-ref", type=Path)
    parser.add_argument("--mamba-new", type=Path)
    parser.add_argument("--mamba-ref", type=Path)
    parser.add_argument("--dpo-new", type=Path)
    parser.add_argument("--dpo-ref", type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--generation-checkpoint", type=Path)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/guardrails"))
    return parser.parse_args()


def _run(cmd: List[str], env: Optional[dict] = None) -> Tuple[bool, str]:
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    except FileNotFoundError as exc:
        return False, str(exc)
    ok = result.returncode == 0
    output = (result.stdout + "\n" + result.stderr).strip()
    return ok, output


def _compare(label: str, new_path: Optional[Path], ref_path: Optional[Path]) -> Tuple[bool, str]:
    if not new_path and not ref_path:
        return True, f"{label}: SKIP (no paths provided)"
    if not new_path or not ref_path:
        return False, f"{label}: FAIL (both --{label}-new and --{label}-ref required)"
    ok, out = _run([
        sys.executable,
        "scripts/check_regression.py",
        "--new",
        str(new_path),
        "--reference",
        str(ref_path),
    ])
    status = "PASS" if ok else "FAIL"
    return ok, f"{label}: {status}" + ("" if ok else f"\n{out}")


def main() -> None:
    args = parse_args()
    results: List[Tuple[bool, str]] = []

    results.append(_compare("transformer", args.transformer_new, args.transformer_ref))
    results.append(_compare("mamba", args.mamba_new, args.mamba_ref))
    results.append(_compare("dpo", args.dpo_new, args.dpo_ref))

    if not args.skip_smoke:
        env = {**os.environ, "PYTHONPATH": "src"}
        ok, _ = _run([sys.executable, "scripts/dev/verify_backbone.py"], env=env)
        results.append((ok, "smoke/verify_backbone: PASS" if ok else "smoke/verify_backbone: FAIL"))
        ok, _ = _run([sys.executable, "scripts/dev/verify_mamba.py"], env=env)
        results.append((ok, "smoke/verify_mamba: PASS" if ok else "smoke/verify_mamba: FAIL"))

        if args.generation_checkpoint:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            gen_cmd = [
                sys.executable,
                "scripts/generate.py",
                "--checkpoint",
                str(args.generation_checkpoint),
                "--model-config",
                str(args.model_config),
                "--num-samples",
                "2",
                "--length",
                "16",
                "--steps",
                "2",
                "--output-dir",
                str(args.output_dir),
            ]
            ok, _ = _run(gen_cmd, env=env)
            results.append((ok, "smoke/generate: PASS" if ok else "smoke/generate: FAIL"))
        else:
            results.append((False, "smoke/generate: FAIL (missing --generation-checkpoint)"))

    failures = [text for ok, text in results if not ok]
    for _, text in results:
        print(text)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
