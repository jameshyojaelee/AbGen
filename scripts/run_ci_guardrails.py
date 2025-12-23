#!/usr/bin/env python3
"""CI-friendly guardrails runner.

This script trains tiny Transformer + Mamba models on the tracked fixture dataset,
builds a small preference dataset, runs a tiny DPO alignment, evaluates each
checkpoint with `scripts/run_benchmarks.py`, and compares the resulting
`regression.json` outputs against committed baselines in `benchmarks/results/`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ci_guardrails"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--dpo-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--write-baselines", action="store_true", help="Overwrite baseline JSONs with this run.")

    parser.add_argument("--train-config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data_ci.yaml"))
    parser.add_argument("--bench-config", type=Path, default=Path("configs/benchmarks_ci.yaml"))
    parser.add_argument("--seeds", type=Path, default=Path("tests/fixtures/toy_sequences.fa"))

    parser.add_argument("--baseline-transformer", type=Path, default=Path("benchmarks/results/baseline_ci_transformer.json"))
    parser.add_argument("--baseline-mamba", type=Path, default=Path("benchmarks/results/baseline_ci_mamba.json"))
    parser.add_argument("--baseline-dpo", type=Path, default=Path("benchmarks/results/baseline_ci_dpo.json"))
    return parser.parse_args(argv)


def _run(cmd: List[str], *, env: Dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_env = dict(os.environ)
    base_env.setdefault("PYTHONPATH", "src")

    # Force CPU determinism for baseline generation and CI parity.
    if args.device == "cpu":
        base_env["CUDA_VISIBLE_DEVICES"] = ""

    transformer_dir = output_dir / "transformer"
    mamba_dir = output_dir / "mamba"
    preferences_path = output_dir / "preferences" / "pairs.jsonl"
    dpo_dir = output_dir / "dpo"
    guardrails_dir = output_dir / "guardrails"
    guardrails_dir.mkdir(parents=True, exist_ok=True)

    def build_train_overrides(run_dir: Path, encoder_type: str) -> str:
        return " ".join(
            [
                "train.max_steps={}".format(args.train_steps),
                "train.eval_interval={}".format(max(1, args.train_steps // 2)),
                "train.checkpoint_interval={}".format(max(1, args.train_steps // 2)),
                "train.batch_size={}".format(args.batch_size),
                "train.num_workers=0",
                "train.warmup_steps=0",
                "train.precision=fp32",
                f"train.log_dir={run_dir}/logs",
                f"train.checkpoint_dir={run_dir}/checkpoints",
                "train.report_interval=1",
                # Reduce noise and runtime
                "model.d_model=64",
                "model.nhead=4",
                "model.num_layers=2",
                "model.dim_feedforward=256",
                "model.dropout=0.0",
                f"model.encoder_type={encoder_type}",
            ]
        )

    # 1) Train tiny Transformer on fixture data
    _run(
        [
            sys.executable,
            "scripts/train.py",
            "--config-path",
            str(args.train_config),
            "--data-config",
            str(args.data_config),
            "--model-config",
            str(args.model_config),
            "--config-overrides",
            build_train_overrides(transformer_dir, "transformer"),
            "--output-dir",
            str(transformer_dir),
        ],
        env=base_env,
    )
    transformer_ckpt = transformer_dir / "checkpoints" / "best.pt"
    if not transformer_ckpt.exists():
        raise FileNotFoundError(f"Expected transformer checkpoint at {transformer_ckpt}")

    # 2) Train tiny Mamba on fixture data
    _run(
        [
            sys.executable,
            "scripts/train.py",
            "--config-path",
            str(args.train_config),
            "--data-config",
            str(args.data_config),
            "--model-config",
            str(args.model_config),
            "--config-overrides",
            build_train_overrides(mamba_dir, "mamba"),
            "--output-dir",
            str(mamba_dir),
        ],
        env=base_env,
    )
    mamba_ckpt = mamba_dir / "checkpoints" / "best.pt"
    if not mamba_ckpt.exists():
        raise FileNotFoundError(f"Expected mamba checkpoint at {mamba_ckpt}")

    # 3) Build preferences on toy seeds (deterministic mutations + reward builder)
    _run(
        [
            sys.executable,
            "scripts/make_preferences.py",
            "--input",
            str(args.seeds),
            "--output",
            str(preferences_path),
            "--seed",
            "42",
            "--device",
            args.device,
        ],
        env=base_env,
    )

    # 4) Tiny DPO alignment (uses preferences file)
    registry_path = output_dir / "registry.json"
    if registry_path.exists():
        registry_path.unlink()
    _run(
        [
            sys.executable,
            "scripts/train_dpo.py",
            "--preferences",
            str(preferences_path),
            "--policy-checkpoint",
            str(transformer_ckpt),
            "--ref-checkpoint",
            str(transformer_ckpt),
            "--model-config",
            str(args.model_config),
            "--output-dir",
            str(dpo_dir),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--max-steps",
            str(args.dpo_steps),
            "--log-interval",
            "10",
            "--registry",
            str(registry_path),
            "--model-id",
            "ci-dpo",
        ],
        env=base_env,
    )
    dpo_ckpt = dpo_dir / "checkpoints" / "dpo_aligned.pt"
    if not dpo_ckpt.exists():
        raise FileNotFoundError(f"Expected DPO checkpoint at {dpo_ckpt}")

    # 5) Benchmarks -> regression.json for each checkpoint
    def run_benchmarks(label: str, checkpoint: Path) -> Path:
        bench_out = output_dir / "benchmarks" / label
        _run(
            [
                sys.executable,
                "scripts/run_benchmarks.py",
                "--checkpoint",
                str(checkpoint),
                "--config",
                str(args.bench_config),
                "--model-config",
                str(args.model_config),
                "--benchmarks",
                "perplexity",
                "cdr_classification",
                "liability",
                "--output-dir",
                str(bench_out),
                "--batch-size",
                str(args.batch_size),
                "--max-samples",
                str(args.max_samples),
                "--device",
                args.device,
                "--no-mlflow",
                "--strict",
            ],
            env=base_env,
        )
        regression_path = bench_out / "regression.json"
        if not regression_path.exists():
            raise FileNotFoundError(f"Expected regression JSON at {regression_path}")
        return regression_path

    transformer_reg = run_benchmarks("transformer", transformer_ckpt)
    mamba_reg = run_benchmarks("mamba", mamba_ckpt)
    dpo_reg = run_benchmarks("dpo", dpo_ckpt)

    transformer_new = guardrails_dir / "transformer.json"
    mamba_new = guardrails_dir / "mamba.json"
    dpo_new = guardrails_dir / "dpo.json"
    _copy(transformer_reg, transformer_new)
    _copy(mamba_reg, mamba_new)
    _copy(dpo_reg, dpo_new)

    if args.write_baselines:
        _copy(transformer_new, args.baseline_transformer)
        _copy(mamba_new, args.baseline_mamba)
        _copy(dpo_new, args.baseline_dpo)
        print("\n✓ Wrote baseline JSONs.")
        return

    # 6) Guardrails: regression checks + smoke checks + tiny generation run
    _run(
        [
            sys.executable,
            "scripts/run_guardrails.py",
            "--transformer-new",
            str(transformer_new),
            "--transformer-ref",
            str(args.baseline_transformer),
            "--mamba-new",
            str(mamba_new),
            "--mamba-ref",
            str(args.baseline_mamba),
            "--dpo-new",
            str(dpo_new),
            "--dpo-ref",
            str(args.baseline_dpo),
            "--generation-checkpoint",
            str(transformer_ckpt),
            "--output-dir",
            str(guardrails_dir),
        ],
        env=base_env,
    )


if __name__ == "__main__":
    main()
