#!/usr/bin/env python3
"""Benchmark transformer vs Mamba encoder runtime by sequence length."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch

from abprop.models import AbPropModel, TransformerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lengths", nargs="*", type=int, default=[32, 64, 128, 256, 512])
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def _benchmark(model: AbPropModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int, warmup: int) -> float:
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids, attention_mask, tasks=("mlm",))
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(steps):
            _ = model(input_ids, attention_mask, tasks=("mlm",))
        if device.type == "cuda":
            torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / max(1, steps)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    results: List[Dict[str, float]] = []

    for seq_len in args.lengths:
        input_ids = torch.randint(0, 25, (args.batch_size, seq_len), device=device)
        attention_mask = torch.ones((args.batch_size, seq_len), device=device)

        mamba_cfg = TransformerConfig(
            encoder_type="mamba",
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=0.0,
        )
        transformer_cfg = TransformerConfig(
            encoder_type="transformer",
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dropout=0.0,
        )

        mamba_model = AbPropModel(mamba_cfg).to(device)
        transformer_model = AbPropModel(transformer_cfg).to(device)

        mamba_time = _benchmark(mamba_model, input_ids, attention_mask, args.steps, args.warmup)
        transformer_time = _benchmark(transformer_model, input_ids, attention_mask, args.steps, args.warmup)

        results.append(
            {
                "seq_len": float(seq_len),
                "mamba_seconds": float(mamba_time),
                "transformer_seconds": float(transformer_time),
                "speedup": float(transformer_time / mamba_time) if mamba_time > 0 else float("inf"),
            }
        )

    header = f"{'length':>8} | {'mamba (ms)':>12} | {'transformer (ms)':>17} | {'xform/mamba':>12}"
    print(header)
    print("-" * len(header))
    for row in results:
        print(
            f"{int(row['seq_len']):>8} |"
            f" {row['mamba_seconds'] * 1000:>12.3f} |"
            f" {row['transformer_seconds'] * 1000:>17.3f} |"
            f" {row['speedup']:>12.3f}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump({"device": str(device), "results": results}, handle, indent=2)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
