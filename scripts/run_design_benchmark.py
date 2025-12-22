#!/usr/bin/env python3
"""Run the design benchmark (generation + constraint scoring)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from abprop.benchmarks.design_benchmark import DesignBenchmark, DesignBenchmarkConfig
from abprop.models import AbPropModel, TransformerConfig
from abprop.utils import extract_model_config, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--seeds", type=Path, required=True, help="FASTA or newline seed sequences.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/benchmarks"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--liability-threshold", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model(checkpoint: Path, model_config: Path, device: torch.device) -> AbPropModel:
    cfg = load_yaml_config(model_config)
    if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
        cfg = cfg["model"]
    state = torch.load(checkpoint, map_location="cpu")
    checkpoint_cfg = extract_model_config(state)
    if checkpoint_cfg:
        cfg = {**cfg, **checkpoint_cfg}
    config = TransformerConfig(**cfg) if cfg else TransformerConfig()
    model = AbPropModel(config).to(device)
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args.checkpoint, args.model_config, device)

    config = DesignBenchmarkConfig(
        data_path=args.seeds,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        device=str(device),
        num_candidates=args.num_candidates,
        steps=args.steps,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_length=args.max_length,
        liability_threshold=args.liability_threshold,
    )

    benchmark = DesignBenchmark(config)
    result = benchmark.run(model)

    summary_path = args.output_dir / "design" / "summary.json"
    if summary_path.exists():
        print(f"Summary written to {summary_path}")
    print(f"Design benchmark metrics: {result.metrics}")


if __name__ == "__main__":
    main()
