#!/usr/bin/env python3
"""Build a preference dataset from sequences by scoring candidate edits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

from abprop.models import AbPropModel, TransformerConfig
from abprop.rewards import build_reward
from abprop.utils import extract_model_config, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="FASTA or newline-delimited sequences.")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--candidates", type=int, default=8, help="Number of candidate edits per sequence.")
    parser.add_argument("--edits", type=int, default=2, help="Number of mutations per candidate.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--use-humanness-proxy", action="store_true")
    return parser.parse_args()


def load_sequences(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return []
    if any(line.startswith(">") for line in lines):
        sequences: List[str] = []
        buffer: List[str] = []
        for line in lines:
            if line.startswith(">"):
                if buffer:
                    sequences.append("".join(buffer).upper())
                    buffer = []
                continue
            buffer.append(line)
        if buffer:
            sequences.append("".join(buffer).upper())
        return sequences
    return [line.upper() for line in lines]


def mutate_sequence(seq: str, rng: random.Random, edits: int) -> str:
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    seq_list = list(seq)
    if not seq_list:
        return seq
    positions = rng.sample(range(len(seq_list)), k=min(edits, len(seq_list)))
    for pos in positions:
        original = seq_list[pos]
        choices = [aa for aa in alphabet if aa != original]
        seq_list[pos] = rng.choice(choices)
    return "".join(seq_list)


def load_model(checkpoint: Path, model_config: Path, device: torch.device) -> AbPropModel:
    cfg = load_yaml_config(model_config)
    if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
        cfg = cfg["model"]

    state = torch.load(checkpoint, map_location="cpu")
    checkpoint_cfg = extract_model_config(state)
    if checkpoint_cfg:
        cfg = {**cfg, **checkpoint_cfg}

    if cfg:
        allowed = set(TransformerConfig.__dataclass_fields__.keys())
        cfg = {key: value for key, value in cfg.items() if key in allowed}
    config = TransformerConfig(**cfg) if cfg else TransformerConfig()
    model = AbPropModel(config).to(device)
    model_state = state.get("model_state", state)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    sequences = load_sequences(args.input)
    if not sequences:
        raise ValueError("No sequences found in input file.")

    device = torch.device(args.device)
    model = None
    if args.checkpoint:
        model = load_model(args.checkpoint, args.model_config, device)

    records: List[Dict[str, object]] = []
    for idx, seed_seq in enumerate(sequences):
        candidates = [mutate_sequence(seed_seq, rng, args.edits) for _ in range(args.candidates)]
        rewards = [
            build_reward(
                cand,
                model=model,
                device=device,
                mc_samples=args.mc_samples,
                use_humanness_proxy=args.use_humanness_proxy,
            )
            for cand in candidates
        ]

        best_idx = max(range(len(rewards)), key=lambda i: rewards[i].total_reward)
        worst_idx = min(range(len(rewards)), key=lambda i: rewards[i].total_reward)

        chosen = candidates[best_idx]
        rejected = candidates[worst_idx]

        record = {
            "prompt": seed_seq,
            "chosen": chosen,
            "rejected": rejected,
            "metadata": {
                "seed_index": idx,
                "reward_chosen": rewards[best_idx].to_dict(),
                "reward_rejected": rewards[worst_idx].to_dict(),
            },
        }
        records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
