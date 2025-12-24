#!/usr/bin/env python3
"""Generate antibody sequences via masked-token sampling."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch

from abprop.generation import decode_sequences, random_sequences, sample_mlm_edit
from abprop.models import AbPropModel
from abprop.models.loading import load_model_from_checkpoint
from abprop.tokenizers import TOKEN_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--length", type=int, default=128, help="Length without special tokens.")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--mask-rate", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generation"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--score-liabilities", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=0, help="MC dropout samples for uncertainty.")
    return parser.parse_args()


def load_model(checkpoint: Path, model_config: Path, device: torch.device) -> AbPropModel:
    model, _, _ = load_model_from_checkpoint(checkpoint, model_config, device)
    return model


def add_special_tokens(raw_ids: torch.Tensor) -> torch.Tensor:
    pad_id = TOKEN_TO_ID["<pad>"]
    bos_id = TOKEN_TO_ID["<bos>"]
    eos_id = TOKEN_TO_ID["<eos>"]
    batch_size, length = raw_ids.shape
    input_ids = torch.full((batch_size, length + 2), pad_id, dtype=torch.long, device=raw_ids.device)
    input_ids[:, 0] = bos_id
    input_ids[:, -1] = eos_id
    input_ids[:, 1:-1] = raw_ids
    return input_ids


def score_liabilities(model: AbPropModel, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[Dict[str, float]]:
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, tasks=("reg",))
        preds = outputs["regression"].detach().cpu().numpy()
    keys = model.config.liability_keys
    return [dict(zip(keys, row.tolist())) for row in preds]


def score_uncertainty(
    model: AbPropModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mc_samples: int,
) -> List[Dict[str, float]]:
    if mc_samples <= 0:
        return []
    samples = model.stochastic_forward(
        input_ids,
        attention_mask,
        tasks=("reg",),
        mc_samples=mc_samples,
        enable_dropout=True,
        no_grad=True,
    )
    stacked = torch.stack([out["regression"].detach().cpu() for out in samples], dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0, unbiased=False)
    keys = model.config.liability_keys
    results = []
    for idx in range(mean.size(0)):
        results.append(
            {
                "mean": dict(zip(keys, mean[idx].tolist())),
                "std": dict(zip(keys, std[idx].tolist())),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.model_config, device)

    max_len = int(model.config.max_position_embeddings)
    if args.length + 2 > max_len:
        raise ValueError(
            f"Requested length {args.length} exceeds max_position_embeddings={max_len} (including specials)."
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    raw_ids = random_sequences(args.num_samples, args.length, generator=generator, device=device)
    input_ids = add_special_tokens(raw_ids)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    sampled_ids = sample_mlm_edit(
        model,
        input_ids,
        attention_mask,
        steps=args.steps,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        generator=generator,
    )

    sequences = decode_sequences(sampled_ids)

    liabilities = score_liabilities(model, sampled_ids, attention_mask) if args.score_liabilities else None
    uncertainties = score_uncertainty(model, sampled_ids, attention_mask, args.mc_samples) if args.mc_samples else None

    records: List[Dict[str, object]] = []
    for idx, seq in enumerate(sequences):
        record: Dict[str, object] = {
            "id": f"cand_{idx:04d}",
            "sequence": seq,
            "length": len(seq),
        }
        if liabilities is not None:
            record["liabilities"] = liabilities[idx]
        if uncertainties is not None:
            record["uncertainty"] = uncertainties[idx]
        records.append(record)

    out_path = output_dir / "candidates.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    meta = {
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "num_samples": args.num_samples,
        "length": args.length,
        "steps": args.steps,
        "mask_rate": args.mask_rate,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "seed": args.seed,
        "device": str(device),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} candidates to {out_path}")


if __name__ == "__main__":
    main()
