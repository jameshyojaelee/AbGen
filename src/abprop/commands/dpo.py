"""DPO training command entrypoint."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader

from abprop.data.preferences import PreferenceDataset, PreferenceExample, collate_preference_batch
from abprop.models import AbPropModel
from abprop.models.loading import load_model_from_checkpoint
from abprop.registry import ModelRegistry
from abprop.tokenizers import TOKEN_TO_ID
from abprop.eval.pseudolikelihood import build_mlm_mask, mlm_pseudologp
from abprop.train.dpo import DPOLoss
from abprop.utils import DEFAULT_OUTPUT_DIR, seed_all


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a policy model with DPO.")
    parser.add_argument("--preferences", type=Path, help="JSONL preference dataset.")
    parser.add_argument("--synthetic", action="store_true", help="Use a synthetic preference dataset.")
    parser.add_argument("--synthetic-samples", type=int, default=64)
    parser.add_argument("--synthetic-length", type=int, default=48)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--ref-checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "dpo")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--registry", type=Path, default=Path("models/registry.json"))
    parser.add_argument("--model-id", type=str, default=None)
    return parser


def _build_synthetic_preferences(num_samples: int, length: int, seed: int) -> PreferenceDataset:
    rng = random.Random(seed)
    examples: List[PreferenceExample] = []
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    for _ in range(num_samples):
        chosen = "".join(rng.choice(alphabet) for _ in range(length))
        rejected = "".join(rng.choice(alphabet) for _ in range(length))
        examples.append(PreferenceExample(prompt="", chosen=chosen, rejected=rejected, metadata={}))
    return PreferenceDataset(examples)


def _load_model(
    checkpoint: Path,
    model_config: Path,
    device: torch.device,
) -> AbPropModel:
    model, _, _ = load_model_from_checkpoint(checkpoint, model_config, device)
    return model


def _ensure_masked_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    masked_input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    special_ids = {
        TOKEN_TO_ID["<pad>"],
        TOKEN_TO_ID["<bos>"],
        TOKEN_TO_ID["<eos>"],
        TOKEN_TO_ID["<mask>"],
    }
    for row in range(labels.size(0)):
        if (labels[row] != -100).any():
            continue
        for idx in range(input_ids.size(1)):
            if not attention_mask[row, idx]:
                continue
            token_id = int(input_ids[row, idx])
            if token_id in special_ids:
                continue
            labels[row, idx] = token_id
            masked_input_ids[row, idx] = TOKEN_TO_ID["<mask>"]
            break


def _compute_logps(
    model: AbPropModel,
    masked_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return mlm_pseudologp(
        model,
        masked_input_ids,
        attention_mask,
        labels,
        average_log_prob=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    seed_all(args.seed)
    device = torch.device(args.device)

    if args.synthetic or args.preferences is None:
        dataset = _build_synthetic_preferences(args.synthetic_samples, args.synthetic_length, args.seed)
    else:
        dataset = PreferenceDataset.from_jsonl(args.preferences)

    policy = _load_model(args.policy_checkpoint, args.model_config, device)
    reference = _load_model(args.ref_checkpoint, args.model_config, device)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    max_length = int(policy.config.max_position_embeddings)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_preference_batch(batch, max_length=max_length),
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    loss_fn = DPOLoss(beta=args.beta, label_smoothing=args.label_smoothing)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_log: List[Dict[str, float]] = []

    policy.train()
    step = 0
    for batch in loader:
        if args.max_steps and step >= args.max_steps:
            break

        chosen_ids = batch["chosen_input_ids"].to(device)
        chosen_mask = batch["chosen_attention_mask"].to(device)
        rejected_ids = batch["rejected_input_ids"].to(device)
        rejected_mask = batch["rejected_attention_mask"].to(device)

        mask_seed = args.seed + step * 2
        masked_chosen, chosen_labels = build_mlm_mask(
            chosen_ids,
            chosen_mask,
            mlm_probability=args.mlm_probability,
            mask_seed=mask_seed,
        )
        masked_rejected, rejected_labels = build_mlm_mask(
            rejected_ids,
            rejected_mask,
            mlm_probability=args.mlm_probability,
            mask_seed=mask_seed + 1,
        )
        _ensure_masked_positions(chosen_ids, chosen_mask, masked_chosen, chosen_labels)
        _ensure_masked_positions(rejected_ids, rejected_mask, masked_rejected, rejected_labels)

        policy_chosen_logps = _compute_logps(policy, masked_chosen, chosen_mask, chosen_labels)
        policy_rejected_logps = _compute_logps(policy, masked_rejected, rejected_mask, rejected_labels)
        with torch.no_grad():
            ref_chosen_logps = _compute_logps(reference, masked_chosen, chosen_mask, chosen_labels)
            ref_rejected_logps = _compute_logps(reference, masked_rejected, rejected_mask, rejected_labels)

        loss, reward_chosen, reward_rejected = loss_fn(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % args.log_interval == 0:
            avg_chosen = float(reward_chosen.mean().item())
            avg_rejected = float(reward_rejected.mean().item())
            print(
                f"step={step} loss={loss.item():.4f} "
                f"reward_chosen={avg_chosen:.4f} reward_rejected={avg_rejected:.4f}"
            )
        metrics_log.append(
            {
                "step": float(step),
                "loss": float(loss.item()),
                "reward_chosen": float(reward_chosen.mean().item()),
                "reward_rejected": float(reward_rejected.mean().item()),
            }
        )
        step += 1

    checkpoint_path = ckpt_dir / "dpo_aligned.pt"
    torch.save(
        {
            "model_state": policy.state_dict(),
            "model_config": asdict(policy.config),
            "dpo_config": {
                "beta": args.beta,
                "label_smoothing": args.label_smoothing,
                "mlm_probability": args.mlm_probability,
            },
        },
        checkpoint_path,
    )

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_log, indent=2), encoding="utf-8")

    registry = ModelRegistry(args.registry)
    model_id = args.model_id or f"dpo-{checkpoint_path.stem}"
    record = registry.register(
        model_id=model_id,
        checkpoint=checkpoint_path,
        config=asdict(policy.config),
        metrics={"dpo_loss": float(metrics_log[-1]["loss"]) if metrics_log else float("nan")},
        tags=["dpo"],
    )
    card_path = output_dir / f"model_card_{model_id}.md"
    registry.export_card(model_id, card_path)

    print(f"Saved DPO checkpoint to {checkpoint_path}")
    print(f"Registered model {record.model_id} in {args.registry}")
    print(f"Wrote model card to {card_path}")


if __name__ == "__main__":
    main()
