"""Design benchmark evaluating generation quality and constraints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import random
import torch
from torch.utils.data import DataLoader, Dataset

from abprop.benchmarks.registry import Benchmark, BenchmarkConfig, BenchmarkResult, register_benchmark
from abprop.generation import decode_sequences, sample_mlm_edit
from abprop.rewards import CANONICAL_AMINO_ACIDS, build_reward
from abprop.tokenizers import batch_encode


@dataclass
class DesignBenchmarkConfig(BenchmarkConfig):
    num_candidates: int = 8
    steps: int = 6
    mask_rate: float = 0.15
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    max_length: int = 128
    liability_threshold: float = 0.2
    max_pairs: int = 2000


class _SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[str]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def _load_sequences(path: Path) -> List[str]:
    sequences: List[str] = []
    buffer: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if buffer:
                    sequences.append("".join(buffer).upper())
                    buffer = []
                continue
            buffer.append(stripped)
    if buffer:
        sequences.append("".join(buffer).upper())
    return sequences


def _mean_pairwise_identity(seqs: Sequence[str], max_pairs: int, seed: int = 0) -> float:
    if len(seqs) < 2:
        return 1.0
    pairs = [(i, j) for i in range(len(seqs)) for j in range(i + 1, len(seqs))]
    if len(pairs) > max_pairs:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, k=max_pairs)
    scores = []
    for i, j in pairs:
        a, b = seqs[i], seqs[j]
        length = min(len(a), len(b))
        if length == 0:
            continue
        matches = sum(1 for k in range(length) if a[k] == b[k])
        scores.append(matches / length)
    return float(sum(scores) / max(1, len(scores)))


@register_benchmark("design")
class DesignBenchmark(Benchmark):
    """Benchmark generation quality with constraint satisfaction and diversity."""

    def __init__(self, config: DesignBenchmarkConfig) -> None:
        super().__init__(config)
        self.config = config

    def load_data(self) -> DataLoader:
        sequences = _load_sequences(Path(self.config.data_path))
        dataset = _SequenceDataset(sequences)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

    def evaluate(self, model: torch.nn.Module, dataloader: DataLoader) -> Dict[str, Any]:
        device = torch.device(self.config.device)
        model.to(device)
        generator = torch.Generator(device=device).manual_seed(123)

        all_candidates: List[str] = []
        reward_deltas: List[float] = []
        valid_flags: List[bool] = []
        liability_flags: List[bool] = []

        for batch in dataloader:
            seeds: List[str] = [seq for seq in batch]
            seed_batch = batch_encode(
                seeds,
                add_special=True,
                max_length=self.config.max_length,
                invalid_policy="replace",
            )
            input_ids = seed_batch["input_ids"].to(device)
            attention_mask = seed_batch["attention_mask"].to(device)

            repeated_ids = input_ids.repeat_interleave(self.config.num_candidates, dim=0)
            repeated_mask = attention_mask.repeat_interleave(self.config.num_candidates, dim=0)
            sampled = sample_mlm_edit(
                model,
                repeated_ids,
                repeated_mask,
                steps=self.config.steps,
                mask_rate=self.config.mask_rate,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                generator=generator,
            )
            candidates = decode_sequences(sampled)
            all_candidates.extend(candidates)

            # Reward deltas vs seed
            for idx, seed in enumerate(seeds):
                start = idx * self.config.num_candidates
                end = start + self.config.num_candidates
                candidate_chunk = candidates[start:end]
                seed_reward = build_reward(seed).total_reward
                chunk_rewards = [build_reward(c).total_reward for c in candidate_chunk]
                reward_deltas.append(max(chunk_rewards) - seed_reward)

            # Constraint checks
            for seq in candidates:
                valid = all(res in CANONICAL_AMINO_ACIDS for res in seq)
                valid_flags.append(valid)
                motif_penalty = build_reward(seq).motif_penalty
                liability_flags.append(motif_penalty <= self.config.liability_threshold)

        unique_fraction = len(set(all_candidates)) / max(1, len(all_candidates))
        mean_identity = _mean_pairwise_identity(all_candidates, self.config.max_pairs)

        results = {
            "num_candidates": len(all_candidates),
            "valid_fraction": float(sum(valid_flags) / max(1, len(valid_flags))),
            "liability_pass_fraction": float(sum(liability_flags) / max(1, len(liability_flags))),
            "unique_fraction": float(unique_fraction),
            "mean_pairwise_identity": float(mean_identity),
            "mean_reward_delta": float(sum(reward_deltas) / max(1, len(reward_deltas))),
            "reward_deltas": reward_deltas,
        }
        return results

    def report(self, results: Dict[str, Any]) -> BenchmarkResult:
        output_dir = self.config.output_dir / self.name
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "summary.json"
        with summary_path.open("w", encoding="utf-8") as handle:
            json_payload = {k: v for k, v in results.items() if k != "reward_deltas"}
            import json
            json.dump(json_payload, handle, indent=2)

        plot_path = output_dir / "reward_delta_hist.png"
        try:
            import matplotlib.pyplot as plt

            plt.hist(results.get("reward_deltas", []), bins=20, color="#4C72B0")
            plt.title("Reward Delta (Best Candidate - Seed)")
            plt.xlabel("Reward Delta")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        except ImportError:
            plot_path = Path("")

        metrics = {
            "valid_fraction": float(results["valid_fraction"]),
            "liability_pass_fraction": float(results["liability_pass_fraction"]),
            "unique_fraction": float(results["unique_fraction"]),
            "mean_pairwise_identity": float(results["mean_pairwise_identity"]),
            "mean_reward_delta": float(results["mean_reward_delta"]),
        }
        plots = {"reward_delta_hist": plot_path} if plot_path else {}
        return BenchmarkResult(
            benchmark_name=self.name,
            metrics=metrics,
            plots=plots,
            metadata={"num_candidates": results["num_candidates"]},
        )


__all__ = ["DesignBenchmark", "DesignBenchmarkConfig"]
