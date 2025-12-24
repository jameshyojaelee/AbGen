from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from abprop.benchmarks.cdr_classification_benchmark import CDRClassificationBenchmark
from abprop.benchmarks.registry import BenchmarkConfig
from abprop.data import build_collate_fn


class _ToyCdrDataset(Dataset):
    def __init__(self) -> None:
        self.samples = [
            {
                "sequence": "ACDE",
                "chain": "H",
                "liability_ln": {},
                "cdr_mask": [1, 0, 1, 0],
            }
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class _PerfectCdrModel(nn.Module):
    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        token_labels: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if token_labels is None:
            raise ValueError("token_labels required for this test model.")
        logits = torch.zeros((*token_labels.shape, 2), device=token_labels.device)
        logits[..., 0] = 1.0
        mask = token_labels == 1
        logits[..., 1][mask] = 2.0
        logits[..., 0][mask] = 0.0
        return {"cls_logits": logits}


def test_cdr_benchmark_position_accuracy_ignores_special_tokens(tmp_path: Path) -> None:
    dataset = _ToyCdrDataset()
    collate = build_collate_fn(generate_mlm=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)

    config = BenchmarkConfig(
        data_path=Path("."),
        batch_size=1,
        output_dir=tmp_path,
        device="cpu",
    )
    benchmark = CDRClassificationBenchmark(config)
    results = benchmark.evaluate(_PerfectCdrModel(), loader)

    position_accuracy = results["position_accuracy"]
    position_total = results["position_total"]

    for acc, total in zip(position_accuracy, position_total):
        if total > 0:
            assert acc == 1.0
