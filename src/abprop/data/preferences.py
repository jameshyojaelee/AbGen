"""Preference dataset utilities for DPO training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from abprop.tokenizers import batch_encode


@dataclass
class PreferenceExample:
    prompt: str
    chosen: str
    rejected: str
    metadata: Dict[str, Any]


class PreferenceDataset(Dataset):
    def __init__(self, examples: Sequence[PreferenceExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> PreferenceExample:
        return self.examples[idx]

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "PreferenceDataset":
        examples = list(load_preference_jsonl(path))
        return cls(examples)


def load_preference_jsonl(path: Path | str) -> Iterable[PreferenceExample]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            data = json.loads(stripped)
            if not isinstance(data, dict):
                raise ValueError(f"Invalid JSONL entry at line {line_num}: {data}")
            prompt = str(data.get("prompt", ""))
            chosen = data.get("chosen")
            rejected = data.get("rejected")
            if chosen is None or rejected is None:
                raise ValueError(f"Missing chosen/rejected at line {line_num}.")
            metadata = data.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be a dict at line {line_num}.")
            yield PreferenceExample(prompt=prompt, chosen=str(chosen), rejected=str(rejected), metadata=metadata)


def collate_preference_batch(
    examples: Sequence[PreferenceExample],
    *,
    max_length: int,
    add_special: bool = True,
    invalid_policy: str = "replace",
) -> Dict[str, torch.Tensor | List[str] | List[Dict[str, Any]]]:
    chosen_sequences = [ex.prompt + ex.chosen for ex in examples]
    rejected_sequences = [ex.prompt + ex.rejected for ex in examples]

    chosen_batch = batch_encode(
        chosen_sequences,
        add_special=add_special,
        max_length=max_length,
        invalid_policy=invalid_policy,
    )
    rejected_batch = batch_encode(
        rejected_sequences,
        add_special=add_special,
        max_length=max_length,
        invalid_policy=invalid_policy,
    )

    return {
        "chosen_input_ids": chosen_batch["input_ids"],
        "chosen_attention_mask": chosen_batch["attention_mask"],
        "rejected_input_ids": rejected_batch["input_ids"],
        "rejected_attention_mask": rejected_batch["attention_mask"],
        "chosen_sequences": chosen_sequences,
        "rejected_sequences": rejected_sequences,
        "metadata": [ex.metadata for ex in examples],
    }


__all__ = [
    "PreferenceExample",
    "PreferenceDataset",
    "load_preference_jsonl",
    "collate_preference_batch",
]
