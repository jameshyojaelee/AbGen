from __future__ import annotations

import json
from pathlib import Path

from abprop.data.preferences import PreferenceDataset, collate_preference_batch
from abprop.tokenizers import decode


def test_preference_collate_unconditional(tmp_path: Path) -> None:
    records = [
        {"prompt": "", "chosen": "ACD", "rejected": "WQX", "metadata": {"seed_sequence": "SEED"}},
    ]
    jsonl_path = tmp_path / "prefs.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    dataset = PreferenceDataset.from_jsonl(jsonl_path)
    batch = collate_preference_batch(dataset.examples, max_length=8)

    assert batch["chosen_sequences"] == ["ACD"]
    assert batch["rejected_sequences"] == ["WQX"]

    # Ensure tokenization matches the intended sequences (no prompt duplication).
    decoded_chosen = decode(batch["chosen_input_ids"][0].tolist(), strip_special=True)
    decoded_rejected = decode(batch["rejected_input_ids"][0].tolist(), strip_special=True)
    assert decoded_chosen == "ACD"
    assert decoded_rejected == "WQX"
