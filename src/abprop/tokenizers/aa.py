"""Amino acid tokenizer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<mask>"]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]

PAD_TOKEN_ID = 0
_SPECIAL_OFFSET = len(SPECIAL_TOKENS)

VOCAB: List[str] = SPECIAL_TOKENS + AMINO_ACIDS
TOKEN_TO_ID: Dict[str, int] = {token: idx for idx, token in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {idx: token for token, idx in TOKEN_TO_ID.items()}


def encode(sequence: str, add_special: bool = True) -> torch.Tensor:
    """Encode a sequence into token ids."""
    sequence = (sequence or "").upper()
    ids: List[int] = []
    if add_special:
        ids.append(TOKEN_TO_ID["<bos>"])
    for residue in sequence:
        ids.append(TOKEN_TO_ID.get(residue, TOKEN_TO_ID["X"]))
    if add_special:
        ids.append(TOKEN_TO_ID["<eos>"])
    return torch.tensor(ids, dtype=torch.long)


def decode(ids: Sequence[int], strip_special: bool = True) -> str:
    """Decode token ids back to a sequence."""
    tokens = [ID_TO_TOKEN.get(int(idx), "X") for idx in ids]
    if strip_special:
        tokens = [tok for tok in tokens if tok not in {"<bos>", "<eos>", "<pad>"}]
    return "".join(tokens)


def collate_batch(
    sequences: Sequence[str],
    add_special: bool = True,
) -> Dict[str, torch.Tensor]:
    """Convert a batch of sequences into padded tensors."""
    if not sequences:
        raise ValueError("collate_batch expects at least one sequence.")

    encoded = [encode(seq, add_special=add_special) for seq in sequences]
    max_len = max(item.size(0) for item in encoded)
    batch_size = len(encoded)

    input_ids = torch.full((batch_size, max_len), PAD_TOKEN_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for idx, tensor in enumerate(encoded):
        length = tensor.size(0)
        input_ids[idx, :length] = tensor
        attention_mask[idx, :length] = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def batch_encode(
    sequences: Sequence[str],
    *,
    add_special: bool = True,
    max_length: int | None = None,
    invalid_policy: str = "replace",
) -> Dict[str, torch.Tensor]:
    """Encode a batch of sequences with validation for inference-time usage."""
    if not sequences:
        raise ValueError("batch_encode expects at least one sequence.")

    cleaned: List[str] = []
    valid_tokens = set(AMINO_ACIDS)
    for seq in sequences:
        seq = (seq or "").strip().upper()
        if not seq:
            raise ValueError("Empty sequence encountered; provide non-empty sequences.")
        if invalid_policy not in {"replace", "error"}:
            raise ValueError(f"Invalid policy '{invalid_policy}' (expected 'replace' or 'error').")

        if invalid_policy == "error":
            invalid = [res for res in seq if res not in valid_tokens]
            if invalid:
                raise ValueError(f"Invalid residue(s) encountered: {sorted(set(invalid))}.")
            cleaned.append(seq)
        else:
            cleaned.append("".join(res if res in valid_tokens else "X" for res in seq))

        if max_length is not None:
            token_length = len(cleaned[-1]) + (2 if add_special else 0)
            if token_length > max_length:
                raise ValueError(
                    f"Sequence length {token_length} exceeds max_length={max_length}."
                )

    batch = collate_batch(cleaned, add_special=add_special)
    batch["sequences"] = cleaned
    return batch


def apply_mlm_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    mlm_probability: float = 0.15,
    rng: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply MLM-style masking to input ids and return masked inputs + labels."""
    if mlm_probability <= 0:
        labels = input_ids.clone()
        labels[:] = -100
        return input_ids, labels

    pad_id = TOKEN_TO_ID["<pad>"]
    bos_id = TOKEN_TO_ID["<bos>"]
    eos_id = TOKEN_TO_ID["<eos>"]
    mask_id = TOKEN_TO_ID["<mask>"]
    vocab_size = len(VOCAB)

    labels = input_ids.clone()
    special_mask = (input_ids == pad_id) | (input_ids == bos_id) | (input_ids == eos_id)
    candidate_mask = attention_mask.bool() & ~special_mask

    rand = torch.rand(input_ids.shape, device=input_ids.device, generator=rng)
    masked_indices = rand < mlm_probability
    masked_indices &= candidate_mask
    labels[~masked_indices] = -100

    # 80% replace with <mask>
    replace_rand = torch.rand(input_ids.shape, device=input_ids.device, generator=rng)
    indices_replaced = replace_rand < 0.8
    indices_replaced &= masked_indices
    masked_input_ids = input_ids.clone()
    masked_input_ids[indices_replaced] = mask_id

    # 10% replace with random token
    random_rand = torch.rand(input_ids.shape, device=input_ids.device, generator=rng)
    indices_random = random_rand < 0.5
    indices_random &= masked_indices & ~indices_replaced
    random_tokens = torch.randint(vocab_size, input_ids.shape, device=input_ids.device, dtype=torch.long)
    masked_input_ids[indices_random] = random_tokens[indices_random]

    return masked_input_ids, labels


__all__ = [
    "SPECIAL_TOKENS",
    "AMINO_ACIDS",
    "VOCAB",
    "TOKEN_TO_ID",
    "ID_TO_TOKEN",
    "PAD_TOKEN_ID",
    "encode",
    "decode",
    "collate_batch",
    "batch_encode",
    "apply_mlm_mask",
]
