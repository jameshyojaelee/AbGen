from __future__ import annotations

import torch

from abprop.eval.pseudolikelihood import build_mlm_mask
from abprop.train.dpo import DPOLoss


def test_dpo_loss_prefers_higher_policy_advantage() -> None:
    loss_fn = DPOLoss(beta=1.0)

    # Policy prefers chosen much more than reference does -> lower loss expected.
    policy_chosen = torch.tensor([2.0])
    policy_rejected = torch.tensor([0.0])
    ref_chosen = torch.tensor([1.0])
    ref_rejected = torch.tensor([0.0])
    loss_good, _, _ = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected)

    # Weaker preference -> higher loss.
    policy_chosen_weak = torch.tensor([0.2])
    policy_rejected_weak = torch.tensor([0.0])
    loss_bad, _, _ = loss_fn(policy_chosen_weak, policy_rejected_weak, ref_chosen, ref_rejected)

    assert loss_good < loss_bad


def test_build_mlm_mask_deterministic() -> None:
    input_ids = torch.tensor([[1, 5, 6, 7, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    masked_a, labels_a = build_mlm_mask(input_ids, attention_mask, mlm_probability=0.5, mask_seed=123)
    masked_b, labels_b = build_mlm_mask(input_ids, attention_mask, mlm_probability=0.5, mask_seed=123)

    assert torch.equal(masked_a, masked_b)
    assert torch.equal(labels_a, labels_b)
