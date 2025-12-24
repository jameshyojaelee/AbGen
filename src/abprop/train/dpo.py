"""
Direct Preference Optimization (DPO) module for AbProp.

This implements the DPO loss function from 'Direct Preference Optimization: Your Language Model is Secretly a Reward Model' (Rafailov et al., 2023).
It allows aligning the generative antibody model towards desirable properties (e.g., predicted low immunogenicity, high binding affinity)
using preference pairs rather than reinforcement learning (PPO).

Novelty: Enables "Generative Design" by aligning the language model probability mass directly with biophysical constraints.
"""

import torch
import torch.nn.functional as F
from torch import nn

class DPOLoss(nn.Module):
    """
    DPO Loss for antibody sequence alignment.
    
    Usage:
        loss_fn = DPOLoss(beta=0.1)
        loss, rewards_chosen, rewards_rejected = loss_fn(
            policy_chosen_logps, 
            policy_rejected_logps, 
            ref_chosen_logps, 
            ref_rejected_logps
        )
    """
    
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of the 'chosen' (better) sequence under policy model. [Batch]
            policy_rejected_logps: Log probs of the 'rejected' (worse) sequence under policy model. [Batch]
            ref_chosen_logps: Log probs of the chosen sequence under reference model. [Batch]
            ref_rejected_logps: Log probs of the rejected sequence under reference model. [Batch]
            
        Returns:
            loss: The DPO loss scalar.
            chosen_rewards: Estimated implicit reward for chosen samples.
            rejected_rewards: Estimated implicit reward for rejected samples.
        """
        
        # pi_logratios = log(pi(y_w) / pi(y_l)) = log pi(y_w) - log pi(y_l)
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        
        # ref_logratios = log(ref(y_w) / ref(y_l))
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        
        # logits = beta * (pi_logratios - ref_logratios)
        logits = self.beta * (policy_logratios - ref_logratios)
        
        # DPO Loss = -log sigmoid(logits)
        # Using softplus for stability: -log(sigmoid(x)) = log(1 + exp(-x)) = softplus(-x)
        if self.label_smoothing > 0:
            target = torch.full_like(logits, 1.0 - float(self.label_smoothing))
            losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        else:
            losses = F.softplus(-logits)

        reward_chosen = self.beta * (policy_chosen_logps - ref_chosen_logps).detach()
        reward_rejected = self.beta * (policy_rejected_logps - ref_rejected_logps).detach()
        return losses.mean(), reward_chosen, reward_rejected

def get_batch_logps(
    logits: torch.Tensor, 
    labels: torch.Tensor, 
    average_log_prob: bool = False
) -> torch.Tensor:
    """
    Utility to extract log probabilities of the labels from logits.
    
    Args:
        logits: [Batch, SeqLen, Vocab]
        labels: [Batch, SeqLen]
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits and labels shape mismatch.")

    labels = labels.clone()
    loss_mask = labels != -100

    # dummy token for gathering
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        denom = loss_mask.sum(-1).clamp_min(1)
        return (per_token_logps * loss_mask).sum(-1) / denom
    return (per_token_logps * loss_mask).sum(-1)
