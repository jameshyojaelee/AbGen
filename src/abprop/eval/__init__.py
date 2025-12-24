"""Evaluation helpers for AbProp models."""

from .metrics import classification_summary, compute_perplexity, regression_per_key, regression_summary
from .uncertainty import (
    SampleStatistics,
    TemperatureScaler,
    causal_perplexity_from_logits,
    combine_ensemble,
    expected_calibration_error,
    mean_variance,
    mlm_perplexity_from_logits,
    regression_uncertainty_summary,
    stack_samples,
)
from .pseudolikelihood import build_mlm_mask, mlm_pseudologp, mlm_pseudologp_from_logits
from .stratified import (
    StratifiedEvalConfig,
    StratifiedEvaluationResult,
    StratumMetrics,
    discover_strata,
    evaluate_strata,
)

__all__ = [
    "compute_perplexity",
    "classification_summary",
    "regression_summary",
    "regression_per_key",
    "StratifiedEvalConfig",
    "StratifiedEvaluationResult",
    "StratumMetrics",
    "SampleStatistics",
    "TemperatureScaler",
    "mlm_perplexity_from_logits",
    "causal_perplexity_from_logits",
    "combine_ensemble",
    "expected_calibration_error",
    "mean_variance",
    "regression_uncertainty_summary",
    "stack_samples",
    "discover_strata",
    "evaluate_strata",
    "build_mlm_mask",
    "mlm_pseudologp",
    "mlm_pseudologp_from_logits",
]
