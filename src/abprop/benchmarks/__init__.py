"""Benchmark suite for comprehensive evaluation of AbProp models.

This module provides a modular benchmark infrastructure for evaluating antibody
property prediction models across multiple dimensions including:
- Perplexity on natural sequences
- CDR3 span classification (substring-derived labels)
- Liability prediction
- Developability assessment
- Zero-shot generalization

Each benchmark implements a standard interface with load_data(), evaluate(),
and report() methods.

Usage:
    from abprop.benchmarks import get_registry
    from abprop.benchmarks.registry import BenchmarkConfig

    # Get the global registry
    registry = get_registry()

    # Create a benchmark
    config = BenchmarkConfig(data_path="data/processed/oas")
    benchmark = registry.create("perplexity", config)

    # Run the benchmark
    result = benchmark.run(model)
"""

from __future__ import annotations

from .registry import Benchmark, BenchmarkConfig, BenchmarkRegistry, BenchmarkResult, get_registry

_MODULE_ALIASES = {
    "cdr_classification_benchmark": "abprop.benchmarks.cdr_classification_benchmark",
    "design_benchmark": "abprop.benchmarks.design_benchmark",
    "developability_benchmark": "abprop.benchmarks.developability_benchmark",
    "liability_benchmark": "abprop.benchmarks.liability_benchmark",
    "perplexity_benchmark": "abprop.benchmarks.perplexity_benchmark",
    "stratified_benchmark": "abprop.benchmarks.stratified_benchmark",
    "zero_shot_benchmark": "abprop.benchmarks.zero_shot_benchmark",
}


def __getattr__(name: str):  # pragma: no cover - import-time dispatch
    module = _MODULE_ALIASES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return __import__(module, fromlist=[name])

__all__ = [
    "Benchmark",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkRegistry",
    "get_registry",
    "cdr_classification_benchmark",
    "design_benchmark",
    "developability_benchmark",
    "liability_benchmark",
    "perplexity_benchmark",
    "stratified_benchmark",
    "zero_shot_benchmark",
]
