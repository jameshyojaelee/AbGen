"""FastAPI server for AbProp inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
try:  # Optional dependency for serving.
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - exercised in non-serve installs
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]

from abprop.eval.perplexity import mlm_perplexity_from_logits
from abprop.models import AbPropModel, TransformerConfig
from abprop.models.loading import load_model_from_checkpoint
from abprop.tokenizers import apply_mlm_mask, batch_encode
from abprop.utils import load_yaml_config


class ModelWrapper:
    def __init__(
        self,
        checkpoint: Union[Path, List[Path]],
        model_config: Path,
        device: Optional[str] = None,
        ensemble_mode: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.ensemble_mode = ensemble_mode

        cfg = load_yaml_config(model_config)
        if isinstance(cfg, dict) and isinstance(cfg.get("model"), dict):
            cfg = cfg["model"]

        checkpoint_paths = checkpoint if isinstance(checkpoint, list) else [checkpoint]
        first_model, model_cfg, _ = load_model_from_checkpoint(
            checkpoint_paths[0],
            model_config,
            self.device,
        )
        self.model_cfg = model_cfg
        self.mlm_probability = float(cfg.get("mlm_probability", 0.15) if isinstance(cfg, dict) else 0.15)

        # Load single or multiple models
        if isinstance(checkpoint, list):
            self.models = []
            for idx, ckpt in enumerate(checkpoint_paths):
                if idx == 0:
                    model = first_model
                else:
                    model, _, _ = load_model_from_checkpoint(ckpt, model_config, self.device)
                self.models.append(model)
            self.ensemble_mode = True
        else:
            self.model = first_model
            self.models = [self.model]

        self.max_length = int(self.model_cfg.max_position_embeddings)

    def score_perplexity(
        self, sequences: List[str], return_std: bool = False
    ) -> Union[List[float], Dict[str, List[float]]]:
        """Score perplexity with optional ensemble std deviation."""
        batch = batch_encode(
            sequences,
            add_special=True,
            max_length=self.max_length,
            invalid_policy="replace",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        masked_input_ids, labels = apply_mlm_mask(
            input_ids,
            attention_mask,
            mlm_probability=self.mlm_probability,
        )

        all_perplexities = []

        for model in self.models:
            with torch.no_grad():
                outputs = model(masked_input_ids, attention_mask, tasks=("mlm",))
                logits = outputs["mlm_logits"]
            perplexities = mlm_perplexity_from_logits(logits, labels)
            all_perplexities.append(perplexities)

        # Average across models if ensemble
        stacked = torch.stack(all_perplexities)
        mean_perplexity = stacked.mean(dim=0).tolist()

        if return_std and self.ensemble_mode and len(self.models) > 1:
            std_perplexity = stacked.std(dim=0).tolist()
            return {"mean": mean_perplexity, "std": std_perplexity}

        return mean_perplexity

    def score_liabilities(
        self, sequences: List[str], return_std: bool = False
    ) -> Union[List[Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
        """Score liabilities with optional ensemble std deviation."""
        batch = batch_encode(
            sequences,
            add_special=True,
            max_length=self.max_length,
            invalid_policy="replace",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        masked_input_ids, labels = apply_mlm_mask(
            input_ids,
            attention_mask,
            mlm_probability=self.mlm_probability,
        )

        all_preds = []

        for model in self.models:
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, tasks=("reg",))
                preds = outputs["regression"]
                all_preds.append(preds)

        # Average across models if ensemble
        stacked = torch.stack(all_preds)
        mean_preds = stacked.mean(dim=0).cpu().numpy()
        mean_results = [
            dict(zip(self.model_cfg.liability_keys, row)) for row in mean_preds
        ]

        if return_std and self.ensemble_mode and len(self.models) > 1:
            std_preds = stacked.std(dim=0).cpu().numpy()
            std_results = [
                dict(zip(self.model_cfg.liability_keys, row)) for row in std_preds
            ]
            return {"mean": mean_results, "std": std_results}

        return mean_results

    def _build_mlm_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mask_seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator(device=input_ids.device)
        seed = 0 if mask_seed is None else int(mask_seed)
        generator.manual_seed(seed)
        return apply_mlm_mask(
            input_ids,
            attention_mask,
            mlm_probability=self.mlm_probability,
            rng=generator,
        )

    def score_perplexity_uncertainty(
        self,
        sequences: List[str],
        *,
        mc_samples: int = 20,
        dropout: bool = True,
        mask_seed: int | None = None,
    ) -> Dict[str, List[float]]:
        """Return mean and variance of perplexity estimates via MC dropout / ensembles."""
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive.")
        batch = batch_encode(
            sequences,
            add_special=True,
            max_length=self.max_length,
            invalid_policy="replace",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        masked_input_ids, labels = self._build_mlm_mask(
            input_ids,
            attention_mask,
            mask_seed=mask_seed,
        )

        sample_perplexities: List[torch.Tensor] = []
        mc_passes = mc_samples if dropout else 1

        with torch.no_grad():
            for model in self.models:
                passes = model.stochastic_forward(
                    masked_input_ids,
                    attention_mask,
                    tasks=("mlm",),
                    mc_samples=mc_passes,
                    enable_dropout=dropout,
                    no_grad=True,
                )
                for out in passes:
                    sample_perplexities.append(
                        mlm_perplexity_from_logits(
                            out["mlm_logits"].detach(),
                            labels,
                        )
                        .detach()
                        .cpu()
                    )

        samples = torch.stack(sample_perplexities, dim=0)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0, unbiased=False)
        std = variance.clamp_min(0.0).sqrt()
        return {
            "mean": mean.tolist(),
            "variance": variance.tolist(),
            "std": std.tolist(),
        }

    def score_liabilities_uncertainty(
        self,
        sequences: List[str],
        *,
        mc_samples: int = 20,
        dropout: bool = True,
    ) -> Dict[str, List[Dict[str, float]]]:
        """Return mean and variance of liability predictions."""
        if mc_samples <= 0:
            raise ValueError("mc_samples must be positive.")
        batch = batch_encode(
            sequences,
            add_special=True,
            max_length=self.max_length,
            invalid_policy="replace",
        )
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        sample_preds: List[torch.Tensor] = []
        mc_passes = mc_samples if dropout else 1

        with torch.no_grad():
            for model in self.models:
                passes = model.stochastic_forward(
                    input_ids,
                    attention_mask,
                    tasks=("reg",),
                    mc_samples=mc_passes,
                    enable_dropout=dropout,
                    no_grad=True,
                )
                for out in passes:
                    sample_preds.append(out["regression"].detach().cpu())

        stacked = torch.stack(sample_preds, dim=0)
        mean = stacked.mean(dim=0)
        variance = stacked.var(dim=0, unbiased=False)
        std = variance.clamp_min(0.0).sqrt()

        def _to_dict(tensor: torch.Tensor) -> List[Dict[str, float]]:
            results = []
            for row in tensor:
                results.append(
                    {
                        key: float(value)
                        for key, value in zip(self.model_cfg.liability_keys, row.tolist())
                    }
                )
            return results

        return {
            "mean": _to_dict(mean),
            "variance": _to_dict(variance),
            "std": _to_dict(std),
        }


def create_app(
    checkpoint: Union[Path, List[Path]],
    model_config: Path,
    device: Optional[str] = None,
    ensemble_mode: bool = False,
) -> FastAPI:
    """
    Create FastAPI app with single or ensemble model inference.

    Args:
        checkpoint: Path to single checkpoint or list of checkpoints for ensemble
        model_config: Path to model configuration YAML
        device: Device to run inference on (default: auto-detect)
        ensemble_mode: Whether to enable ensemble mode with multiple models

    Returns:
        FastAPI application instance
    """
    if FastAPI is None or HTTPException is None:  # pragma: no cover - guarded for optional dep
        raise RuntimeError("fastapi is required to create the inference app (install with .[serve]).")
    wrapper = ModelWrapper(checkpoint, model_config, device, ensemble_mode)
    app = FastAPI(title="AbProp Inference API")

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": str(wrapper.device),
            "ensemble_mode": wrapper.ensemble_mode,
            "n_models": len(wrapper.models),
        }

    @app.post("/score/perplexity")
    def score_perplexity(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        return_std = payload.get("return_std", False)
        try:
            scores = wrapper.score_perplexity(sequences, return_std=return_std)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"perplexity": scores}

    @app.post("/score/liabilities")
    def score_liabilities(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        return_std = payload.get("return_std", False)
        try:
            scores = wrapper.score_liabilities(sequences, return_std=return_std)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"liabilities": scores}

    @app.post("/score/perplexity/uncertainty")
    def score_perplexity_uncertainty(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        mc_samples = int(payload.get("mc_samples", 20))
        dropout = bool(payload.get("dropout", True))
        mask_seed = payload.get("mask_seed")
        mask_seed = int(mask_seed) if mask_seed is not None else None
        try:
            stats = wrapper.score_perplexity_uncertainty(
                sequences,
                mc_samples=mc_samples,
                dropout=dropout,
                mask_seed=mask_seed,
            )
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"perplexity": stats}

    @app.post("/score/liabilities/uncertainty")
    def score_liabilities_uncertainty(payload: Dict[str, Any]) -> Dict[str, Any]:
        sequences = payload.get("sequences")
        if not isinstance(sequences, list) or not sequences:
            raise HTTPException(status_code=400, detail="provide non-empty 'sequences' list")
        mc_samples = int(payload.get("mc_samples", 20))
        dropout = bool(payload.get("dropout", True))
        try:
            stats = wrapper.score_liabilities_uncertainty(sequences, mc_samples=mc_samples, dropout=dropout)
        except ValueError as err:
            raise HTTPException(status_code=400, detail=str(err))
        return {"liabilities": stats}

    return app


def create_ensemble_app_from_cv(
    cv_dir: Path, model_config: Path, device: Optional[str] = None
) -> FastAPI:
    """
    Create FastAPI app with ensemble inference from CV fold checkpoints.

    Args:
        cv_dir: Directory containing CV fold subdirectories
        model_config: Path to model configuration YAML
        device: Device to run inference on (default: auto-detect)

    Returns:
        FastAPI application instance with ensemble inference
    """
    # Discover all fold checkpoints
    fold_dirs = sorted([d for d in cv_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    checkpoints = []

    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / "checkpoints" / "best.pt"
        if ckpt_path.exists():
            checkpoints.append(ckpt_path)

    if not checkpoints:
        raise ValueError(f"No fold checkpoints found in {cv_dir}")

    return create_app(checkpoints, model_config, device, ensemble_mode=True)


__all__ = ["create_app", "create_ensemble_app_from_cv", "ModelWrapper"]
