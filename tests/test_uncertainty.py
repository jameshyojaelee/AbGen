import pytest

torch = pytest.importorskip("torch", reason="PyTorch is required for uncertainty tests.")
nn = pytest.importorskip("torch.nn", reason="PyTorch is required for uncertainty tests.")

from abprop.eval.uncertainty import (
    TemperatureScaler,
    expected_calibration_error,
    regression_uncertainty_summary,
    mlm_perplexity_from_logits,
)


def test_mlm_perplexity_ignores_unmasked_tokens():
    labels = torch.tensor([[-100, 2, -100, -100]], dtype=torch.long)
    vocab_size = 4
    logits = torch.full((1, labels.size(1), vocab_size), -5.0, dtype=torch.float32)
    logits[0, 1, 2] = 5.0  # correct only for the masked token

    perplexity = mlm_perplexity_from_logits(logits, labels)
    assert perplexity.shape == (1,)
    assert perplexity.item() == pytest.approx(1.0, rel=1e-2)


def test_regression_uncertainty_summary_coverage():
    mean = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    variance = torch.tensor([[1.0], [0.04]], dtype=torch.float32)  # std = 1.0 and 0.2
    targets = torch.tensor([[0.5], [1.3]], dtype=torch.float32)

    summary = regression_uncertainty_summary(mean, variance, targets, sigma_levels=(1.0, 2.0))
    coverage = summary["coverage"]
    assert coverage["1.0σ"] == pytest.approx(0.5, rel=1e-5)
    assert coverage["2.0σ"] == pytest.approx(1.0, rel=1e-5)
    assert summary["mean_std"] == pytest.approx(0.6, rel=1e-5)
    assert summary["mean_abs_error"] > 0.0


def test_temperature_scaler_reduces_cross_entropy():
    logits = torch.tensor(
        [
            [3.5, 0.1],
            [0.2, 2.8],
            [2.2, 0.3],
            [0.1, 2.5],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    scaler = TemperatureScaler(initial_temperature=5.0)
    criterion = nn.CrossEntropyLoss()
    loss_before = criterion(scaler(logits), labels)
    scaler.fit(logits, labels, max_iter=25)
    loss_after = criterion(scaler(logits), labels)

    assert scaler.temperature.item() > 0
    assert loss_after <= loss_before + 1e-5


def test_expected_calibration_error_small_for_correct_predictions():
    probabilities = torch.tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    ece = expected_calibration_error(probabilities, labels, n_bins=5)
    assert ece >= 0.0
    assert ece == pytest.approx(0.1333, rel=1e-2)
