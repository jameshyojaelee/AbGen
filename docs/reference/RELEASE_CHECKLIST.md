# Release Checklist

Use this checklist before tagging a release or publishing results.

## Data & provenance
- [ ] Record dataset version(s) and SHA256 checksums in `data/DATA_PROVENANCE.md`.
- [ ] Confirm `data/raw/` is empty in git (raw downloads are untracked).
- [ ] Ensure fixture datasets under `tests/fixtures/` are unchanged or updated with new checksums.

## Reproducibility
- [ ] Fix random seed(s) in `configs/train.yaml` (or document overrides).
- [ ] Capture `outputs/**/config_snapshot.json` and `git_commit.txt` for each training run.
- [ ] Record any non‑default CLI overrides used for a run.

## Key commands (expected artifacts)
- [ ] Train baseline: `python scripts/train.py ...`
  - Expect: `outputs/<run>/checkpoints/best.pt`
- [ ] Eval baseline: `python scripts/eval.py ...`
  - Expect: `outputs/eval/metrics.json` + plots
- [ ] Generation: `python scripts/generate.py ...`
  - Expect: `outputs/generation/<run>/candidates.jsonl`
- [ ] DPO: `python scripts/train_dpo.py ...`
  - Expect: `outputs/dpo_run/checkpoints/dpo_aligned.pt`
- [ ] Benchmarks: `python scripts/run_benchmarks.py ...`
  - Expect: `outputs/benchmarks/<benchmark>/metrics.json`
- [ ] Design benchmark: `python scripts/run_design_benchmark.py ...`
  - Expect: `outputs/benchmarks/design/summary.json`

## Documentation
- [ ] Update `docs/reference/CLAIMS.md` with the evidence for each claim.
- [ ] Ensure README model zoo is accurate (or marked illustrative).
- [ ] Update `docs/evaluation/RESULTS.md` with the latest numbers and dataset labels.

## Sanity checks
- [ ] `python3 -m compileall -q src scripts`
- [ ] `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q` (expect 1 CPU‑only skip for CUDA AMP)
