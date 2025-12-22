# Repository Layout

## Inventory (Top-Level)

- `src/abprop/` — product code (models, data, eval, training, server)
- `scripts/` — CLI entrypoints and workflows (core + dev helpers)
- `configs/` — YAML/json configuration files
- `docs/` — documentation, figures, and references
- `data/` — datasets and provenance (raw/interim/processed)
- `outputs/` — training/eval artifacts (generated)
- `models/` — model registry + cards
- `tests/` — unit and integration tests
- `examples/` — small example inputs
- `notebooks/` — exploratory notebooks
- `benchmarks/`, `mlruns/`, `logs/` — experiment artifacts (generated)

## Target Structure

```
repo/
├── src/abprop/               # product code
├── scripts/                  # user-facing CLIs
│   ├── dev/                  # smoke + ad-hoc benchmarks
│   └── README.md             # script catalog
├── configs/                  # runtime configs
│   ├── legacy/               # deprecated configs
│   └── README.md             # config catalog
├── docs/                     # documentation hub
│   ├── training/
│   ├── design/
│   ├── evaluation/
│   ├── data/
│   ├── reference/
│   └── README.md             # navigation hub
├── data/                     # datasets + provenance
├── models/                   # registry + model cards
├── outputs/                  # generated artifacts
└── tests/
```

## Conventions

- **Docs** live under `docs/` subfolders. Old paths remain as stubs that point to new locations.
- **Scripts** in `scripts/` are the canonical entrypoints; `scripts/dev/` holds smoke/bench helpers.
- **Configs** under `configs/legacy/` are kept for reference but not used in the main workflows.
- **Artifacts** (`outputs/`, `mlruns/`, `benchmarks/`, `logs/`) are generated and not sources of truth.
