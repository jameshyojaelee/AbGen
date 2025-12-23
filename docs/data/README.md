# Data & ETL

## Data locations

- `data/raw/` — raw downloads (gitignored)
- `data/interim/` — intermediate artifacts (gitignored)
- `data/processed/` — parquet exports (gitignored)
- `tests/fixtures/` — small, tracked fixtures for tests/CI

## ETL

```bash
python scripts/process_real_data_etl.py --input <raw_csv> --output data/processed/oas_real_full
```

For data provenance and checksums, see `data/DATA_PROVENANCE.md`.

