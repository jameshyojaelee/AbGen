# Data

- Primary processed dataset: `data/processed/oas_real_full/`
- Benchmark overrides: `configs/benchmarks_local.yaml` points benchmarks at the processed OAS export.
- Provenance: `data/DATA_PROVENANCE.md`
- Raw data is ignored by git; use the ETL in `scripts/etl.py` or `scripts/process_real_data_etl.py` to refresh.
