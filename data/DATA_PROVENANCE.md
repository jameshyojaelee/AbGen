# Data Provenance

| Dataset | Location | Source | Notes |
|---------|----------|--------|-------|
| OAS processed splits | `data/processed/oas_real_full` | Internal ETL via `scripts/process_real_data_etl.py` | Contains train/val/test parquet partitions with chain/species metadata. |
| Benchmark fixtures | `data/benchmarks/` | Derived from OAS + curated therapeutic set | Used for deterministic regression checks. |
| Demo sequences | `examples/attention_success.fa`, `examples/attention_failure.fa` | Synthetic heavy chains | Safe for public demos. |
| CI fixture dataset | `tests/fixtures/oas_fixture.csv` | Handcrafted synthetic fixture | Used by `configs/data_ci.yaml` and `configs/benchmarks_ci.yaml`. |
| Toy sequences | `tests/fixtures/toy_sequences.fa` | Synthetic FASTA seeds | Used for preference building + design benchmark smoke tests. |

## Policy

- `data/raw/` is ignored by git and should be populated via the fetch/ETL scripts.
- Small, curated fixtures live under `tests/fixtures/` for unit tests.

## Fixture checksums

SHA256:

- `tests/fixtures/oas_fixture.csv`: `aebf23be7b093c55a895060be5541005437bd4670e3f5a8361e1b14c973d6229`
- `tests/fixtures/toy_sequences.fa`: `34756fc2931660a6ec29e114fc2c856f6947fd2944c344b406a341c1f7aa32b2`

## Regeneration

1. Download raw OAS data following the licensing terms.
2. Run `python scripts/process_real_data_etl.py --input <raw_csv> --output data/processed/oas_real_full`.
3. Log dataset version + checksum in this file for every refresh.
