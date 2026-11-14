# Phase 1 notes: data layer

## What was built

- `housing_mlops.data.schema`: the feature and target column names, per-column value
  bounds, a Pandera `DataFrameSchema` for the full dataset and one for features only,
  and two `validate_*` functions that run in lazy mode so every failure is reported.
- `housing_mlops.data.download`: `load_raw_dataset` returns the dataset as a
  DataFrame, reading `data/raw/california_housing.parquet` when it exists and
  otherwise calling a fetcher (by default scikit-learn's downloader) and caching the
  result. The fetcher is an injectable function.
- `housing_mlops.data.split`: `split_dataset` makes a seeded 70/10/20
  train/validation/test split, plus `write_splits` and `read_splits` for parquet files
  in `data/processed`.
- `housing_mlops.data.hashing`: `dataframe_sha256`, a stable fingerprint of a
  DataFrame that the training metadata will record in phase 2.
- `housing_mlops.data.prepare`: `prepare_dataset` chains load, validate, split and
  write. `scripts/download_data.py` is a thin CLI over it.
- Tests: 59 offline unit tests for the data package (schema, split, download, prepare,
  hashing) and 3 tests marked `dataset` that check the real cached file.

Real run on this machine: 20,640 rows, data hash `8004a491...`, split sizes
train 14,448 / validation 2,064 / test 4,128, second run served from cache.

## Design choices and why

**Injectable fetcher instead of mocking scikit-learn.** `load_raw_dataset` takes a
`fetcher` callable. Unit tests pass a counting fake and never touch the network, and
the production path is unchanged. The scikit-learn wrapper itself is tested by
monkeypatching `sklearn.datasets.fetch_california_housing`, so even that path is
covered without a download.

**Parquet for the cache and the splits.** Parquet preserves dtypes exactly (CSV would
turn floats into text and lose the index), is compact, and `pyarrow` was already a
transitive dependency through Streamlit. It is now pinned explicitly because the data
layer depends on it directly.

**Bounds that are loose but physical.** The schema rejects impossible values (zero or
negative income, rooms or occupancy, coordinates outside California, prices above the
dataset's cap) while leaving room for unusual yet legitimate rows. Drift is a separate
concern handled in phase 4; a drifted request is unusual, not invalid.

**A dataframe-level check for bedrooms versus rooms.** Average bedrooms cannot exceed
average rooms because both are divided by the same household count. Cross-column
rules are where single-column schemas usually fall short, so one is included.

**`coerce=True` on columns.** Integers in a numeric column (for example a `HouseAge`
read from CSV) are converted to float rather than rejected. Strings still fail.

**Lazy validation.** `lazy=True` collects every failure before raising `SchemaErrors`,
so a broken file reports all bad columns at once instead of one per run.

**Two-step split with a rescaled validation fraction.** `train_test_split` is applied
twice: first to carve out the test set, then to split the remainder. The validation
fraction is divided by `(1 - test_fraction)` so that 10% of the whole dataset lands in
validation, not 10% of the remainder.

**Original row indices are kept.** The splits are not re-indexed, so a row can be
traced back to the raw dataset and the disjointness test can compare index sets.

**Hash includes the index and column names.** `pd.util.hash_pandas_object` hashes
values per row with a fixed key, so the digest is the same across processes and
machines. Reordering rows or renaming a column changes it, which is the intended
meaning of "the exact training data".

**Failure precedence in `prepare_dataset`.** Validation runs before splitting, and the
test asserts no processed files are written when validation fails, so a broken
download cannot leave half-written outputs behind.

## Alternatives rejected

- **Pandera `DataFrameModel` (class-based) instead of `DataFrameSchema`.** The class
  style is elegant but the bounds are needed as plain data in phase 3 (Pydantic
  request validation) and phase 4 (drift bins). A dict of bounds feeding a
  `DataFrameSchema` keeps one source of truth.
- **Stratified split on the target.** Useful for skewed regression targets, but
  `MedHouseVal` is fairly well spread and stratifying would complicate the
  "reproducible from a seed" story without a measurable benefit here.
- **Skipping the real-dataset tests when the file is missing.** A skipped test
  silently passes in CI forever. The tests fail with a message telling you to run the
  download script instead. They are kept out of the unit job by the `dataset` marker.
- **Checking the dataset into git.** 800 KB of parquet is small, but data belongs in a
  cache that is regenerated, not in history. The script makes regeneration one command.
- **Hashing the parquet file bytes.** Parquet encoding is not byte-stable across
  library versions, so the same rows could hash differently after an upgrade.

## Places where a code comment would help

1. `schema.py`, `FEATURE_BOUNDS`: a note that these are validity bounds, not drift
   bounds, and how they were chosen from the real data ranges.
2. `schema.py`, `_bedrooms_within_rooms`: why the inequality must hold.
3. `split.py`, the `relative_validation` line: why the fraction is rescaled.
4. `hashing.py`: why column names and the index are folded into the digest.
5. `download.py`, the import inside `fetch_from_sklearn`: it keeps scikit-learn's
   dataset module out of the import path of the API process.
6. `tests/unit/data/test_schema.py`, `column_level_failures`: why dataframe-level
   failure cases are excluded.

## Five questions to test your understanding

1. The schema uses `Check.gt(lower)` rather than `Check.ge(lower)`. For which columns
   does that distinction matter in the real data, and for which is it irrelevant?
2. If someone changes `validation_fraction` to 0.2 without touching `test_fraction`,
   how many rows land in each split for 20,640 rows, and why is the second
   `train_test_split` call given `0.25` rather than `0.2`?
3. Why does `test_prepare_is_reproducible_across_runs` assert `fetcher.calls == 1`
   and what bug would slip through if that line were removed?
4. `dataframe_sha256` changes when rows are reordered. Give one situation where that
   is exactly what you want and one where it would cause a false "data changed" alarm.
5. The dataframe-level bedrooms check reports a failure case for every column. How
   did that affect the first version of the bound tests, and what did the fix change
   about the test rather than the schema?
