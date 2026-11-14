from housing_mlops.data.hashing import dataframe_sha256


def test_hash_is_deterministic(housing_frame):
    assert dataframe_sha256(housing_frame) == dataframe_sha256(housing_frame.copy())


def test_hash_is_hex_sha256(housing_frame):
    digest = dataframe_sha256(housing_frame)
    assert len(digest) == 64
    int(digest, 16)


def test_hash_changes_when_a_value_changes(housing_frame):
    changed = housing_frame.copy()
    changed.loc[0, "MedInc"] += 0.001
    assert dataframe_sha256(changed) != dataframe_sha256(housing_frame)


def test_hash_changes_when_a_column_is_renamed(housing_frame):
    renamed = housing_frame.rename(columns={"MedInc": "Income"})
    assert dataframe_sha256(renamed) != dataframe_sha256(housing_frame)


def test_hash_changes_when_rows_are_reordered(housing_frame):
    reordered = housing_frame.iloc[::-1]
    assert dataframe_sha256(reordered) != dataframe_sha256(housing_frame)
