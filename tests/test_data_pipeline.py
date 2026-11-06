"""
Unit tests for Phase B: Data Loading, Split Contract, and Anti-Leakage Preprocessing.
Enforces Gate B requirements from docs/07_VERIFICATION.md.
"""

import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest

from dddcl.data.hf import (
    DatasetBundle,
    load_dataset_from_release,
    validate_csv_bytes,
)
from dddcl.data.splits import (
    SplitBundle,
    create_stratified_split,
    save_split,
    load_split,
    compute_dataset_fingerprint,
)
from dddcl.data.preprocess import (
    prepare_processed_split,
    get_scaler,
)


@pytest.fixture
def release_dir() -> Path:
    return Path("hf_release")


@pytest.fixture
def haberman_bundle(release_dir) -> DatasetBundle:
    return load_dataset_from_release(release_dir, "haberman")


@pytest.fixture
def thyroid_bundle(release_dir) -> DatasetBundle:
    return load_dataset_from_release(release_dir, "new_thyroid")


@pytest.fixture
def banknote_bundle(release_dir) -> DatasetBundle:
    return load_dataset_from_release(release_dir, "banknote")


# 1. Dataset loading and verification tests
def test_dataset_bundle_properties(haberman_bundle, thyroid_bundle, banknote_bundle):
    # Haberman: 306 rows, 3 features, 2 classes
    assert len(haberman_bundle.frame) == 306
    assert len(haberman_bundle.feature_columns) == 3
    assert haberman_bundle.n_classes == 2
    assert set(haberman_bundle.y.unique()) == {0, 1}
    assert len(haberman_bundle.row_ids) == 306
    assert len(set(haberman_bundle.row_ids)) == 306

    # New-Thyroid: 215 rows, 5 features, 3 classes
    assert len(thyroid_bundle.frame) == 215
    assert len(thyroid_bundle.feature_columns) == 5
    assert thyroid_bundle.n_classes == 3
    assert set(thyroid_bundle.y.unique()) == {0, 1, 2}

    # Banknote: 1372 rows, 4 features, 2 classes
    assert len(banknote_bundle.frame) == 1372
    assert len(banknote_bundle.feature_columns) == 4
    assert banknote_bundle.n_classes == 2
    assert set(banknote_bundle.y.unique()) == {0, 1}


# 2. Strict corruption and integrity rejection tests
def test_integrity_rejections():
    entry = {
        "filename": "data/dummy.csv",
        "row_id_column": "row_id",
        "feature_columns": ["f1", "f2"],
        "label_column": "label",
        "n_rows": 2,
        "class_counts": {"0": 1, "1": 1},
        "sha256": "",
    }
    valid_csv = "row_id,f1,f2,label\nid1,1.0,2.0,0\nid2,3.0,4.0,1\n"
    blob = valid_csv.encode("utf-8")
    entry["sha256"] = hashlib.sha256(blob).hexdigest()

    # Valid case passes
    rows = validate_csv_bytes(blob, entry)
    assert len(rows) == 2

    # Checksum mismatch
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_csv_bytes(blob + b"corrupt", entry)

    # Schema mismatch: extra column
    bad_schema = "row_id,f1,f2,extra,label\nid1,1.0,2.0,0,0\nid2,3.0,4.0,0,1\n".encode("utf-8")
    entry_bad_schema = copy.deepcopy(entry)
    entry_bad_schema["sha256"] = hashlib.sha256(bad_schema).hexdigest()
    with pytest.raises(ValueError, match="schema/column order mismatch"):
        validate_csv_bytes(bad_schema, entry_bad_schema)

    # Duplicate row ID
    dup_ids = "row_id,f1,f2,label\nid1,1.0,2.0,0\nid1,3.0,4.0,1\n".encode("utf-8")
    entry_dup = copy.deepcopy(entry)
    entry_dup["sha256"] = hashlib.sha256(dup_ids).hexdigest()
    with pytest.raises(ValueError, match="unique row_id"):
        validate_csv_bytes(dup_ids, entry_dup)

    # Nonfinite feature value (NaN or inf)
    nan_csv = "row_id,f1,f2,label\nid1,nan,2.0,0\nid2,3.0,4.0,1\n".encode("utf-8")
    entry_nan = copy.deepcopy(entry)
    entry_nan["sha256"] = hashlib.sha256(nan_csv).hexdigest()
    with pytest.raises(ValueError, match="Non-finite feature"):
        validate_csv_bytes(nan_csv, entry_nan)


# 3. Stratified Split Contract tests
def test_stratified_split_contract(haberman_bundle, thyroid_bundle, banknote_bundle):
    for bundle in [haberman_bundle, thyroid_bundle, banknote_bundle]:
        split = create_stratified_split(bundle, seed=42)
        total_rows = len(bundle.frame)

        # Disjointness
        s_train = set(split.train_ids)
        s_val = set(split.validation_ids)
        s_test = set(split.test_ids)
        assert len(s_train & s_val) == 0, "Train and val overlap"
        assert len(s_train & s_test) == 0, "Train and test overlap"
        assert len(s_val & s_test) == 0, "Val and test overlap"

        # Completeness
        assert (s_train | s_val | s_test) == set(bundle.row_ids)

        # Proportions ~ 70% / 20% / 10%
        assert 0.68 <= len(split.train_ids) / total_rows <= 0.72
        assert 0.18 <= len(split.validation_ids) / total_rows <= 0.22
        assert 0.08 <= len(split.test_ids) / total_rows <= 0.12

        # Every class present in each split partition
        for partition in ["train", "validation", "test"]:
            counts = split.class_counts[partition]
            assert len(counts) == bundle.n_classes, f"Missing class in {partition} for {bundle.dataset_id}"
            assert all(c > 0 for c in counts.values())


def test_split_determinism_and_pairing(haberman_bundle):
    split_a = create_stratified_split(haberman_bundle, seed=10)
    split_b = create_stratified_split(haberman_bundle, seed=10)
    assert split_a.train_ids == split_b.train_ids
    assert split_a.validation_ids == split_b.validation_ids
    assert split_a.test_ids == split_b.test_ids

    split_c = create_stratified_split(haberman_bundle, seed=20)
    assert split_a.test_ids != split_c.test_ids, "Different seeds should produce different splits"


# 4. Anti-leakage Preprocessing Sentinel Test
def test_train_only_scaler_anti_leakage(haberman_bundle):
    """
    Sentinel test: Changing test set feature values or labels must NOT change
    the fitted scaler parameters (mean_, scale_) or train scaled features.
    """
    split = create_stratified_split(haberman_bundle, seed=42)
    processed_orig = prepare_processed_split(haberman_bundle, split, scaler_type="standard")

    # Clone bundle frame and corrupt the test set with massive outliers
    corrupted_frame = haberman_bundle.frame.copy()
    test_mask = corrupted_frame[haberman_bundle.row_id_column].isin(split.test_ids)
    for col in haberman_bundle.feature_columns:
        corrupted_frame.loc[test_mask, col] = 999999.0

    corrupted_bundle = DatasetBundle(
        dataset_id=haberman_bundle.dataset_id,
        frame=corrupted_frame,
        feature_columns=haberman_bundle.feature_columns,
        label_column=haberman_bundle.label_column,
        row_id_column=haberman_bundle.row_id_column,
        metadata_columns=haberman_bundle.metadata_columns,
        label_mapping=haberman_bundle.label_mapping,
        label_meanings=haberman_bundle.label_meanings,
        data_revision=haberman_bundle.data_revision,
        manifest_hash=haberman_bundle.manifest_hash,
        n_classes=haberman_bundle.n_classes,
    )

    processed_corrupted = prepare_processed_split(corrupted_bundle, split, scaler_type="standard")

    # Verify scaler mean_ and scale_ are 100% identical
    np.testing.assert_array_equal(processed_orig.scaler.mean_, processed_corrupted.scaler.mean_)
    np.testing.assert_array_equal(processed_orig.scaler.scale_, processed_corrupted.scaler.scale_)
    # Verify train scaled arrays are 100% identical
    np.testing.assert_array_equal(processed_orig.X_train_scaled, processed_corrupted.X_train_scaled)
    # Verify test scaled arrays differ due to corrupted inputs
    assert not np.array_equal(processed_orig.X_test_scaled, processed_corrupted.X_test_scaled)


# 5. Split serialization and deserialization
def test_split_serialization_roundtrip(tmp_path, haberman_bundle):
    split = create_stratified_split(haberman_bundle, seed=99)
    saved_path = save_split(split, out_dir=tmp_path)
    loaded_split = load_split(saved_path, bundle=haberman_bundle)

    assert loaded_split.protocol_id == split.protocol_id
    assert loaded_split.dataset_id == split.dataset_id
    assert loaded_split.split_seed == split.split_seed
    assert loaded_split.train_ids == split.train_ids
    assert loaded_split.validation_ids == split.validation_ids
    assert loaded_split.test_ids == split.test_ids
    assert loaded_split.dataset_fingerprint == split.dataset_fingerprint

