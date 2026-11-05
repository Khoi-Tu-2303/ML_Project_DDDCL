"""
Dataset loading, integrity validation, and DatasetBundle definition.
Supports hosted Hugging Face downloads and local release verification.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    """
    Immutable container for a verified dataset.
    Excludes row IDs, labels, and metadata from the feature space X.
    """
    dataset_id: str
    frame: pd.DataFrame
    feature_columns: List[str]
    label_column: str
    row_id_column: str
    metadata_columns: List[str]
    label_mapping: Dict[str, int]
    label_meanings: Dict[str, str]
    data_revision: str
    manifest_hash: str
    n_classes: int

    @property
    def X(self) -> pd.DataFrame:
        """Return the feature matrix only, preserving exact column order."""
        return self.frame[self.feature_columns].copy()

    @property
    def y(self) -> pd.Series:
        """Return the label series."""
        return self.frame[self.label_column].copy()

    @property
    def row_ids(self) -> List[str]:
        """Return the list of unique row IDs."""
        return self.frame[self.row_id_column].tolist()


def checked_config(path: Path) -> Dict[str, Any]:
    """
    Validate HF data configuration.
    Rejects placeholders, floating revisions (e.g. 'main', 'latest'), and invalid hashes.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data configuration file not found: {path}")

    config = json.loads(path.read_text(encoding="utf-8"))
    repo = config.get("repo_id", "")
    if repo.count("/") != 1 or "YOUR_" in repo or " " in repo:
        raise ValueError(
            f"Invalid repo_id '{repo}'. Please set a real HF dataset repo_id (e.g., 'username/dataset-name')."
        )
    revision = config.get("revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(
            f"Invalid revision '{revision}'. HF revision must be a full 40-character commit SHA, not main/latest."
        )
    manifest_sha = config.get("manifest_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha):
        raise ValueError(
            f"Invalid manifest_sha256 '{manifest_sha}'. Expected 64-character hex SHA-256."
        )
    return config


def validate_csv_bytes(blob: bytes, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate CSV bytes strictly against manifest entry:
    - Verifies SHA-256 checksum
    - Verifies exact schema and column ordering
    - Verifies row count and row_id uniqueness
    - Verifies class counts and contiguous integer labels 0..C-1
    - Verifies all features are finite numbers
    """
    actual_sha = hashlib.sha256(blob).hexdigest()
    expected_sha = entry.get("sha256", "")
    if actual_sha != expected_sha:
        raise ValueError(f"Dataset checksum mismatch: expected {expected_sha}, got {actual_sha}")

    text = blob.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    expected_fields = [entry["row_id_column"], *entry["feature_columns"], entry["label_column"]]
    if reader.fieldnames != expected_fields:
        raise ValueError(
            f"Dataset schema/column order mismatch: expected {expected_fields}, got {reader.fieldnames}"
        )

    rows = list(reader)
    row_count = len(rows)
    expected_rows = entry["n_rows"]
    if row_count != expected_rows:
        raise ValueError(f"Dataset row count mismatch: expected {expected_rows}, got {row_count}")

    row_id_col = entry["row_id_column"]
    ids = [r[row_id_col] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Dataset unique row_id constraint failed: duplicate row IDs detected")

    label_col = entry["label_column"]
    actual_counts = dict(Counter(r[label_col] for r in rows))
    expected_counts = entry["class_counts"]
    # Handle int/str keys in class_counts
    norm_expected = {str(k): int(v) for k, v in expected_counts.items()}
    norm_actual = {str(k): int(v) for k, v in actual_counts.items()}
    if norm_actual != norm_expected:
        raise ValueError(f"Dataset class counts differ from manifest: expected {norm_expected}, got {norm_actual}")

    # Contiguous integer labels 0..C-1
    labels = sorted(int(k) for k in norm_expected.keys())
    if labels != list(range(len(labels))):
        raise ValueError(f"Canonical labels must be contiguous integers 0..C-1, found {labels}")

    # Check finite float features
    for r in rows:
        for col in entry["feature_columns"]:
            val = float(r[col])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite feature value detected in column '{col}': {val}")

    return rows


def load_dataset_from_release(
    release_dir: Path,
    dataset_id: str,
    manifest_name: str = "manifest.json"
) -> DatasetBundle:
    """
    Load a verified DatasetBundle from a local release folder (e.g. hf_release).
    """
    manifest_file = release_dir / manifest_name
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found in release dir: {manifest_file}")

    manifest_bytes = manifest_file.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    if dataset_id not in manifest["datasets"]:
        raise KeyError(f"Dataset '{dataset_id}' not found in manifest. Available: {list(manifest['datasets'].keys())}")

    entry = manifest["datasets"][dataset_id]
    csv_file = release_dir / entry["filename"]
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    blob = csv_file.read_bytes()
    rows = validate_csv_bytes(blob, entry)

    df = pd.DataFrame(rows)
    # Cast features to float and label to int
    for col in entry["feature_columns"]:
        df[col] = df[col].astype(float)
    df[entry["label_column"]] = df[entry["label_column"]].astype(int)

    n_classes = len(entry["class_counts"])

    return DatasetBundle(
        dataset_id=dataset_id,
        frame=df,
        feature_columns=list(entry["feature_columns"]),
        label_column=entry["label_column"],
        row_id_column=entry["row_id_column"],
        metadata_columns=[],
        label_mapping=entry.get("label_mapping", {}),
        label_meanings=entry.get("label_meanings", {}),
        data_revision=entry.get("sha256", "")[:12],
        manifest_hash=manifest_hash,
        n_classes=n_classes,
    )


def load_dataset_from_hf(config_path: Path, dataset_id: str) -> DatasetBundle:
    """
    Download pinned manifest and dataset CSV from Hugging Face Hub,
    verify checksums and schema, and return a verified DatasetBundle.
    """
    from huggingface_hub import hf_hub_download

    config = checked_config(config_path)
    kwargs = {
        "repo_id": config["repo_id"],
        "repo_type": "dataset",
        "revision": config["revision"],
        "cache_dir": config.get("cache_dir", ".cache/huggingface"),
    }

    manifest_filename = config.get("manifest_filename", "manifest.json")
    manifest_path = Path(hf_hub_download(filename=manifest_filename, **kwargs))
    raw_manifest = manifest_path.read_bytes()

    actual_manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
    if actual_manifest_sha != config["manifest_sha256"]:
        raise ValueError(
            f"Manifest checksum mismatch: expected {config['manifest_sha256']}, got {actual_manifest_sha}"
        )

    manifest = json.loads(raw_manifest.decode("utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema version")

    if dataset_id not in manifest["datasets"]:
        raise KeyError(f"Dataset '{dataset_id}' not found in HF manifest")

    entry = manifest["datasets"][dataset_id]
    if entry.get("status") not in {"verified_primary_source", "verified_hosted_mirror"}:
        raise ValueError(f"Dataset provenance not approved: {entry.get('status')}")

    csv_path = Path(hf_hub_download(filename=entry["filename"], **kwargs))
    blob = csv_path.read_bytes()
    rows = validate_csv_bytes(blob, entry)

    df = pd.DataFrame(rows)
    for col in entry["feature_columns"]:
        df[col] = df[col].astype(float)
    df[entry["label_column"]] = df[entry["label_column"]].astype(int)

    return DatasetBundle(
        dataset_id=dataset_id,
        frame=df,
        feature_columns=list(entry["feature_columns"]),
        label_column=entry["label_column"],
        row_id_column=entry["row_id_column"],
        metadata_columns=[],
        label_mapping=entry.get("label_mapping", {}),
        label_meanings=entry.get("label_meanings", {}),
        data_revision=config["revision"],
        manifest_hash=actual_manifest_sha,
        n_classes=len(entry["class_counts"]),
    )

