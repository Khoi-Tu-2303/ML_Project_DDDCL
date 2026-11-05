"""
Data splitting contract and SplitBundle definition.
Enforces stratified 70/20/10 splits by row ID without index leakage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dddcl.data.hf import DatasetBundle


@dataclass(frozen=True)
class SplitBundle:
    """
    Immutable container for row ID splits of a dataset.
    Preserves exact string IDs, ensuring all algorithms evaluate on identical test partitions.
    """
    protocol_id: str
    dataset_id: str
    split_seed: int
    dataset_fingerprint: str
    train_ids: List[str]
    validation_ids: List[str]
    test_ids: List[str]
    class_counts: Dict[str, Dict[str, int]]

    @property
    def all_ids(self) -> List[str]:
        return self.train_ids + self.validation_ids + self.test_ids

    def validate_integrity(self, bundle: Optional[DatasetBundle] = None) -> None:
        """
        Verify:
        1. Disjoint partitions
        2. Union matches bundle row IDs
        3. Dataset fingerprint matches bundle
        """
        s_train = set(self.train_ids)
        s_val = set(self.validation_ids)
        s_test = set(self.test_ids)

        if len(s_train) != len(self.train_ids):
            raise ValueError("Duplicate IDs found in train_ids")
        if len(s_val) != len(self.validation_ids):
            raise ValueError("Duplicate IDs found in validation_ids")
        if len(s_test) != len(self.test_ids):
            raise ValueError("Duplicate IDs found in test_ids")

        if s_train & s_val:
            raise ValueError("train_ids and validation_ids intersect")
        if s_train & s_test:
            raise ValueError("train_ids and test_ids intersect")
        if s_val & s_test:
            raise ValueError("validation_ids and test_ids intersect")

        if bundle is not None:
            actual_fp = compute_dataset_fingerprint(bundle)
            if actual_fp != self.dataset_fingerprint:
                raise ValueError(
                    f"Dataset fingerprint mismatch: split expects {self.dataset_fingerprint}, bundle is {actual_fp}"
                )
            bundle_ids = set(bundle.row_ids)
            union_ids = s_train | s_val | s_test
            if union_ids != bundle_ids:
                raise ValueError("Union of split IDs does not match dataset row IDs")


def compute_dataset_fingerprint(bundle: DatasetBundle) -> str:
    """
    Compute a deterministic SHA-256 fingerprint from sorted (row_id, label) pairs and manifest hash.
    """
    hasher = hashlib.sha256()
    hasher.update(bundle.dataset_id.encode("utf-8"))
    hasher.update(bundle.manifest_hash.encode("utf-8"))
    for row_id, label in zip(bundle.row_ids, bundle.y):
        hasher.update(f"{row_id}:{label}".encode("utf-8"))
    return hasher.hexdigest()


def create_stratified_split(
    bundle: DatasetBundle,
    seed: int,
    protocol_id: str = "paper_interpretation_clean_v1",
    train_ratio: float = 0.70,
    val_ratio: float = 0.20,
    test_ratio: float = 0.10,
) -> SplitBundle:
    """
    Create a stratified 70/20/10 split on row IDs using scikit-learn.
    First separates test (10%), then validation (20/90 ~ 22.22% of the remaining 90%).
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    df = bundle.frame[[bundle.row_id_column, bundle.label_column]].copy()
    y = df[bundle.label_column].values
    ids = df[bundle.row_id_column].values

    # Step 1: 10% test split stratified
    rem_ids, test_ids, rem_y, test_y = train_test_split(
        ids,
        y,
        test_size=test_ratio,
        stratify=y,
        random_state=seed,
    )

    # Step 2: 20% validation from remaining 90%
    val_fraction_of_rem = val_ratio / (train_ratio + val_ratio)
    train_ids, val_ids, train_y, val_y = train_test_split(
        rem_ids,
        rem_y,
        test_size=val_fraction_of_rem,
        stratify=rem_y,
        random_state=seed,
    )

    # Count breakdown
    def counts(arr_y: np.ndarray) -> Dict[str, int]:
        unique, counts = np.unique(arr_y, return_counts=True)
        return {str(k): int(v) for k, v in zip(unique, counts)}

    class_breakdown = {
        "train": counts(train_y),
        "validation": counts(val_y),
        "test": counts(test_y),
        "total": counts(y),
    }

    split = SplitBundle(
        protocol_id=protocol_id,
        dataset_id=bundle.dataset_id,
        split_seed=seed,
        dataset_fingerprint=compute_dataset_fingerprint(bundle),
        train_ids=train_ids.tolist(),
        validation_ids=val_ids.tolist(),
        test_ids=test_ids.tolist(),
        class_counts=class_breakdown,
    )
    split.validate_integrity(bundle)
    return split


def save_split(split: SplitBundle, out_dir: Path = Path("outputs/splits")) -> Path:
    """
    Save split configuration to disk in JSON format.
    Location: out_dir / protocol_id / dataset_id / seed_{split_seed}.json
    """
    target_dir = out_dir / split.protocol_id / split.dataset_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"seed_{split.split_seed}.json"

    data = asdict(split)
    target_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return target_file


def load_split(split_file: Path, bundle: Optional[DatasetBundle] = None) -> SplitBundle:
    """
    Load a split configuration from JSON and validate its integrity against the dataset bundle.
    """
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    data = json.loads(split_file.read_text(encoding="utf-8"))
    split = SplitBundle(
        protocol_id=data["protocol_id"],
        dataset_id=data["dataset_id"],
        split_seed=int(data["split_seed"]),
        dataset_fingerprint=data["dataset_fingerprint"],
        train_ids=list(data["train_ids"]),
        validation_ids=list(data["validation_ids"]),
        test_ids=list(data["test_ids"]),
        class_counts=data["class_counts"],
    )
    split.validate_integrity(bundle)
    return split

