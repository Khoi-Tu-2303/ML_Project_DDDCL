"""
Data pipeline components for DDDCL.
Handles loading, schema verification, stratified splitting, and anti-leakage preprocessing.
"""

from dddcl.data.hf import (
    DatasetBundle,
    checked_config,
    load_dataset_from_hf,
    load_dataset_from_release,
    validate_csv_bytes,
)
from dddcl.data.splits import (
    SplitBundle,
    compute_dataset_fingerprint,
    create_stratified_split,
    load_split,
    save_split,
)
from dddcl.data.preprocess import (
    ProcessedSplit,
    get_scaler,
    prepare_processed_split,
)

__all__ = [
    "DatasetBundle",
    "checked_config",
    "load_dataset_from_hf",
    "load_dataset_from_release",
    "validate_csv_bytes",
    "SplitBundle",
    "compute_dataset_fingerprint",
    "create_stratified_split",
    "load_split",
    "save_split",
    "ProcessedSplit",
    "get_scaler",
    "prepare_processed_split",
]

