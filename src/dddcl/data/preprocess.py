"""
Train-only preprocessing and anti-leakage feature scaling.
Enforces strict train-only fitting for scalers and transformers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from dddcl.data.hf import DatasetBundle
from dddcl.data.splits import SplitBundle


@dataclass(frozen=True)
class ProcessedSplit:
    """
    Container for feature matrices and labels partitioned and scaled for ML models.
    Guarantees scaler was fit ONLY on train.
    """
    dataset_id: str
    protocol_id: str
    split_seed: int
    feature_names: List[str]
    # Scaled feature matrices
    X_train_scaled: np.ndarray
    y_train: np.ndarray
    train_ids: List[str]
    X_val_scaled: np.ndarray
    y_val: np.ndarray
    val_ids: List[str]
    X_test_scaled: np.ndarray
    y_test: np.ndarray
    test_ids: List[str]
    # Raw unscaled features for curriculum geometry
    X_train_raw: np.ndarray
    X_val_raw: np.ndarray
    X_test_raw: np.ndarray
    # Scaler artifact
    scaler: Any


def get_scaler(scaler_type: str = "standard") -> Any:
    """Factory for standard preprocessing scalers."""
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    elif scaler_type == "minmax":
        return MinMaxScaler()
    elif scaler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scaler_type: '{scaler_type}'. Choose from 'standard', 'robust', 'minmax', 'none'.")


def prepare_processed_split(
    bundle: DatasetBundle,
    split: SplitBundle,
    scaler_type: str = "standard",
) -> ProcessedSplit:
    """
    Extract train/validation/test partitions by row ID, fit scaler STRICTLY on train,
    and transform all partitions. Returns ProcessedSplit with both raw and scaled representations.
    """
    split.validate_integrity(bundle)

    # Index dataframe by row_id for stable extraction
    df = bundle.frame.set_index(bundle.row_id_column)
    cols = bundle.feature_columns
    y_col = bundle.label_column

    # Extract raw arrays
    df_train = df.loc[split.train_ids]
    df_val = df.loc[split.validation_ids]
    df_test = df.loc[split.test_ids]

    X_train_raw = df_train[cols].values.astype(np.float32)
    y_train = df_train[y_col].values.astype(np.int64)

    X_val_raw = df_val[cols].values.astype(np.float32)
    y_val = df_val[y_col].values.astype(np.int64)

    X_test_raw = df_test[cols].values.astype(np.float32)
    y_test = df_test[y_col].values.astype(np.int64)

    # Fit scaler ONLY on train
    scaler = get_scaler(scaler_type)
    if scaler is not None:
        scaler.fit(X_train_raw)
        X_train_scaled = scaler.transform(X_train_raw).astype(np.float32)
        X_val_scaled = scaler.transform(X_val_raw).astype(np.float32)
        X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)
    else:
        X_train_scaled = X_train_raw.copy()
        X_val_scaled = X_val_raw.copy()
        X_test_scaled = X_test_raw.copy()

    return ProcessedSplit(
        dataset_id=bundle.dataset_id,
        protocol_id=split.protocol_id,
        split_seed=split.split_seed,
        feature_names=cols,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        train_ids=list(split.train_ids),
        X_val_scaled=X_val_scaled,
        y_val=y_val,
        val_ids=list(split.validation_ids),
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        test_ids=list(split.test_ids),
        X_train_raw=X_train_raw,
        X_val_raw=X_val_raw,
        X_test_raw=X_test_raw,
        scaler=scaler,
    )

