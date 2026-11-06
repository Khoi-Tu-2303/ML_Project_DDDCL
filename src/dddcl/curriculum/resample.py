"""
Train-only SMOTE resampling within pooled curriculum bins.
Implements Decision D06 & Assumption A05 (within_bin_minority_v1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE


@dataclass(frozen=True)
class ResampleResult:
    """
    Immutable container for resampled features, labels, IDs, bins, and normalized distances.
    """
    X: np.ndarray
    y: np.ndarray
    sample_ids: List[str]
    bin_assignments: np.ndarray
    normalized_distances: np.ndarray
    n_synthetic: int
    resampling_report: Dict[str, Any]


def resample_curriculum_bins(
    X_raw: np.ndarray,
    y: np.ndarray,
    sample_ids: List[str],
    bin_assignments: np.ndarray,
    normalized_distances: np.ndarray,
    centroids: Dict[int, np.ndarray],
    class_norms: Dict[int, float],
    dataset_id: str,
    smote_policy: str = "none",
    k_neighbors: int = 3,
    seed: int = 42,
) -> ResampleResult:
    """
    Apply train-only SMOTE oversampling within pooled curriculum bins.
    
    If smote_policy == 'none':
        Returns original arrays untouched.
    If smote_policy == 'within_bin_minority_v1':
        Oversamples minority classes within bins that have >= 2 classes and >= k+1 minority samples.
        Labels synthetic samples with clear traceable IDs (e.g. 'synth_...').
    """
    if smote_policy == "none":
        return ResampleResult(
            X=X_raw.copy(),
            y=y.copy(),
            sample_ids=list(sample_ids),
            bin_assignments=bin_assignments.copy(),
            normalized_distances=normalized_distances.copy(),
            n_synthetic=0,
            resampling_report={"policy": "none", "synthetic_generated": 0},
        )

    if smote_policy != "within_bin_minority_v1":
        raise ValueError(f"Unknown smote_policy: '{smote_policy}'. Supported: 'none', 'within_bin_minority_v1'.")

    all_X = [X_raw]
    all_y = [y]
    all_ids = list(sample_ids)
    all_bins = [bin_assignments]
    all_dists = [normalized_distances]

    unique_bins = np.unique(bin_assignments)
    bin_reports: List[Dict[str, Any]] = []
    total_synthetic = 0

    for b in unique_bins:
        mask = (bin_assignments == b)
        X_b = X_raw[mask]
        y_b = y[mask]
        classes_in_bin, counts_in_bin = np.unique(y_b, return_counts=True)

        # Check eligibility: >= 2 classes
        if len(classes_in_bin) < 2:
            bin_reports.append({"bin": int(b), "status": "single_class_skipped", "n_samples": len(y_b)})
            continue

        min_count = int(np.min(counts_in_bin))
        max_count = int(np.max(counts_in_bin))

        if min_count == max_count:
            bin_reports.append({"bin": int(b), "status": "already_balanced", "n_samples": len(y_b)})
            continue

        # Check if minority has at least k_neighbors + 1 samples
        if min_count < k_neighbors + 1:
            bin_reports.append({
                "bin": int(b),
                "status": f"insufficient_samples_for_k{k_neighbors}",
                "min_count": min_count,
                "n_samples": len(y_b)
            })
            continue

        # Apply SMOTE to this bin
        try:
            smote = SMOTE(k_neighbors=k_neighbors, random_state=seed)
            X_res_b, y_res_b = smote.fit_resample(X_b, y_b)
            n_new = len(y_res_b) - len(y_b)

            if n_new > 0:
                X_synth = X_res_b[len(y_b):]
                y_synth = y_res_b[len(y_b):]

                # Compute normalized distance for synthetic samples using class centroids
                synth_norm_dists = np.zeros(n_new, dtype=np.float64)
                synth_ids: List[str] = []

                for idx, (x_s, y_s) in enumerate(zip(X_synth, y_synth)):
                    c = int(y_s)
                    mu_c = centroids[c]
                    raw_d = float(np.sqrt(np.sum((x_s - mu_c) ** 2)))
                    norm_scale = class_norms.get(c, 1.0)
                    synth_norm_dists[idx] = raw_d / norm_scale if norm_scale > 1e-12 else 0.0
                    synth_ids.append(f"synth_{dataset_id}_bin{b}_c{c}_{total_synthetic + idx}")

                all_X.append(X_synth)
                all_y.append(y_synth)
                all_ids.extend(synth_ids)
                all_bins.append(np.full(n_new, b, dtype=int))
                all_dists.append(synth_norm_dists)
                total_synthetic += n_new

                bin_reports.append({
                    "bin": int(b),
                    "status": "smote_applied",
                    "synthetic_generated": n_new,
                    "before": dict(zip([int(x) for x in classes_in_bin], [int(x) for x in counts_in_bin])),
                })
        except Exception as e:
            bin_reports.append({"bin": int(b), "status": "smote_failed", "error": str(e)})

    return ResampleResult(
        X=np.concatenate(all_X, axis=0),
        y=np.concatenate(all_y, axis=0),
        sample_ids=all_ids,
        bin_assignments=np.concatenate(all_bins, axis=0),
        normalized_distances=np.concatenate(all_dists, axis=0),
        n_synthetic=total_synthetic,
        resampling_report={
            "policy": smote_policy,
            "synthetic_generated": total_synthetic,
            "bin_details": bin_reports,
        },
    )

