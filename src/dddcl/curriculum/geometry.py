"""
Centroid determination, Euclidean distance computation, and distance normalization.
Implements Decision D04 & Assumption A03 with degenerate class handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class GeometryResult:
    """
    Immutable container for computed centroids, Euclidean distances, and normalized distances.
    """
    centroids: Dict[int, np.ndarray]
    raw_distances: np.ndarray
    normalized_distances: np.ndarray
    class_norms: Dict[int, float]


def compute_centroids(X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Compute Euclidean mean centroid mu_c = (1/|X_c|) * sum(x) for each class c.
    Handles single-point and degenerate classes.
    """
    if len(X) != len(y):
        raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

    unique_classes = np.unique(y)
    centroids: Dict[int, np.ndarray] = {}

    for c in unique_classes:
        mask = (y == c)
        X_c = X[mask]
        if len(X_c) == 0:
            raise ValueError(f"Class {c} has zero samples")
        # Analytical mean centroid
        centroids[int(c)] = np.mean(X_c, axis=0)

    return centroids


def compute_class_distances(
    X: np.ndarray,
    y: np.ndarray,
    centroids: Dict[int, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, Dict[int, float]]:
    """
    Compute Euclidean distances r_i = ||x_i - mu_c||_2 and normalized distances
    r_hat_i = r_i / ||r_c||_2 per class.
    
    Degenerate handling:
    - If a class has zero distance norm (all samples identical or single sample),
      normalized distance is set to 0.0.
    """
    n_samples = len(X)
    raw_distances = np.zeros(n_samples, dtype=np.float64)
    normalized_distances = np.zeros(n_samples, dtype=np.float64)
    class_norms: Dict[int, float] = {}

    for c, mu_c in centroids.items():
        mask = (y == c)
        indices = np.where(mask)[0]
        X_c = X[indices]

        # Euclidean distance to centroid
        diff = X_c - mu_c
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        raw_distances[indices] = dists

        # Class L2 norm of distances: sqrt(sum(r_j^2))
        l2_norm = float(np.sqrt(np.sum(dists ** 2)))
        class_norms[c] = l2_norm

        if l2_norm > 1e-12:
            normalized_distances[indices] = dists / l2_norm
        else:
            # All samples at centroid or single sample
            normalized_distances[indices] = 0.0

    return raw_distances, normalized_distances, class_norms


def compute_geometry(X: np.ndarray, y: np.ndarray) -> GeometryResult:
    """
    End-to-end geometry pipeline: centroids -> raw distances -> normalized distances.
    """
    centroids = compute_centroids(X, y)
    raw_dists, norm_dists, class_norms = compute_class_distances(X, y, centroids)
    return GeometryResult(
        centroids=centroids,
        raw_distances=raw_dists,
        normalized_distances=norm_dists,
        class_norms=class_norms,
    )

