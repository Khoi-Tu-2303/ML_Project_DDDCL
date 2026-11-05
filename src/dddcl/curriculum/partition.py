"""
KDE-based Quantile and Density Partitioning.
Implements Decision D05 & Assumption A04 (kde_level_partition_v1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import gaussian_kde


@dataclass(frozen=True)
class PartitionResult:
    """
    Immutable container for partition boundaries and sample bin assignments.
    """
    boundaries: Dict[int, List[float]]  # class_id -> list of bin boundaries [b0, b1, ...]
    bin_assignments: np.ndarray        # (n_samples,) integer local bin indices 0..M-1
    n_bins_per_class: Dict[int, int]
    fallback_classes: List[int]


def kde_level_partition_v1(
    distances: np.ndarray,
    n_levels: int = 6,
    grid_points: int = 1000,
    tolerance: float = 1e-4,
) -> Tuple[List[float], np.ndarray, bool]:
    """
    Partition 1D normalized distances using horizontal density slices from a KDE estimate.
    
    Returns:
    - boundaries: Sorted monotonic list of floats defining bin edges [b0, b1, ..., bM]
    - bin_indices: Integer array of bin assignment for each sample (0 <= idx < len(boundaries)-1)
    - used_fallback: True if constant distance or KDE failure forced a single-bin fallback
    """
    n = len(distances)
    if n == 0:
        return [0.0, 1.0], np.array([], dtype=int), True

    min_val = float(np.min(distances))
    max_val = float(np.max(distances))

    # Degenerate case: all distances identical or variance near zero
    if max_val - min_val < 1e-7 or n < 3:
        boundaries = [min_val - 1e-5, max_val + 1e-5]
        bin_indices = np.zeros(n, dtype=int)
        return boundaries, bin_indices, True

    # Fit 1D Gaussian KDE
    try:
        kde = gaussian_kde(distances, bw_method="scott")
    except Exception:
        boundaries = [min_val - 1e-5, max_val + 1e-5]
        bin_indices = np.zeros(n, dtype=int)
        return boundaries, bin_indices, True

    # Grid covering observed range with slight padding
    pad = (max_val - min_val) * 0.05
    x_grid = np.linspace(max(0.0, min_val - pad), max_val + pad, grid_points)
    density = kde.evaluate(x_grid)

    peak_density = float(np.max(density))
    if peak_density < 1e-12:
        boundaries = [min_val - 1e-5, max_val + 1e-5]
        bin_indices = np.zeros(n, dtype=int)
        return boundaries, bin_indices, True

    # Find horizontal crossing points for levels k = 1..n_levels-1
    crossings: List[float] = [min_val, max_val]
    for k in range(1, n_levels):
        target_density = peak_density * (k / n_levels)
        # Sign changes indicate crossing
        diff = density - target_density
        sign_changes = np.where(np.diff(np.sign(diff)))[0]
        for idx in sign_changes:
            # Linear interpolation for precise x coordinate
            x0, x1 = x_grid[idx], x_grid[idx + 1]
            y0, y1 = diff[idx], diff[idx + 1]
            if abs(y1 - y0) > 1e-12:
                x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
                if min_val <= x_cross <= max_val:
                    crossings.append(float(x_cross))

    # Deduplicate and sort
    crossings = sorted(crossings)
    merged_bounds = [crossings[0]]
    for val in crossings[1:]:
        if val - merged_bounds[-1] > tolerance:
            merged_bounds.append(val)

    # Ensure at least 2 boundaries (1 bin)
    if len(merged_bounds) < 2:
        merged_bounds = [min_val - 1e-5, max_val + 1e-5]

    # Assign samples to bins: b_j <= r < b_{j+1}, last bin closed [b_{M-1}, b_M]
    n_bins = len(merged_bounds) - 1
    bin_indices = np.zeros(n, dtype=int)

    for i, val in enumerate(distances):
        # Clip to boundaries range for numerical safety
        v = np.clip(val, merged_bounds[0], merged_bounds[-1])
        # Find index
        idx = int(np.searchsorted(merged_bounds, v, side="right") - 1)
        idx = min(max(0, idx), n_bins - 1)
        bin_indices[i] = idx

    return merged_bounds, bin_indices, False


def partition_by_class(
    normalized_distances: np.ndarray,
    y: np.ndarray,
    n_levels: int = 6,
) -> PartitionResult:
    """
    Execute KDE level partition separately for each class, returning unified bin assignments.
    """
    n_samples = len(normalized_distances)
    bin_assignments = np.zeros(n_samples, dtype=int)
    boundaries: Dict[int, List[float]] = {}
    n_bins_per_class: Dict[int, int] = {}
    fallback_classes: List[int] = []

    unique_classes = np.unique(y)
    for c in unique_classes:
        mask = (y == c)
        indices = np.where(mask)[0]
        class_dists = normalized_distances[indices]

        bounds, local_bins, fallback = kde_level_partition_v1(class_dists, n_levels=n_levels)
        boundaries[int(c)] = bounds
        bin_assignments[indices] = local_bins
        n_bins_per_class[int(c)] = len(bounds) - 1
        if fallback:
            fallback_classes.append(int(c))

    return PartitionResult(
        boundaries=boundaries,
        bin_assignments=bin_assignments,
        n_bins_per_class=n_bins_per_class,
        fallback_classes=fallback_classes,
    )
