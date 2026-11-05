"""
Curriculum scoring and sample ordering strategies.
Implements DDCL-Density, DDCL-Point, No Curriculum, and Self-Paced baselines.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class CurriculumResult:
    """
    Immutable container representing the curriculum order of samples for training.
    Strictly separates IDs, distances, group IDs, and scores.
    """
    strategy: str
    ordered_train_ids: List[str]
    distances: Dict[str, float]
    group_ids: Dict[str, int]
    scores: Dict[str, float]
    resampling_report: Dict[str, Any]
    policy_hash: str


def compute_policy_hash(strategy: str, params: Dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of curriculum configuration."""
    hasher = hashlib.sha256()
    hasher.update(strategy.encode("utf-8"))
    for k in sorted(params.keys()):
        hasher.update(f"{k}:{params[k]}".encode("utf-8"))
    return hasher.hexdigest()[:16]


def create_curriculum(
    strategy: str,
    sample_ids: List[str],
    normalized_distances: np.ndarray,
    bin_assignments: np.ndarray,
    seed: int = 42,
    resampling_report: Optional[Dict[str, Any]] = None,
) -> CurriculumResult:
    """
    Create ordered train IDs and scores for one of the 6 strategies:
    - no_curriculum
    - ddcl_density
    - ddcl_point
    - self_paced (uses random base order, dynamic pace in training)
    - sp_ddcl_density (uses density base order, dynamic pace in training)
    - sp_ddcl_point (uses point base order, dynamic pace in training)
    """
    n_samples = len(sample_ids)
    if len(normalized_distances) != n_samples or len(bin_assignments) != n_samples:
        raise ValueError("Length mismatch between sample_ids, distances, and bin_assignments")

    dist_map = {sid: float(d) for sid, d in zip(sample_ids, normalized_distances)}
    bin_map = {sid: int(b) for sid, b in zip(sample_ids, bin_assignments)}
    report = resampling_report or {"policy": "none"}

    valid_strategies = {
        "no_curriculum",
        "ddcl_density",
        "ddcl_point",
        "self_paced",
        "sp_ddcl_density",
        "sp_ddcl_point",
    }
    if strategy not in valid_strategies:
        raise ValueError(f"Unknown strategy: '{strategy}'. Supported: {sorted(valid_strategies)}")

    if strategy in {"no_curriculum", "self_paced"}:
        # Fixed pseudo-random permutation derived from seed
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n_samples)
        ordered_ids = [sample_ids[i] for i in perm]
        scores = {sid: float(rank) for rank, sid in enumerate(ordered_ids)}

    elif strategy in {"ddcl_point", "sp_ddcl_point"}:
        # Point scoring: sort all samples by normalized distance ascending (shorter distance = easier)
        # Stable tie-break by sample_id string
        items = list(zip(sample_ids, normalized_distances))
        items.sort(key=lambda item: (item[1], str(item[0])))
        ordered_ids = [item[0] for item in items]
        scores = {sid: float(d) for sid, d in items}

    elif strategy in {"ddcl_density", "sp_ddcl_density"}:
        # Density scoring: sort quantiles/bins by sample count descending (higher count = easier)
        # Tie-break bins by bin index ascending
        bin_counts = Counter(bin_assignments)
        sorted_bins = sorted(bin_counts.keys(), key=lambda b: (-bin_counts[b], b))

        # Group sample items by bin
        bin_to_samples: Dict[int, List[Tuple[str, float]]] = {b: [] for b in sorted_bins}
        for sid, b, d in zip(sample_ids, bin_assignments, normalized_distances):
            bin_to_samples[b].append((sid, float(d)))

        # Within each bin, sort stably by distance ascending, then sample_id
        ordered_ids = []
        scores = {}
        current_rank = 0
        for b in sorted_bins:
            samples_in_bin = bin_to_samples[b]
            samples_in_bin.sort(key=lambda s: (s[1], str(s[0])))
            density_score = float(bin_counts[b])
            for sid, _ in samples_in_bin:
                ordered_ids.append(sid)
                scores[sid] = density_score  # higher density = higher score = easier
                current_rank += 1

    policy_hash = compute_policy_hash(strategy, {"seed": seed, "n_samples": n_samples})

    return CurriculumResult(
        strategy=strategy,
        ordered_train_ids=ordered_ids,
        distances=dist_map,
        group_ids=bin_map,
        scores=scores,
        resampling_report=report,
        policy_hash=policy_hash,
    )
