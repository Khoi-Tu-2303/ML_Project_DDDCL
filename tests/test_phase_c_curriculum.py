"""
Unit tests for Phase C: Curriculum Geometry, KDE Partition, Resampling, Scoring, and SPL Pace Decisions.
Enforces Gate C requirements from docs/07_VERIFICATION.md.
"""

from pathlib import Path
import numpy as np
import pytest

from dddcl.data.hf import load_dataset_from_release
from dddcl.data.splits import create_stratified_split
from dddcl.curriculum.geometry import (
    compute_centroids,
    compute_class_distances,
    compute_geometry,
)
from dddcl.curriculum.partition import (
    kde_level_partition_v1,
    partition_by_class,
)
from dddcl.curriculum.resample import (
    resample_curriculum_bins,
)
from dddcl.curriculum.scoring import (
    create_curriculum,
)
from dddcl.training.pace import (
    compute_pace_decision,
    update_lambda,
    should_reintroduce_hard,
)


# 1. Centroid & distance hand-calculated oracle test
def test_centroid_and_distances_hand_calculated():
    # Points (0, 0) and (2, 0) in class 0
    X = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=float)
    y = np.array([0, 0], dtype=int)

    centroids = compute_centroids(X, y)
    assert np.allclose(centroids[0], np.array([1.0, 0.0])), f"Expected centroid (1, 0), got {centroids[0]}"

    raw_dists, norm_dists, class_norms = compute_class_distances(X, y, centroids)
    # Both points are distance 1.0 from (1, 0)
    assert np.allclose(raw_dists, np.array([1.0, 1.0]))
    # Class L2 norm is sqrt(1^2 + 1^2) = sqrt(2)
    assert np.isclose(class_norms[0], np.sqrt(2.0))
    # Normalized distances are 1 / sqrt(2) ~ 0.7071
    assert np.allclose(norm_dists, np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)]))


def test_geometry_degenerate_cases():
    # Single sample in a class
    X_single = np.array([[5.0, 5.0]], dtype=float)
    y_single = np.array([0], dtype=int)
    geom_single = compute_geometry(X_single, y_single)
    assert np.allclose(geom_single.centroids[0], np.array([5.0, 5.0]))
    assert geom_single.raw_distances[0] == 0.0
    assert geom_single.normalized_distances[0] == 0.0

    # Multiple identical points in a class (norm zero)
    X_ident = np.array([[3.0, 3.0], [3.0, 3.0]], dtype=float)
    y_ident = np.array([1, 1], dtype=int)
    geom_ident = compute_geometry(X_ident, y_ident)
    assert np.allclose(geom_ident.centroids[1], np.array([3.0, 3.0]))
    assert np.allclose(geom_ident.raw_distances, [0.0, 0.0])
    assert np.allclose(geom_ident.normalized_distances, [0.0, 0.0])


# 2. Density scoring oracle test
def test_density_scoring_oracle():
    # Groups of sizes [2, 5, 1]
    # Group 0: 2 samples
    # Group 1: 5 samples
    # Group 2: 1 sample
    sample_ids = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"]
    bin_assignments = np.array([0, 0, 1, 1, 1, 1, 1, 2], dtype=int)
    norm_dists = np.array([0.2, 0.3, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5], dtype=float)

    curriculum = create_curriculum(
        strategy="ddcl_density",
        sample_ids=sample_ids,
        normalized_distances=norm_dists,
        bin_assignments=bin_assignments,
    )

    ordered = curriculum.ordered_train_ids
    # Group 1 (5 samples) must appear first, then Group 0 (2 samples), then Group 2 (1 sample)
    group1_samples = {"s2", "s3", "s4", "s5", "s6"}
    group0_samples = {"s0", "s1"}
    group2_samples = {"s7"}

    assert set(ordered[0:5]) == group1_samples, "Group with 5 samples should be first"
    assert set(ordered[5:7]) == group0_samples, "Group with 2 samples should be second"
    assert set(ordered[7:8]) == group2_samples, "Group with 1 sample should be last"


# 3. Point scoring oracle test
def test_point_scoring_oracle():
    # Distances [0.8, 0.1, 0.5] with IDs ['id0', 'id1', 'id2']
    sample_ids = ["id0", "id1", "id2"]
    norm_dists = np.array([0.8, 0.1, 0.5], dtype=float)
    bin_assignments = np.array([0, 0, 0], dtype=int)

    curriculum = create_curriculum(
        strategy="ddcl_point",
        sample_ids=sample_ids,
        normalized_distances=norm_dists,
        bin_assignments=bin_assignments,
    )

    # Sorted by distance ascending: id1 (0.1) -> id2 (0.5) -> id0 (0.8)
    assert curriculum.ordered_train_ids == ["id1", "id2", "id0"]

    # Tie-break consistency test: identical distances sorted by ID string
    sample_ids_tie = ["b_sample", "a_sample"]
    norm_dists_tie = np.array([0.4, 0.4], dtype=float)
    curriculum_tie = create_curriculum(
        strategy="ddcl_point",
        sample_ids=sample_ids_tie,
        normalized_distances=norm_dists_tie,
        bin_assignments=np.array([0, 0]),
    )
    assert curriculum_tie.ordered_train_ids == ["a_sample", "b_sample"]


# 4. Partition coverage and monotonicity test
def test_partition_coverage_and_monotonicity():
    rng = np.random.RandomState(42)
    distances = rng.exponential(scale=1.0, size=100)
    bounds, bins, fallback = kde_level_partition_v1(distances, n_levels=6)

    # Monotonically increasing bounds
    assert all(bounds[i] < bounds[i + 1] for i in range(len(bounds) - 1))
    # Full coverage: each sample assigned to a valid bin
    n_bins = len(bounds) - 1
    assert len(bins) == 100
    assert np.all(bins >= 0) and np.all(bins < n_bins)
    assert not fallback

    # Constant distance fallback test
    const_dists = np.full(50, 0.5)
    bounds_c, bins_c, fallback_c = kde_level_partition_v1(const_dists, n_levels=6)
    assert fallback_c
    assert len(bins_c) == 50
    assert np.all(bins_c == 0)


# 5. Pace decision oracle test
def test_pace_decision_oracle():
    # Losses [0.2, 0.8, 0.4, 0.65], lambda = 0.5
    losses = [0.2, 0.8, 0.4, 0.65]
    decision = compute_pace_decision(losses, lambda_val=0.5)

    # Easy: loss < 0.5 -> batch 0 (0.2), batch 2 (0.4)
    # Sorted: [0, 2]
    assert decision.ordered_easy_batches == [0, 2]
    assert set(decision.easy_batch_ids) == {0, 2}

    # Hard: loss >= 0.5 -> batch 1 (0.8), batch 3 (0.65)
    # Sorted: 0.65 < 0.8 -> [3, 1]
    assert decision.ordered_hard_batches == [3, 1]
    assert set(decision.hard_batch_ids) == {1, 3}
    assert decision.fallback_reason is None

    # Strict equality test: loss == lambda belongs to HARD
    losses_eq = [0.3, 0.5, 0.7]
    decision_eq = compute_pace_decision(losses_eq, lambda_val=0.5)
    assert decision_eq.ordered_easy_batches == [0]
    assert decision_eq.ordered_hard_batches == [1, 2]  # 0.5 is in hard


def test_pace_decision_empty_easy_fallback():
    # All losses >= lambda: [0.7, 0.9, 0.6] with lambda = 0.5
    losses = [0.7, 0.9, 0.6]
    decision = compute_pace_decision(losses, lambda_val=0.5, empty_easy_fallback="min_loss_batch")

    # Min loss batch is index 2 (loss 0.6)
    assert decision.ordered_easy_batches == [2]
    assert decision.ordered_hard_batches == [0, 1]  # remaining hard: 0.7, 0.9
    assert decision.fallback_reason == "empty_easy_picked_min_loss"


# 6. Lambda update and cap test
def test_lambda_update_and_cap():
    current_lambda = 0.4
    growth = 1.1
    lambda_max = 0.6

    history = [current_lambda]
    for _ in range(10):
        current_lambda = update_lambda(current_lambda, lambda_max=lambda_max, growth=growth)
        history.append(current_lambda)

    # Increases monotonically
    assert all(history[i] <= history[i + 1] for i in range(len(history) - 1))
    # Never exceeds lambda_max = 0.6
    assert all(val <= 0.6 + 1e-9 for val in history)
    assert np.isclose(history[-1], 0.6)


# 7. Real dataset curriculum end-to-end test
def test_real_dataset_curriculum_strategies():
    bundle = load_dataset_from_release(Path("hf_release"), "haberman")
    split = create_stratified_split(bundle, seed=42)

    df_train = bundle.frame.set_index(bundle.row_id_column).loc[split.train_ids]
    X_train = df_train[bundle.feature_columns].values
    y_train = df_train[bundle.label_column].values

    # Step 1: Geometry
    geom = compute_geometry(X_train, y_train)
    assert len(geom.centroids) == 2
    assert len(geom.normalized_distances) == len(X_train)

    # Step 2: Partition
    partition = partition_by_class(geom.normalized_distances, y_train, n_levels=6)
    assert len(partition.bin_assignments) == len(X_train)

    # Step 3: Resample
    resample = resample_curriculum_bins(
        X_raw=X_train,
        y=y_train,
        sample_ids=split.train_ids,
        bin_assignments=partition.bin_assignments,
        normalized_distances=geom.normalized_distances,
        centroids=geom.centroids,
        class_norms=geom.class_norms,
        dataset_id=bundle.dataset_id,
        smote_policy="none",
    )
    assert len(resample.sample_ids) == len(X_train)

    # Step 4: Test all 6 strategies
    strategies = [
        "no_curriculum",
        "ddcl_density",
        "ddcl_point",
        "self_paced",
        "sp_ddcl_density",
        "sp_ddcl_point",
    ]
    for strat in strategies:
        res = create_curriculum(
            strategy=strat,
            sample_ids=resample.sample_ids,
            normalized_distances=resample.normalized_distances,
            bin_assignments=resample.bin_assignments,
            seed=42,
        )
        assert len(res.ordered_train_ids) == len(X_train)
        assert set(res.ordered_train_ids) == set(split.train_ids)
        assert len(res.scores) == len(X_train)
