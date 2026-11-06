"""Unit tests and gate checks for Phase E: Ensemble Evaluation & Alignment.

Verifies according to docs/07_VERIFICATION.md (E / ensemble):
1. Hand-calculated hard voting: [0,0,1,1,1,1] -> 1.
2. Deterministic 3-3 tie-breaking using train class prior, strictly independent of y_test.
3. ID alignment and automatic re-ordering when predictions arrive out-of-order.
4. Integrity rejections for missing IDs, duplicate IDs, or mismatched y_true labels.
5. Pairwise disagreement diversity calculation.
6. End-to-end controlled 6-NN ensemble functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from dddcl.ensemble import (
    MemberPrediction,
    align_member_predictions,
    compute_pairwise_disagreement,
    evaluate_controlled_6nn_ensemble,
    hard_voting_with_tie_break,
)


class TestHardVotingAndTieBreak:
    """Verify voting mathematics and tie-breaking contracts."""

    def test_hand_calculated_votes(self) -> None:
        """Hand-calculated vote oracle: [0,0,1,1,1,1] -> 1."""
        # Shape: (2 samples, 6 members)
        # Sample 0: [0, 0, 1, 1, 1, 1] -> winner 1 (4 votes vs 2 votes)
        # Sample 1: [0, 0, 0, 1, 2, 2] -> winner 0 (3 votes vs 1 vs 2)
        votes = np.array([
            [0, 0, 1, 1, 1, 1],
            [0, 0, 0, 1, 2, 2],
        ], dtype=np.int64)

        preds, ties = hard_voting_with_tie_break(votes)
        assert preds[0] == 1, f"Expected 1, got {preds[0]}"
        assert preds[1] == 0, f"Expected 0, got {preds[1]}"
        assert ties == 0

    def test_tie_break_respects_train_prior_not_test(self) -> None:
        """3-3 tie broken by train class prior, completely agnostic to test labels."""
        # 3 votes for class 0, 3 votes for class 1
        votes = np.array([[0, 0, 0, 1, 1, 1]], dtype=np.int64)

        # Case A: Train set has more class 0 -> class 0 wins
        prior_favor_0 = {0: 100, 1: 20}
        preds_a, ties_a = hard_voting_with_tie_break(votes, train_class_prior=prior_favor_0)
        assert preds_a[0] == 0
        assert ties_a == 1

        # Case B: Train set has more class 1 -> class 1 wins
        prior_favor_1 = {0: 20, 1: 100}
        preds_b, ties_b = hard_voting_with_tie_break(votes, train_class_prior=prior_favor_1)
        assert preds_b[0] == 1
        assert ties_b == 1

        # Case C: Balanced train set -> tie broken by lowest label index (0 < 1)
        prior_equal = {0: 50, 1: 50}
        preds_c, ties_c = hard_voting_with_tie_break(votes, train_class_prior=prior_equal)
        assert preds_c[0] == 0
        assert ties_c == 1


class TestIDAlignmentAndReordering:
    """Verify that predictions from members are aligned strictly by row ID."""

    def test_automatic_reordering_by_id(self) -> None:
        """Shuffled prediction rows must be re-ordered to match target IDs."""
        target_ids = ["id_A", "id_B", "id_C"]
        y_true = [0, 1, 0]

        # Member 1 in natural order
        m1 = MemberPrediction(
            member_id="m1",
            row_ids=["id_A", "id_B", "id_C"],
            y_true=[0, 1, 0],
            y_pred=[0, 1, 0],
            probabilities=np.zeros((3, 2)),
        )

        # Member 2 in shuffled order: C, A, B
        m2 = MemberPrediction(
            member_id="m2",
            row_ids=["id_C", "id_A", "id_B"],
            y_true=[0, 0, 1],
            y_pred=[1, 0, 1],  # id_C -> 1, id_A -> 0, id_B -> 1
            probabilities=np.zeros((3, 2)),
        )

        aligned_ids, aligned_true, votes = align_member_predictions(
            [m1, m2],
            expected_ids=target_ids,
        )

        assert aligned_ids == target_ids
        assert aligned_true == y_true
        # Check that Member 2 predictions are mapped to correct sample rows:
        # id_A: m1=0, m2=0
        assert list(votes[0]) == [0, 0]
        # id_B: m1=1, m2=1
        assert list(votes[1]) == [1, 1]
        # id_C: m1=0, m2=1
        assert list(votes[2]) == [0, 1]

    def test_missing_id_raises_value_error(self) -> None:
        """Member missing a required row ID raises ValueError."""
        m1 = MemberPrediction("m1", ["id_A", "id_B"], [0, 1], [0, 1], np.zeros((2, 2)))
        m2 = MemberPrediction("m2", ["id_A"], [0], [0], np.zeros((1, 2)))

        with pytest.raises(ValueError, match="row ID mismatch"):
            align_member_predictions([m1, m2], expected_ids=["id_A", "id_B"])

    def test_duplicate_id_raises_value_error(self) -> None:
        """Member containing duplicate row IDs raises ValueError."""
        m_dup = MemberPrediction("m_dup", ["id_A", "id_A"], [0, 0], [0, 0], np.zeros((2, 2)))
        with pytest.raises(ValueError, match="duplicate"):
            align_member_predictions([m_dup])

    def test_inconsistent_ground_truth_raises_value_error(self) -> None:
        """Members disagreeing on true label for the same row ID raises ValueError."""
        m1 = MemberPrediction("m1", ["id_A"], [0], [0], np.zeros((1, 2)))
        m2 = MemberPrediction("m2", ["id_A"], [1], [0], np.zeros((1, 2)))  # y_true is 1 vs 0

        with pytest.raises(ValueError, match="Inconsistent ground truth label"):
            align_member_predictions([m1, m2], expected_ids=["id_A"])


class TestDiversityAndControlledEnsemble:
    """Verify diversity metrics and controlled 6-NN baseline."""

    def test_pairwise_disagreement_calculation(self) -> None:
        """Compute exact pairwise disagreement fraction."""
        # 2 samples, 2 members
        # Sample 0: agree [0, 0]
        # Sample 1: disagree [0, 1]
        # Disagreement rate = 1 / 2 = 0.5
        votes = np.array([[0, 0], [0, 1]], dtype=np.int64)
        dis = compute_pairwise_disagreement(votes)
        assert np.isclose(dis, 0.5)

        # 3 identical members -> 0.0
        votes_identical = np.zeros((4, 3), dtype=np.int64)
        assert compute_pairwise_disagreement(votes_identical) == 0.0

    def test_controlled_6nn_ensemble_synthetic(self) -> None:
        """End-to-end test of controlled 6-NN ensemble on synthetic linearly separable data."""
        np.random.seed(42)
        X_train = np.random.randn(20, 2).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(np.int64)
        train_ids = [f"train_{i}" for i in range(20)]

        X_test = np.random.randn(10, 2).astype(np.float32)
        y_test = (X_test[:, 0] > 0).astype(np.int64)
        test_ids = [f"test_{i}" for i in range(10)]

        res = evaluate_controlled_6nn_ensemble(
            processed_X_train=X_train,
            y_train=y_train,
            train_ids=train_ids,
            processed_X_test=X_test,
            y_test=y_test,
            test_ids=test_ids,
            dataset_id="test_synthetic",
            base_seed=42,
            epochs=3,
        )

        assert res.ensemble_type == "controlled_6nn_ensemble"
        assert res.num_members == 6
        assert len(res.member_ids) == 6
        assert "accuracy" in res.metrics
        assert res.metrics["sample_count"] == 10
        assert res.actual_compute_cost["total_optimizer_steps"] > 0
