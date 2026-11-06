"""Ensemble evaluation module for DDDCL.

Implements protocols from docs/03_METHOD_SPEC.md, docs/04_PROTOCOLS.md,
and docs/07_VERIFICATION.md:
- Strict ID alignment across all ensemble members.
- DDCL 6-Strategy Ensemble with equal hard voting.
- Deterministic Tie-Breaking (Decision D09):
  * Ties broken by train class prior (higher prior wins).
  * If priors are equal, broken by lowest integer class label index.
  * Strictly independent of test set ground truth.
- Standard Paper Baseline Ensemble (1 NN + 1 SVM RBF).
- Controlled 6-NN Baseline Ensemble (6 independent random inits, no curriculum).
- Comprehensive diversity and compute cost accounting.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import torch

from .metrics import compute_classification_metrics, save_predictions_csv
from .models.mlp import create_mlp_model
from .training.loop import create_curriculum_batches, train_dddcl_model
from .curriculum.scoring import CurriculumResult


@dataclass(frozen=True)
class MemberPrediction:
    """Container for predictions of a single ensemble member."""
    member_id: str
    row_ids: list[str]
    y_true: list[int]
    y_pred: list[int]
    probabilities: np.ndarray


@dataclass
class EnsembleResult:
    """Container for ensemble evaluation results."""
    ensemble_type: str
    dataset_id: str
    seed: int
    num_members: int
    member_ids: list[str]
    metrics: dict[str, Any]
    tie_count: int = 0
    tie_ratio: float = 0.0
    diversity_stats: dict[str, Any] = field(default_factory=dict)
    actual_compute_cost: dict[str, Any] = field(default_factory=dict)


def load_predictions_csv(filepath: str | Path, member_id: str = "") -> MemberPrediction:
    """Load and validate predictions CSV."""
    p = Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {p}")

    df = pd.read_csv(p)
    required = {"row_id", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns in {p}. Expected {required}, found {list(df.columns)}")

    row_ids = [str(r) for r in df["row_id"].tolist()]
    y_true = [int(y) for y in df["y_true"].tolist()]
    y_pred = [int(y) for y in df["y_pred"].tolist()]

    prob_cols = [c for c in df.columns if c.startswith("prob_class_")]
    if prob_cols:
        probs = df[prob_cols].to_numpy(dtype=np.float32)
    else:
        probs = np.zeros((len(df), len(set(y_true))), dtype=np.float32)

    return MemberPrediction(
        member_id=member_id or p.stem,
        row_ids=row_ids,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probs,
    )


def align_member_predictions(
    members: Sequence[MemberPrediction],
    expected_ids: Sequence[str] | None = None,
) -> tuple[list[str], list[int], np.ndarray]:
    """Align predictions from multiple members by row ID.
    
    Returns
    -------
    tuple
        (aligned_row_ids, aligned_y_true, aligned_preds_matrix)
        where aligned_preds_matrix has shape (num_samples, num_members).
    """
    if not members:
        raise ValueError("No ensemble members provided.")

    ref_member = members[0]
    target_ids = list(expected_ids) if expected_ids is not None else ref_member.row_ids

    # Validate target IDs uniqueness
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("Target row IDs contain duplicates.")

    target_id_set = set(target_ids)
    num_samples = len(target_ids)
    num_members = len(members)

    aligned_preds = np.zeros((num_samples, num_members), dtype=np.int64)
    ref_true_map = dict(zip(ref_member.row_ids, ref_member.y_true))

    for m_idx, m in enumerate(members):
        m_set = set(m.row_ids)
        if len(m.row_ids) != len(m_set):
            raise ValueError(f"Member '{m.member_id}' contains duplicate row IDs.")
        if m_set != target_id_set:
            missing = target_id_set - m_set
            extra = m_set - target_id_set
            raise ValueError(
                f"Member '{m.member_id}' row ID mismatch! Missing: {len(missing)}, Extra: {len(extra)}"
            )

        m_dict = dict(zip(m.row_ids, m.y_pred))
        m_true = dict(zip(m.row_ids, m.y_true))

        for s_idx, rid in enumerate(target_ids):
            # Check y_true consistency
            if m_true[rid] != ref_true_map[rid]:
                raise ValueError(
                    f"Inconsistent ground truth label for row ID '{rid}' in member '{m.member_id}'!"
                )
            aligned_preds[s_idx, m_idx] = m_dict[rid]

    aligned_y_true = [ref_true_map[rid] for rid in target_ids]
    return target_ids, aligned_y_true, aligned_preds


def hard_voting_with_tie_break(
    votes_matrix: np.ndarray,
    train_class_prior: Mapping[int, int] | Mapping[int, float] | None = None,
    num_classes: int | None = None,
) -> tuple[np.ndarray, int]:
    """Execute majority voting across member predictions with deterministic tie-breaking.
    
    Parameters
    ----------
    votes_matrix : np.ndarray
        Array of shape (num_samples, num_members) containing discrete predictions.
    train_class_prior : Mapping[int, int | float] | None
        Sample counts or prior probabilities per class in the training split.
    num_classes : int | None
        Total number of classes.

    Returns
    -------
    tuple
        (ensemble_predictions, tie_count)
    """
    num_samples, num_members = votes_matrix.shape
    ensemble_preds = np.zeros(num_samples, dtype=np.int64)
    tie_count = 0

    # Ensure train_class_prior format
    priors: dict[int, float] = {}
    if train_class_prior is not None:
        priors = {int(k): float(v) for k, v in train_class_prior.items()}

    for i in range(num_samples):
        sample_votes = votes_matrix[i]
        unique_labels, counts = np.unique(sample_votes, return_counts=True)
        max_votes = np.max(counts)
        candidates = unique_labels[counts == max_votes]

        if len(candidates) == 1:
            ensemble_preds[i] = int(candidates[0])
        else:
            # Tie detected!
            tie_count += 1
            # Sort candidates by:
            # 1. Negative prior (highest prior first)
            # 2. Integer class label (lowest index first)
            candidates_sorted = sorted(
                candidates,
                key=lambda c: (-priors.get(int(c), 0.0), int(c)),
            )
            ensemble_preds[i] = int(candidates_sorted[0])

    return ensemble_preds, tie_count


def compute_pairwise_disagreement(votes_matrix: np.ndarray) -> float:
    """Compute pairwise disagreement rate across ensemble members."""
    num_samples, num_members = votes_matrix.shape
    if num_members < 2:
        return 0.0

    total_pairs = 0
    disagreed_pairs = 0
    for j1 in range(num_members):
        for j2 in range(j1 + 1, num_members):
            disagreed = np.sum(votes_matrix[:, j1] != votes_matrix[:, j2])
            disagreed_pairs += int(disagreed)
            total_pairs += num_samples

    return float(disagreed_pairs / total_pairs) if total_pairs > 0 else 0.0


def evaluate_ddcl_6strategy_ensemble(
    dataset_id: str,
    seed: int,
    runs_dir: str | Path = "outputs/runs",
    splits_dir: str | Path = "outputs/splits",
    protocol_id: str = "paper_interpretation_clean_v1",
) -> EnsembleResult:
    """Evaluate DDCL 6-Strategy Ensemble from executed runs."""
    runs_dir = Path(runs_dir)
    splits_dir = Path(splits_dir)

    strategies = [
        "no_curriculum",
        "ddcl_density",
        "ddcl_point",
        "self_paced",
        "sp_ddcl_density",
        "sp_ddcl_point",
    ]

    member_preds: list[MemberPrediction] = []
    total_opt_steps = 0
    total_exposures = 0
    total_wall_time = 0.0

    for s in strategies:
        run_id = f"{dataset_id}_{s}_s{seed}"
        run_path = runs_dir / run_id
        pred_file = run_path / "predictions.csv"
        metrics_file = run_path / "metrics.json"

        if not pred_file.is_file():
            raise FileNotFoundError(
                f"Missing prediction file for member '{run_id}' at {pred_file}"
            )

        mp = load_predictions_csv(pred_file, member_id=s)
        member_preds.append(mp)

        if metrics_file.is_file():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    mdata = json.load(f)
                    total_opt_steps += mdata.get("total_optimizer_steps", 0)
                    total_exposures += mdata.get("total_sample_exposures", 0)
                    total_wall_time += mdata.get("duration_seconds", 0.0)
            except Exception:
                pass

    # Load split to obtain test_ids and train_class_prior
    split_file = splits_dir / protocol_id / dataset_id / f"seed_{seed}.json"
    train_prior = None
    expected_test_ids = None

    if split_file.is_file():
        try:
            with open(split_file, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                expected_test_ids = sdata.get("test_ids")
                train_counts = sdata.get("class_counts", {}).get("train", {})
                train_prior = {int(k): int(v) for k, v in train_counts.items()}
        except Exception:
            pass

    target_ids, y_true, votes = align_member_predictions(
        member_preds,
        expected_ids=expected_test_ids,
    )

    ensemble_preds, tie_count = hard_voting_with_tie_break(
        votes_matrix=votes,
        train_class_prior=train_prior,
    )

    metrics = compute_classification_metrics(y_true, ensemble_preds)
    disagreement = compute_pairwise_disagreement(votes)

    return EnsembleResult(
        ensemble_type="ddcl_6strategy_ensemble",
        dataset_id=dataset_id,
        seed=seed,
        num_members=len(strategies),
        member_ids=strategies,
        metrics=metrics,
        tie_count=tie_count,
        tie_ratio=float(tie_count / len(y_true)) if y_true else 0.0,
        diversity_stats={"pairwise_disagreement_rate": disagreement},
        actual_compute_cost={
            "total_optimizer_steps": total_opt_steps,
            "total_sample_exposures": total_exposures,
            "total_wall_time_seconds": total_wall_time,
        },
    )


def evaluate_standard_paper_baseline(
    processed_X_train: np.ndarray,
    y_train: np.ndarray,
    processed_X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: list[str],
    dataset_id: str,
    seed: int,
    nn_prediction_file: str | Path,
) -> EnsembleResult:
    """Evaluate Standard Paper Baseline: 1 NN (no_curriculum) + 1 SVM RBF."""
    # 1. Load NN predictions
    nn_mp = load_predictions_csv(nn_prediction_file, member_id="nn_no_curriculum")

    # 2. Fit SVM RBF on train set
    svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
    svm.fit(processed_X_train, y_train)

    svm_preds = svm.predict(processed_X_test)
    svm_probs = svm.predict_proba(processed_X_test)

    svm_mp = MemberPrediction(
        member_id="svm_rbf",
        row_ids=list(test_ids),
        y_true=list(y_test),
        y_pred=svm_preds.tolist(),
        probabilities=svm_probs,
    )

    # 3. Align and combine (Average Probabilities / Soft Voting)
    target_ids, aligned_true, votes = align_member_predictions(
        [nn_mp, svm_mp],
        expected_ids=test_ids,
    )

    # For 2 members, average probabilities is more standard than hard tie
    # Or hard vote with train prior tie-break
    unique_classes, train_counts = np.unique(y_train, return_counts=True)
    prior_map = {int(c): int(cnt) for c, cnt in zip(unique_classes, train_counts)}

    ensemble_preds, tie_count = hard_voting_with_tie_break(
        votes_matrix=votes,
        train_class_prior=prior_map,
    )

    metrics = compute_classification_metrics(aligned_true, ensemble_preds)
    disagreement = compute_pairwise_disagreement(votes)

    return EnsembleResult(
        ensemble_type="standard_paper_baseline_nn_svm",
        dataset_id=dataset_id,
        seed=seed,
        num_members=2,
        member_ids=["nn_no_curriculum", "svm_rbf"],
        metrics=metrics,
        tie_count=tie_count,
        tie_ratio=float(tie_count / len(aligned_true)) if aligned_true else 0.0,
        diversity_stats={"pairwise_disagreement_rate": disagreement},
        actual_compute_cost={},
    )


def evaluate_controlled_6nn_ensemble(
    processed_X_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: list[str],
    processed_X_test: np.ndarray,
    y_test: np.ndarray,
    test_ids: list[str],
    dataset_id: str,
    base_seed: int,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
) -> EnsembleResult:
    """Evaluate Controlled 6-NN Ensemble: 6 NNs with independent seeds, no curriculum."""
    member_preds: list[MemberPrediction] = []
    num_classes = len(np.unique(y_train))
    total_opt_steps = 0
    total_exposures = 0

    unique_classes, train_counts = np.unique(y_train, return_counts=True)
    prior_map = {int(c): int(cnt) for c, cnt in zip(unique_classes, train_counts)}

    # Dummy curriculum result for no_curriculum
    curr_res = CurriculumResult(
        strategy="no_curriculum",
        ordered_train_ids=train_ids,
        distances={sid: 0.0 for sid in train_ids},
        group_ids={sid: 0 for sid in train_ids},
        scores={sid: 0.0 for sid in train_ids},
        resampling_report={"policy": "none"},
        policy_hash="controlled",
    )

    batches = create_curriculum_batches(
        X_train=processed_X_train,
        y_train=y_train,
        train_ids=train_ids,
        curriculum_result=curr_res,
        batch_size=batch_size,
    )

    for i in range(6):
        member_seed = base_seed * 100 + i
        torch.manual_seed(member_seed)
        np.random.seed(member_seed)

        model = create_mlp_model(
            input_dim=processed_X_train.shape[1],
            num_classes=num_classes,
            hidden_dims=(64, 32),
            dropout_rate=0.0,
            seed=member_seed,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        trained_m, history = train_dddcl_model(
            model=model,
            optimizer=optimizer,
            batches=batches,
            strategy="no_curriculum",
            epochs=epochs,
        )

        preds = trained_m.predict(processed_X_test)
        probs = trained_m.predict_proba(processed_X_test)

        if history:
            total_opt_steps += history[-1].get("total_optimizer_steps", 0)
            total_exposures += history[-1].get("total_sample_exposures", 0)

        member_preds.append(MemberPrediction(
            member_id=f"controlled_nn_{i}",
            row_ids=list(test_ids),
            y_true=list(y_test),
            y_pred=preds.tolist(),
            probabilities=probs,
        ))

    target_ids, aligned_true, votes = align_member_predictions(
        member_preds,
        expected_ids=test_ids,
    )

    ensemble_preds, tie_count = hard_voting_with_tie_break(
        votes_matrix=votes,
        train_class_prior=prior_map,
    )

    metrics = compute_classification_metrics(aligned_true, ensemble_preds)
    disagreement = compute_pairwise_disagreement(votes)

    return EnsembleResult(
        ensemble_type="controlled_6nn_ensemble",
        dataset_id=dataset_id,
        seed=base_seed,
        num_members=6,
        member_ids=[f"controlled_nn_{i}" for i in range(6)],
        metrics=metrics,
        tie_count=tie_count,
        tie_ratio=float(tie_count / len(aligned_true)) if aligned_true else 0.0,
        diversity_stats={"pairwise_disagreement_rate": disagreement},
        actual_compute_cost={
            "total_optimizer_steps": total_opt_steps,
            "total_sample_exposures": total_exposures,
        },
    )


def save_ensemble_result(filepath: str | Path, result: EnsembleResult) -> Path:
    """Save ensemble result to formatted JSON."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2, sort_keys=True)
    return p
