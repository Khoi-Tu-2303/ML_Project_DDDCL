"""Official R2 Final Ensemble Runner with 600 epochs per neural network member.

Implements the protocol from docs/04_PROTOCOLS.md and Table 3 of Dynamic DDCL (2025):
- 600 epochs per neural network member
- Three ensemble configurations:
  1. DDCL-6 Strategy Ensemble (Hard voting among 6 curriculum strategies)
  2. Standard Paper Baseline Ensemble (1 NN no_curriculum + 1 SVM RBF)
  3. Controlled 6-NN Ensemble (6 independent random initializations, no curriculum)
- Deterministic Decision D09 tie-breaking (train class prior -> class label index)
- Atomic caching: saves to outputs/ensembles_600/<ensemble_type>/<dataset_id>_s<seed>.json
"""

import argparse
import json
from pathlib import Path
import time
from typing import Sequence
import numpy as np
import torch
from sklearn.svm import SVC

from dddcl.data.hf import load_dataset_from_release
from dddcl.data.splits import load_split
from dddcl.data.preprocess import prepare_processed_split
from dddcl.curriculum.geometry import compute_geometry
from dddcl.curriculum.partition import partition_by_class
from dddcl.curriculum.scoring import create_curriculum, CurriculumResult
from dddcl.training.loop import create_curriculum_batches, train_dddcl_model
from dddcl.models.mlp import create_mlp_model
from dddcl.ensemble import (
    MemberPrediction,
    EnsembleResult,
    align_member_predictions,
    hard_voting_with_tie_break,
    compute_pairwise_disagreement,
    save_ensemble_result,
)
from dddcl.metrics import compute_classification_metrics


def train_single_member_600(
    proc,
    strategy: str,
    seed: int,
    epochs: int = 600,
    batch_size: int = 32,
    lr: float = 0.001,
) -> MemberPrediction:
    """Train a single neural network member for 600 epochs and return MemberPrediction."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Setup curriculum
    if strategy in ("no_curriculum", "self_paced"):
        curr_res = CurriculumResult(
            strategy=strategy,
            ordered_train_ids=proc.train_ids,
            distances={sid: 0.0 for sid in proc.train_ids},
            group_ids={sid: 0 for sid in proc.train_ids},
            scores={sid: 0.0 for sid in proc.train_ids},
            resampling_report={"policy": "none"},
            policy_hash="none",
        )
    else:
        geom = compute_geometry(proc.X_train_raw, proc.y_train)
        part = partition_by_class(geom.normalized_distances, proc.y_train, n_levels=5)
        curr_res = create_curriculum(
            strategy=strategy,
            sample_ids=proc.train_ids,
            normalized_distances=geom.normalized_distances,
            bin_assignments=part.bin_assignments,
            seed=seed,
        )

    batches = create_curriculum_batches(
        X_train=proc.X_train_scaled,
        y_train=proc.y_train,
        train_ids=proc.train_ids,
        curriculum_result=curr_res,
        batch_size=batch_size,
    )

    num_classes = len(np.unique(proc.y_train))
    model = create_mlp_model(
        input_dim=proc.X_train_scaled.shape[1],
        num_classes=num_classes,
        hidden_dims=(64, 32),
        dropout_rate=0.0,
        seed=seed,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    trained_m, _ = train_dddcl_model(
        model=model,
        optimizer=optimizer,
        batches=batches,
        strategy=strategy,
        epochs=epochs,
        lambda_init=0.4,
        lambda_max=0.6,
        lambda_growth=1.1,
    )

    preds = trained_m.predict(proc.X_test_scaled)
    probs = trained_m.predict_proba(proc.X_test_scaled)

    return MemberPrediction(
        member_id=strategy,
        row_ids=list(proc.test_ids),
        y_true=list(proc.y_test),
        y_pred=preds.tolist(),
        probabilities=probs,
    )


def evaluate_seed_600(
    dataset_id: str,
    seed: int,
    out_dir: Path,
    splits_dir: Path = Path("outputs/splits"),
    rel_dir: Path = Path("hf_release"),
    protocol_id: str = "paper_interpretation_clean_v1",
    epochs: int = 600,
    include_controlled: bool = True,
) -> dict[str, float]:
    """Evaluate all 3 ensemble configurations for a single dataset and seed with 600 epochs."""
    ds_bundle = load_dataset_from_release(rel_dir, dataset_id)
    split_file = splits_dir / protocol_id / dataset_id / f"seed_{seed}.json"
    split_bundle = load_split(split_file, bundle=ds_bundle)
    proc = prepare_processed_split(ds_bundle, split_bundle)

    # Class prior map for tie-breaking
    unique_classes, train_counts = np.unique(proc.y_train, return_counts=True)
    prior_map = {int(c): int(cnt) for c, cnt in zip(unique_classes, train_counts)}

    ddcl_file = out_dir / "ddcl_6strategy" / f"{dataset_id}_s{seed}.json"
    std_file = out_dir / "standard_nn_svm" / f"{dataset_id}_s{seed}.json"
    ctrl_file = out_dir / "controlled_6nn" / f"{dataset_id}_s{seed}.json"

    accuracies = {}

    # Check if DDCL-6 and Standard NN+SVM are already done
    ddcl_done = ddcl_file.is_file() and ddcl_file.stat().st_size > 0
    std_done = std_file.is_file() and std_file.stat().st_size > 0

    if ddcl_done and std_done:
        with open(ddcl_file, "r", encoding="utf-8") as f:
            accuracies["ddcl_6strategy"] = json.load(f)["metrics"]["accuracy"]
        with open(std_file, "r", encoding="utf-8") as f:
            accuracies["standard_nn_svm"] = json.load(f)["metrics"]["accuracy"]
    else:
        strategies = [
            "no_curriculum",
            "ddcl_density",
            "ddcl_point",
            "self_paced",
            "sp_ddcl_density",
            "sp_ddcl_point",
        ]
        member_preds = []
        no_curr_pred = None

        for s in strategies:
            mp = train_single_member_600(proc, strategy=s, seed=seed, epochs=epochs)
            member_preds.append(mp)
            if s == "no_curriculum":
                no_curr_pred = mp

        # 1. DDCL-6 Ensemble
        target_ids, aligned_true, votes = align_member_predictions(
            member_preds,
            expected_ids=proc.test_ids,
        )
        ensemble_preds, tie_count = hard_voting_with_tie_break(
            votes_matrix=votes,
            train_class_prior=prior_map,
        )
        metrics_ddcl = compute_classification_metrics(aligned_true, ensemble_preds)
        dis_ddcl = compute_pairwise_disagreement(votes)
        res_ddcl = EnsembleResult(
            ensemble_type="ddcl_6strategy_ensemble",
            dataset_id=dataset_id,
            seed=seed,
            num_members=6,
            member_ids=strategies,
            metrics=metrics_ddcl,
            tie_count=tie_count,
            tie_ratio=float(tie_count / len(aligned_true)),
            diversity_stats={"pairwise_disagreement_rate": dis_ddcl},
            actual_compute_cost={"epochs_per_member": epochs},
        )
        save_ensemble_result(ddcl_file, res_ddcl)
        accuracies["ddcl_6strategy"] = metrics_ddcl["accuracy"]

        # 2. Standard Paper Baseline (NN + SVM RBF)
        svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
        svm.fit(proc.X_train_scaled, proc.y_train)
        svm_preds = svm.predict(proc.X_test_scaled)
        svm_probs = svm.predict_proba(proc.X_test_scaled)
        svm_mp = MemberPrediction(
            member_id="svm_rbf",
            row_ids=list(proc.test_ids),
            y_true=list(proc.y_test),
            y_pred=svm_preds.tolist(),
            probabilities=svm_probs,
        )

        target_ids, aligned_true_std, votes_std = align_member_predictions(
            [no_curr_pred, svm_mp],
            expected_ids=proc.test_ids,
        )
        preds_std, tie_std = hard_voting_with_tie_break(
            votes_matrix=votes_std,
            train_class_prior=prior_map,
        )
        metrics_std = compute_classification_metrics(aligned_true_std, preds_std)
        dis_std = compute_pairwise_disagreement(votes_std)
        res_std = EnsembleResult(
            ensemble_type="standard_paper_baseline_nn_svm",
            dataset_id=dataset_id,
            seed=seed,
            num_members=2,
            member_ids=["nn_no_curriculum", "svm_rbf"],
            metrics=metrics_std,
            tie_count=tie_std,
            tie_ratio=float(tie_std / len(aligned_true_std)),
            diversity_stats={"pairwise_disagreement_rate": dis_std},
            actual_compute_cost={"epochs_per_member": epochs},
        )
        save_ensemble_result(std_file, res_std)
        accuracies["standard_nn_svm"] = metrics_std["accuracy"]

    # 3. Controlled 6-NN Ensemble
    if include_controlled:
        ctrl_done = ctrl_file.is_file() and ctrl_file.stat().st_size > 0
        if ctrl_done:
            with open(ctrl_file, "r", encoding="utf-8") as f:
                accuracies["controlled_6nn"] = json.load(f)["metrics"]["accuracy"]
        else:
            ctrl_members = []
            for i in range(6):
                m_seed = seed * 100 + i
                mp = train_single_member_600(proc, strategy="no_curriculum", seed=m_seed, epochs=epochs)
                ctrl_members.append(mp)

            target_ids, aligned_true_ctrl, votes_ctrl = align_member_predictions(
                ctrl_members,
                expected_ids=proc.test_ids,
            )
            preds_ctrl, tie_ctrl = hard_voting_with_tie_break(
                votes_matrix=votes_ctrl,
                train_class_prior=prior_map,
            )
            metrics_ctrl = compute_classification_metrics(aligned_true_ctrl, preds_ctrl)
            dis_ctrl = compute_pairwise_disagreement(votes_ctrl)
            res_ctrl = EnsembleResult(
                ensemble_type="controlled_6nn_ensemble",
                dataset_id=dataset_id,
                seed=seed,
                num_members=6,
                member_ids=[f"controlled_nn_{i}" for i in range(6)],
                metrics=metrics_ctrl,
                tie_count=tie_ctrl,
                tie_ratio=float(tie_ctrl / len(aligned_true_ctrl)),
                diversity_stats={"pairwise_disagreement_rate": dis_ctrl},
                actual_compute_cost={"epochs_per_member": epochs},
            )
            save_ensemble_result(ctrl_file, res_ctrl)
            accuracies["controlled_6nn"] = metrics_ctrl["accuracy"]

    return accuracies


def run_r2_600_matrix(
    datasets: list[str],
    seeds: list[int],
    out_dir: Path = Path("outputs/ensembles_600"),
    epochs: int = 600,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ["ddcl_6strategy", "standard_nn_svm", "controlled_6nn"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"Executing R2 Ensemble Benchmark with {epochs} Epochs/Member")
    print(f"Datasets: {datasets}")
    print(f"Seeds: {min(seeds)}..{max(seeds)} ({len(seeds)} total)")
    print(f"Output directory: {out_dir}")
    print("=" * 65)

    t_start = time.time()
    for ds_id in datasets:
        print(f"\n[*] Processing Dataset: {ds_id}")
        for s in seeds:
            t0 = time.time()
            accs = evaluate_seed_600(
                dataset_id=ds_id,
                seed=s,
                out_dir=out_dir,
                epochs=epochs,
            )
            dur = time.time() - t0
            ddcl_acc = accs.get("ddcl_6strategy", 0.0) * 100.0
            std_acc = accs.get("standard_nn_svm", 0.0) * 100.0
            ctrl_acc = accs.get("controlled_6nn", 0.0) * 100.0
            print(f"  [Seed {s:2d}] DDCL-6: {ddcl_acc:5.2f}% | Std (NN+SVM): {std_acc:5.2f}% | Ctrl 6-NN: {ctrl_acc:5.2f}% | {dur:4.1f}s")

    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"[+] R2 {epochs}-EPOCH ENSEMBLE EVALUATIONS COMPLETE ({elapsed:.1f}s = {elapsed/60:.1f} mins)")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run R2 600-Epoch Ensemble Benchmark")
    parser.add_argument("--dataset", type=str, default="new_thyroid", choices=["haberman", "new_thyroid", "banknote", "all"])
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds")
    parser.add_argument("--epochs", type=int, default=600, help="Epochs per neural network member")
    args = parser.parse_args()

    datasets = ["haberman", "new_thyroid"] if args.dataset == "all" else [args.dataset]
    run_r2_600_matrix(
        datasets=datasets,
        seeds=list(range(args.seeds)),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()

