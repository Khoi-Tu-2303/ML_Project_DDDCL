"""Quick diagnostic runner: few seeds, few epochs, one dataset, all 6 strategies.

Purpose
-------
Cheap, separate script for analysis questions that do not need the full 30-seed
matrix, for example:
  * Do the 6 strategy members make the SAME errors (so a majority vote cannot help)?
  * Does validation loss rise late in training (a sign of over-fitting)?

It reuses the official pipeline (`dddcl.runner.execute_single_run`), so models,
splits, preprocessing and metrics are identical to the main experiments. Only the
epoch count, the seed list and the output directory differ.

Rules respected
---------------
* Writes ONLY under --out-dir (default outputs/quick_analysis/e<epochs>); it refuses
  to use the official output directories.
* Splits (outputs/splits) and data (hf_release) are read-only.
* The test set is evaluated once per run by the standard runner. This script only
  re-reads the saved test predictions for DIAGNOSTICS. Do NOT use its numbers to
  choose hyper-parameters or strategies (see docs/CLAUDE.md, "Rules").

Usage (run from the ML_Project_DDDCL folder)
--------------------------------------------
  python tools/run_quick_analysis.py --dataset haberman --seeds 5 --epochs 100
  python tools/run_quick_analysis.py --dataset haberman --seeds 3 --epochs 600
  python tools/run_quick_analysis.py --dataset haberman --seeds 5 --epochs 100 --analyze-only
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.svm import SVC

import dddcl.runner as _runner_module
from dddcl.data.hf import load_dataset_from_release
from dddcl.data.preprocess import prepare_processed_split
from dddcl.data.splits import load_split
from dddcl.ensemble import (
    align_member_predictions,
    compute_pairwise_disagreement,
    evaluate_standard_paper_baseline,
    hard_voting_with_tie_break,
    load_predictions_csv,
)
from dddcl.metrics import compute_classification_metrics
from dddcl.models.mlp import MLPClassifier
import dddcl.training.loop as _loop_module
from dddcl.runner import execute_single_run, load_yaml_or_json_config

ALL_STRATEGIES = [
    "no_curriculum",
    "ddcl_density",
    "ddcl_point",
    "self_paced",
    "sp_ddcl_density",
    "sp_ddcl_point",
]

# Directories that hold official results; this script must never write there.
PROTECTED_DIRS = {
    "outputs/runs_final",
    "outputs/runs_pilot",
    "outputs/runs",
    "outputs/ensemble",
    "outputs/ensemble_600ep",
    "outputs/ensembles",
    "outputs/sensitivity_final",
    "outputs/sensitivity",
    "outputs/splits",
}


# Architecture presets. "default" = the project's fixed MLP (64, 32), ReLU, no dropout.
# "author_tuned" = per-dataset hidden layers / activations / dropout taken from the
# hyper-parameter table in the authors' public repository (values only, no code).
# Its single dropout layer sits just before the output layer, as in that table's model.
# Everything else (init, optimiser, loss, batches, SPL, splits) stays the project's own.
ARCH_PRESETS: dict[str, dict | None] = {
    "default": None,
    "author_tuned": {
        "haberman": {
            "layers": [(352, "relu"), (512, "tanh"), (16, "relu"), (16, "relu"), (16, "relu")],
            "dropout_before_output": 0.1,
        },
        "banknote": {
            "layers": [(480, "tanh")],
            "dropout_before_output": 0.4,
        },
    },
}

_ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh}


class LayerwiseMLP(MLPClassifier):
    """MLPClassifier with per-layer activations and one dropout layer before the output.

    Reuses the project's forward / loss / predict / initialisation code; only the layer
    stack (self.network) is different.
    """

    def __init__(self, input_dim: int, num_classes: int, layers: list[tuple[int, str]],
                 dropout_before_output: float, seed: int | None = None) -> None:
        super().__init__(input_dim=input_dim, num_classes=num_classes,
                         hidden_dims=[u for u, _ in layers], dropout_rate=0.0, seed=None)
        stack: list[nn.Module] = []
        prev = input_dim
        for units, act in layers:
            stack.append(nn.Linear(prev, units))
            stack.append(_ACTIVATIONS[act]())
            prev = units
        if dropout_before_output > 0.0:
            stack.append(nn.Dropout(p=dropout_before_output))
        stack.append(nn.Linear(prev, self.output_dim))
        self.network = nn.Sequential(*stack)
        self.dropout_rate = dropout_before_output
        self.reset_parameters(seed=seed)  # same deterministic init scheme as the project


def install_architecture(preset: str, dataset: str) -> dict | None:
    """Swap the model factory used by the official runner for THIS process only."""
    table = ARCH_PRESETS[preset]
    if table is None:
        return None
    if dataset not in table:
        sys.exit(f"Architecture preset '{preset}' has no entry for dataset '{dataset}'.")
    spec = table[dataset]

    def factory(input_dim, num_classes, hidden_dims=None, dropout_rate=0.0,
                activation="relu", seed=None):
        return LayerwiseMLP(input_dim, num_classes, spec["layers"],
                            spec["dropout_before_output"], seed=seed)

    _runner_module.create_mlp_model = factory
    return spec


def install_checkpoint_retry(max_attempts: int = 8, base_delay: float = 0.2) -> None:
    """Make checkpoint saving tolerate short file locks on Windows.

    `os.replace` can raise PermissionError (WinError 5) when antivirus, an indexer or a
    sync client briefly holds the checkpoint file. The shared function in
    src/dddcl/training/checkpoint.py is left untouched; this wrapper only exists for
    runs started by THIS script. It retries the same call, so saved content is identical.
    """
    original = _loop_module.save_checkpoint
    if getattr(original, "_has_retry", False):
        return

    def save_with_retry(*args, **kwargs):
        for attempt in range(1, max_attempts + 1):
            try:
                return original(*args, **kwargs)
            except PermissionError:
                if attempt == max_attempts:
                    raise
                time.sleep(base_delay * attempt)

    save_with_retry._has_retry = True  # type: ignore[attr-defined]
    _loop_module.save_checkpoint = save_with_retry


def _is_protected(path: Path) -> bool:
    resolved = path.resolve()
    for p in PROTECTED_DIRS:
        protected = Path(p).resolve()
        if resolved == protected or protected in resolved.parents:
            return True
    return False


def majority_reference_binary(p: float, n_members: int) -> float:
    """Accuracy of a majority vote if members err independently with accuracy p.

    Binary-style reference (a tie counts as half correct). For multi-class data it is
    only a rough reference, because wrong votes can split across several classes.
    """
    total = 0.0
    for k in range(n_members + 1):
        prob = math.comb(n_members, k) * (p ** k) * ((1 - p) ** (n_members - k))
        if 2 * k > n_members:
            total += prob
        elif 2 * k == n_members:
            total += 0.5 * prob
    return total


def read_trace(trace_file: Path) -> list[dict]:
    """Read trace.jsonl, keeping the last record per epoch (robust to re-runs)."""
    if not trace_file.is_file():
        return []
    by_epoch: dict[int, dict] = {}
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "epoch" in rec:
                by_epoch[int(rec["epoch"])] = rec
    return [by_epoch[e] for e in sorted(by_epoch)]


def validation_curve_summary(records: list[dict]) -> dict:
    """Summarise the validation-loss curve of one run."""
    vals = [(int(r["epoch"]), r.get("val_loss")) for r in records if r.get("val_loss") is not None]
    if not vals:
        return {}
    epochs = np.array([e for e, _ in vals])
    losses = np.array([v for _, v in vals], dtype=float)
    best_idx = int(np.argmin(losses))
    return {
        "n_epochs": int(epochs[-1]),
        "best_epoch": int(epochs[best_idx]),
        "min_val_loss": float(losses[best_idx]),
        "final_val_loss": float(losses[-1]),
        # > 0 means validation loss ended higher than its minimum (possible over-fitting)
        "rise_after_min": float(losses[-1] - losses[best_idx]),
        "final_val_accuracy": float(records[-1].get("val_accuracy", float("nan"))),
    }


def run_training(args, cfg: dict, runs_dir: Path, seeds: list[int], strategies: list[str]) -> None:
    total = len(strategies) * len(seeds)
    print("=" * 66)
    print(f"Quick analysis training: dataset={args.dataset} epochs={args.epochs} seeds={seeds}")
    print(f"Runs directory: {runs_dir}")
    print(f"Total fits: {total}")
    print("=" * 66)
    done = 0
    t_start = time.time()
    for seed in seeds:
        for strat in strategies:
            done += 1
            t0 = time.time()
            res = execute_single_run(
                raw_config=cfg,
                dataset_id=args.dataset,
                strategy=strat,
                seed=seed,
                runs_dir=runs_dir,
                splits_dir=Path(args.splits_dir),
                release_dir=Path(args.release_dir),
                resume=False,
            )
            dur = time.time() - t0
            if res.get("status") == "skipped_already_completed":
                print(f"[{done:3d}/{total}] {res['run_id']:<34} skipped (already completed)")
            else:
                acc = res.get("test_accuracy", 0.0) * 100.0
                f1 = res.get("test_macro_f1", 0.0) * 100.0
                print(f"[{done:3d}/{total}] {res['run_id']:<34} acc={acc:5.2f}% f1={f1:5.2f}% {dur:5.1f}s")
    print(f"Training finished in {(time.time() - t_start) / 60:.1f} min")


def analyse(args, runs_dir: Path, seeds: list[int], strategies: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ds_bundle = load_dataset_from_release(Path(args.release_dir), args.dataset)
    seed_rows = []
    curve_rows = []

    for seed in seeds:
        run_paths = {s: runs_dir / f"{args.dataset}_{s}_s{seed}" for s in strategies}
        missing = [s for s, p in run_paths.items() if not (p / "predictions.csv").is_file()]
        if missing:
            print(f"[seed {seed}] missing predictions for {missing}; skipped")
            continue

        # Class prior of the training split (tie-break rule of the ensemble code).
        split_file = Path(args.splits_dir) / str(args.protocol_id) / args.dataset / f"seed_{seed}.json"
        split_bundle = load_split(split_file, bundle=ds_bundle)
        proc = prepare_processed_split(ds_bundle, split_bundle)
        classes, counts = np.unique(proc.y_train, return_counts=True)
        prior_map = {int(c): int(n) for c, n in zip(classes, counts)}

        members = [load_predictions_csv(run_paths[s] / "predictions.csv", member_id=s) for s in strategies]
        _, y_true, votes = align_member_predictions(members, expected_ids=list(proc.test_ids))
        y_true = np.asarray(y_true)

        correct = votes == y_true[:, None]
        member_acc = correct.mean(axis=0)
        mean_member_acc = float(member_acc.mean())

        ens_pred, tie_count = hard_voting_with_tie_break(votes, train_class_prior=prior_map)
        ens_acc = float((ens_pred == y_true).mean())
        ens_f1 = float(compute_classification_metrics(y_true, ens_pred)["macro_f1"])

        # Standard Ensemble (1 NN no_curriculum + 1 SVM RBF), same official function.
        std = {}
        if "no_curriculum" in strategies:
            std_res = evaluate_standard_paper_baseline(
                processed_X_train=proc.X_train_scaled, y_train=proc.y_train,
                processed_X_test=proc.X_test_scaled, y_test=proc.y_test,
                test_ids=list(proc.test_ids), dataset_id=args.dataset, seed=seed,
                nn_prediction_file=run_paths["no_curriculum"] / "predictions.csv",
            )
            svm = SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)
            svm.fit(proc.X_train_scaled, proc.y_train)
            std = {
                "std_ens_acc": float(std_res.metrics["accuracy"]),
                "std_ens_macro_f1": float(std_res.metrics["macro_f1"]),
                "std_ens_ties": int(std_res.tie_count),
                "svm_acc": float((svm.predict(proc.X_test_scaled) == proc.y_test).mean()),
            }

        n_wrong = (~correct).sum(axis=1)
        n_members = len(strategies)
        frac_all_wrong = float((n_wrong == n_members).mean())
        frac_none_wrong = float((n_wrong == 0).mean())
        frac_some_wrong = 1.0 - frac_all_wrong - frac_none_wrong
        any_wrong = float((n_wrong > 0).mean())
        shared_error_share = frac_all_wrong / any_wrong if any_wrong > 0 else float("nan")

        row = {
            "seed": seed,
            "n_test": int(len(y_true)),
            "mean_member_acc": mean_member_acc,
            "best_member_acc": float(member_acc.max()),
            "ensemble_acc": ens_acc,
            "ensemble_macro_f1": ens_f1,
            "majority_class_share": float(np.bincount(y_true).max() / len(y_true)),
            **std,
            "gain_vs_mean_member": ens_acc - mean_member_acc,
            "oracle_acc_any_member_right": 1.0 - frac_all_wrong,
            "frac_all_members_wrong": frac_all_wrong,
            "frac_some_members_wrong": frac_some_wrong,
            "frac_no_member_wrong": frac_none_wrong,
            "shared_error_share": shared_error_share,
            "pairwise_disagreement": float(compute_pairwise_disagreement(votes)),
            "tie_count": int(tie_count),
            "indep_majority_ref_binary": majority_reference_binary(mean_member_acc, n_members),
        }
        for s, a in zip(strategies, member_acc):
            row[f"acc_{s}"] = float(a)
        seed_rows.append(row)

        for s in strategies:
            summ = validation_curve_summary(read_trace(run_paths[s] / "trace.jsonl"))
            if summ:
                curve_rows.append({"seed": seed, "strategy": s, **summ})

    return pd.DataFrame(seed_rows), pd.DataFrame(curve_rows)


def print_report(seed_df: pd.DataFrame, curve_df: pd.DataFrame, args) -> None:
    print("\n" + "=" * 66)
    print(f"DIAGNOSTIC REPORT  dataset={args.dataset}  epochs={args.epochs}")
    print("(diagnostic only: do not use to select hyper-parameters or strategies)")
    print("=" * 66)

    if seed_df.empty:
        print("No completed seeds to analyse.")
        return

    pct = lambda x: f"{100 * x:6.2f}%"
    print(f"Seeds analysed: {len(seed_df)}   test samples per seed: {int(seed_df['n_test'].iloc[0])}")
    print("\n[1] Ensemble vs members (mean over seeds)")
    print(f"  mean member accuracy         : {pct(seed_df['mean_member_acc'].mean())}")
    print(f"  best member accuracy         : {pct(seed_df['best_member_acc'].mean())}")
    print(f"  majority-vote ensemble       : {pct(seed_df['ensemble_acc'].mean())}")
    print(f"  gain vs mean member (pp)     : {100 * seed_df['gain_vs_mean_member'].mean():+.2f}")
    print(f"  oracle (any member correct)  : {pct(seed_df['oracle_acc_any_member_right'].mean())}")
    print(f"  independent-error reference* : {pct(seed_df['indep_majority_ref_binary'].mean())}")
    print(f"  DDCL-6 ensemble macro-F1     : {pct(seed_df['ensemble_macro_f1'].mean())}")
    print(f"  always-majority-class        : {pct(seed_df['majority_class_share'].mean())}")
    if "std_ens_acc" in seed_df.columns:
        print("\n[1b] Standard Ensemble (NN no_curriculum + SVM RBF, mean over seeds)")
        print(f"  NN no_curriculum accuracy    : {pct(seed_df['acc_no_curriculum'].mean())}")
        print(f"  SVM accuracy                 : {pct(seed_df['svm_acc'].mean())}")
        print(f"  Standard ensemble accuracy   : {pct(seed_df['std_ens_acc'].mean())}")
        print(f"  Standard ensemble macro-F1   : {pct(seed_df['std_ens_macro_f1'].mean())}")
        print(f"  mean tie count (1-1 votes)   : {seed_df['std_ens_ties'].mean():.2f}")

    print("\n[2] Error overlap among the 6 members (mean over seeds)")
    print(f"  samples where ALL members wrong : {pct(seed_df['frac_all_members_wrong'].mean())}")
    print(f"  samples where SOME members wrong: {pct(seed_df['frac_some_members_wrong'].mean())}")
    print(f"  samples where NO member wrong   : {pct(seed_df['frac_no_member_wrong'].mean())}")
    print(f"  share of erroneous samples that are shared by all members: {pct(seed_df['shared_error_share'].mean())}")
    print(f"  pairwise disagreement           : {pct(seed_df['pairwise_disagreement'].mean())}")
    print(f"  mean tie count per seed         : {seed_df['tie_count'].mean():.2f}")

    acc_cols = [c for c in seed_df.columns if c.startswith("acc_")]
    print("\n[3] Per-strategy test accuracy (mean +- std over seeds)")
    for c in acc_cols:
        print(f"  {c[4:]:<18}: {pct(seed_df[c].mean())} +- {100 * seed_df[c].std(ddof=1) if len(seed_df) > 1 else 0.0:5.2f}")

    if not curve_df.empty:
        print("\n[4] Validation-loss curve (mean over seeds)")
        print("  rise_after_min > 0 means validation loss ended above its minimum")
        agg = curve_df.groupby("strategy").agg(
            best_epoch=("best_epoch", "mean"),
            min_val_loss=("min_val_loss", "mean"),
            final_val_loss=("final_val_loss", "mean"),
            rise_after_min=("rise_after_min", "mean"),
        )
        agg = agg.reindex([s for s in ALL_STRATEGIES if s in agg.index])
        print(agg.to_string(float_format=lambda v: f"{v:8.4f}"))

    print("\n* binary-style reference: majority accuracy if members erred independently with the")
    print("  same mean accuracy (tie = half correct). Only a rough guide for multi-class data.")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Quick diagnostic runner (few seeds, few epochs)")
    parser.add_argument("--dataset", default="haberman", choices=["haberman", "new_thyroid", "banknote"])
    parser.add_argument("--seeds", type=int, default=5, help="Use seeds 0..N-1 (splits must exist)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--strategies", default="all", help="Comma-separated strategies or 'all'")
    parser.add_argument("--config", type=Path, default=Path("configs/single_model.yaml"))
    parser.add_argument("--splits-dir", default="outputs/splits")
    parser.add_argument("--release-dir", default="hf_release")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: outputs/quick_analysis/e<epochs>")
    parser.add_argument("--analyze-only", action="store_true", help="Skip training, analyse existing runs")
    parser.add_argument("--arch", default="default", choices=sorted(ARCH_PRESETS),
                        help="Model architecture preset (default = project MLP (64, 32))")
    args = parser.parse_args()

    cfg = load_yaml_or_json_config(args.config)
    args.protocol_id = cfg.get("protocol_id", "paper_interpretation_clean_v1")

    # The epoch count is part of the directory name because run ids do not contain it
    # (a finished run would otherwise be silently reused with a different epoch count).
    default_name = f"e{args.epochs}" if args.arch == "default" else f"{args.arch}_e{args.epochs}"
    out_dir = args.out_dir or Path("outputs/quick_analysis") / default_name
    if _is_protected(out_dir):
        sys.exit(f"Refusing to write into protected official directory: {out_dir}")
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    strategies = ALL_STRATEGIES if args.strategies == "all" else [
        s.strip() for s in args.strategies.split(",") if s.strip() in ALL_STRATEGIES
    ]
    if len(strategies) < 2:
        sys.exit("Need at least 2 valid strategies for the ensemble analysis.")
    seeds = list(range(args.seeds))

    cfg = dict(cfg)
    cfg["epochs"] = args.epochs
    cfg["runs_dir"] = str(runs_dir)

    arch_spec = install_architecture(args.arch, args.dataset)
    with open(out_dir / "architecture.json", "w", encoding="utf-8") as f:
        json.dump({"preset": args.arch, "dataset": args.dataset,
                   "spec": arch_spec or {"hidden_dims": [64, 32], "activation": "relu", "dropout": 0.0}},
                  f, indent=2)
    print(f"Architecture preset: {args.arch} -> {arch_spec or 'project default (64, 32) relu, no dropout'}")

    if not args.analyze_only:
        install_checkpoint_retry()
        run_training(args, cfg, runs_dir, seeds, strategies)

    seed_df, curve_df = analyse(args, runs_dir, seeds, strategies)
    print_report(seed_df, curve_df, args)

    if not seed_df.empty:
        seed_df.to_csv(out_dir / "per_seed_summary.csv", index=False)
    if not curve_df.empty:
        curve_df.to_csv(out_dir / "validation_curves.csv", index=False)
    print(f"\nSaved CSV files under: {out_dir}")


if __name__ == "__main__":
    main()
