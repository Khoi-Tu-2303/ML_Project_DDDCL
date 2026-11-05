"""Experiment runner for DDDCL.

Handles config resolution, execution planning (--dry-run), idempotent run execution,
reproducible pipeline orchestration, and checkpoint resumption.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping
import numpy as np
import torch
import torch.optim as optim
import yaml

from .curriculum.geometry import compute_geometry
from .curriculum.partition import partition_by_class
from .curriculum.resample import resample_curriculum_bins
from .curriculum.scoring import CurriculumResult, create_curriculum
from .data.hf import DatasetBundle, load_dataset_from_hf, load_dataset_from_release
from .data.preprocess import ProcessedSplit, prepare_processed_split
from .data.splits import SplitBundle, load_split
from .metrics import compute_classification_metrics, save_metrics_json, save_predictions_csv
from .models.mlp import create_mlp_model
from .tracking import ExperimentTracker
from .training.loop import create_curriculum_batches, evaluate_dataset, train_dddcl_model


def load_yaml_or_json_config(path: str | Path) -> dict[str, Any]:
    """Load config from YAML or JSON file."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Configuration file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        if p.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def compute_run_id(
    protocol_id: str,
    dataset_id: str,
    strategy: str,
    seed: int,
) -> str:
    """Generate a stable, readable run identifier."""
    return f"{dataset_id}_{strategy}_s{seed}"


def compute_run_fingerprint(resolved_config: Mapping[str, Any]) -> str:
    """Compute deterministic SHA-256 fingerprint for a resolved run configuration."""
    canonical_json = json.dumps(resolved_config, sort_keys=True, default=str)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()[:16]


def dry_run_matrix(config_path: str | Path) -> dict[str, Any]:
    """Inspect and summarize the experiment matrix without running training."""
    raw_cfg = load_yaml_or_json_config(config_path)
    datasets = raw_cfg.get("datasets", [])
    strategies = raw_cfg.get("strategies", [])
    seeds = raw_cfg.get("seeds", [])
    epochs = raw_cfg.get("epochs", 5)
    protocol_id = raw_cfg.get("protocol_id", "paper_interpretation_clean_v1")
    runs_dir = Path("outputs/runs")

    total_fits = len(datasets) * len(strategies) * len(seeds)
    runs_info: list[dict[str, Any]] = []

    for d in datasets:
        for s in strategies:
            for seed in seeds:
                run_id = compute_run_id(protocol_id, d, s, seed)
                run_path = runs_dir / run_id
                status_file = run_path / "status.json"
                is_completed = False
                if status_file.is_file():
                    try:
                        with open(status_file, "r", encoding="utf-8") as f:
                            sdata = json.load(f)
                            is_completed = (sdata.get("status") == "completed")
                    except Exception:
                        pass
                runs_info.append({
                    "run_id": run_id,
                    "dataset": d,
                    "strategy": s,
                    "seed": seed,
                    "completed": is_completed,
                })

    return {
        "config_file": str(config_path),
        "protocol_id": protocol_id,
        "datasets": datasets,
        "strategies": strategies,
        "seeds": seeds,
        "epochs": epochs,
        "total_fits": total_fits,
        "completed_fits": sum(1 for r in runs_info if r["completed"]),
        "runs": runs_info,
    }


def prepare_curriculum_for_run(
    processed_train_X: np.ndarray,
    raw_train_X: np.ndarray,
    train_y: np.ndarray,
    train_ids: list[str],
    dataset_id: str,
    strategy: str,
    seed: int,
    smote_policy: str = "none",
    n_levels: int = 5,
) -> tuple[CurriculumResult, np.ndarray, np.ndarray, list[str]]:
    """Compute geometry, partitions, optional SMOTE, and ordering for one run."""
    # 1. Centroids and distances computed on raw features according to D04
    geom = compute_geometry(raw_train_X, train_y)

    # 2. KDE Partition
    part = partition_by_class(geom.normalized_distances, train_y, n_levels=n_levels)

    # 3. Optional SMOTE within bins
    resampled = resample_curriculum_bins(
        X_raw=processed_train_X,
        y=train_y,
        sample_ids=train_ids,
        bin_assignments=part.bin_assignments,
        normalized_distances=geom.normalized_distances,
        centroids=geom.centroids,
        class_norms=geom.class_norms,
        dataset_id=dataset_id,
        smote_policy=smote_policy,
        seed=seed,
    )

    # 4. Generate curriculum ordering
    curr_result = create_curriculum(
        strategy=strategy,
        sample_ids=resampled.sample_ids,
        normalized_distances=resampled.normalized_distances,
        bin_assignments=resampled.bin_assignments,
        seed=seed,
        resampling_report=resampled.resampling_report,
    )

    return curr_result, resampled.X, resampled.y, resampled.sample_ids


def get_dataset_bundle_smart(
    dataset_id: str,
    raw_config: Mapping[str, Any],
    release_dir: str | Path = "hf_release",
) -> DatasetBundle:
    """Load dataset bundle either from local verified release or HF config."""
    rel_path = Path(release_dir)
    if (rel_path / "manifest.json").is_file():
        return load_dataset_from_release(rel_path, dataset_id)
    
    data_cfg_path = raw_config.get("data_config")
    if data_cfg_path and Path(data_cfg_path).is_file():
        return load_dataset_from_hf(Path(data_cfg_path), dataset_id)

    raise FileNotFoundError(
        f"Cannot find dataset source for '{dataset_id}'. Checked {rel_path} and data_config."
    )


def execute_single_run(
    raw_config: Mapping[str, Any],
    dataset_id: str,
    strategy: str,
    seed: int,
    runs_dir: str | Path = "outputs/runs",
    splits_dir: str | Path = "outputs/splits",
    release_dir: str | Path = "hf_release",
    resume: bool = False,
) -> dict[str, Any]:
    """Execute one specific run of the experiment matrix."""
    runs_dir = Path(runs_dir)
    splits_dir = Path(splits_dir)

    protocol_id = str(raw_config.get("protocol_id", "paper_interpretation_clean_v1"))
    run_id = compute_run_id(protocol_id, dataset_id, strategy, seed)
    run_path = runs_dir / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    status_file = run_path / "status.json"

    # Idempotence check
    if not resume and status_file.is_file():
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                sdata = json.load(f)
                if sdata.get("status") == "completed":
                    return {"run_id": run_id, "status": "skipped_already_completed"}
        except Exception:
            pass

    # Setup deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load dataset bundle
    ds_bundle = get_dataset_bundle_smart(dataset_id, raw_config, release_dir=release_dir)

    # Load split bundle
    split_bundle_path = splits_dir / protocol_id / dataset_id / f"seed_{seed}.json"
    if not split_bundle_path.is_file():
        raise FileNotFoundError(
            f"Split file missing at {split_bundle_path}. Please run 'make-splits' first."
        )
    split_bundle = load_split(split_bundle_path, bundle=ds_bundle)

    # Preprocess split (StandardScaler fit on train only!)
    processed = prepare_processed_split(ds_bundle, split_bundle)

    # Resolve run parameters
    epochs = int(raw_config.get("epochs", 5))
    batch_size = int(raw_config.get("batch_size", 32))
    opt_cfg = raw_config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 0.001))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))

    spl_cfg = raw_config.get("spl", {})
    lambda_init = float(spl_cfg.get("lambda_initial", 0.4))
    lambda_max = float(spl_cfg.get("lambda_max", 0.6))
    lambda_growth = float(spl_cfg.get("growth", 1.1))

    pace_policy = str(raw_config.get("pace_policy", "replay_easy_then_union_v1"))
    smote_policy = str(raw_config.get("smote_policy", "none"))

    resolved_config = {
        "run_id": run_id,
        "protocol_id": protocol_id,
        "dataset_id": dataset_id,
        "strategy": strategy,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "lambda_init": lambda_init,
        "lambda_max": lambda_max,
        "lambda_growth": lambda_growth,
        "pace_policy": pace_policy,
        "smote_policy": smote_policy,
        "data_revision": ds_bundle.data_revision,
        "manifest_hash": ds_bundle.manifest_hash,
        "dataset_fingerprint": split_bundle.dataset_fingerprint,
    }
    run_fp = compute_run_fingerprint(resolved_config)

    # Save resolved config
    with open(run_path / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(resolved_config, f, indent=2, sort_keys=True)

    curr_res, final_X_train, final_y_train, final_train_ids = prepare_curriculum_for_run(
        processed_train_X=processed.X_train_scaled,
        raw_train_X=processed.X_train_raw,
        train_y=processed.y_train,
        train_ids=processed.train_ids,
        dataset_id=dataset_id,
        strategy=strategy,
        seed=seed,
        smote_policy=smote_policy,
    )

    batches = create_curriculum_batches(
        X_train=final_X_train,
        y_train=final_y_train,
        train_ids=final_train_ids,
        curriculum_result=curr_res,
        batch_size=batch_size,
    )

    # Create MLP model
    model = create_mlp_model(
        input_dim=processed.X_train_scaled.shape[1],
        num_classes=ds_bundle.n_classes,
        hidden_dims=(64, 32),
        dropout_rate=0.0,
        seed=seed,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Tracking
    tracking_cfg = raw_config.get("tracking", {})
    wandb_mode = tracking_cfg.get("mode", "offline")
    tracker = ExperimentTracker(
        run_dir=run_path,
        config=resolved_config,
        wandb_mode=wandb_mode,
    )

    start_time = time.time()

    # Full training loop
    trained_model, history = train_dddcl_model(
        model=model,
        optimizer=optimizer,
        batches=batches,
        strategy=strategy,
        epochs=epochs,
        X_val=processed.X_val_scaled,
        y_val=processed.y_val,
        num_classes=ds_bundle.n_classes,
        lambda_init=lambda_init,
        lambda_max=lambda_max,
        lambda_growth=lambda_growth,
        pace_policy=pace_policy,
        run_fingerprint=run_fp,
        checkpoint_dir=run_path,
        checkpoint_freq=1,
        tracker=tracker,
        resume=resume,
    )

    duration = time.time() - start_time
    tracker.finish()

    # Final Test Split Evaluation (Single pass only!)
    test_loss, test_metrics = evaluate_dataset(
        model=trained_model,
        X=processed.X_test_scaled,
        y=processed.y_test,
        num_classes=ds_bundle.n_classes,
    )

    # Predictions CSV
    test_preds = trained_model.predict(processed.X_test_scaled)
    test_probs = trained_model.predict_proba(processed.X_test_scaled)
    save_predictions_csv(
        filepath=run_path / "predictions.csv",
        row_ids=processed.test_ids,
        y_true=processed.y_test.tolist(),
        y_pred=test_preds.tolist(),
        y_prob=test_probs,
    )

    # Metrics summary
    final_metrics_summary = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "strategy": strategy,
        "seed": seed,
        "duration_seconds": duration,
        "final_epoch": epochs,
        "test_loss": test_loss,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_per_class_recall": test_metrics["per_class_recall"],
        "test_confusion_matrix": test_metrics["confusion_matrix"],
        "last_epoch_train_loss": history[-1]["train_loss"] if history else None,
        "last_epoch_val_accuracy": history[-1]["val_accuracy"] if history else None,
        "total_optimizer_steps": history[-1]["total_optimizer_steps"] if history else 0,
        "total_sample_exposures": history[-1]["total_sample_exposures"] if history else 0,
    }
    save_metrics_json(run_path / "metrics.json", final_metrics_summary)

    # Write completion status
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump({
            "status": "completed",
            "run_id": run_id,
            "run_fingerprint": run_fp,
            "completed_at": time.time(),
            "duration_seconds": duration,
        }, f, indent=2)

    return final_metrics_summary


def run_experiment_matrix(
    config_path: str | Path,
    dry_run: bool = False,
    resume: bool = False,
    runs_dir: str | Path = "outputs/runs",
    splits_dir: str | Path = "outputs/splits",
    release_dir: str | Path = "hf_release",
) -> dict[str, Any]:
    """Run full experiment matrix according to configuration."""
    if dry_run:
        return dry_run_matrix(config_path)

    raw_cfg = load_yaml_or_json_config(config_path)
    datasets = raw_cfg.get("datasets", [])
    strategies = raw_cfg.get("strategies", [])
    seeds = raw_cfg.get("seeds", [])

    results: list[dict[str, Any]] = []

    for d in datasets:
        for s in strategies:
            for seed in seeds:
                res = execute_single_run(
                    raw_config=raw_cfg,
                    dataset_id=d,
                    strategy=s,
                    seed=seed,
                    runs_dir=runs_dir,
                    splits_dir=splits_dir,
                    release_dir=release_dir,
                    resume=resume,
                )
                results.append(res)

    return {
        "config_file": str(config_path),
        "total_runs": len(results),
        "results": results,
    }
