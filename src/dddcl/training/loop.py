"""Independent training loop and step accounting for DDDCL.

Follows specifications from docs/03_METHOD_SPEC.md, docs/06_ARCHITECTURE.md,
and docs/07_VERIFICATION.md:
- Pure PyTorch CPU training loop with deterministic execution.
- Stable batch partitions derived directly from curriculum order without random shuffle.
- SPL loop:
  * Epoch begins: model.eval(), torch.no_grad(), compute exact per-batch loss.
  * compute_pace_decision() separates Easy and Hard batches with stable sort.
  * Pass 1: train Easy batches with model.train() and gradients.
  * Replay check: evaluate re-computed mean loss on Easy batches. If < lambda,
    execute second pass on union or hard-only according to policy.
  * Update lambda: min(lambda_max, lambda * lambda_growth).
- Non-SPL loop:
  * Sequential training over all batches in curriculum order.
- Complete exposure and optimizer step accounting.
- Atomic checkpoint saving per epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Callable, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..curriculum.scoring import CurriculumResult
from ..metrics import compute_classification_metrics
from ..tracking import ExperimentTracker
from .checkpoint import load_checkpoint, restore_rng_states, save_checkpoint
from .pace import compute_pace_decision, update_lambda


@dataclass
class BatchData:
    """Container for a single training mini-batch."""
    batch_id: int
    sample_ids: list[str]
    X: torch.Tensor
    y: torch.Tensor


@dataclass
class EpochAccounting:
    """Detailed accounting of computational effort within an epoch."""
    epoch: int
    optimizer_steps: int = 0
    sample_exposures: int = 0
    easy_batch_ids: list[int] = field(default_factory=list)
    hard_batch_ids: list[int] = field(default_factory=list)
    replayed: bool = False
    fallback_used: bool = False
    lambda_value: float = 0.0
    mean_easy_loss_eval: float | None = None
    pass1_loss: float = 0.0
    pass2_loss: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    val_macro_f1: float = 0.0
    duration_seconds: float = 0.0


def create_curriculum_batches(
    X_train: torch.Tensor | np.ndarray,
    y_train: torch.Tensor | np.ndarray,
    train_ids: Sequence[str],
    curriculum_result: CurriculumResult,
    batch_size: int,
) -> list[BatchData]:
    """Create stable mini-batches ordered by curriculum results.
    
    Parameters
    ----------
    X_train : torch.Tensor | np.ndarray
        Training feature array, shape (N, d).
    y_train : torch.Tensor | np.ndarray
        Training labels, shape (N,).
    train_ids : Sequence[str]
        Original list of row IDs corresponding to X_train and y_train rows.
    curriculum_result : CurriculumResult
        Curriculum result containing ordered_train_ids.
    batch_size : int
        Size of each mini-batch (batch_size >= 1).

    Returns
    -------
    list[BatchData]
        List of mini-batches strictly preserving curriculum ordering.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if not isinstance(X_train, torch.Tensor):
        X_train_t = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    else:
        X_train_t = X_train.float()

    if not isinstance(y_train, torch.Tensor):
        y_train_t = torch.from_numpy(np.asarray(y_train))
    else:
        y_train_t = y_train

    id_to_idx = {rid: idx for idx, rid in enumerate(train_ids)}
    ordered_ids = curriculum_result.ordered_train_ids

    # Group into batches
    batches: list[BatchData] = []
    num_samples = len(ordered_ids)

    for b_idx, start_pos in enumerate(range(0, num_samples, batch_size)):
        end_pos = min(start_pos + batch_size, num_samples)
        batch_ids = ordered_ids[start_pos:end_pos]
        indices = [id_to_idx[rid] for rid in batch_ids]

        b_X = X_train_t[indices]
        b_y = y_train_t[indices]

        batches.append(BatchData(
            batch_id=b_idx,
            sample_ids=batch_ids,
            X=b_X,
            y=b_y,
        ))

    return batches


def evaluate_batch_losses(
    model: nn.Module,
    batches: Sequence[BatchData],
) -> list[float]:
    """Evaluate mean per-example loss for each batch independently with no gradients."""
    was_training = model.training
    model.eval()
    losses: list[float] = []

    try:
        with torch.no_grad():
            for b in batches:
                logits = model(b.X)
                b_loss = model.compute_loss(logits, b.y, reduction="mean")
                losses.append(float(b_loss.item()))
    finally:
        model.train(was_training)

    return losses


def evaluate_dataset(
    model: nn.Module,
    X: torch.Tensor | np.ndarray,
    y: torch.Tensor | np.ndarray,
    num_classes: int | None = None,
) -> tuple[float, dict[str, Any]]:
    """Compute overall loss and full metrics dictionary on evaluation split."""
    was_training = model.training
    model.eval()

    if not isinstance(X, torch.Tensor):
        X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    else:
        X_t = X.float()

    if not isinstance(y, torch.Tensor):
        y_t = torch.from_numpy(np.asarray(y))
    else:
        y_t = y

    try:
        with torch.no_grad():
            logits = model(X_t)
            loss_tensor = model.compute_loss(logits, y_t, reduction="mean")
            val_loss = float(loss_tensor.item())

            preds = model.predict(X_t)
            targets = y_t.cpu().numpy()
            metrics = compute_classification_metrics(targets, preds, num_classes=num_classes)
            return val_loss, metrics
    finally:
        model.train(was_training)


def train_single_epoch_spl(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batches: Sequence[BatchData],
    current_lambda: float,
    pace_policy: str = "replay_easy_then_union_v1",
) -> tuple[EpochAccounting, float]:
    """Execute one epoch of Self-Paced Learning according to DDDCL specifications."""
    t0 = time.time()
    batch_map = {b.batch_id: b for b in batches}

    # Step 1: Score per-batch loss in eval mode with no gradients
    batch_losses = evaluate_batch_losses(model, batches)

    # Step 2: Make pace decision
    decision = compute_pace_decision(
        batch_losses=batch_losses,
        lambda_val=current_lambda,
    )

    accounting = EpochAccounting(
        epoch=0,
        easy_batch_ids=decision.easy_batch_ids,
        hard_batch_ids=decision.hard_batch_ids,
        fallback_used=(decision.fallback_reason is not None),
        lambda_value=current_lambda,
    )

    model.train()

    # Step 3: Pass 1 on Easy batches (ordered from smallest loss)
    pass1_losses: list[float] = []
    for b_id in decision.ordered_easy_batches:
        b = batch_map[b_id]
        optimizer.zero_grad()
        logits = model(b.X)
        loss = model.compute_loss(logits, b.y, reduction="mean")
        loss.backward()
        optimizer.step()

        accounting.optimizer_steps += 1
        accounting.sample_exposures += len(b.y)
        pass1_losses.append(float(loss.item()))

    accounting.pass1_loss = float(np.mean(pass1_losses)) if pass1_losses else 0.0

    # Step 4: Conditional Replay check
    # Re-evaluate mean loss on the easy batches
    easy_batches = [batch_map[bid] for bid in decision.easy_batch_ids]
    if easy_batches:
        re_eval_losses = evaluate_batch_losses(model, easy_batches)
        mean_easy_loss = float(np.mean(re_eval_losses))
    else:
        mean_easy_loss = accounting.pass1_loss
    accounting.mean_easy_loss_eval = mean_easy_loss

    # If easy loss is below threshold, trigger replay pass
    pass2_losses: list[float] = []
    if mean_easy_loss < current_lambda:
        accounting.replayed = True
        model.train()

        if pace_policy == "replay_easy_then_union_v1":
            # Replay all batches (easy + hard) sorted by original loss
            all_sorted_ids = sorted(range(len(batches)), key=lambda idx: (batch_losses[idx], idx))
            replay_batch_ids = all_sorted_ids
        elif pace_policy == "hard_only_second_pass":
            # Only train on hard batches
            replay_batch_ids = decision.hard_batch_ids
        else:
            replay_batch_ids = []

        for b_id in replay_batch_ids:
            b = batch_map[b_id]
            optimizer.zero_grad()
            logits = model(b.X)
            loss = model.compute_loss(logits, b.y, reduction="mean")
            loss.backward()
            optimizer.step()

            accounting.optimizer_steps += 1
            accounting.sample_exposures += len(b.y)
            pass2_losses.append(float(loss.item()))

    accounting.pass2_loss = float(np.mean(pass2_losses)) if pass2_losses else 0.0
    accounting.duration_seconds = time.time() - t0

    return accounting, current_lambda


def train_single_epoch_standard(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batches: Sequence[BatchData],
) -> EpochAccounting:
    """Execute one epoch of standard sequential training (no SPL filtering)."""
    t0 = time.time()
    model.train()
    losses: list[float] = []
    steps = 0
    exposures = 0

    for b in batches:
        optimizer.zero_grad()
        logits = model(b.X)
        loss = model.compute_loss(logits, b.y, reduction="mean")
        loss.backward()
        optimizer.step()

        steps += 1
        exposures += len(b.y)
        losses.append(float(loss.item()))

    return EpochAccounting(
        epoch=0,
        optimizer_steps=steps,
        sample_exposures=exposures,
        easy_batch_ids=[b.batch_id for b in batches],
        hard_batch_ids=[],
        replayed=False,
        fallback_used=False,
        lambda_value=0.0,
        pass1_loss=float(np.mean(losses)) if losses else 0.0,
        duration_seconds=time.time() - t0,
    )


def train_dddcl_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    batches: Sequence[BatchData],
    strategy: str,
    epochs: int,
    X_val: torch.Tensor | np.ndarray | None = None,
    y_val: torch.Tensor | np.ndarray | None = None,
    num_classes: int | None = None,
    lambda_init: float = 0.4,
    lambda_max: float = 0.6,
    lambda_growth: float = 1.1,
    pace_policy: str = "replay_easy_then_union_v1",
    run_fingerprint: str = "pilot_run",
    checkpoint_dir: str | Path | None = None,
    checkpoint_freq: int = 1,
    tracker: ExperimentTracker | None = None,
    resume: bool = False,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    """Full independent training loop with checkpointing and tracking.
    
    Parameters
    ----------
    model : nn.Module
        MLP model instance.
    optimizer : optim.Optimizer
        Configured optimizer.
    batches : Sequence[BatchData]
        Pre-partitioned mini-batches in curriculum order.
    strategy : str
        One of the 6 strategies: 'no_curriculum', 'ddcl_density', 'ddcl_point',
        'self_paced', 'sp_ddcl_density', 'sp_ddcl_point'.
    epochs : int
        Total epochs to train.
    X_val, y_val : arrays, optional
        Validation split data.
    num_classes : int, optional
        Total target classes.
    lambda_init, lambda_max, lambda_growth : float
        SPL pace parameters.
    pace_policy : str
        SPL pace policy.
    run_fingerprint : str
        Unique experiment hash.
    checkpoint_dir : str | Path, optional
        Directory to store checkpoint.pt.
    checkpoint_freq : int, default=1
        Save checkpoint every N epochs.
    tracker : ExperimentTracker, optional
        Tracker for logging epoch metrics.
    resume : bool, default=False
        If True and checkpoint exists, resume training.

    Returns
    -------
    tuple[nn.Module, list[dict[str, Any]]]
        Trained model and list of epoch trace records.
    """
    is_spl = strategy in ("self_paced", "sp_ddcl_density", "sp_ddcl_point")
    start_epoch = 1
    current_lambda = float(lambda_init)
    history: list[dict[str, Any]] = []
    total_opt_steps = 0
    total_exposures = 0

    ckpt_path = Path(checkpoint_dir) / "checkpoint.pt" if checkpoint_dir is not None else None

    # Handle resume if requested
    if resume and ckpt_path is not None and ckpt_path.is_file():
        ckpt = load_checkpoint(ckpt_path, expected_fingerprint=run_fingerprint)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        restore_rng_states(ckpt["rng_states"])
        start_epoch = ckpt["epoch"] + 1
        current_lambda = float(ckpt.get("current_lambda", current_lambda))
        history = ckpt.get("history", [])
        acct = ckpt.get("accounting", {})
        total_opt_steps = acct.get("total_optimizer_steps", 0)
        total_exposures = acct.get("total_sample_exposures", 0)

    for epoch in range(start_epoch, epochs + 1):
        if is_spl:
            epoch_acct, _ = train_single_epoch_spl(
                model=model,
                optimizer=optimizer,
                batches=batches,
                current_lambda=current_lambda,
                pace_policy=pace_policy,
            )
            # Update lambda for next epoch
            next_lambda = update_lambda(current_lambda, lambda_max=lambda_max, growth=lambda_growth)
        else:
            epoch_acct = train_single_epoch_standard(
                model=model,
                optimizer=optimizer,
                batches=batches,
            )
            next_lambda = 0.0

        epoch_acct.epoch = epoch
        total_opt_steps += epoch_acct.optimizer_steps
        total_exposures += epoch_acct.sample_exposures

        # Validation evaluation
        if X_val is not None and y_val is not None:
            val_loss, val_metrics = evaluate_dataset(
                model=model,
                X=X_val,
                y=y_val,
                num_classes=num_classes,
            )
            epoch_acct.val_loss = val_loss
            epoch_acct.val_accuracy = float(val_metrics["accuracy"])
            epoch_acct.val_macro_f1 = float(val_metrics["macro_f1"])

        record = {
            "epoch": epoch,
            "train_loss": epoch_acct.pass1_loss,
            "pass2_loss": epoch_acct.pass2_loss,
            "val_loss": epoch_acct.val_loss,
            "val_accuracy": epoch_acct.val_accuracy,
            "val_macro_f1": epoch_acct.val_macro_f1,
            "lambda": current_lambda if is_spl else None,
            "optimizer_steps": epoch_acct.optimizer_steps,
            "total_optimizer_steps": total_opt_steps,
            "sample_exposures": epoch_acct.sample_exposures,
            "total_sample_exposures": total_exposures,
            "easy_count": len(epoch_acct.easy_batch_ids),
            "hard_count": len(epoch_acct.hard_batch_ids),
            "replayed": epoch_acct.replayed,
            "fallback_used": epoch_acct.fallback_used,
            "duration_seconds": epoch_acct.duration_seconds,
        }
        history.append(record)

        if tracker is not None:
            tracker.log_epoch(record)

        if is_spl:
            current_lambda = next_lambda

        # Save periodic checkpoint
        if ckpt_path is not None and (epoch % checkpoint_freq == 0 or epoch == epochs):
            save_checkpoint(
                filepath=ckpt_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                current_lambda=current_lambda,
                run_fingerprint=run_fingerprint,
                accounting={
                    "total_optimizer_steps": total_opt_steps,
                    "total_sample_exposures": total_exposures,
                },
                history=history,
            )

    return model, history
