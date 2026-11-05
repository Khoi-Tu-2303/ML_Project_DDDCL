"""Atomic checkpoint saving and exact RNG resume for DDDCL training.

Follows protocols from docs/03_METHOD_SPEC.md, docs/06_ARCHITECTURE.md, and docs/07_VERIFICATION.md:
- Atomic file writes using temporary files and atomic replacement.
- Complete state preservation: model weights, optimizer state, epoch, lambda,
  accounting metrics, and full RNG states (Python random, NumPy, PyTorch CPU).
- Strict fingerprint validation to prevent loading mismatched experiment runs.
"""

from __future__ import annotations

import os
from pathlib import Path
import random
from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def capture_rng_states() -> dict[str, Any]:
    """Capture current state of all pseudo-random number generators."""
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }


def restore_rng_states(rng_states: Mapping[str, Any]) -> None:
    """Restore pseudo-random number generator states exactly."""
    if "python_random" in rng_states:
        random.setstate(rng_states["python_random"])
    if "numpy_random" in rng_states:
        np.random.set_state(rng_states["numpy_random"])
    if "torch_cpu" in rng_states:
        torch.set_rng_state(rng_states["torch_cpu"])


def save_checkpoint(
    filepath: str | Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    current_lambda: float,
    run_fingerprint: str,
    accounting: Mapping[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> Path:
    """Save training checkpoint atomically.
    
    Parameters
    ----------
    filepath : str | Path
        Target checkpoint file path (e.g. checkpoint.pt).
    epoch : int
        Completed epoch number.
    model : nn.Module
        Model whose state_dict is saved.
    optimizer : optim.Optimizer
        Optimizer whose state_dict is saved.
    current_lambda : float
        Current SPL lambda threshold.
    run_fingerprint : str
        Hash identifier of the running experiment.
    accounting : Mapping[str, Any] | None
        Accumulated step counts and exposures.
    history : list[dict[str, Any]] | None
        List of historical epoch metric summaries.

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "current_lambda": float(current_lambda),
        "run_fingerprint": str(run_fingerprint),
        "accounting": dict(accounting) if accounting is not None else {},
        "history": list(history) if history is not None else [],
        "rng_states": capture_rng_states(),
    }

    # Save to temp file first, then atomically replace
    torch.save(payload, temp_path)
    os.replace(temp_path, path)
    return path


def load_checkpoint(
    filepath: str | Path,
    expected_fingerprint: str | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load and validate training checkpoint.
    
    Parameters
    ----------
    filepath : str | Path
        Path to checkpoint file.
    expected_fingerprint : str | None
        If provided, validates that the checkpoint belongs to the exact experiment.
    device : str, default='cpu'
        Target device for tensors.

    Returns
    -------
    dict[str, Any]
        Loaded checkpoint dictionary.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    # Use weights_only=False because checkpoint contains custom dicts and RNG states
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if expected_fingerprint is not None:
        actual_fp = ckpt.get("run_fingerprint")
        if actual_fp != expected_fingerprint:
            raise ValueError(
                f"Checkpoint fingerprint mismatch! Expected {expected_fingerprint}, got {actual_fp}"
            )

    return ckpt
