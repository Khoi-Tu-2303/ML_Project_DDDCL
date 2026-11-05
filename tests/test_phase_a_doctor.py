"""
Tests for Phase A Environment & Doctor Verification.
Verifies system environment, PyTorch CPU determinism, and CLI doctor.
"""

import os
import platform
import subprocess
import sys
import pytest


def test_python_version():
    """Verify runtime is Python 3.11.x as required by reference environment."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11, f"Expected Python 3.11, found {sys.version}"


def test_core_imports():
    """Verify all locked core scientific packages are importable."""
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "sklearn",
        "imblearn",
        "torch",
        "matplotlib",
        "seaborn",
        "huggingface_hub",
        "wandb",
        "yaml",
        "dotenv",
    ]
    for pkg in packages:
        mod = __import__(pkg)
        assert hasattr(mod, "__file__"), f"Failed to import {pkg}"


def test_torch_cpu_forward_backward():
    """Verify PyTorch CPU forward, backward, loss finiteness, and weight update."""
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)

    model = torch.nn.Linear(2, 1)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    initial_weight = model.weight.detach().clone()
    optimizer.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).item(), "Loss is not finite"
    assert not torch.equal(initial_weight, model.weight.detach()), "Weights were not updated"


def test_torch_cpu_determinism():
    """Verify identical execution trajectories across identical seed initializations."""
    import torch

    def run_step(seed: int = 123):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        model = torch.nn.Linear(4, 2)
        x = torch.randn(8, 4)
        y = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        out = model(x)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        return float(loss.item()), model.weight.detach().clone()

    loss1, w1 = run_step(123)
    loss2, w2 = run_step(123)
    assert abs(loss1 - loss2) < 1e-7, "Losses differ between runs with same seed"
    assert torch.equal(w1, w2), "Weights differ between runs with same seed"

    # Different seed should produce different weights
    _, w3 = run_step(456)
    assert not torch.equal(w1, w3), "Weights are identical despite different seeds"


def test_cli_doctor():
    """Verify `python -m dddcl.cli doctor` runs and returns exit code 0."""
    from dddcl.cli import run_doctor
    exit_code = run_doctor()
    assert exit_code == 0, "CLI doctor reported failures"

