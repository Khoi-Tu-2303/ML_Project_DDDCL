"""Tests for the pure helpers of tools/sync_wandb.py (no network, no wandb import)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "sync_wandb.py"
_spec = importlib.util.spec_from_file_location("sync_wandb", _SCRIPT)
sync_wandb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sync_wandb)


def _make_offline_run(runs_dir: Path, run_id: str, stamp: str, synced: bool = False) -> Path:
    offline = runs_dir / run_id / "wandb" / f"offline-run-{stamp}-{run_id}"
    offline.mkdir(parents=True)
    wandb_file = offline / f"run-{run_id}.wandb"
    wandb_file.write_bytes(b"x")
    if synced:
        wandb_file.with_suffix(".wandb.synced").write_bytes(b"")
    return wandb_file


def test_find_and_synced_detection(tmp_path: Path) -> None:
    a = _make_offline_run(tmp_path, "banknote_ddcl_density_s0", "20261115_100000")
    b = _make_offline_run(tmp_path, "banknote_ddcl_point_s0", "20261115_100001", synced=True)
    files = sync_wandb.find_wandb_files(tmp_path)
    assert files == sorted([a, b])
    assert not sync_wandb.is_synced(a)
    assert sync_wandb.is_synced(b)
    assert sync_wandb.run_dir_of(a) == tmp_path / "banknote_ddcl_density_s0"


def test_build_summary_picks_scalars_and_per_class_recall() -> None:
    metrics = {
        "run_id": "x",
        "test_accuracy": 0.97,
        "test_macro_f1": 0.96,
        "test_loss": 0.12,
        "duration_seconds": 3.5,
        "test_per_class_recall": {"0": 0.98, "1": 0.95},
        "test_confusion_matrix": [[1, 0], [0, 1]],
        "strategy": "ddcl_density",
    }
    summary = sync_wandb.build_summary(metrics)
    assert summary["test_accuracy"] == 0.97
    assert summary["test_recall_class_0"] == 0.98
    assert summary["test_recall_class_1"] == 0.95
    assert "test_confusion_matrix" not in summary
    assert "strategy" not in summary


def test_read_env_key(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("# comment\nHF_TOKEN=\nWANDB_API_KEY = \"abc123\"\n", encoding="utf-8")
    assert sync_wandb.read_env_key(env) == "abc123"
    env.write_text("WANDB_API_KEY=\n", encoding="utf-8")
    assert sync_wandb.read_env_key(env) is None
    assert sync_wandb.read_env_key(tmp_path / "missing.env") is None


def test_chunked() -> None:
    items = [Path(str(i)) for i in range(5)]
    assert [len(c) for c in sync_wandb.chunked(items, 2)] == [2, 2, 1]
