"""Tests for W&B tag and project handling in the experiment tracker."""

from __future__ import annotations

import sys
import types

from dddcl.tracking import DEFAULT_WANDB_PROJECT, ExperimentTracker, build_run_tags


def test_build_run_tags_adds_dataset_and_strategy() -> None:
    cfg = {"tags": ["reproduction", "final", "headline"]}
    assert build_run_tags(cfg, "banknote", "ddcl_density") == [
        "reproduction",
        "final",
        "headline",
        "banknote",
        "ddcl_density",
    ]


def test_build_run_tags_deduplicates_and_handles_missing_config() -> None:
    assert build_run_tags({"tags": ["final", "banknote"]}, "banknote", "self_paced") == [
        "final",
        "banknote",
        "self_paced",
    ]
    assert build_run_tags(None, "haberman", "no_curriculum") == ["haberman", "no_curriculum"]
    assert build_run_tags({"tags": None}, "haberman", "no_curriculum") == ["haberman", "no_curriculum"]


def test_tracker_passes_project_and_tags_to_wandb(tmp_path, monkeypatch) -> None:
    captured: dict = {}

    class _FakeRun:
        url = None

        def log(self, *args, **kwargs) -> None:
            pass

        def finish(self) -> None:
            pass

    def _fake_init(**kwargs):
        captured.update(kwargs)
        return _FakeRun()

    fake_wandb = types.SimpleNamespace(init=_fake_init)
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    ExperimentTracker(
        run_dir=tmp_path,
        config={"run_id": "banknote_ddcl_density_s0"},
        wandb_mode="offline",
        tags=["reproduction", "final", "headline", "banknote", "ddcl_density"],
    )

    assert captured["project"] == DEFAULT_WANDB_PROJECT == "dynamic-ddcl"
    assert "entity" not in captured
    assert captured["tags"] == ["reproduction", "final", "headline", "banknote", "ddcl_density"]
    assert captured["id"] == "banknote_ddcl_density_s0"


def test_tracker_disabled_does_not_import_wandb(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "wandb", None)  # any import would raise
    tracker = ExperimentTracker(run_dir=tmp_path, config={}, wandb_mode="disabled")
    assert (tmp_path / "wandb_url.txt").read_text(encoding="utf-8").strip() == "disabled"
    tracker.finish()
