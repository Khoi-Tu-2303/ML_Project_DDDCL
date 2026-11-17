"""Vẽ đường cong huấn luyện (mean ± std qua các seed) cho từng dataset.

Dùng:  python tools/plot_training_curves.py haberman [new_thyroid banknote ...]
       python tools/plot_training_curves.py haberman --runs-dir outputs/quick_analysis/author_tuned_e600/runs --tag _author_tuned_e600 --title-note "..."
Đầu vào : outputs/runs_final/<dataset>_<strategy>_s<seed>/{trace.jsonl,metrics.json}
Đầu ra  : outputs/figures/<dataset>_curves.png  (train/val loss, val acc, val macro-F1 theo epoch)
          outputs/figures/<dataset>_test.png    (test metrics cuối cùng, mean ± std)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs_final"   # có thể đổi bằng --runs-dir
TAG = ""; TAG_TITLE = ""                                  # hậu tố tên file/tiêu đề, đổi bằng --tag
OUT = ROOT / "outputs" / "figures"

# Thứ tự cố định = slot categorical 1..6 của palette tham chiếu (qua adjacent-pair CVD check).
STRATEGIES = [
    ("no_curriculum", "No curriculum", "#2a78d6"),
    ("self_paced", "Self-paced", "#eb6834"),
    ("ddcl_point", "DDCL point", "#1baf7a"),
    ("ddcl_density", "DDCL density", "#eda100"),
    ("sp_ddcl_point", "SP-DDCL point", "#e87ba4"),
    ("sp_ddcl_density", "SP-DDCL density", "#6250d6"),
]
CURVE_PANELS = [
    ("train_loss", "Train loss"),
    ("val_loss", "Validation loss"),
    ("val_accuracy", "Validation accuracy"),
    ("val_macro_f1", "Validation macro-F1"),
]
TEST_PANELS = [
    ("test_loss", "Test loss"),
    ("test_accuracy", "Test accuracy"),
    ("test_balanced_accuracy", "Test balanced accuracy"),
    ("test_macro_f1", "Test macro-F1"),
]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load_trace(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    """Đọc trace.jsonl; nếu một epoch bị ghi trùng thì giữ dòng cuối."""
    by_epoch: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_epoch[row["epoch"]] = row
    epochs = sorted(by_epoch)
    out = {"epoch": np.array(epochs)}
    for k in keys:
        out[k] = np.array([by_epoch[e][k] for e in epochs], dtype=float)
    return out


def collect(dataset: str):
    curves, tests = {}, {}
    for sid, _, _ in STRATEGIES:
        dirs = sorted(RUNS.glob(f"{dataset}_{sid}_s*"), key=lambda p: int(p.name.rsplit("_s", 1)[1]))
        dirs = [d for d in dirs if (d / "trace.jsonl").exists()]
        if not dirs:
            continue
        traces = [load_trace(d / "trace.jsonl", [k for k, _ in CURVE_PANELS]) for d in dirs]
        n = min(len(t["epoch"]) for t in traces)
        curves[sid] = {
            "epoch": traces[0]["epoch"][:n],
            **{k: np.stack([t[k][:n] for t in traces]) for k, _ in CURVE_PANELS},  # (seeds, epochs)
        }
        metrics = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in dirs]
        tests[sid] = {k: np.array([m[k] for m in metrics], dtype=float) for k, _ in TEST_PANELS}
    return curves, tests


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=9)


# Panel hiển thị: (tiêu đề, [(key, nét)]) — train & val loss chung một panel.
PANELS = [
    ("Loss (train: nét liền, validation: nét đứt)", [("train_loss", "-"), ("val_loss", "--")]),
    ("Validation accuracy", [("val_accuracy", "-")]),
    ("Validation macro-F1", [("val_macro_f1", "-")]),
]


def robust_ylim(curves: dict, keys: list[str], sids: list[str], skip: int = 10) -> tuple[float, float]:
    """Khoảng y theo mean±std từ epoch `skip` trở đi (bỏ các epoch đầu có std rất lớn)."""
    lo, hi = np.inf, -np.inf
    for sid in sids:
        for key in keys:
            y = curves[sid][key][:, skip:]
            mu, sd = y.mean(0), y.std(0, ddof=1)
            lo, hi = min(lo, (mu - sd).min()), max(hi, (mu + sd).max())
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def draw(ax, c: dict, key: str, ls: str, color: str, *, band: bool, lw: float, label=None, alpha=1.0):
    x, y = c["epoch"], c[key]
    mu, sd = y.mean(0), y.std(0, ddof=1)
    if band:
        ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
    ax.plot(x, mu, color=color, lw=lw, ls=ls, label=label, alpha=alpha)


RED, BLUE = "#e34948", "#2a78d6"   # train = đỏ, validation = xanh dương
KEY_COLOR = {"train_loss": RED, "val_loss": BLUE, "val_accuracy": BLUE, "val_macro_f1": BLUE}


def plot_curves(dataset: str, curves: dict) -> Path:
    """Lưới: mỗi cột một chiến lược (No curriculum đứng riêng ở cột đầu), mỗi hàng một nhóm metric.
    Train loss = đỏ (nét liền), validation = xanh dương (loss nét đứt)."""
    sids = [s for s, _, _ in STRATEGIES if s in curves]
    names = {s: l for s, l, _ in STRATEGIES}
    fig, axes = plt.subplots(len(PANELS), len(sids), figsize=(3.1 * len(sids), 2.7 * len(PANELS)),
                             dpi=150, facecolor="white", sharex=True, squeeze=False)
    n_seeds = sorted({c["train_loss"].shape[0] for c in curves.values()})
    for r, (title, series) in enumerate(PANELS):
        ylim = robust_ylim(curves, [k for k, _ in series], list(curves))
        for c, sid in enumerate(sids):
            ax = axes[r, c]
            style(ax)
            for key, ls in series:
                draw(ax, curves[sid], key, ls, KEY_COLOR[key], band=True, lw=1.8)
            ax.set_ylim(*ylim)
            ax.set_xlim(1, curves[sid]["epoch"][-1])
            if r == 0:
                ax.set_title(names[sid], fontsize=11, color=INK, fontweight="bold", loc="left")
            if c == 0:
                ax.set_ylabel(title.split(" (")[0], color=INK2, fontsize=9.5)
            else:
                ax.tick_params(labelleft=False)
            if r == len(PANELS) - 1:
                ax.set_xlabel("Epoch", color=INK2, fontsize=9)
    from matplotlib.lines import Line2D
    key_handles = [Line2D([], [], color=RED, ls="-", lw=2, label="Train loss"),
                   Line2D([], [], color=BLUE, ls="--", lw=2, label="Validation loss"),
                   Line2D([], [], color=BLUE, ls="-", lw=2, label="Validation accuracy / macro-F1")]
    fig.legend(handles=key_handles, loc="lower center", ncol=3, frameon=False, fontsize=10,
               labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{dataset}{TAG_TITLE} — đường cong huấn luyện theo chiến lược (mean ± 1 std, n = {'/'.join(map(str, n_seeds))} seed)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset}{TAG}_curves.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def plot_test(dataset: str, tests: dict) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), dpi=150, facecolor="white", sharey=False)
    sids = [s for s, _, _ in STRATEGIES if s in tests]
    ypos = np.arange(len(sids))[::-1]
    for ax, (key, title) in zip(axes, TEST_PANELS):
        style(ax)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.grid(axis="y", visible=False)
        for y, sid in zip(ypos, sids):
            color = next(c for s, _, c in STRATEGIES if s == sid)
            v = tests[sid][key]
            mu, sd = v.mean(), v.std(ddof=1)
            ax.errorbar(mu, y, xerr=sd, fmt="o", color=color, ecolor=color, elinewidth=2.2,
                        capsize=4, markersize=8, markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            ax.text(mu, y + 0.28, f"{mu:.3f}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_yticks(ypos)
        ax.set_yticklabels([next(l for s, l, _ in STRATEGIES if s == sid) for sid in sids] if ax is axes[0] else [])
        ax.set_ylim(-0.7, len(sids) - 0.2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    n = sorted({len(v["test_loss"]) for v in tests.values()})
    fig.suptitle(f"{dataset}{TAG_TITLE} — kết quả test cuối cùng (chấm = mean, thanh = ±1 std, n = {'/'.join(map(str, n))} seed)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = OUT / f"{dataset}{TAG}_test.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def main(datasets: list[str]) -> None:
    for ds in datasets:
        curves, tests = collect(ds)
        if not curves:
            print(f"[skip] không có run nào cho {ds}")
            continue
        print(plot_curves(ds, curves))
        print(plot_test(ds, tests))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", default=["haberman"])
    ap.add_argument("--runs-dir", help="thư mục chứa <dataset>_<strategy>_s<seed>/ (mặc định outputs/runs_final)")
    ap.add_argument("--tag", default="", help="hậu tố tên file, vd _author_tuned_e600")
    ap.add_argument("--title-note", default="", help="ghi chú thêm vào tiêu đề hình")
    a = ap.parse_args()
    if a.runs_dir:
        RUNS = (ROOT / a.runs_dir) if not Path(a.runs_dir).is_absolute() else Path(a.runs_dir)
    TAG, TAG_TITLE = a.tag, (f" [{a.title_note}]" if a.title_note else "")
    main(a.datasets)
