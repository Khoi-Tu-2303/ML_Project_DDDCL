"""Vẽ kết quả chạy lại code tác giả (outputs/author_rerun) và so với paper + bản PyTorch của nhóm.

Chạy bằng môi trường chính (không cần TensorFlow):
    python tools/plot_author_rerun.py Haberman

Đầu ra (outputs/figures/):
    <ds>_author_rerun_curves.png      train/val loss + accuracy theo epoch (mean ± 1 std)
    <ds>_author_rerun_vs_paper.png    test accuracy: paper vs code tác giả chạy lại vs PyTorch của nhóm
    <ds>_author_rerun_summary.md      bảng số liệu
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RERUN = ROOT / "outputs" / "author_rerun"
OUT = ROOT / "outputs" / "figures"

# (tên thư mục của tác giả, tên chiến lược trong repo nhóm, nhãn)
STRATS = [
    ("NO_CL", "no_curriculum", "No curriculum"),
    ("SELF_PACED", "self_paced", "Self-paced"),
    ("CL_POINT", "ddcl_point", "DDCL point"),
    ("CL_DENSITY", "ddcl_density", "DDCL density"),
    ("SP_POINT", "sp_ddcl_point", "SP-DDCL point"),
    ("SP_DENSITY", "sp_ddcl_density", "SP-DDCL density"),
]
# Table 1 của paper (accuracy test %, mean ± SD). Nguồn: PROGRESS.md §bảng R1 (trích từ Table 1, p. 8).
PAPER = {
    "Haberman": {
        "no_curriculum": (65.833, 8.898), "ddcl_density": (69.167, 10.353), "ddcl_point": (67.222, 9.424),
        "self_paced": (64.861, 8.785), "sp_ddcl_density": (69.028, 10.741), "sp_ddcl_point": (68.056, 10.054),
    },
}
RED, BLUE, GREY, GREEN = "#e34948", "#2a78d6", "#8a8984", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def style(ax):
    ax.set_facecolor("white")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c3")
    ax.tick_params(colors=INK2, labelsize=9)


def load(dataset: str) -> dict[str, list[dict]]:
    out = {}
    for author_name, key, _ in STRATS:
        files = sorted((RERUN / dataset / author_name).glob("run_*.json"))
        if files:
            out[key] = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return out


def band(ax, y, color, ls, lw=1.8):
    x = np.arange(1, y.shape[1] + 1)
    mu, sd = y.mean(0), y.std(0, ddof=1)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.16, lw=0)
    ax.plot(x, mu, color=color, lw=lw, ls=ls)


def ylim(arrs, skip=10):
    lo = min((a.mean(0)[skip:] - a.std(0, ddof=1)[skip:]).min() for a in arrs)
    hi = max((a.mean(0)[skip:] + a.std(0, ddof=1)[skip:]).max() for a in arrs)
    pad = 0.04 * (hi - lo)
    return lo - pad, hi + pad


def plot_curves(dataset: str, runs: dict) -> Path:
    keys = [k for _, k, _ in STRATS if k in runs]
    names = {k: l for _, k, l in STRATS}
    rows = [("Loss", [("loss", RED, "-"), ("val_loss", BLUE, "--")]),
            ("Accuracy", [("accuracy", RED, "-"), ("val_accuracy", BLUE, "--")])]
    fig, axes = plt.subplots(len(rows), len(keys), figsize=(3.1 * len(keys), 2.9 * len(rows)),
                             dpi=150, facecolor="white", sharex=True, squeeze=False)
    for r, (title, series) in enumerate(rows):
        data = {h: {k: np.array([run["history"][h] for run in runs[k]]) for k in keys} for h, _, _ in series}
        lims = ylim([a for d in data.values() for a in d.values()])
        for c, k in enumerate(keys):
            ax = axes[r, c]
            style(ax)
            for h, color, ls in series:
                band(ax, data[h][k], color, ls)
            ax.set_ylim(*lims)
            ax.set_xlim(1, data[series[0][0]][k].shape[1])
            if r == 0:
                ax.set_title(names[k], fontsize=11, color=INK, fontweight="bold", loc="left")
            if c == 0:
                ax.set_ylabel(title, color=INK2, fontsize=9.5)
            else:
                ax.tick_params(labelleft=False)
            if r == len(rows) - 1:
                ax.set_xlabel("Epoch", color=INK2, fontsize=9)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=RED, ls="-", lw=2, label="Train (Keras history, dropout bật)"),
                        Line2D([], [], color=BLUE, ls="--", lw=2, label="Validation (32 mẫu, drop_remainder)")],
               loc="lower center", ncol=2, frameon=False, fontsize=10, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    n = sorted({len(v) for v in runs.values()})
    fig.suptitle(f"{dataset} — chạy lại code tác giả (TF 2.17, kiến trúc tác giả, 200 epoch; mean ± 1 std, n = {'/'.join(map(str, n))} lần)",
                 x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{dataset.lower()}_author_rerun_curves.png"
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def ours(dataset: str) -> dict[str, tuple[float, float]]:
    res = {}
    with open(ROOT / "outputs" / "r1_summary.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["dataset_id"] == dataset.lower():
                res[row["strategy"]] = (float(row["mean_accuracy_pct"]), float(row["std_accuracy_ddof0"]))
    return res


def stats(runs, key="Accuracy"):
    v = np.array([r["test"][key] for r in runs]) * 100
    return v.mean(), v.std(ddof=0), v


def plot_vs_paper(dataset: str, runs: dict) -> Path:
    keys = [k for _, k, _ in STRATS if k in runs]
    names = {k: l for _, k, l in STRATS}
    paper, mine = PAPER[dataset], ours(dataset)
    series = [("Paper (Table 1)", GREY, lambda k: paper[k]),
              ("Code tác giả chạy lại (TF)", RED, lambda k: stats(runs[k])[:2]),
              ("PyTorch của nhóm (R1)", BLUE, lambda k: mine[k])]
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=150, facecolor="white")
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    ypos = np.arange(len(keys))[::-1]
    off = [0.24, 0.0, -0.24]
    for (label, color, getter), dy in zip(series, off):
        for y, k in zip(ypos, keys):
            mu, sd = getter(k)
            ax.errorbar(mu, y + dy, xerr=sd, fmt="o", color=color, ecolor=color, elinewidth=2, capsize=3.5,
                        markersize=7, markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                        label=label if k == keys[0] else None)
            ax.text(mu, y + dy + 0.085, f"{mu:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[k] for k in keys], color=INK2)
    ax.set_xlabel("Test accuracy (%)  —  chấm = mean, thanh = ±1 std (ddof=0, như code tác giả)", color=INK2, fontsize=9)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=False, fontsize=9.5, labelcolor=INK2)
    n = sorted({len(v) for v in runs.values()})
    ax.set_title(f"{dataset} — test accuracy: paper vs. code tác giả chạy lại vs. PyTorch nhóm (n = {'/'.join(map(str, n))})",
                 loc="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    path = OUT / f"{dataset.lower()}_author_rerun_vs_paper.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(dataset: str, runs: dict) -> Path:
    paper, mine = PAPER[dataset], ours(dataset)
    lines = [f"# {dataset}: chạy lại code tác giả vs paper vs PyTorch nhóm (test accuracy %, std ddof=0)", "",
             "| Chiến lược | n | Paper | Code tác giả (chạy lại) | Δ so với paper | PyTorch nhóm | F1 nhị phân (tác giả, TB) | val acc cuối (TB) | n_test |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, k, label in STRATS:
        if k not in runs:
            continue
        mu, sd, _ = stats(runs[k])
        f1 = np.mean([r["test"]["F1_Score"] for r in runs[k]]) * 100
        va = np.mean([r["history"]["val_accuracy"][-1] for r in runs[k]]) * 100
        nt = sorted({r["n_test"] for r in runs[k]})
        lines.append(f"| {label} | {len(runs[k])} | {paper[k][0]:.2f} ± {paper[k][1]:.2f} | {mu:.2f} ± {sd:.2f} | "
                     f"{mu - paper[k][0]:+.2f} | {mine[k][0]:.2f} ± {mine[k][1]:.2f} | {f1:.1f} | {va:.1f} | {'/'.join(map(str, nt))} |")
    path = OUT / f"{dataset.lower()}_author_rerun_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["Haberman"]:
        r = load(ds)
        print(plot_curves(ds, r))
        print(plot_vs_paper(ds, r))
        print(write_summary(ds, r))
