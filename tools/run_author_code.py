"""Chạy lại code tác giả (ECOLS-research-group/DDCL, bản clone ở ML/DDCL) trên một dataset.

Mục tiêu: dùng NGUYÊN VĂN code tác giả (Dataset, curriculum_lib, custom_callback) — không sửa file nào trong
ML/DDCL. File này chỉ thay `distribution.py` (script gốc cần hộp thoại tkinter chọn file và ghi file kết quả)
bằng một vòng lặp tương đương, và ghi lại thêm những gì code gốc bỏ đi:
  * toàn bộ `history` của model.fit (loss, accuracy, val_loss, val_accuracy theo epoch) — code gốc chỉ giữ `loss`;
  * metric test của từng lần chạy (Accuracy/Recall/Precision/F1 do chính `compute_result_metrics` của tác giả tính).

Phải chạy bằng môi trường TensorFlow riêng (Python 3.12, TF 2.17.0, Keras 3.4.1):
    ML/.venv-author-tf/Scripts/python.exe tools/run_author_code.py --dataset Haberman --strategy 0 --runs 30

Mỗi tiến trình chạy 1 chiến lược (0..5 theo constants.TrainingStrategies), tuần tự `--runs` lần.
Kết quả: outputs/author_rerun/<Dataset>/<strategy_name>/run_<k>.json  (chạy lại sẽ bỏ qua run đã xong).

Khác biệt DUY NHẤT so với chạy gốc: đặt seed cho numpy/random/TF theo (chiến lược, lần chạy) để lặp lại được.
Code tác giả không đặt seed (mỗi lần chạy là một split ngẫu nhiên khác nhau); seed ở đây chỉ chọn một chuỗi
split cụ thể, không làm đổi thuật toán.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")       # khuyến nghị của tác giả (KMeans)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLBACKEND", "Agg")           # dataset.py vẽ KDE bằng matplotlib, không cần hiển thị

ROOT = Path(__file__).resolve().parents[1]
AUTHOR_DIR = ROOT.parent / "DDCL"
OUT_ROOT = ROOT / "outputs" / "author_rerun"
sys.path.insert(0, str(AUTHOR_DIR))

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402
import tensorflow as tf                 # noqa: E402

tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

import curriculum_lib as cl             # noqa: E402  (code tác giả)
from constants import CLASS_COL, TRAIN_STRATEGY_FULLNAMES, TrainingStrategies as TS  # noqa: E402
from dataset import Dataset             # noqa: E402  (code tác giả)

BATCH_SIZE = 32  # config.ini của tác giả
SIZES_RE = re.compile(r"Using (\d+) samples for training, (\d+) for validation and (\d+) for testing")


def build_train_dataset(experiment_dataset: Dataset, strategy: int):
    """Sao y các nhánh của distribution.py."""
    oversample = strategy in (TS.CL_DENSITY, TS.CL_POINT, TS.SP_DENSITY, TS.SP_POINT)
    spl = strategy in (TS.SELF_PACED, TS.SP_DENSITY, TS.SP_POINT)
    if strategy in (TS.NO_CL, TS.SELF_PACED):
        train = pd.concat(experiment_dataset.all_data, ignore_index=True)
    elif strategy in (TS.CL_DENSITY, TS.SP_DENSITY):
        train = experiment_dataset.prepare_curriculum(oversample, False)
    else:  # CL_POINT, SP_POINT
        train = experiment_dataset.prepare_curriculum(oversample, True)
    return train, spl


def install_history_capture(store: dict) -> None:
    """Bọc dataset_create_model để lấy History của model.fit mà không đổi hành vi."""
    original = cl.dataset_create_model

    def wrapped(*args, **kwargs):
        model = original(*args, **kwargs)
        original_fit = model.fit

        def fit(*a, **k):
            history = original_fit(*a, **k)
            store["history"] = {key: [float(v) for v in vals] for key, vals in history.history.items()}
            return history

        model.fit = fit
        return model

    cl.dataset_create_model = wrapped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Haberman", help="tên file trong RealData/ecols (không .csv), khớp hyperparameter_values")
    ap.add_argument("--strategy", type=int, required=True, choices=range(6))
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--first-run", type=int, default=1)
    ap.add_argument("--out", default=str(OUT_ROOT))
    args = ap.parse_args()

    strategy_name = TS.get_name(args.strategy)
    out_root = Path(args.out).resolve()   # phải resolve TRƯỚC khi chdir
    dataset_path = AUTHOR_DIR / "RealData" / "ecols" / f"{args.dataset}.csv"
    out_dir = out_root / args.dataset / strategy_name
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_root / "_work" / f"{args.dataset}_{strategy_name}"   # nơi code tác giả có thể ghi file tạm
    workdir.mkdir(parents=True, exist_ok=True)
    os.chdir(workdir)

    captured: dict = {}
    install_history_capture(captured)

    for run in range(args.first_run, args.first_run + args.runs):
        target = out_dir / f"run_{run:02d}.json"
        if target.exists():
            continue
        seed = args.strategy * 1000 + run
        random.seed(seed)
        np.random.seed(seed)
        tf.keras.utils.set_random_seed(seed)
        captured.clear()
        cl.metrics_list.clear()

        t0 = time.time()
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            experiment_dataset = Dataset(str(dataset_path), CLASS_COL)
            train_dataset, spl = build_train_dataset(experiment_dataset, args.strategy)
            predictions: list = []
            cl.experiments_neural_network(train_dataset, CLASS_COL, BATCH_SIZE, args.dataset,
                                          "train_and_predict", f"NN-{args.dataset}-{strategy_name}",
                                          run, spl, False, predictions)
        text = log.getvalue()
        sizes = SIZES_RE.search(text)
        record = {
            "dataset": args.dataset,
            "strategy": strategy_name,
            "strategy_fullname": TRAIN_STRATEGY_FULLNAMES[args.strategy],
            "run": run,
            "seed": seed,
            "n_train": int(sizes.group(1)) if sizes else None,
            "n_val": int(sizes.group(2)) if sizes else None,
            "n_test": int(sizes.group(3)) if sizes else None,
            "test": cl.metrics_list[-1],
            "history": captured["history"],
            "duration_seconds": time.time() - t0,
        }
        target.write_text(json.dumps(record), encoding="utf-8")
        print(f"[{strategy_name}] run {run:02d} acc={record['test']['Accuracy']:.4f} "
              f"sizes={record['n_train']}/{record['n_val']}/{record['n_test']} {record['duration_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
