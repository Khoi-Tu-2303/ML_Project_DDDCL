"""Script to generate and execute notebooks/03_training_walkthrough.ipynb.

Demonstrates:
1. MLP architecture creation (binary & multiclass).
2. Data preparation & curriculum batching for pilot datasets.
3. Step-by-step SPL execution: scoring, Easy/Hard partition, Pass 1, Replay check, lambda update.
4. Step accounting: optimizer steps vs sample exposures.
5. Atomic Checkpointing and bit-identical resume verification.
6. Execution of experiment dry-run inspection matrix.
"""

from pathlib import Path
import nbformat as nbf
from nbclient import NotebookClient


def create_training_walkthrough_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "DDDCL Local",
            "language": "python",
            "name": "ddcl-local",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    }

    cells = []

    # Title & Markdown Overview
    cells.append(nbf.v4.new_markdown_cell(
        "# Notebook 03: Vòng lặp Huấn luyện Độc lập, Checkpoint & Step Accounting (Phase D)\n\n"
        "Notebook này kiểm chứng và minh họa kiến trúc huấn luyện của gói `dddcl`:\n"
        "1. **Kiến trúc PyTorch MLP Classifier**: Khởi tạo trọng số deterministic, hỗ trợ Binary (`BCEWithLogitsLoss`) và Multiclass (`CrossEntropyLoss`).\n"
        "2. **Tạo Mini-batches Ổn định**: Giữ nguyên thứ tự mẫu từ Curriculum mà không dùng random shuffle gây xáo trộn.\n"
        "3. **Vòng lặp Huấn luyện SPL Từng Bước**:\n"
        "   - Đánh giá lỗi $L_b$ của từng batch ở chế độ `eval()` không gradient.\n"
        "   - Lọc batch Easy ($L_b < \\lambda$) và Hard ($L_b \\ge \\lambda$).\n"
        "   - Lượt 1: Huấn luyện trên batch Easy.\n"
        "   - Kiểm tra điều kiện Replay (Algorithm 2) khi lỗi trung bình Easy $< \\lambda$.\n"
        "   - Cập nhật ngưỡng $\\lambda$.\n"
        "4. **Kế toán Điện toán (Accounting)**: Ghi nhận chính xác số bước `optimizer_steps` và số lượt tiếp xúc `sample_exposures`.\n"
        "5. **Cơ chế Checkpoint Atomic & Resume**: Đảm bảo khôi phục bit-identical trạng thái mô hình và RNG."
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_code_cell(
        "import torch\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from pathlib import Path\n"
        "import tempfile\n\n"
        "from dddcl.models.mlp import create_mlp_model\n"
        "from dddcl.curriculum.geometry import compute_geometry\n"
        "from dddcl.curriculum.partition import partition_by_class\n"
        "from dddcl.curriculum.scoring import create_curriculum\n"
        "from dddcl.curriculum.resample import resample_curriculum_bins\n"
        "from dddcl.data.hf import load_dataset_from_release\n"
        "from dddcl.data.splits import load_split\n"
        "from dddcl.data.preprocess import prepare_processed_split\n"
        "from dddcl.training.loop import (\n"
        "    create_curriculum_batches,\n"
        "    evaluate_batch_losses,\n"
        "    evaluate_dataset,\n"
        "    train_single_epoch_spl,\n"
        "    train_dddcl_model,\n"
        ")\n"
        "from dddcl.training.checkpoint import save_checkpoint, load_checkpoint\n"
        "from dddcl.runner import dry_run_matrix\n\n"
        "print('Tất cả các module của dddcl đã sẵn sàng!')"
    ))

    # Cell 2: Markdown Section 1
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Khởi tạo Mô hình MLP & Kiểm tra Forward Pass\n\n"
        "Mô hình `MLPClassifier` được thiết kế linh hoạt với seed khởi tạo deterministic, "
        "đầu ra trả về raw logits và tính toán loss phù hợp theo số lượng lớp."
    ))

    # Cell 3: Code Section 1
    cells.append(nbf.v4.new_code_cell(
        "# Kiểm tra mô hình binary (Haberman: 3 features, 2 classes)\n"
        "mlp_binary = create_mlp_model(input_dim=3, num_classes=2, hidden_dims=(64, 32), seed=42)\n"
        "x_dummy = torch.randn(4, 3)\n"
        "logits_bin = mlp_binary(x_dummy)\n"
        "probs_bin = mlp_binary.predict_proba(x_dummy)\n"
        "preds_bin = mlp_binary.predict(x_dummy)\n\n"
        "print('Binary Model Architecture:')\n"
        "print(mlp_binary)\n"
        "print('\\nLogits shape:', logits_bin.shape)  # (4, 1)\n"
        "print('Probabilities shape:', probs_bin.shape)  # (4, 2)\n"
        "print('Predictions:', preds_bin)"
    ))

    # Cell 4: Markdown Section 2
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Nạp Dữ liệu Pilot & Xây dựng Mini-batches theo Curriculum\n\n"
        "Chúng ta nạp tập dữ liệu `haberman` (đã xác thực từ `hf_release`), "
        "nạp file split phân tầng seed 0, chuẩn hóa dữ liệu trên train-only, "
        "tính toán thứ tự curriculum (`sp_ddcl_point`) và tạo các mini-batches ổn định."
    ))

    # Cell 5: Code Section 2
    cells.append(nbf.v4.new_code_cell(
        "release_dir = Path('../hf_release') if Path('../hf_release').exists() else Path('hf_release')\n"
        "ds_bundle = load_dataset_from_release(release_dir, 'haberman')\n"
        "splits_dir = Path('../outputs/splits') if Path('../outputs/splits').exists() else Path('outputs/splits')\n"
        "split_file = splits_dir / 'paper_interpretation_clean_v1' / 'haberman' / 'seed_0.json'\n"
        "split_bundle = load_split(split_file, bundle=ds_bundle)\n\n"
        "# Tiền xử lý (StandardScaler fit trên train)\n"
        "processed = prepare_processed_split(ds_bundle, split_bundle)\n\n"
        "# Tính curriculum point\n"
        "geom = compute_geometry(processed.X_train_raw, processed.y_train)\n"
        "part = partition_by_class(geom.normalized_distances, processed.y_train, n_levels=5)\n"
        "curr_res = create_curriculum(\n"
        "    strategy='sp_ddcl_point',\n"
        "    sample_ids=processed.train_ids,\n"
        "    normalized_distances=geom.normalized_distances,\n"
        "    bin_assignments=part.bin_assignments,\n"
        "    seed=0,\n"
        ")\n\n"
        "# Tạo các batches kích thước 32\n"
        "batches = create_curriculum_batches(\n"
        "    X_train=processed.X_train_scaled,\n"
        "    y_train=processed.y_train,\n"
        "    train_ids=processed.train_ids,\n"
        "    curriculum_result=curr_res,\n"
        "    batch_size=32,\n"
        ")\n\n"
        "print(f'Tổng số mẫu train: {len(processed.train_ids)}')\n"
        "print(f'Số lượng mini-batches: {len(batches)}')\n"
        "for b in batches:\n"
        "    print(f'  Batch {b.batch_id}: {len(b.sample_ids)} mẫu, X shape: {b.X.shape}')"
    ))

    # Cell 6: Markdown Section 3
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Thực thi Vòng lặp SPL 1 Epoch và Kiểm tra Kế toán Chi tiết\n\n"
        "Theo dõi cơ chế tách batch Easy/Hard ở đầu epoch, cập nhật trọng số trên Easy batches, "
        "và kiểm tra điều kiện Replay."
    ))

    # Cell 7: Code Section 3
    cells.append(nbf.v4.new_code_cell(
        "model = create_mlp_model(input_dim=processed.X_train_scaled.shape[1], num_classes=2, seed=0)\n"
        "optimizer = torch.optim.Adam(model.parameters(), lr=0.01)\n\n"
        "# Đánh giá losses trước update\n"
        "initial_losses = evaluate_batch_losses(model, batches)\n"
        "print('Batch losses ban đầu khi chưa huấn luyện:', [f'{l:.4f}' for l in initial_losses])\n\n"
        "# Chạy 1 epoch SPL với ngưỡng lambda = 0.4\n"
        "epoch_acct, next_lambda = train_single_epoch_spl(\n"
        "    model=model,\n"
        "    optimizer=optimizer,\n"
        "    batches=batches,\n"
        "    current_lambda=0.4,\n"
        "    pace_policy='replay_easy_then_union_v1',\n"
        ")\n\n"
        "print(f'\\nKết quả Epoch 1 SPL:')\n"
        "print(f'  - Ngưỡng lambda sử dụng: {epoch_acct.lambda_value}')\n"
        "print(f'  - Các batch Easy: {epoch_acct.easy_batch_ids}')\n"
        "print(f'  - Các batch Hard: {epoch_acct.hard_batch_ids}')\n"
        "print(f'  - Có kích hoạt Replay không? {epoch_acct.replayed}')\n"
        "print(f'  - Số bước optimizer thực hiện: {epoch_acct.optimizer_steps}')\n"
        "print(f'  - Số lượt mẫu tiếp xúc: {epoch_acct.sample_exposures}')\n"
        "print(f'  - Thời gian thực thi: {epoch_acct.duration_seconds:.4f}s')"
    ))

    # Cell 8: Markdown Section 4
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Xác minh Tính Tái lập & Khôi phục Trạng thái (Bit-identical Checkpoint Resume)\n\n"
        "Chúng ta thực hiện kiểm thử thực nghiệm: huấn luyện 4 epochs liên tục "
        "so sánh với việc dừng ở epoch 2, lưu checkpoint, khởi động lại và resume 2 epochs tiếp theo."
    ))

    # Cell 9: Code Section 4
    cells.append(nbf.v4.new_code_cell(
        "# Chạy 4 epochs liên tục\n"
        "torch.manual_seed(99)\n"
        "m_unint = create_mlp_model(input_dim=3, num_classes=2, seed=99)\n"
        "opt_unint = torch.optim.Adam(m_unint.parameters(), lr=0.005)\n"
        "_, hist_unint = train_dddcl_model(\n"
        "    model=m_unint,\n"
        "    optimizer=opt_unint,\n"
        "    batches=batches,\n"
        "    strategy='sp_ddcl_point',\n"
        "    epochs=4,\n"
        "    lambda_init=0.4,\n"
        "    lambda_max=0.6,\n"
        "    lambda_growth=1.1,\n"
        "    run_fingerprint='walkthrough_demo',\n"
        ")\n\n"
        "# Chạy 2 epochs + Checkpoint\n"
        "with tempfile.TemporaryDirectory() as tmpdir:\n"
        "    ckpt_dir = Path(tmpdir)\n"
        "    torch.manual_seed(99)\n"
        "    m_res = create_mlp_model(input_dim=3, num_classes=2, seed=99)\n"
        "    opt_res = torch.optim.Adam(m_res.parameters(), lr=0.005)\n"
        "    train_dddcl_model(\n"
        "        model=m_res,\n"
        "        optimizer=opt_res,\n"
        "        batches=batches,\n"
        "        strategy='sp_ddcl_point',\n"
        "        epochs=2,\n"
        "        lambda_init=0.4,\n"
        "        lambda_max=0.6,\n"
        "        lambda_growth=1.1,\n"
        "        run_fingerprint='walkthrough_demo',\n"
        "        checkpoint_dir=ckpt_dir,\n"
        "        checkpoint_freq=1,\n"
        "    )\n"
        "    \n"
        "    # Resume 2 epochs tiếp theo\n"
        "    _, hist_res = train_dddcl_model(\n"
        "        model=m_res,\n"
        "        optimizer=opt_res,\n"
        "        batches=batches,\n"
        "        strategy='sp_ddcl_point',\n"
        "        epochs=4,\n"
        "        lambda_init=0.4,\n"
        "        lambda_max=0.6,\n"
        "        lambda_growth=1.1,\n"
        "        run_fingerprint='walkthrough_demo',\n"
        "        checkpoint_dir=ckpt_dir,\n"
        "        resume=True,\n"
        "    )\n\n"
        "# Kiểm tra bit-identical\n"
        "weights_match = all(\n"
        "    torch.equal(m_unint.state_dict()[k], m_res.state_dict()[k])\n"
        "    for k in m_unint.state_dict()\n"
        ")\n"
        "loss_unint = [h['train_loss'] for h in hist_unint]\n"
        "loss_res = [h['train_loss'] for h in hist_res]\n\n"
        "print('Loss 4 epochs liên tục:', [f'{l:.6f}' for l in loss_unint])\n"
        "print('Loss 2 + resume 2 epochs:', [f'{l:.6f}' for l in loss_res])\n"
        "print(f'Mô hình weights giống nhau 100% bit-identical: {weights_match}')\n"
        "assert weights_match, 'Resume weights mismatch!'"
    ))

    # Cell 10: Markdown Section 5
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Kiểm tra Kế hoạch Thực nghiệm Matrix (--dry-run)\n\n"
        "Gọi hàm `dry_run_matrix` để kiểm tra toàn bộ 54 runs của cấu hình pilot `configs/pilot.yaml` "
        "mà không thực hiện huấn luyện hay sửa đổi dữ liệu."
    ))

    # Cell 11: Code Section 5
    cells.append(nbf.v4.new_code_cell(
        "cfg_path = Path('../configs/pilot.yaml') if Path('../configs/pilot.yaml').exists() else Path('configs/pilot.yaml')\n"
        "plan = dry_run_matrix(cfg_path)\n\n"
        "df_runs = pd.DataFrame(plan['runs'])\n"
        "print(f'Config: {plan[\"config_file\"]}')\n"
        "print(f'Tổng số runs: {plan[\"total_fits\"]}')\n"
        "print(f'Số lượng dataset: {len(plan[\"datasets\"])} {plan[\"datasets\"]}')\n"
        "print(f'Số lượng strategy: {len(plan[\"strategies\"])} {plan[\"strategies\"]}')\n"
        "print(f'Số lượng seed: {len(plan[\"seeds\"])} {plan[\"seeds\"]}\\n')\n"
        "print('Trích đoạn ma trận thực nghiệm:')\n"
        "display(df_runs.head(12))"
    ))

    nb.cells = cells
    return nb


def main():
    proj_root = Path(__file__).resolve().parent.parent
    nb_path = proj_root / "notebooks" / "03_training_walkthrough.ipynb"

    print(f"[*] Building notebook: {nb_path}")
    nb = create_training_walkthrough_notebook()

    # Save initial notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    # Execute in-place using NotebookClient with kernel ddcl-local
    print("[*] Executing notebook using kernel 'ddcl-local'...")
    client = NotebookClient(nb, timeout=600, kernel_name="ddcl-local")
    client.execute(cwd=str(proj_root / "notebooks"))

    # Save executed notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"[+] Notebook successfully executed and saved to {nb_path}!")


if __name__ == "__main__":
    main()
