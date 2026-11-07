# Dự án Tái lập và Khảo sát Dynamic DDCL (ML_Project_DDDCL)

Dự án tái lập độc lập và đánh giá thực nghiệm bài báo:
> **Shonal Chaudhry & Anuraganand Sharma**, *Dynamic Data Distribution-based Curriculum Learning*, **Information Sciences** 702 (2025) 121924. DOI: `10.1016/j.ins.2025.121924`.

---

## 1. Phân định Phạm vi Trách nhiệm: Phần việc Cá nhân vs Toàn nhóm

> [!IMPORTANT]
> **Nguyên tắc phân định phạm vi**:
> - **Phần việc cá nhân (User phụ trách)**: Thực hiện độc lập trên 3 tập dữ liệu được phân công gồm `Haberman's Survival`, `New-Thyroid`, và `Banknote`. **Trạng thái phần cá nhân: Hoàn tất tính toán 100%, public W&B pending**.
> - **Phạm vi & Tiến độ toàn nhóm**: **Chưa được xác nhận đầy đủ**. Các tập dữ liệu và thí nghiệm còn lại trong bài báo gốc thuộc phân công của các thành viên khác trong nhóm. Tuyệt đối **không tự ý thu hẹp phạm vi đồ án** theo phần việc cá nhân.

---

## 2. Bảng Bàn giao Nhiệm vụ Thực nghiệm (Handover Matrix)

| Tập dữ liệu (Dataset) | Nguồn tham chiếu Paper / Giao thức | Thí nghiệm | Người phụ trách | Trạng thái | Ghi chú & Artifacts |
|---|---|---|---|---|---|
| **Haberman's Survival** | Table 1 (p. 8) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/haberman_*`) |
| **Haberman's Survival** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/haberman_*`) |
| **Haberman's Survival** | Table 3 (p. 9) | R2 Ensemble (600 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_600/*/haberman_*`) |
| **Haberman's Survival** | Phân tích bổ sung của nhóm | R3 Sensitivity (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/sensitivity_final/`) |
| **New-Thyroid** | Table 1 (p. 8) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/new_thyroid_*`) |
| **New-Thyroid** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/new_thyroid_*`) |
| **New-Thyroid** | Table 3 (p. 9) | R2 Ensemble (600 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_600/*/new_thyroid_*`) |
| **New-Thyroid** | Phân tích bổ sung của nhóm | R3 Sensitivity (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/sensitivity_final/`) |
| **Banknote** | Table 2 (p. 9) | R1 Headline (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 180 runs (`outputs/runs_final/banknote_*`) |
| **Banknote** | Ngân sách bổ sung của nhóm | R2 Ensemble (200 ep, 30 seeds) | Người dùng (User) | **Hoàn tất tính toán** | 90 evals (`outputs/ensembles_final/*/banknote_*`) |
| **Banknote** | Paper Table 3 (không có) | R2 Ensemble (600 ep) | Người dùng (User) | **Không áp dụng** | Không xuất hiện trong Table 3 bài báo gốc |
| **Breast Cancer Wisconsin (Diagnostic)** | Table 1 (p. 8) & Table 3 (p. 9) | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Breast Cancer Wisconsin (Diagnostic)** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Cancer** | Table 1 (p. 8) & Table 3 (p. 9) | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Cancer** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Liver Disorder** | Table 1 (p. 8) & Table 3 (p. 9) | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Liver Disorder** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Pima Indians Diabetes** | Table 1 (p. 8) & Table 3 (p. 9) | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Pima Indians Diabetes** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **ISOLET** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Optical Digits** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Clean1** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |

---

## 3. Thống kê Minh bạch Khối lượng Phần việc Cá nhân

*Bóc tách theo artifacts thực tế trên đĩa, tuyệt đối không đếm trùng (trạng thái: hoàn tất tính toán, public W&B pending):*

- **Training runs (huấn luyện thực sự)**: **2,070 runs** (không tính 78 runs pilot):
  * R1 Headline (200 epochs): **540 runs** (3 datasets × 6 strategies × 30 seeds).
  * R2 200 epochs (ngân sách bổ sung của nhóm): **630 runs** (3 datasets × 30 seeds × [1 SVM + 6 Controlled NNs]).
  * R2 600 epochs (giao thức Table 3 paper): **780 runs** (2 datasets: `haberman` [390] và `new_thyroid` [390] × 30 seeds × [6 DDCL NNs + 1 SVM + 6 Controlled NNs]).
  * R3 Sensitivity (phân tích bổ sung của nhóm, 200 epochs, train mới): **120 runs** (2 datasets × 2 ngưỡng mới $\lambda \in \{0.2, 0.6\}$ × 30 seeds).
- **Ensemble evaluations (đánh giá tổ hợp test set)**: **450 evaluations** (không tính 27 evaluations pilot):
  * R2 200 epochs (ngân sách bổ sung): **270 evaluations** (3 datasets × 3 cấu hình × 30 seeds).
  * R2 600 epochs (Table 3 paper): **180 evaluations** (2 datasets: `haberman` [90] và `new_thyroid` [90]; `banknote` [0]).
- **Reused Predictions / Runs (kết quả tái sử dụng)**: **240 lượt**:
  * 90 evaluations của `ddcl_6strategy` ở 200 epochs (tái sử dụng 100% dự đoán R1).
  * 90 evaluations của `standard_nn_svm` ở 200 epochs (tái sử dụng dự đoán `no_curriculum` từ R1).
  * 60 runs của mốc $\lambda=0.4$ trong R3 (tái sử dụng từ 60 runs `sp_ddcl_point` của R1).

---

## 4. Trạng thái Dịch vụ & Minh chứng Công khai

1. **Hosted Hugging Face Dataset**:
   - URL: `https://huggingface.co/datasets/KieranTuVanKhoi/dynamic-ddcl-benchmarks`
   - Revision Commit: `140c06864683d8c20a9cc2441a90a3d89099759a`
   - Trạng thái: **COMPLETED / VERIFIED (100% SHA-256 match)** trên cả 3 bộ dữ liệu.
2. **Weights & Biases (W&B) Dashboard**:
   - Trạng thái: **PENDING (Chưa có public URL / chưa kết nối cloud W&B)**.
   - Cơ chế: Lưu trữ cục bộ 100% qua `trace.jsonl` và `metrics.json`. Sẵn sàng đồng bộ bằng `tools/sync_wandb.py` khi cấu hình API key.

---

## 5. Cấu trúc Thư mục Dự án

```
ML_Project_DDDCL/
├── configs/                  # File cấu hình thí nghiệm (YAML)
├── data/                     # Dữ liệu cục bộ (raw, splits, processed)
├── docs/                     # Tài liệu đặc tả và hướng dẫn
├── notebooks/                # Jupyter Notebooks tương tác (00 đến 04)
├── outputs/                  # Kết quả thực nghiệm (runs, ensembles, sensitivity)
│   ├── runs_final/           # 540 runs R1 chính thức
│   ├── ensembles_final/      # 270 evaluations R2 200 epochs
│   ├── ensembles_600/        # 180 evaluations R2 600 epochs Table 3
│   └── sensitivity_final/    # 180 evaluations / 120 runs R3
├── references/               # Tài liệu bài báo gốc (PDF)
├── reports/                  # Báo cáo tổng hợp Markdown và JSON
├── src/dddcl/                # Mã nguồn gói Python độc lập
├── tests/                    # Bộ kiểm thử tự động PyTest (44/44 pass)
├── tools/                    # Các kịch bản tiện ích (build notebook, check data)
├── PROGRESS.md               # Nhật ký tiến độ chi tiết từng Phase
└── pyproject.toml            # Cấu hình dự án và dependencies
```

---

## 6. Hướng dẫn Kiểm thử Hệ thống

```powershell
# Kích hoạt môi trường ảo
.\.venv\Scripts\Activate.ps1

# Chạy toàn bộ 44 kiểm thử tự động
pytest

# Sinh lại các báo cáo tổng hợp
python -m src.dddcl.reporting.aggregate
```
