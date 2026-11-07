# Tiến độ dự án Dynamic DDCL (PROGRESS.md)

Cập nhật lần cuối: 2026-11-06 (Hoàn thiện các kiểm chứng chuẩn xác của Phase E Final theo PDF gốc, đối chiếu 200/600 epochs, chưa chuyển Phase F).

---

## 1. Trạng thái các Phase

| Phase | Tên Phase | Trạng thái | Ghi chú & Checkpoint Gate |
|---|---|---|---|
| **Phase A** | Đọc nguồn, chốt thiết kế, dựng môi trường | **COMPLETED / PASSED** | Đã đọc docs 01–12, course requirements, 2 PDFs. Đã tạo `pyproject.toml`, `DATA_REGISTRY.md`, `DECISIONS.md`, `ASSUMPTIONS.md`, `src/dddcl/cli.py` (doctor), `tests/test_environment.py`. Đã cài đặt `.venv`, `pip check` pass, `dddcl.cli doctor` pass, `pytest` 5/5 pass, `tools/test_data_tools.py` 9/9 pass, `00_environment.ipynb` executed thành công. |
| **Phase B** | Data hosted và split contract | **COMPLETED / PASSED (100% Hosted HF & Local Verified)** | Dữ liệu đã được publish lên HF `KieranTuVanKhoi/dynamic-ddcl-benchmarks` (revision `140c06864683d8c20a9cc2441a90a3d89099759a`). Đã xác thực tải và kiểm tra SHA-256 từ cloud thành công 100%. Đã triển khai `src/dddcl/data/`. Sinh 90 split files stratified 70/20/10 (30 seeds × 3 datasets). `notebooks/01_data_audit.ipynb` executed thành công. |
| **Phase C** | Curriculum + Pace pure logic | **COMPLETED / PASSED** | Đã triển khai `src/dddcl/curriculum/` (geometry, partition, resample, scoring) và `src/dddcl/training/pace.py`. Toàn bộ 9/9 unit tests tính tay, degenerate cases và oracle pass. `notebooks/02_curriculum_walkthrough.ipynb` executed thành công. |
| **Phase D** | Training loop độc lập & checkpoints | **COMPLETED / PASSED** | Đã triển khai PyTorch MLP model (`src/dddcl/models/mlp.py`), vòng lặp huấn luyện độc lập (`src/dddcl/training/loop.py`), atomic checkpoints (`src/dddcl/training/checkpoint.py`), metrics (`src/dddcl/metrics.py`), tracking JSONL/W&B (`src/dddcl/tracking.py`), runner & CLI (`src/dddcl/runner.py`, `cli.py run --dry-run`). 7/7 unit tests pass. `notebooks/03_training_walkthrough.ipynb` executed thành công. |
| **Phase E** | Reproduction + Ensemble | **HOÀN THIỆN ĐỐI CHIẾU CHUẨN XÁC (Dừng tại Phase E, chưa sang F)** | **Tách bạch hoàn toàn Pilot, Final 200 epochs, và Final 600 epochs**:<br>• **Pilot**: 54 runs R1 (5 epochs, 3 seeds), 9 ensembles R2, 6 runs R3, lưu tại `outputs/*_pilot` & `reports/pilot/`.<br>• **Final Core R1**: **540 training runs** (200 epochs, 30 seeds), lưu tại `outputs/runs_final/`.<br>• **Final Core R2 (200 ep)**: **270 evaluations** (3 ensemble types × 30 seeds), lưu tại `outputs/ensembles_final/`.<br>• **Final Core R2 (600 ep)**: **180 evaluations** (2 datasets × 3 ensemble types × 30 seeds; Banknote = 0 do không có trong Table 3 paper), lưu tại `outputs/ensembles_600/`.<br>• **Final Core R3**: **180 training runs** ($\lambda \in \{0.2, 0.4, 0.6\}$ × 30 seeds), lưu tại `outputs/sensitivity_final/`.<br>• **Bóc tách khối lượng**: 2.130 training runs thực tế, 450 ensemble evaluations, 180 reused predictions (tuyệt đối không gộp chung thành 1.170 thực nghiệm độc lập).<br>• **Kiểm chứng Paper Benchmark**: Đã đối chiếu 100% từng ô số liệu với `references/papers/dynamic-ddcl.pdf` (Elsevier Information Sciences 2025). 44/44 unit tests pass. W&B Dashboard: **PENDING (chưa có public URL)**. |
| **Phase F** | Audit mã nguồn tác giả | **PENDING (Chờ lệnh người dùng)** | Độc lập 100%, chưa clone hoặc xem code tác giả. Dừng đúng tại Phase E theo yêu cầu. |
| **Phase G** | Thực nghiệm phản biện lý thuyết (Empirics) | **PENDING** | Giả thuyết E1 (Centroid/Multimodality) và E2 (Batch-level threshold). |
| **Phase H** | Báo cáo, Slide deck & Poster | **PENDING** | Chuẩn bị artifact bàn giao cho môn học. |

---

## 2. Kết quả Thực nghiệm Phase E (Phân định Pilot vs Final)

### 2.1. Quản lý Dữ liệu & Lưu trữ Tách Biệt
- **Thực nghiệm Pilot (5 epochs, 3 seeds: 0, 1, 2)**:
  - Thư mục lưu trữ: `outputs/runs_pilot/`, `outputs/ensembles_pilot/`, `outputs/sensitivity_pilot/`.
  - Báo cáo Pilot: `reports/pilot/reproduction_pilot_summary.md`, `reports/pilot/reproduction_pilot_metrics.json`.
- **Thực nghiệm Final 200 Epochs (30 seeds: 0..29)**:
  - Thư mục lưu trữ: `outputs/runs_final/`, `outputs/ensembles_final/`, `outputs/sensitivity_final/`.
  - Báo cáo Final: `reports/final/reproduction_final_summary.md`, `reports/final/reproduction_final_metrics.json`.
- **Thực nghiệm Final 600 Epochs (Giao thức Table 3 Bài báo)**:
  - Thư mục lưu trữ: `outputs/ensembles_600/`.

---

### 2.2. Kết quả Thí nghiệm R1 Final (Headline Single-Model Benchmark: 540 runs)
*Đối chiếu với Table 1 (p. 8) và Table 2 (p. 9) của `dynamic-ddcl.pdf` (200 epochs, 30 seeds, 70/20/10 split)*:

| Dataset | Chiến lược | Seeds | Accuracy trung bình (Final) | SD mẫu (ddof=1) | Macro-F1 trung bình | Optimizer Steps | Số liệu Paper Table 1/2 | Delta vs Paper (pp) |
|---|---|---|---|---|---|---|---|---|
| **banknote** | `no_curriculum` | 30 | **100.00%** | 0.00% | 100.00% | 6,000.0 | 99.879 ± 0.309% | +0.12 pp |
| **banknote** | `ddcl_density` | 30 | **100.00%** | 0.00% | 100.00% | 6,000.0 | 99.913 ± 0.261% | +0.09 pp |
| **banknote** | `ddcl_point` | 30 | **100.00%** | 0.00% | 100.00% | 6,000.0 | 99.942 ± 0.217% | +0.06 pp |
| **banknote** | `self_paced` | 30 | **100.00%** | 0.00% | 100.00% | 11,784.6 | 99.939 ± 0.227% | +0.06 pp |
| **banknote** | `sp_ddcl_density` | 30 | **100.00%** | 0.00% | 100.00% | 11,951.9 | 99.884 ± 0.296% | +0.12 pp |
| **banknote** | `sp_ddcl_point` | 30 | **100.00%** | 0.00% | 100.00% | 11,875.9 | 99.884 ± 0.371% | +0.12 pp |
| **new_thyroid** | `no_curriculum` | 30 | **95.61%** | 4.38% | 93.39% | 1,000.0 | 93.137 ± 5.280% | +2.47 pp |
| **new_thyroid** | `ddcl_density` | 30 | **95.30%** | 4.38% | 92.85% | 1,000.0 | 95.490 ± 5.409% | -0.19 pp |
| **new_thyroid** | `ddcl_point` | 30 | **95.15%** | 4.76% | 92.66% | 1,000.0 | 94.118 ± 5.683% | +1.03 pp |
| **new_thyroid** | `self_paced` | 30 | **95.76%** | 4.12% | 93.69% | 1,855.3 | 94.902 ± 4.488% | +0.86 pp |
| **new_thyroid** | `sp_ddcl_density` | 30 | **95.61%** | 4.70% | 93.48% | 1,887.4 | 96.078 ± 4.882% | -0.47 pp |
| **new_thyroid** | `sp_ddcl_point` | 30 | **95.76%** | 3.95% | 93.69% | 1,875.3 | 93.725 ± 6.609% | +2.04 pp |
| **haberman** | `no_curriculum` | 30 | **71.29%** | 6.47% | 56.67% | 1,400.0 | 65.833 ± 8.898% | +5.46 pp |
| **haberman** | `ddcl_density` | 30 | **72.47%** | 7.47% | 59.27% | 1,400.0 | 69.167 ± 10.353% | +3.30 pp |
| **haberman** | `ddcl_point` | 30 | **72.15%** | 7.21% | 59.18% | 1,400.0 | 67.222 ± 9.424% | +4.93 pp |
| **haberman** | `self_paced` | 30 | **69.57%** | 6.87% | 56.45% | 2,626.5 | 64.861 ± 8.785% | +4.71 pp |
| **haberman** | `sp_ddcl_density` | 30 | **71.51%** | 5.75% | 55.56% | 2,423.6 | 69.028 ± 10.741% | +2.48 pp |
| **haberman** | `sp_ddcl_point` | 30 | **71.18%** | 5.68% | 57.22% | 2,440.8 | 68.056 ± 10.054% | +3.12 pp |

---

### 2.3. Kết quả Thí nghiệm R2 Final (Ensemble Comparisons)

#### 2.3a. Giao thức Trung gian (200 epochs/thành viên — 30 seeds, 270 evaluations):

| Dataset | Cấu hình Ensemble | Seeds | Accuracy trung bình | SD mẫu | Macro-F1 | Tỷ lệ hòa trung bình | Bất đồng dự đoán | Paper Table 3 (600ep) | Ghi chú so sánh |
|---|---|---|---|---|---|---|---|---|---|
| **haberman** | DDCL-6 Strategy Ensemble | 30 | **71.08%** | 6.25% | 56.00% | 0.4 mẫu | 5.06% | 96.774% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **haberman** | Standard Paper Baseline (NN+SVM) | 30 | **74.84%** | 5.18% | 53.42% | 3.4 mẫu | 10.97% | 90.323% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **haberman** | Controlled 6-NN Baseline | 30 | **71.08%** | 6.37% | 56.40% | 0.5 mẫu | 4.80% | N/A | Tương đương DDCL-6 (variance reduction) |
| **new_thyroid** | DDCL-6 Strategy Ensemble | 30 | **95.61%** | 4.38% | 93.39% | 0.1 mẫu | 0.98% | 100.000% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **new_thyroid** | Standard Paper Baseline (NN+SVM) | 30 | **94.85%** | 4.73% | 91.44% | 0.8 mẫu | 3.48% | 95.455% | Đánh giá 200 epochs (Table 3 là 600 epochs) |
| **new_thyroid** | Controlled 6-NN Baseline | 30 | **95.61%** | 4.85% | 93.15% | 0.1 mẫu | 1.22% | N/A | Hiệu năng tương đương DDCL-6 |
| **banknote** | DDCL-6 Strategy Ensemble | 30 | **100.00%** | 0.00% | 100.00% | 0.0 mẫu | 0.00% | N/A | Mở rộng độc lập của nhóm |
| **banknote** | Standard Paper Baseline (NN+SVM) | 30 | **100.00%** | 0.00% | 100.00% | 0.0 mẫu | 0.00% | N/A | Mở rộng độc lập của nhóm |
| **banknote** | Controlled 6-NN Baseline | 30 | **100.00%** | 0.00% | 100.00% | 0.0 mẫu | 0.00% | N/A | Mở rộng độc lập của nhóm |

#### 2.3b. Giao thức Chuẩn Paper Table 3 (600 epochs/thành viên — Đúng 180 evaluations tại `outputs/ensembles_600/`):

- **Số lượng evaluations**: Đúng **180 evaluations** (không phải 270):
  * `haberman` × `DDCL-6 Strategy Ensemble`: 30 evaluations (seeds 0..29).
  * `haberman` × `Standard Paper Baseline (NN+SVM)`: 30 evaluations (seeds 0..29).
  * `haberman` × `Controlled 6-NN Baseline`: 30 evaluations (seeds 0..29).
  * `new_thyroid` × `DDCL-6 Strategy Ensemble`: 30 evaluations (seeds 0..29).
  * `new_thyroid` × `Standard Paper Baseline (NN+SVM)`: 30 evaluations (seeds 0..29).
  * `new_thyroid` × `Controlled 6-NN Baseline`: 30 evaluations (seeds 0..29).
  * `banknote`: **0 evaluations** (Do Table 3 của bài báo gốc `dynamic-ddcl.pdf`, p. 9 chỉ báo cáo trên Haberman và New-Thyroid, không có Banknote. Tập Banknote được bảo lưu độc lập ở mốc 200 epochs).

| Dataset | Cấu hình Ensemble | Seeds | Accuracy trung bình | SD mẫu (ddof=1) | Macro-F1 | Tỷ lệ hòa trung bình | Bất đồng dự đoán | Paper Table 3 (600ep) | Delta vs Paper (pp) |
|---|---|---|---|---|---|---|---|---|---|
| **haberman** | DDCL-6 Strategy Ensemble | 30 | **68.39%** | 6.58% | 54.95% | 0.8 mẫu | 6.73% | 96.774% ± 0.000% | -28.39 pp |
| **haberman** | Standard Paper Baseline (NN+SVM) | 30 | **74.52%** | 5.31% | 52.39% | 5.4 mẫu | 17.42% | 90.323% ± 0.000% | -15.81 pp |
| **haberman** | Controlled 6-NN Baseline | 30 | **68.39%** | 6.13% | 55.18% | 0.8 mẫu | 7.35% | N/A (Team Extension) | N/A |
| **new_thyroid** | DDCL-6 Strategy Ensemble | 30 | **95.61%** | 4.38% | 93.53% | 0.0 mẫu | 0.42% | 100.000% ± 0.000% | -4.39 pp |
| **new_thyroid** | Standard Paper Baseline (NN+SVM) | 30 | **94.85%** | 4.73% | 91.44% | 0.9 mẫu | 4.24% | 95.455% ± 0.000% | -0.61 pp |
| **new_thyroid** | Controlled 6-NN Baseline | 30 | **95.61%** | 4.54% | 93.30% | 0.0 mẫu | 1.13% | N/A (Team Extension) | N/A |

- **Nhận định Khách quan Table 3**:
  * Trên `new_thyroid`: `Standard Paper Baseline` đạt 94.85% ± 4.73% (so với 95.455% ± 0.000% của bài báo, lệch -0.61 pp). `DDCL-6 Ensemble` đạt 95.61% ± 4.38% (so với 100.000% ± 0.000% của bài báo, lệch -4.39 pp, tương đương sai biệt khoảng ~1 mẫu trên tập test 22 mẫu).
  * Trên `haberman`: `Standard Paper Baseline (NN+SVM)` đạt 74.52% ± 5.31% (so với 90.323% ± 0.000% của bài báo, lệch -15.81 pp) và `DDCL-6 Ensemble` đạt 68.39% ± 6.58% (so với 96.774% ± 0.000% của bài báo, lệch -28.39 pp).
  * **Về hiện tượng độ lệch chuẩn ± 0.000% trong bài báo gốc**: Bài báo gốc công bố SD = ± 0.000% cho cả Standard Ensemble và DDCL Ensemble. Để bảo đảm tính nghiêm cẩn khoa học, nhóm KHÔNG đưa ra nhận định nhân quả võ đoán về việc tác giả dùng single test split cố định hay giả thuyết SVM overfitting/soft-margin khi chưa kiểm chứng; mọi câu hỏi về phương pháp phân chia dữ liệu và đánh giá thực tế của tác giả sẽ được kiểm chứng có đối chứng trong Phase F (Audit mã nguồn tác giả).
  * `banknote`: Không xuất hiện trong Table 3 của bài báo gốc; ở mốc 600 epochs có đúng **0 evaluations** trên Banknote (kết quả Banknote được khảo sát độc lập ở mốc 200 epochs).

---

### 2.4. Kết quả Thí nghiệm R3 Final (Sensitivity Analysis trên $\lambda_{initial}$: 180 runs qua 30 seeds)

| Dataset | Ngưỡng $\lambda_{initial}$ | Seeds | Epochs | Accuracy trung bình | SD mẫu (ddof=1) | Macro-F1 trung bình | Optimizer Steps |
|---|---|---|---|---|---|---|---|
| **haberman** | $\lambda = 0.2$ | 30 | 200 | **70.75%** | 6.50% | 56.61% | 2,381.8 |
| **haberman** | $\lambda = 0.4$ | 30 | 200 | **71.18%** | 5.68% | 57.22% | 2,440.8 |
| **haberman** | $\lambda = 0.6$ | 30 | 200 | **70.65%** | 5.25% | 56.38% | 2,458.2 |
| **new_thyroid** | $\lambda = 0.2$ | 30 | 200 | **95.76%** | 3.95% | 93.69% | 1,862.5 |
| **new_thyroid** | $\lambda = 0.4$ | 30 | 200 | **95.76%** | 3.95% | 93.69% | 1,875.3 |
| **new_thyroid** | $\lambda = 0.6$ | 30 | 200 | **95.76%** | 3.95% | 93.69% | 1,877.0 |

**Nhận định R3**:
- Khác biệt rõ rệt so với mốc pilot 5 epochs (khi fallback ở $\lambda=0.6$ khiến số bước chỉ có 5), ở mốc 200 epochs thuật toán có đủ thời gian để hàm tăng trưởng $\lambda_{growth}=1.1$ nới rộng ngưỡng, cho phép tất cả các batch được tiếp cận đầy đủ.
- Trên `new_thyroid`: Cả 3 mốc $\lambda \in \{0.2, 0.4, 0.6\}$ đều đạt độ chính xác trung bình 95.76% ± 3.95%.
- Trên `haberman`: Trong 3 mốc khảo sát cục bộ $\lambda \in \{0.2, 0.4, 0.6\}$, mốc $\lambda=0.4$ ghi nhận độ chính xác trung bình mẫu là 71.18% ± 5.68% (cao hơn 70.75% ở $\lambda=0.2$ khoảng 0.43 điểm phần trăm và 70.65% ở $\lambda=0.6$ khoảng 0.53 điểm phần trăm). Khoảng chênh lệch này nằm trong phạm vi độ lệch chuẩn mẫu (~5.3 - 6.5 pp), do đó chỉ phản ánh tương quan trong phạm vi 3 điểm khảo sát, không khẳng định $\lambda=0.4$ là giá trị tối ưu toàn cục.
- Tỷ lệ bất đồng dự đoán (pairwise disagreement) giữa các thành viên ở mức từ 0.4% đến 6.7% ghi nhận sự khác biệt dự đoán ở mức độ mẫu đơn lẻ giữa các chiến lược sắp xếp mẫu, chưa đủ cơ sở để khẳng định quy luật giảm phương sai (variance reduction) toàn diện.

---

### 2.5. Kiểm thử Hệ thống & Artifacts
- **Kiểm thử PyTest**: **44/44 tests PASS 100%**:
  - `tests/test_environment.py`: 5/5 PASS.
  - `tests/test_data_pipeline.py`: 6/6 PASS.
  - `tests/test_curriculum.py`: 9/9 PASS.
  - `tests/test_training.py`: 7/7 PASS.
  - `tests/test_ensemble.py`: 8/8 PASS (Bao gồm test tie-breaking D09, alignment và controlled ensemble).
  - `tools/test_data_tools.py`: 9/9 PASS.
- **Notebook Trực quan hóa**:
  - `notebooks/04_reproduction.ipynb`: Đã tạo mới và thực thi hoàn tất trên kernel `ddcl-local`, bao gồm biểu đồ R1 30 seeds, bảng R2 200 epochs và R2 600 epochs đối chiếu Table 3 bài báo, biểu đồ R2 Ensemble & Disagreement, biểu đồ R3 Sensitivity, và biểu đồ so sánh trực tiếp Pilot vs Final.

### 2.6. Thống kê Minh bạch Khối lượng Thực nghiệm & Trạng thái Công khai

#### Bóc tách Khối lượng (Phân định Training Runs vs Ensemble Evaluations vs Kết quả Tái sử dụng):
*Tuyệt đối không cộng gộp lẫn lộn thành 1.170 thực nghiệm độc lập.*

| Phân loại | Mục / Giao thức | Phạm vi thực nghiệm | Số lượng | Ghi chú phương pháp luận |
|---|---|---|---|---|
| **Training Runs** | R1 Headline (200 ep) | 3 datasets × 6 strategies × 30 seeds | **540 runs** | Huấn luyện mạng MLP 200 epochs thực sự. |
| **Training Runs** | R2 Thành viên mới (200 ep) | 3 datasets × 30 seeds × [1 SVM + 6 Controlled NNs] | **630 runs** | Huấn luyện mới mô hình SVM và 6 mạng kiểm soát. |
| **Training Runs** | R2 Độc lập (600 ep) | 2 datasets × 30 seeds × [6 DDCL NNs + 1 SVM + 6 Controlled NNs] | **780 runs** | Huấn luyện độc lập 600 epochs đối chiếu Table 3. |
| **Training Runs** | R3 Sensitivity (200 ep) | 2 datasets × 3 $\lambda$ × 30 seeds | **180 runs** | Huấn luyện mạng MLP khảo sát độ nhạy cảm $\lambda$. |
| **Tổng Training Runs** | **Toàn bộ Phase E Final** | **Toàn bộ các lượt huấn luyện mô hình** | **2,130 runs** | *Không bao gồm 78 runs của giai đoạn Pilot.* |
| **Ensemble Evaluations** | R2 Final (200 ep) | 3 datasets × 3 cấu hình × 30 seeds | **270 evals** | Bỏ phiếu đa số / tổ hợp dự đoán trên test set. |
| **Ensemble Evaluations** | R2 Table 3 (600 ep) | 2 datasets × 3 cấu hình × 30 seeds | **180 evals** | Đánh giá tổ hợp độc lập 600 epochs (`banknote` = 0). |
| **Tổng Ensemble Evaluations**| **Toàn bộ Phase E Final** | **Toàn bộ lượt đánh giá tổ hợp kiểm thử** | **450 evals** | *Không bao gồm 9 evaluations của Pilot.* |
| **Reused Predictions** | R2 `ddcl_6strategy` (200 ep) | 3 datasets × 30 seeds | **90 evals** | Tái sử dụng 100% dự đoán test set từ 540 runs R1. |
| **Reused Predictions** | R2 `standard_nn_svm` (200 ep)| 3 datasets × 30 seeds | **90 evals** | Tái sử dụng dự đoán test set `no_curriculum` từ R1. |

#### Trạng thái Dịch vụ Công khai (Public Evidence):
1. **Hosted Hugging Face Dataset**:
   - URL: `https://huggingface.co/datasets/KieranTuVanKhoi/dynamic-ddcl-benchmarks`
   - Revision Git Commit: `140c06864683d8c20a9cc2441a90a3d89099759a`
   - Trạng thái: **HOÀN TẤT & ĐÃ XÁC THỰC 100%** (SHA-256 trùng khớp hoàn hảo trên cả 3 tập dữ liệu).
2. **Public Weights & Biases (W&B) Dashboard**:
   - Trạng thái: **PENDING (Chưa có public URL / chưa kết nối cloud W&B)**.
   - Cơ chế lưu trữ: Toàn bộ lịch sử epoch, optimizer steps, loss, accuracy được ghi nhận cục bộ 100% qua `trace.jsonl` và `metrics.json` tại từng thư mục run.
   - Script sẵn sàng: `tools/sync_wandb.py` sẵn sàng đồng bộ 1 lệnh lên dashboard khi có API key / URL.

---

## 3. Blockers và Tuân thủ Nguyên tắc Độc lập

1. **Tuyệt đối không vi phạm nguyên tắc độc lập**:
   - Trong suốt toàn bộ Phase A, B, C, D, E, nhóm **chưa từng clone hay đọc bất kỳ dòng code nào** từ kho lưu trữ của tác giả (`https://github.com/ECOLS-research-group/DDCL`).
   - Toàn bộ kết quả Pilot và Final đạt được 100% từ việc hiện thực hóa độc lập các phương trình và mô tả trong bài báo gốc và file đặc tả thiết kế.
2. **Sẵn sàng chuyển giao sang Phase F**:
   - Phase E đã hoàn thành đầy đủ cả pilot và final scope được giao.
   - Khi người dùng gửi lệnh tiếp tục, hệ thống sẽ tiến hành checkpoint git và chuyển sang Phase F (Audit mã nguồn tác giả).

---

## 4. Kế hoạch Phase F (Audit mã nguồn tác giả)

1. Checkpoint Git toàn bộ mã nguồn và artifacts độc lập của Phase E.
2. Clone kho lưu trữ tham chiếu tác giả `https://github.com/ECOLS-research-group/DDCL` vào thư mục `external/DDCL_author` (bị `.gitignore` bảo vệ tuyệt đối khỏi `src/`).
3. Khảo sát và đối chiếu:
   - Tỷ lệ phân chia split thực tế trong mã nguồn tác giả (kiểm chứng nghi vấn 72/20/8 vs 70/20/10).
   - Công thức tính khoảng cách, chuẩn hóa và phân nhóm quantile trong code tác giả.
   - Logic thực tế của vòng lặp Dynamic SPL trong TensorFlow/Keras so với Algorithm 2.
   - So sánh các đại lượng trung gian (split sizes, order, batch updates) giữa bản độc lập và bản tác giả.
