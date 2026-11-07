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
| **Phase E** | Reproduction + Ensemble | **HOÀN THIỆN ĐỐI CHIẾU CHUẨN XÁC (Dừng tại Phase E, chưa sang F)** | **Tách bạch hoàn toàn Pilot, Final 200 epochs, và Final 600 epochs**:<br>• **Pilot**: 54 runs R1 (5 epochs, 3 seeds), **27 evaluations R2** (3 datasets × 3 cấu hình × 3 seeds), 18 runs R3 (2 datasets × 3 lambdas × 3 seeds, trong đó mốc $\lambda=0.4$ có 6 runs tái sử dụng từ R1), lưu tại `outputs/*_pilot` & `reports/pilot/`.<br>• **Phần cá nhân (User phụ trách) đã hoàn tất**: R1 trên `Haberman`, `New-Thyroid`, `Banknote` (**540 training runs**); R2 200 epochs trên 3 tập (**270 evaluations**); R2 600 epochs trên `Haberman` và `New-Thyroid` (**180 evaluations**; Banknote không có trong Table 3 paper); R3 trên 2 tập (**180 kết quả đánh giá: 120 train mới + 60 tái sử dụng**).<br>• **Phạm vi & Tiến độ toàn nhóm**: Chưa được xác nhận đầy đủ; các datasets còn lại trong bài báo (`Breast Cancer Wisconsin (Diagnostic)`, `Cancer`, `Liver Disorder`, `Pima Indians Diabetes`, `ISOLET`, `Digits`, `Clean1`) thuộc phân công các thành viên khác trong nhóm và để trạng thái "Chưa xác nhận" (không tự ý thu hẹp đồ án).<br>• **Bóc tách khối lượng phần cá nhân**: **2.070 training runs thực tế**, 450 ensemble evaluations, 240 reused predictions/runs (tuyệt đối không đếm trùng hay gộp chung thành 1.170 thực nghiệm độc lập).<br>• **Kiểm chứng Paper Benchmark**: Đã đối chiếu từng ô số liệu với `references/papers/dynamic-ddcl.pdf` (Table 3 gồm 6 datasets: Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman’s Survival, Liver Disorder, Pima Indians Diabetes, New-Thyroid). 44/44 unit tests pass. W&B Dashboard: **PENDING (chưa có public URL)**. |
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
  * `banknote`: **0 evaluations** (Table 3 trong bài báo gốc gồm đúng 6 tập dữ liệu: Breast Cancer Wisconsin (Diagnostic), Cancer, Haberman’s Survival, Liver Disorder, Pima Indians Diabetes và New-Thyroid. Nhóm chọn 2 tập cốt lõi trong core scope của dự án là Haberman và New-Thyroid. Tập Banknote từ Table 2 không xuất hiện trong 6 tập của Table 3 bài báo gốc nên có đúng 0 evaluations ở mốc 600 epochs, và được bảo lưu độc lập ở mốc 200 epochs).

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
  * `banknote`: Không xuất hiện trong 6 tập dữ liệu của Table 3 bài báo gốc; ở mốc 600 epochs có đúng **0 evaluations** trên Banknote (kết quả Banknote được khảo sát mở rộng độc lập ở mốc 200 epochs).

---

### 2.4. Kết quả Thí nghiệm R3 Final (Sensitivity Analysis trên $\lambda_{initial}$: 180 kết quả qua 30 seeds, 120 training runs mới)

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
| **Training Runs** | R2 Độc lập (600 ep) | 2 datasets phân công cá nhân × 30 seeds × [6 DDCL NNs + 1 SVM + 6 Controlled NNs] | **780 runs** | Huấn luyện độc lập 600 epochs trên phần cá nhân phụ trách. |
| **Training Runs** | R3 Sensitivity (200 ep) | 2 datasets × 2 mốc mới $\lambda \in \{0.2, 0.6\}$ × 30 seeds | **120 runs** | Huấn luyện mới mạng MLP khảo sát độ nhạy cảm $\lambda$. |
| **Tổng Training Runs** | **Toàn bộ Phase E Final** | **Toàn bộ các lượt huấn luyện mô hình (không đếm trùng)** | **2,070 runs** | *540 + 630 + 780 + 120 (không tính 78 runs Pilot).* |
| **Ensemble Evaluations** | R2 Final (200 ep) | 3 datasets × 3 cấu hình × 30 seeds | **270 evals** | Bỏ phiếu đa số / tổ hợp dự đoán trên test set. |
| **Ensemble Evaluations** | R2 Table 3 (600 ep) | 2 datasets phân công cá nhân × 3 cấu hình × 30 seeds | **180 evals** | Đánh giá tổ hợp độc lập 600 epochs (`banknote` = 0). |
| **Tổng Ensemble Evaluations**| **Toàn bộ Phase E Final** | **Toàn bộ lượt đánh giá tổ hợp kiểm thử** | **450 evals** | *Không bao gồm 27 evaluations của Pilot (3 datasets × 3 cấu hình × 3 seeds = 27 files json trên đĩa).* |
| **Reused Predictions / Runs**| R2 `ddcl_6strategy` (200 ep)| 3 datasets × 30 seeds | **90 evals** | Tái sử dụng 100% dự đoán test set từ 540 runs R1. |
| **Reused Predictions / Runs**| R2 `standard_nn_svm` (200 ep)| 3 datasets × 30 seeds | **90 evals** | Tái sử dụng dự đoán test set `no_curriculum` từ R1. |
| **Reused Predictions / Runs**| R3 Sensitivity $\lambda=0.4$ | 2 datasets × 30 seeds | **60 runs** | Tái sử dụng 60 runs của R1 `sp_ddcl_point` (mặc định $\lambda=0.4$). |
| **Tổng Reused Results** | **Toàn bộ Phase E Final** | **Toàn bộ lượt tái sử dụng kết quả / runs** | **240 lượt** | *90 + 90 + 60 = 240 lượt tái sử dụng.* |

> [!NOTE]
> - **Phân định rõ ràng**: 3 datasets (`Haberman`, `New-Thyroid`, `Banknote`) là phần việc do cá nhân tôi phụ trách và đã hoàn tất 100%. Phạm vi và tiến độ của toàn nhóm đối với các datasets/thí nghiệm còn lại trong bài báo chưa được xác nhận đầy đủ. Tuyệt đối không tự ý thu hẹp phạm vi đồ án của nhóm theo phần việc cá nhân.
> - **Trạng thái tính toán**: Phần tính toán của cá nhân đã hoàn tất 100% trong phạm vi đã giao; public W&B dashboard vẫn ở trạng thái **PENDING** (toàn bộ logs được lưu trữ cục bộ đầy đủ qua `trace.jsonl` và `metrics.json`). Dừng tại Phase E, không khởi chạy thêm bất kỳ thí nghiệm training nào.

---

### 2.7. Bảng Bàn giao Nhiệm vụ Thực nghiệm (Handover Matrix)

| Tập dữ liệu (Dataset) | Nguồn tham chiếu Paper / Giao thức | Thí nghiệm | Người phụ trách | Trạng thái | Ghi chú Artifacts |
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
| **Banknote** | Paper Table 3 (không có) | R2 Ensemble (600 ep) | Người dùng (User) | **Không áp dụng** | Không thuộc Table 3 bài báo gốc |
| **Breast Cancer Wisconsin (Diagnostic)** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Breast Cancer Wisconsin (Diagnostic)** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Cancer** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Cancer** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Liver Disorder** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Liver Disorder** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **Pima Indians Diabetes** | Table 1 & Table 3 | R1, R2 (600 ep) | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Pima Indians Diabetes** | Phân tích bổ sung của nhóm | R3 Sensitivity | Chưa xác nhận người phụ trách | **Chờ thống nhất** | Chờ thống nhất toàn nhóm có chạy R3 hay không |
| **ISOLET** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Optical Digits** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |
| **Clean1** | Table 2 (p. 9) | R1 Stand-alone | Chưa xác nhận người phụ trách | **Chưa xác nhận** | Chờ tổng hợp toàn nhóm |

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
2. **Trạng thái Gate Phase E**:
   - Phần tính toán và thực nghiệm độc lập của Phase E đối với phần việc cá nhân phụ trách (3 datasets `Haberman`, `New-Thyroid`, `Banknote`) đã hoàn tất 100%. Phạm vi và tiến độ của toàn nhóm đối với các datasets/thí nghiệm còn lại trong bài báo chưa được xác nhận đầy đủ.
   - Trạng thái Public W&B Dashboard vẫn là **PENDING** (toàn bộ logs được lưu trữ cục bộ đầy đủ qua `trace.jsonl` và `metrics.json`).
   - Dừng chính xác tại Phase E, không khởi chạy thêm bất kỳ training nào. Chờ lệnh từ người dùng trước khi chuyển sang Phase F (Audit mã nguồn tác giả).

---

## 4. Kế hoạch Phase F (Audit mã nguồn tác giả)

1. Checkpoint Git toàn bộ mã nguồn và artifacts độc lập của Phase E.
2. Clone kho lưu trữ tham chiếu tác giả `https://github.com/ECOLS-research-group/DDCL` vào thư mục `external/DDCL_author` (bị `.gitignore` bảo vệ tuyệt đối khỏi `src/`).
3. Khảo sát và đối chiếu:
   - Tỷ lệ phân chia split thực tế trong mã nguồn tác giả (kiểm chứng nghi vấn 72/20/8 vs 70/20/10).
   - Công thức tính khoảng cách, chuẩn hóa và phân nhóm quantile trong code tác giả.
   - Logic thực tế của vòng lặp Dynamic SPL trong TensorFlow/Keras so với Algorithm 2.
   - So sánh các đại lượng trung gian (split sizes, order, batch updates) giữa bản độc lập và bản tác giả.
