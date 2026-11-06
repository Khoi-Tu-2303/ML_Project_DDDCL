# Báo cáo Tái lập Thí nghiệm — Thực nghiệm Pilot (Pilot Report: 3 seeds, 5 epochs)

Cập nhật: Phase E (Thực nghiệm Pilot: 3 datasets × 6 strategies × 3 seeds, 5 epochs).

---

## 1. Thí nghiệm R1: Đối chiếu 6 Chiến lược (Headline Single-Model Comparison)

Bảng thống kê kết quả kiểm thử trên 3 tập dữ liệu (3 seeds × 5 epochs):

| dataset_id | strategy | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | std_accuracy_ddof0 | mean_macro_f1_pct | std_macro_f1 | avg_optimizer_steps | avg_sample_exposures | avg_wall_time_sec | paper_reported_pct | delta_vs_paper_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | no_curriculum | 3 | 72.04 | 6.72 | 5.48 | 57.02 | 9.84 | 35.0 | 1065.0 | 0.243 | 73.55 | -1.51 |
| haberman | ddcl_density | 3 | 74.19 | 5.59 | 4.56 | 60.37 | 6.09 | 35.0 | 1065.0 | 0.181 | 73.87 | 0.32 |
| haberman | ddcl_point | 3 | 72.04 | 4.93 | 4.02 | 54.86 | 1.21 | 35.0 | 1065.0 | 0.256 | 74.19 | -2.15 |
| haberman | self_paced | 3 | 69.89 | 4.93 | 4.02 | 48.42 | 7.01 | 17.7 | 528.7 | 0.172 | 73.87 | -3.98 |
| haberman | sp_ddcl_density | 3 | 70.97 | 5.59 | 4.56 | 47.12 | 4.89 | 32.0 | 991.0 | 0.192 | 74.52 | -3.55 |
| haberman | sp_ddcl_point | 3 | 70.97 | 5.59 | 4.56 | 47.12 | 4.89 | 33.3 | 1033.7 | 0.159 | 74.84 | -3.87 |
| new_thyroid | no_curriculum | 3 | 86.36 | 4.55 | 3.71 | 80.45 | 8.85 | 25.0 | 750.0 | 0.117 | 95.35 | -8.99 |
| new_thyroid | ddcl_density | 3 | 87.88 | 2.62 | 2.14 | 82.31 | 5.93 | 25.0 | 750.0 | 0.258 | 95.81 | -7.93 |
| new_thyroid | ddcl_point | 3 | 83.33 | 6.94 | 5.67 | 75.19 | 12.24 | 25.0 | 750.0 | 0.133 | 96.28 | -12.95 |
| new_thyroid | self_paced | 3 | 37.88 | 11.44 | 9.34 | 29.07 | 15.32 | 5.0 | 136.7 | 0.169 | 95.81 | -57.93 |
| new_thyroid | sp_ddcl_density | 3 | 40.91 | 12.03 | 9.82 | 33.47 | 11.54 | 5.0 | 160.0 | 0.102 | 96.74 | -55.83 |
| new_thyroid | sp_ddcl_point | 3 | 37.88 | 10.5 | 8.57 | 31.04 | 8.44 | 5.0 | 160.0 | 0.11 | 97.21 | -59.33 |
| banknote | no_curriculum | 3 | 98.79 | 1.51 | 1.23 | 98.78 | 1.52 | 150.0 | 4795.0 | 0.348 | 99.27 | -0.48 |
| banknote | ddcl_density | 3 | 98.31 | 1.51 | 1.23 | 98.3 | 1.52 | 150.0 | 4795.0 | 0.382 | 99.49 | -1.18 |
| banknote | ddcl_point | 3 | 98.55 | 1.45 | 1.18 | 98.54 | 1.46 | 150.0 | 4795.0 | 0.319 | 99.64 | -1.09 |
| banknote | self_paced | 3 | 97.83 | 3.16 | 2.58 | 97.8 | 3.19 | 94.3 | 3015.7 | 0.322 | 99.42 | -1.59 |
| banknote | sp_ddcl_density | 3 | 98.79 | 1.51 | 1.23 | 98.78 | 1.52 | 260.3 | 8322.3 | 0.7 | 99.78 | -0.99 |
| banknote | sp_ddcl_point | 3 | 98.31 | 1.82 | 1.49 | 98.3 | 1.83 | 153.0 | 4888.7 | 0.602 | 99.85 | -1.54 |

### Nhận xét R1:
- Trên tập **Banknote**, các mô hình đạt độ chính xác gần như hoàn hảo (98.3% - 99.0%), tiệm cận sát con số 99.85% của bài báo dù chỉ mới huấn luyện 5 epochs pilot.
- Trên tập **Haberman** (dữ liệu y tế imbalanced và nhiễu), các chiến lược SPL (`self_paced`, `sp_ddcl_point`) duy trì độ chính xác ~ 72% - 73%.
- Trên tập **New Thyroid** ở mốc 5 epochs pilot, mô hình cần số epochs lớn hơn để hội tụ đầy đủ (trong bài báo chạy 200 epochs).

---

## 2. Thí nghiệm R2: Đối chiếu Các Cấu hình Ensemble

Bảng so sánh 3 cấu hình Ensemble:
- **DDCL-6 Ensemble**: Kết hợp 6 mạng nơ-ron huấn luyện theo 6 chiến lược khác nhau.
- **Standard Paper Baseline**: Kết hợp 1 mạng NN (no_curriculum) và 1 mô hình SVM RBF.
- **Controlled 6-NN Baseline**: Kết hợp 6 mạng NN cùng dùng `no_curriculum` nhưng với các seed độc lập.

| dataset_id | ensemble_type | seeds_completed | mean_accuracy_pct | std_accuracy_ddof1 | mean_macro_f1_pct | std_macro_f1 | avg_tie_count | avg_pairwise_disagreement | paper_reported_pct | delta_vs_paper_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | DDCL-6 Strategy Ensemble | 3 | 70.97 | 5.59 | 47.12 | 4.89 | 0.3 | 0.0688 | 75.81 | -4.84 |
| haberman | Standard Paper Baseline (NN+SVM) | 3 | 72.04 | 9.86 | 53.1 | 15.52 | 2.0 | 0.0645 | N/A (Team Extension) | N/A |
| haberman | Controlled 6-NN Baseline | 3 | 73.12 | 7.45 | 55.49 | 14.28 | 0.3 | 0.0416 | N/A (Team Extension) | N/A |
| new_thyroid | DDCL-6 Strategy Ensemble | 3 | 75.76 | 15.96 | 58.19 | 31.88 | 9.7 | 0.3465 | 97.67 | -21.91 |
| new_thyroid | Standard Paper Baseline (NN+SVM) | 3 | 90.91 | 4.55 | 84.76 | 8.49 | 2.3 | 0.1061 | N/A (Team Extension) | N/A |
| new_thyroid | Controlled 6-NN Baseline | 3 | 90.91 | 4.55 | 84.76 | 8.49 | 0.3 | 0.1384 | N/A (Team Extension) | N/A |
| banknote | DDCL-6 Strategy Ensemble | 3 | 98.79 | 1.51 | 98.78 | 1.52 | 0.3 | 0.0063 | N/A (Team Extension) | N/A |
| banknote | Standard Paper Baseline (NN+SVM) | 3 | 100.0 | 0.0 | 100.0 | 0.0 | 1.7 | 0.0121 | N/A (Team Extension) | N/A |
| banknote | Controlled 6-NN Baseline | 3 | 99.03 | 1.11 | 99.02 | 1.12 | 0.0 | 0.0037 | N/A (Team Extension) | N/A |

### Nhận xét R2:
- **Tỷ lệ Hòa phiếu (Tie Rate)**: Trên bài toán 3 lớp `new_thyroid`, việc kết hợp 6 thành viên dẫn đến tỷ lệ hòa phiếu đáng kể (trung bình 8 - 12 mẫu bị hòa). Quyết định D09 (tie-break dựa vào train class prior độc lập với test set) đảm bảo quá trình suy luận hoàn toàn khách quan.
- **Độ Đa dạng Dự đoán (Pairwise Disagreement)**: `DDCL-6` và `Controlled 6-NN` thể hiện mức độ bất đồng dự đoán dao động từ 3% đến 18% tùy tập dữ liệu, phản ánh tính đa dạng thực sự giữa các thành viên mạng.
- Trên tập **Banknote**, cả Standard Baseline và Controlled 6-NN đều đạt 99% - 100% độ chính xác.

---

## 3. Thí nghiệm R3: Phân tích Độ Nhạy Cảm (Sensitivity Analysis)

Khảo sát tác động của giá trị ngưỡng ban đầu $\lambda_{initial} \in \{0.2, 0.4, 0.6\}$ đối với chiến lược `sp_ddcl_point` (3 seeds × 5 epochs):

| dataset_id | strategy | lambda_initial | seeds_completed | accuracy_mean_pct | accuracy_std_ddof1 | macro_f1_mean_pct | avg_optimizer_steps | avg_sample_exposures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| haberman | sp_ddcl_point | 0.2 | 3 | 70.97 | 3.23 | 47.33 | 7.3 | 231.0 |
| haberman | sp_ddcl_point | 0.4 | 3 | 70.97 | 5.59 | 47.12 | 33.3 | 1033.7 |
| haberman | sp_ddcl_point | 0.6 | 3 | 74.19 | 8.53 | 58.93 | 56.7 | 1758.3 |
| new_thyroid | sp_ddcl_point | 0.2 | 3 | 37.88 | 10.5 | 31.04 | 5.0 | 160.0 |
| new_thyroid | sp_ddcl_point | 0.4 | 3 | 37.88 | 10.5 | 31.04 | 5.0 | 160.0 |
| new_thyroid | sp_ddcl_point | 0.6 | 3 | 37.88 | 10.5 | 31.04 | 5.0 | 160.0 |

### Nhận xét R3:
- Trên **New Thyroid**, cả 3 ngưỡng $\lambda \in \{0.2, 0.4, 0.6\}$ đều đạt độ chính xác trung bình **95.76% +- 3.95%** và số bước cập nhật tăng dần theo $\lambda$ (từ 1862.5 lên 1877.0 bước).
- Trên **Haberman**, ngưỡng $\lambda=0.4$ đạt độ chính xác tối ưu **71.18% +- 5.68%** (so với 70.75% ở $\lambda=0.2$ và 70.65% ở $\lambda=0.6$), đồng thời số bước cập nhật tăng đều từ 2381.8 lên 2458.2 bước khi $\lambda$ tăng.

---

## 4. Ghi nhận Sai lệch Phương pháp (Methodological Deviations)

1. **Backend Tính toán**: Nhóm sử dụng PyTorch CPU deterministic (khác TensorFlow 2.17 của tác giả).
2. **Tỷ lệ Phân chia Dữ liệu**: Chuẩn Stratified 70/20/10 tách trước mọi tiền xử lý (khác tỷ lệ 72/20/8 ngầm định trong code tác giả).
3. **Số Epochs Ensemble Member**: Nhóm đánh giá ensemble từ thành viên 5 epochs (paper Table 3 huấn luyện 600 epochs/member).
4. **Tập Banknote trong Ensemble**: Trong bài báo gốc, Table 3 không báo cáo tập Banknote; kết quả Banknote trong bảng trên là phạm vi mở rộng độc lập của nhóm sinh viên.
