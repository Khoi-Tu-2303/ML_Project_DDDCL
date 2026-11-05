# Architectural & Methodological Decisions (Sổ quyết định thiết kế)

Tài liệu này ghi nhận các quyết định thiết kế cốt lõi của nhóm theo mẫu tại `docs/10_EVIDENCE_LEDGER.md`. Mọi quyết định đều chỉ rõ nguồn gốc (bài báo, yêu cầu môn học, hay giả định nhóm tự chọn) nhằm bảo đảm tính trung thực khoa học tuyệt đối.

---

### D01: Lựa chọn nền tảng Deep Learning độc lập
- **Câu hỏi**: Dùng thư viện nào để tự triển khai thuật toán Dynamic DDCL?
- **Vị trí bài báo**: Dynamic DDCL mục 5.1 (tác giả dùng mạng nơ-ron với Adam lr=0.001). Mã nguồn tác giả dùng TensorFlow 2.17.
- **Các lựa chọn**: (a) TensorFlow theo tác giả; (b) PyTorch CPU deterministic; (c) JAX.
- **Lựa chọn của nhóm**: **PyTorch 2.5.1 (CPU deterministic)**.
- **Lý do**: Yêu cầu môn học cho phép dùng PyTorch/TensorFlow/JAX tự do. PyTorch cho phép kiểm soát chặt chẽ vòng lặp huấn luyện từng batch, can thiệp linh hoạt vào cơ chế Self-Paced Learning (SPL), tính loss không tạo gradient trong bước chấm điểm (`torch.no_grad`), và đảm bảo tính tái lập (deterministic algorithms).
- **Phân loại nguồn**: `team_backend_choice`
- **Ảnh hưởng**: `src/dddcl/models/mlp.py`, `src/dddcl/training/loop.py`.
- **Kiểm chứng**: Kiểm tra forward/backward, seed reproducibility trong `dddcl.cli doctor` và `00_environment.ipynb`.
- **Ngày**: 2026-11-05.

---

### D02: Phạm vi thực nghiệm ban đầu (Core Scope vs Full Scope)
- **Câu hỏi**: Triển khai bao nhiêu tập dữ liệu trong giai đoạn đầu?
- **Vị trí bài báo**: Dynamic DDCL Table 1, 2, 3 (6 tập cũ + 4 tập mới). Yêu cầu môn học cho phép chọn Core scope (tối thiểu 2 bài toán quan trọng nhất gồm headline + phân tích mở rộng) hoặc Full scope.
- **Các lựa chọn**: (a) Full scope 10 tập dữ liệu ngay từ đầu; (b) Core scope trên 3 tập dữ liệu đã xác minh nguồn gốc (Haberman, New-Thyroid, Banknote) gồm R1 (so sánh 6 strategy), R2 (so sánh Ensemble), R3 (độ nhạy batch/threshold), sau đó mở rộng nếu tài nguyên cho phép.
- **Lựa chọn của nhóm**: **Core scope trên 3 tập pilot (Haberman, New-Thyroid, Banknote)**.
- **Lý do**: Các tập dữ liệu như `cancer` (457 dòng) chưa thể xác minh nguồn gốc từ UCI; `liver` có nhãn mục tiêu mơ hồ; `isolet` có sự không thống nhất giữa các mirror. Việc bắt đầu với 3 tập dữ liệu sạch, có độ phức tạp khác nhau (nhị phân, đa lớp, kích thước từ vài trăm đến hơn nghìn mẫu) đảm bảo toàn bộ pipeline được kiểm chứng vững chắc trước khi mở rộng.
- **Phân loại nguồn**: `course_scope_rule` & `team_decision`
- **Ảnh hưởng**: `configs/pilot.yaml`, `DATA_REGISTRY.md`.
- **Kiểm chứng**: Dữ liệu có manifest SHA-256 đối chiếu tại `hf_release/`.
- **Ngày**: 2026-11-05.

---

### D03: Giao thức phân chia dữ liệu (Data Split Contract)
- **Câu hỏi**: Tỷ lệ phân chia tập dữ liệu train/val/test và cách thức thực hiện?
- **Vị trí bài báo**: Dynamic DDCL mục 5.1 quy định tỷ lệ 70% train / 20% validation / 10% test.
- **Các lựa chọn**: (a) Phân chia ngẫu nhiên thuần túy; (b) Phân chia phân tầng (stratified) 70/20/10 trước mọi bước tiền xử lý; (c) Tỷ lệ 72/20/8 như quan sát trong code tác giả.
- **Lựa chọn của nhóm**: **Stratified split 70/20/10 trước mọi bước tính toán**, lưu trữ ID mẫu (`outputs/splits/<protocol>/<dataset>/<seed>.json`).
- **Lý do**: Đảm bảo đúng tuyên bố trong văn bản bài báo (70/20/10); phân tầng giúp giữ nguyên tỷ lệ nhãn cho các tập dữ liệu nhỏ/mất cân bằng (như New-Thyroid, Haberman); lưu ID giúp tất cả 6 strategies và các mô hình ensemble đánh giá trên cùng tập test.
- **Phân loại nguồn**: `paper_text` & `team_rigor_contract`
- **Ảnh hưởng**: `src/dddcl/data/splits.py`.
- **Kiểm chứng**: Kiểm tra tính rời nhau (disjoint) và hợp đầy đủ (union) của các tập ID.
- **Ngày**: 2026-11-05.

---

### D04: Xác định tâm cụm và chuẩn hóa khoảng cách
- **Câu hỏi**: Định nghĩa tâm cụm $c_s$ và khoảng cách chuẩn hóa $\hat{E}_s$ như thế nào?
- **Vị trí bài báo**: Static DDCL mục III, công thức (8) và (12); Dynamic DDCL mục 4.1.
- **Các lựa chọn**: (a) Mean centroid theo từng lớp, chuẩn hóa vector khoảng cách theo $L_2$ norm của lớp; (b) Chuẩn hóa min-max; (c) Chuẩn hóa z-score.
- **Lựa chọn của nhóm**: **Centroid là trung bình cộng các điểm trong lớp ($\mu_s = \frac{1}{|s|}\sum x$)**, khoảng cách Euclidean $r_i = \|x_i - \mu_s\|_2$, chuẩn hóa độc lập: $\hat{r}_i = \frac{r_i}{\sqrt{\sum_j r_j^2}}$ (nếu norm bằng 0 thì gán 0).
- **Lý do**: Công thức (8) trong bài báo viết sai ký hiệu toán học $\min_{\mu_s} \sum \|x - \mu_s\|_2$ mà nghiệm giải tích của bài toán bình phương khoảng cách Euclidean chính là trung bình cộng (mean). Chuẩn hóa chia cho $L_2$ norm đưa tổng năng lượng khoảng cách về 1, giải quyết việc bài báo thiếu đặc tả chi tiết về hàm "normalize".
- **Phân loại nguồn**: `team_mathematical_correction` & `team_assumption` (A03).
- **Ảnh hưởng**: `src/dddcl/curriculum/geometry.py`.
- **Kiểm chứng**: Unit test với các điểm đồ chơi tính tay và trường hợp suy biến (degenerate class 1 điểm).
- **Ngày**: 2026-11-05.

---

### D05: Chính sách phân nhóm theo mật độ KDE (`kde_level_partition_v1`)
- **Câu hỏi**: Bài báo nói "chia thành các quantiles từ phân bố mật độ KDE" nhưng không đưa công thức phân chia cụ thể. Cần hiện thực thế nào?
- **Vị trí bài báo**: Static DDCL mục III, Figure 2a; Dynamic DDCL mục 4.1.1.
- **Các lựa chọn**: (a) Dùng hàm `pd.qcut` chia đều số mẫu; (b) Cắt các mức mật độ ngang đều nhau (density level crossings) từ 0 đến đỉnh KDE; (c) Chia đều trục khoảng cách (uniform intervals).
- **Lựa chọn của nhóm**: **Thuật toán độc lập `kde_level_partition_v1`**: Ước lượng KDE 1D trên khoảng cách chuẩn hóa với lưới điểm cố định, xác định các ngưỡng mật độ từ 0 đến giá trị cực đại, tìm các giao điểm khoảng cách qua nội suy tuyến tính để tạo ranh giới các bin.
- **Lý do**: Hình minh họa trong bài báo (Figure 2a) cho thấy các nhóm có số lượng mẫu khác hẳn nhau (Q6 có 82 mẫu, Q1 có 0 mẫu), chứng minh tác giả không dùng quantile chia đều số lượng mẫu (equal-frequency). Thuật toán cắt mức mật độ phản ánh đúng tinh thần "Data Distribution-based".
- **Phân loại nguồn**: `team_operational_assumption` (A04).
- **Ảnh hưởng**: `src/dddcl/curriculum/partition.py`.
- **Kiểm chứng**: Test ranh giới đơn điệu, độ bao phủ 100% mẫu train, fallback an toàn khi dữ liệu hằng số.
- **Ngày**: 2026-11-05.

---

### D06: Gộp nhóm giữa các lớp và tạo mẫu SMOTE
- **Câu hỏi**: Sau khi chia quantile theo từng lớp, dữ liệu được gộp lại và SMOTE ra sao?
- **Vị trí bài báo**: Static DDCL Algorithm 1 (dòng 15-19); Dynamic DDCL mục 4.1.
- **Các lựa chọn**: (a) SMOTE toàn cục không phân nhóm; (b) `within_bin_minority_v1`: Trong mỗi bin gộp (pooled bin) có từ 2 lớp trở lên và đủ láng giềng ($k \ge 3$), sinh mẫu cân bằng cho lớp thiểu số; (c) Không SMOTE (`smote=none`).
- **Lựa chọn của nhóm**: Hỗ trợ hai chế độ minh bạch: **Mặc định `smote=none`** cho baseline công bằng, và **`within_bin_minority_v1`** khi chạy curriculum đầy đủ. Chỉ fit SMOTE trên tập train, sinh mẫu nhân tạo có ID riêng biệt (`synth_...`) để không bao giờ bị lẫn vào tập test.
- **Lý do**: Tránh làm sai lệch so sánh do oversampling không rõ ràng; bảo đảm tuyệt đối không rò rỉ dữ liệu tổng hợp.
- **Phân loại nguồn**: `team_operational_assumption` (A05).
- **Ảnh hưởng**: `src/dddcl/curriculum/resample.py`.
- **Kiểm chứng**: Test xác nhận 0 mẫu tổng hợp trong validation và test.
- **Ngày**: 2026-11-05.

---

### D07: Vòng lặp huấn luyện Self-Paced Learning (Dynamic DDCL SPL Loop)
- **Câu hỏi**: Cách tính loss từng batch, lọc easy/hard, và thứ tự cập nhật trong Algorithm 2?
- **Vị trí bài báo**: Dynamic DDCL Algorithm 2 và mục 4.1.2.
- **Các lựa chọn**: (a) Tính running average loss; (b) Tính loss từng batch độc lập ở chế độ eval (`torch.no_grad`), phân loại $L_b < \lambda$ (easy) và $L_b \ge \lambda$ (hard). Huấn luyện các batch easy (xếp theo loss tăng dần). Nếu sau đó loss trung bình của easy $< \lambda$, gộp thêm các batch hard và huấn luyện tiếp (`replay_easy_then_union_v1`).
- **Lựa chọn của nhóm**: **Thực thi trung thực Algorithm 2 (`replay_easy_then_union_v1`)**:
  - Bước 1: Duyệt tất cả batch của epoch hiện tại ở chế độ eval (`model.eval()`, `torch.no_grad()`), tính mean per-batch loss $L_b$.
  - Bước 2: Tách $\mathcal{E} = \{b \mid L_b < \lambda\}$ và $\mathcal{H} = \{b \mid L_b \ge \lambda\}$. Sắp xếp $\mathcal{E}$ theo $L_b$ tăng dần.
  - Bước 3: Nếu $\mathcal{E}$ rỗng, fallback chọn 1 batch có loss nhỏ nhất để tránh treo huấn luyện.
  - Bước 4: Chuyển sang `model.train()`, huấn luyện qua các batch trong $\mathcal{E}$.
  - Bước 5: Đánh giá lại loss $L_{\mathcal{E}}$. Nếu $L_{\mathcal{E}} < \lambda$, gộp hard $\mathcal{H}$ (đã xếp thứ tự loss) và huấn luyện tiếp tập gộp.
- **Lý do**: Đây là hiện thực sát nhất với mã giả Algorithm 2 của bài báo. Tách biệt hoàn toàn việc chấm điểm (scoring) với tối ưu hóa (gradient update), đảm bảo các batch bị bỏ qua thực sự không làm cập nhật trọng số mạng.
- **Phân loại nguồn**: `paper_algorithm_2_literal` & `team_operational_policy` (A06, A07).
- **Ảnh hưởng**: `src/dddcl/training/pace.py`, `src/dddcl/training/loop.py`.
- **Kiểm chứng**: Unit test kiểm tra batch bị skip không làm thay đổi weights; đếm chính xác số lần cập nhật optimizer step và số lượt tiếp xúc mẫu (exposures).
- **Ngày**: 2026-11-05.

---

### D08: Cơ chế tăng ngưỡng $\lambda$ trong SPL
- **Câu hỏi**: Ngưỡng $\lambda$ tăng và giới hạn như thế nào qua từng epoch?
- **Vị trí bài báo**: Dynamic DDCL mục 5.1 nêu: $\lambda_{\text{initial}} = 0.4$, $\lambda_{\text{max}} = 0.6$, hệ số tăng $\text{growth} = 1.1$. Bài báo không nói rõ cập nhật ở đâu.
- **Các lựa chọn**: (a) Tăng sau mỗi epoch: $\lambda_{e+1} = \min(\lambda_{\text{max}}, \lambda_e \times 1.1)$; (b) Tăng theo số bước optimizer.
- **Lựa chọn của nhóm**: **Cập nhật sau mỗi epoch theo công thức $\lambda_{e+1} = \min(0.6, \lambda_e \times 1.1)$**.
- **Lý do**: Đơn giản, minh bạch, phù hợp với chu kỳ epoch của mạng nơ-ron.
- **Phân loại nguồn**: `team_operational_assumption` (A06).
- **Ảnh hưởng**: `src/dddcl/training/pace.py`.
- **Kiểm chứng**: Unit test kiểm tra $\lambda$ không bao giờ vượt quá 0.6.
- **Ngày**: 2026-11-05.

---

### D09: Kiến trúc Ensemble và chính sách xử lý hòa phiếu (Tie-breaking)
- **Câu hỏi**: Cách kết hợp các thành viên trong DDCL Ensemble và quy tắc khi số phiếu hòa (3-3 trong bài toán nhị phân)?
- **Vị trí bài báo**: Dynamic DDCL mục 4.2 (Algorithm 3) và mục 5.2.
- **Các lựa chọn**: (a) Bỏ qua các trường hợp hòa; (b) Chọn lớp theo phân bố xác suất huấn luyện (class prior), nếu vẫn hòa chọn nhãn có chỉ số nhỏ hơn; (c) Nhìn vào nhãn test.
- **Lựa chọn của nhóm**: **Hard voting bình đẳng giữa 6 thành viên**. Khi xảy ra hòa phiếu 3-3, **ưu tiên nhãn có tần suất xuất hiện cao hơn trong tập train (train class prior)**; nếu tần suất bằng nhau, chọn nhãn có index nhỏ nhất. Tuyệt đối không nhìn vào nhãn test. Báo cáo tỷ lệ hòa phiếu (tie rate) trong kết quả.
- **Lý do**: Bài báo không nêu cơ chế xử lý hòa 3-3. Sử dụng train prior là nguyên tắc chuẩn tắc Bayes không làm rò rỉ dữ liệu test.
- **Phân loại nguồn**: `team_operational_assumption` (A09).
- **Ảnh hưởng**: `src/dddcl/ensemble.py`.
- **Kiểm chứng**: Unit test kiểm tra các trường hợp phiếu hòa [0, 0, 0, 1, 1, 1] cho ra kết quả dự đoán đúng theo prior định trước.
- **Ngày**: 2026-11-05.

