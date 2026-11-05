# Data Registry: Danh mục dữ liệu thực nghiệm Dynamic DDCL

Tài liệu này xác định hợp đồng dữ liệu, trạng thái xác minh nguồn, cấu trúc đặc trưng và kế hoạch nạp dữ liệu cho toàn bộ các tập dữ liệu trong nghiên cứu Dynamic DDCL (2025) và tài liệu nền Static DDCL (2024).

---

## 1. Nguyên tắc quản lý dữ liệu (Data Contract)

1. **Nguồn Hosted công khai**: Theo yêu cầu môn học, dữ liệu phải được tải tự động trong runtime từ Hugging Face Hub (hoặc Kaggle) với **Full commit revision SHA** và **SHA-256 manifest**. Không đọc file cục bộ tùy ý hoặc nạp lén từ UCI trong loop huấn luyện.
2. **Canonical Format**: Mỗi tập dữ liệu là CSV UTF-8 gồm `row_id`, các cột đặc trưng `feature_1..feature_d`, và `label` (integer `0..C-1`). Cột định danh (patient ID, molecule ID) và metadata được tách khỏi ma trận đặc trưng $X$.
3. **Train-Only Preprocessing**: Mọi phép chuẩn hóa (StandardScaler), ước lượng mật độ (KDE), tâm cụm (Centroid) và tạo mẫu nhân tạo (SMOTE) chỉ được fit trên tập `train`. Tuyệt đối không để rò rỉ dữ liệu (data leakage) sang `validation` hoặc `test`.
4. **Phân tách Split cố định**: Tỷ lệ mặc định theo bài báo là 70% train / 20% validation / 10% test (stratified theo nhãn). Mỗi seed sinh ra một danh sách `row_id` duy nhất và được lưu tại `outputs/splits/` để mọi thuật toán so sánh công bằng trên cùng tập dữ liệu.

---

## 2. Bảng đăng ký các tập dữ liệu (Registry Table)

| Dataset ID | N (dòng) | d (đặc trưng) | C (lớp) | Nguồn gốc / Nguồn tác giả | Trạng thái xác minh | Phân loại phạm vi |
|---|---|---|---|---|---|---|
| `haberman` | 306 | 3 | 2 | UCI ID 43 (Haberman's Survival); raw label 1 (sống $\ge$ 5 năm), 2 (chết < 5 năm) | **VERIFIED (Primary UCI + HF Release pilot)** | Pilot / Core R1, R2, R3 |
| `new_thyroid` | 215 | 5 | 3 | UCI ID 102 (file `new-thyroid.data`, nhãn 1, 2, 3) | **VERIFIED (Primary UCI + HF Release pilot)** | Pilot / Core R1, R2, R3 |
| `banknote` | 1372 | 4 | 2 | UCI ID 267 (Banknote Authentication) | **VERIFIED (Primary UCI + HF Release pilot)** | Pilot / Core R1, R3 (Mở rộng nhóm) |
| `breast_cancer_diagnostic` | 569 | 30 | 2 | UCI ID 17 (WDBC). File tác giả có 31 cột (kèm ID bệnh nhân) | **VERIFIED UCI; Cần drop cột ID khi đưa vào clean track** | Mở rộng Phase D/E |
| `cancer` | 457 | 9 | 2 | Repo tác giả có file 457 dòng x 9 đặc trưng; UCI gốc có 286 dòng hoặc Wisconsin Breast Cancer gốc 699 dòng | **BLOCKED (Chưa xác minh nguồn gốc tập con 457 dòng)** | Tạm hoãn đến khi audit Phase F |
| `pima` | 768 | 8 | 2 | UCI ID 34 (Pima Indians Diabetes). Có các giá trị 0 sinh học đặc thù | **VERIFIED UCI; Cần kiểm tra quy tắc giá trị zero** | Mở rộng Phase D/E |
| `liver` | 345 | 6 | 2 | UCI ID 60 (BUPA Liver Disorders). Cột 7 là `selector`, không phải chẩn đoán bệnh | **VERIFIED UCI; Cần quy định rõ nhãn nhị phân** | Mở rộng Phase D/E |
| `clean1` | 476 | 166 | 2 | UCI ID 74 (Musk Version 1); 92 phân tử, 476 cấu hình | **VERIFIED UCI; Có cấu trúc phân nhóm (group metadata)** | Mở rộng Phase D/E & E1 |
| `optical_digits` | 3823 / 1797 | 64 | 10 | UCI ID 80 (Optical Recognition of Handwritten Digits). Tác giả dùng 3823 train | **VERIFIED UCI; Cần phân biệt train/test gốc vs random split** | Mở rộng Phase D/E |
| `isolet` | 7797 | 617 | 26 | UCI ID 54 (ISOLET). Một số mirror chỉ có 6238 dòng | **Cần kiểm tra bản đầy đủ 7797 dòng** | Mở rộng Phase D/E |
| `synthetic_multimodal` | Tùy biến | 2..10 | 2 | Dữ liệu nhân tạo có nhiều mode và Bayes error biết trước do nhóm tự sinh | **Thiết kế riêng cho thực nghiệm E1 (Hạn chế lý thuyết)** | Thực nghiệm E1 |

---

## 3. Trạng thái Release Pilot

Hiện tại nhóm đã chuẩn bị gói pilot release sẵn tại `hf_release/`:
- `manifest.json`: Chứa metadata, danh sách cột, nhãn, SHA256 từng file CSV.
- `data/haberman.csv`: SHA-256 đã xác minh (`03ff26...`)
- `data/new_thyroid.csv`: SHA-256 đã xác minh (`d4d3ce...`)
- `data/banknote.csv`: SHA-256 đã xác minh (`43b177...`)
- `provenance/`: Lưu URL nguồn gốc UCI, timestamp tải và các bước trích xuất.

**Gate Hosted**: Cần người dùng cung cấp thông tin tài khoản Hugging Face (`HF_DATASET_REPO` và quyền upload) để đưa lên HF Hub công khai và khóa commit revision SHA trước khi chạy benchmark chính thức.

