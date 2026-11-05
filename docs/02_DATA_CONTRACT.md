# Dataset contract và cổng kiểm tra dữ liệu

## Phạm vi

Pilot: Haberman, New-Thyroid, Banknote. Final core tối thiểu kiểm tra nhiều lớp và nhiều quy mô; mặc định ba bộ pilot cho hai nhóm thí nghiệm chính, ghi rõ giới hạn coverage. Mở rộng đến 10 bộ nếu tài nguyên đủ; không gọi kết quả 3 bộ là full reproduction.

| ID nội bộ | N / đặc trưng / lớp quan sát hoặc nguồn | Trạng thái |
|---|---|---|
| haberman | 306 / 3 / 2 | UCI 43; raw nhãn 1,2; pilot |
| new_thyroid | 215 / 5 / 3 | File **new-thyroid.data** trong UCI 102; không phải mọi Thyroid Disease đều là bộ này |
| banknote | 1372 / 4 / 2 | UCI 267; pilot |
| breast_cancer_diagnostic | 569 / 30 thật / 2 | Repo tác giả có thêm ID thành 31; chuẩn loại ID, track đối chiếu có ID riêng |
| cancer | 457 / 9 / 2 trong CSV tác giả | Chưa xác minh nguồn/subset; không thay bằng UCI Breast Cancer 286 dòng |
| pima | 768 / 8 / 2 | Xác minh bản, quy tắc zeros; không tự coi tất cả 0 là missing |
| liver | 345 / 6 / 2 trong repo | Kiểm tra selector BUPA; không diễn giải nhãn thành có bệnh khi chưa chứng minh |
| clean1 | 476 / 166 / 2 | Musk v1, 92 phân tử; giữ group metadata khi có |
| optical_digits | 3823 train + 1797 test gốc / 64 / 10 | CSV standalone tác giả là 3823; không thay bằng sklearn digits 1797 rồi coi tương đương |
| isolet | UCI full 7797 / 617 / 26 | ZIP tác giả chưa giải nén xác minh trong khảo sát; nhiều mirror chỉ có 6238 |

Quan sát source tác giả dựa trên commit `c72c34478f3cf523eb43016f19057bd7ba64c2f7`; số liệu phải được Agent kiểm chứng khi đối chiếu, không coi là nguồn nhãn đã chứng nhận. Diabetes130 thuộc bài Static, không nằm trong 10 bộ Dynamic.

## Canonical records

Mỗi dataset có file CSV UTF-8: `row_id, feature_1, ..., feature_d, label` (tên semantic được khuyến khích). `row_id` là chuỗi ổn định từ dataset_id + vị trí raw; không sắp lại raw trước khi cấp ID. Metadata như patient_id/molecule_id tách khỏi X. Không dùng row_id/label/group làm feature. Nhãn integer 0..C-1; mapping raw được lưu, ví dụ raw1->0/raw2->1 có thể đảo so với tác giả nhưng accuracy vẫn so sánh được; per-class metrics phải decode đúng.

Không deduplicate, impute, standardize hoặc oversample canonical release một cách ngầm định. Ghi duplicate exact-feature và feature+label; đưa quyết định dedup vào protocol riêng. Missingness ghi count theo cột. Preserve zero values trừ khi protocol tuyên bố imputation có lý do.

## Manifest version 1

Root manifest: `schema_version`, `release_name`, `datasets` (mapping dataset_id). Entry: `filename`, `sha256`, `n_rows`, `feature_columns` có thứ tự, `label_column`, `row_id_column`, `class_counts`, `label_mapping`, `provenance_file`, `source_url`, `source_sha256`, `license`, `status`.

HF config: `repo_id`, FULL `revision` SHA 40 ký tự, `manifest_filename`, `manifest_sha256`, `cache_dir`. Không cần nhét SHA HF vào chính manifest vì tạo vòng phụ thuộc nội dung/commit. Checksum manifest được lưu ở Git config sau build.

Loader fail nếu SHA sai, cột/shape/labels sai, feature không finite, duplicate row_id, dataset không approved. Không im lặng lấy revision mới, không tạo synthetic fallback giả làm benchmark. Các feature có missing chỉ được cho qua khi entry khai báo và pipeline có policy đã test.

## Nguồn và publish

Ưu tiên public HF/Kaggle copy đã kiểm tra so với UCI và phiên bản cần dùng. Đường tự chuẩn bị UCI->HF là fallback có ghi lý do, không tự tuyên bố đã tìm hết mọi mirror. Giữ citations/license/source checksum, mô tả biến đổi. Run train phải đi qua HF/Kaggle adapter đã pin; không fallback tải UCI khi HF lỗi.

Tách **bootstrap một lần** khỏi **experiment loader**. Script bootstrap có thể cần UCI/network; sau release, experiment ở môi trường trống chỉ cần snapshot hosted. Test download sạch bằng cache tạm mới, không xóa cache người dùng.

## Split contract

Lưu `row_id` train/validation/test theo từng dataset + split_seed + protocol, original label counts, content fingerprint; ba tập không giao nhau và hợp đầy đủ eligible rows. Cùng seed dùng cùng split mọi strategy. Pilot datasets stratified row split 70/20/10 với rounding công khai. Clean1 có thêm group split experiment riêng, không trộn với row split. Official split ISOLET/Optical là protocol khác random split trong paper; quyết định bằng ledger.
