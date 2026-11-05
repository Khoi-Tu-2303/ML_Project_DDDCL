# Đặc tả phương pháp và ranh giới paper/assumption

## Nguyên tắc

Đọc hai PDF, nhất là Dynamic mục 4–5. Đây là bản chỉ dẫn triển khai độc lập, không thay paper. Mọi lựa chọn chưa được paper xác định phải vào `ASSUMPTIONS.md` và run config trước khi xem test. Code của nhóm không được sao chép source tác giả.

## Sáu strategy cần triển khai

| Strategy ID | Chuẩn bị trước train | Lựa chọn batch trong train |
|---|---|---|
| no_curriculum | Thứ tự ngẫu nhiên cố định từ seed cho comparator kiểm soát | Tất cả batch |
| ddcl_density | Nhóm từ phân bố khoảng cách; đông mẫu trước | Tất cả batch |
| ddcl_point | Khoảng cách đến tâm chuẩn hóa, nhỏ trước | Tất cả batch |
| self_paced | Không DDCL | Loss và ngưỡng |
| sp_ddcl_density | DDCL Density | Loss và ngưỡng |
| sp_ddcl_point | DDCL Point | Loss và ngưỡng |

Random order của No Curriculum là lựa chọn kiểm soát của nhóm; phải ghi khác biệt nếu code tham chiếu group theo nhãn. Không dùng full-batch optimizer rồi kỳ vọng order đơn thuần sẽ tác động giống mini-batch.

## Static preprocessing từ mô tả paper

1. Chỉ lấy train, chia theo y.
2. Tính centroid Euclidean một tâm/lớp. Mean là nghiệm tổng bình phương khoảng cách; có thể dùng mean trực tiếp, không cần copy công thức sai ký hiệu trong pseudocode.
3. Khoảng cách `r_i = ||x_i - mu_class||_2`. Không đưa y/ID vào feature vector. Geometry raw_features là track gần thứ tự trong paper; thêm standardized geometry là ablation riêng.
4. Khoảng cách chuẩn hóa: PDF không chỉ rõ phép chuẩn hóa. Đề xuất độc lập `r_i / sqrt(sum_j r_j^2)` theo lớp, xử lý norm 0 -> tất cả 0. Gắn assumption; không gọi công thức này là phương trình paper.
5. KDE 1 chiều trên khoảng cách chuẩn hóa. Giữ kernel/bandwidth/grid explicit trong config. Với lớp constant hoặc quá ít điểm, có fallback một nhóm và ghi event.
6. Chia nhóm: paper không định nghĩa đầy đủ hàm quantile. **Không dùng qcut như thể chắc chắn đúng tác giả.** Thiết kế độc lập đề xuất `kde_level_partition_v1`: KDE grid xác định, các mức density đều nhau từ 0 đến peak, lấy crossing x bằng nội suy, thêm min/max observed distance, sort/dedup, gán bin half-open cuối đóng. Số mức là config, không bảo đảm số nhóm bằng số mức. Đây là operational assumption dựa trên ý tưởng phân bố, không phải sao chép implementation gốc. Được phép so với equal-frequency bins như sensitivity riêng, không thay mặc định ngầm.
7. Định danh nhóm theo `(class_label, local_bin)`; quy tắc pooling nhóm giữa các lớp là một ambiguity. Chọn explicit `pool_by_local_bin_index_v1` nếu muốn thử gộp cùng local index, ghi thiếu đảm bảo tương đương vị trí giữa các lớp; so với class_specific nếu nghiên cứu. Tất cả liên quan phải log.
8. SMOTE: paper cho phép oversampling nhóm ít mẫu nếu đủ mẫu nhưng điều kiện chưa đủ rõ. Hỗ trợ `none` và `within_bin_minority_v1`: trong bin pooled có >=2 lớp và mỗi lớp liên quan đủ k+1 (k=3 là lựa chọn nhóm), cân bằng minority trong train; nhóm quá ít/empty skip có log. Lựa chọn bin nào là cấu hình có lý do, ví dụ nửa số eligible bins ít mẫu nhất. Không gọi đây là hoàn nguyên chắc chắn của paper; phải có ablation off. Synthetic samples lưu IDs riêng và parent/source provenance khi API cho phép; không trộn vào test.
9. Density: xếp nhóm theo cardinality đã chọn trước/sau SMOTE (mặc định sau, ghi rõ), tie-break ổn định. Point: sort toàn bộ mẫu theo khoảng cách tăng, không sort_index. Với synthetic points, tính lại khoảng cách bằng train centroids và cùng distance scale. Không đưa dist/bin vào input NN.

Default KDE/n_bins/SMOTE policy phải chốt ở phase C qua validation, không dò test. Nếu chưa đủ căn cứ, label kết quả `paper_interpretation_clean_v1`; chỉ track faithful khi đã đủ evidence.

## MLP của nhóm

PyTorch MLP nhận X đã transform từ scaler train-only; binary một logit + BCEWithLogitsLoss, multiclass C logits + CrossEntropyLoss. Không sigmoid/softmax trước loss nhận logits. Inference binary sigmoid>=0.5; multiclass argmax. Dropout chỉ train. Adam lr=0.001; tất cả tham số còn lại explicit. Đọc paper về Bayesian tuning; nếu dùng model nhỏ cố định để pilot, ghi `pilot_fixed_backbone`, không so như cấu hình final tác giả.

## SPL: vòng lặp explicit, có thể quan sát

Mô tả paper: mỗi epoch đánh giá batch loss, chia easy loss<lambda và hard loss>=lambda; train easy từ loss thấp; nếu loss sau train easy dưới ngưỡng thì đưa hard trở lại. Không lọc chỉ sau khi backward đã xảy ra rồi gọi là đã skip batch.

Chọn operational policy độc lập:

- Epoch bắt đầu: eval/no_grad, tính **mean per-example loss của từng batch riêng**, không dùng running average metric. Dropout tắt khi scoring. Dùng batch partition ổn định tạo từ curriculum, không random shuffle phá order.
- Lưu easy/hard list, batch IDs, sample IDs, score trước update. Stable sort theo (loss,batch_id).
- Nếu easy rỗng: dùng batch có loss nhỏ nhất; ghi fallback, không no-op mãi, không lấy test để điều chỉnh.
- Train easy một lượt bằng gradients; sau đó eval lại mean loss trên easy. Nếu dưới lambda, theo Algorithm 2 literal dùng tập ordered easy+hard để train một lượt tiếp, tức easy có thể cập nhật hai lần. Gắn `replay_easy_then_union_v1`; cung cấp `hard_only_second_pass` như ablation, không đổi giữa run.
- Nếu hard/easy rỗng, không chia 0; giữ accounting của từng sample exposure và optimizer step.
- Lambda: chọn tăng mỗi epoch `min(lambda_max, lambda*growth)` với 0.4/0.6/1.1 theo giá trị paper. Paper thiếu quy tắc cập nhật đầy đủ; đây là assumption. Không tự thêm warmup 10 epoch từ source tác giả vào bản dựa paper.
- Train/eval mode phải phục hồi đúng. Không update mô hình khi scoring loss. Threshold maximum thật sự cap. Chỉ train loss hoặc validation được chọn policy; test không được tham gia.

Tác giả nói dynamic feedback ở batch level. Thay đổi batch_size làm đổi decision unit; log cả batch-size và phần batch cuối. `drop_last=false` tránh loại mẫu; đây là lựa chọn nhóm và có thể khác implementation gốc.

## Ensemble

DDCL: 6 thành viên cùng backbone/config theo 6 strategy; cùng train/val/test, init policy được ghi. Hard voting bằng nhau. Binary 3–3: tie-break chọn class prior train cao hơn, nếu vẫn hòa chọn label nhỏ hơn; paper không quy định, nên phải ghi và báo tỷ lệ ties. Không chọn tie theo test.

Standard paper baseline: 1 NN no curriculum + SVM RBF C=1.0, gamma='scale' là operational reading của mô tả. Thêm controlled baseline 6 NN no curriculum với independent init; báo cả compute, diversity, accuracy. Giữ cùng test sample IDs và aligned label mappings, không ensemble theo vị trí mảng chưa kiểm tra ID.
