# W&B, dữ liệu kết quả và báo cáo

## Tracking

Một W&B project public cho cả reproduction và empirics. Tags tối thiểu `reproduction|empirics`, `pilot|final`, protocol_id, dataset_id, strategy. Run name chỉ là label, run_id là hash ổn định và W&B id lưu riêng để resume.

Config allowlist: dataset snapshot/revision+hash, split fingerprint, seeds, model/backbone/optimizer, epochs/batch policy, geometry/KDE/SMOTE/pace/tie settings, Git commit+dirty flag/diff hash, lock hash, OS/Python/backend/device, budget. **Không log os.environ hoặc .env nguyên file.**

Mỗi epoch: optimization loss với đúng mẫu số exposures, eval losses tách biệt, lambda, easy/hard/selected counts theo class, unique samples, optimizer steps, fallback events, wall time. Không log test mỗi epoch để chọn checkpoint. Final test một lần sau model-selection freeze theo protocol.

Local artifacts: resolved_config.json, environment.json, data_provenance.json, split.json, trace.jsonl, metrics.json, predictions.csv có row_id/y_true/y_pred/probabilities, checkpoint, status.json, wandb_url.txt. W&B artifacts giữ model/output lớn khi cần; không đưa vào Git mặc định. Không ghi một accuracy final không rõ split nào.

## Aggregation

Một dòng/run với dataset/protocol/strategy/seed/metrics/compute/status/W&B URL. Chỉ aggregate các run completed cùng protocol/data revision/backbone/seed scheme, báo số failed và missing. Mean/sd/CI và paired deltas đúng unit. Paper table transcription có table/row/metric/value/unit/source; kiểm tra bằng mắt PDF, không copy OCR thiếu dấu ± thành dữ liệu thật.

## README và output cho thầy

README giải thích setup, hosted data source, exact rerun/resume command, scope full/core và lý do, assumptions/deviations, reuse citations, limitations và evidence. Bảng bắt buộc: experiment_id -> script/notebook -> W&B URL -> điều run cho thấy. Không dùng W&B offline path thay public URL.

Reading+reproduction: deck20–40 slides gồm appendix, notes mỗi slide<=500 từ, IEEE refs tại slide có claim, không references slide riêng, tự vẽ figures, <20MB; repo link nộp riêng Moodle.

Analysis+empirics: 1slide portraitA2 trong1PPTX; theory argument cạnh measured evidence, chọn2notable proven weaknesses, còn lại README. Có>=3strengths,>=2theoretical limitations,>=2future directions gắn hạn chế. Không gọi “chạy code lỗi” là thay thế đủ theoretical critique. Không tạo deck/poster tự động trong phase code nếu người dùng chưa giao; tạo figures/tables và outline để sẵn bàn giao.

Không có yêu cầu “chỉ được dùng mô hình bài giảng Kaggle” từ project mass-spectra áp dụng ở đây. Theo yêu cầu lab này được dùng torch/tensorflow/jax khi paper cần deep learning; không trộn constraints hai đồ án.
