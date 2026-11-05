# Kiểm tra và tiêu chuẩn hoàn thành

Agent phải chạy kiểm tra và ghi command, exit code, evidence path, môi trường. Test fail là blocker đúng phạm vi, không sửa assertion cho khớp output sai.

## A / environment

Fresh venv trên máy người dùng từ lock, `pip check`, import packages, tiny CPU forward/backward finite. Đúng kernel notebook. Ghi Python patch/OS/device/BLAS/torch/build/dependency freeze/code hash. Resolve lock trên Linux không được ghi là Windows runtime đã test.

## B / dữ liệu

- Kiểm tra hosted download đúng revision bằng cache mới tạm, verify manifest hash và từng file hash. Corrupt byte -> reject. Placeholder repo/SHA -> reject, không fallback synthetic.
- N/d/C đúng registry; mapping raw->canonical có round-trip; unique row_id; feature excludes IDs/label; giữ order columns.
- Train/val/test disjoint, union đúng IDs, class counts và rounding; chiến lược khác seed pair load cùng IDs.
- Fit scaler/centroid/KDE từ train; thay cực trị/labels ở test không làm fitted train artifact đổi. Dùng test sentinel có ý nghĩa, không chỉ inspect code.
- Với synthetic oversampling, không có generated rows trong val/test; parentage nếu không có từ thư viện thì ghi limitation chứ không bịa.

## C / curriculum và SPL

- Centroid/distances với ví dụ tính tay: class points (0,0),(2,0) -> tâm(1,0), distances 1,1; norm zero/one-sample/constant xử lý đúng.
- Density groups sizes [2,5,1] phải xếp 5,2,1; Point distances [0.8,0.1,0.5] -> IDs 1,2,0, không theo index cũ. Tie reproducible.
- Partition bounds tăng, coverage tất cả train finite, mỗi sample đúng một group; boundary value không nhân đôi; KDE constant fallback không crash.
- Pace oracle losses [0.2,0.8,0.4,0.65], lambda .5 -> easy IDs[0,2], hard[3,1] theo sort; equality nằm hard. Empty easy, no hard, tiny final batch tests.
- Kiểm tra batch bị skip **không làm optimizer update**; selected ids thật sự xuất hiện trong optimizer steps, không chỉ list do callback tạo. Batch scoring không đổi weights/optimizer/RNG dropout.
- Independent batch loss: không dùng running average epoch. Dùng batch biết loss khác rõ và kiểm tra selection đúng.
- Tăng/cap lambda đúng policy; replay-easy accounting chính xác. No Curriculum cố định policy order, CL order không bị DataLoader shuffle phá.

## D / train

Tiny dataset có khả năng overfit, optimizer thực sự cập nhật; binary/multiclass shapes và logits-loss đúng. Chạy twice same CPU/env/config/seed -> same splits/order, loss trajectories/metrics trong tolerance đã công bố; khác seed đổi init mà không phá paired split.

Resume test: uninterrupted 4 epochs so với 2 epochs + process restart + resume 2; so weights/optimizer/pace/RNG/IDs/metrics. Resume sai manifest/config/code hash phải fail. Completed run rerun phải skip, không ghi duplicate run. Lỗi giữa epoch chỉ rollback epoch safe boundary, giữ error log.

## E / ensemble

Votes tay [0,0,1,1,1,1]->1; 3–3 theo tie rule không dựa y_test. Predictions bị đảo ID phải realign, missing/duplicate IDs phải fail. Mọi strategy cùng true y cho cùng IDs; không dùng ID theo thứ tự mảng mặc định. Equal compute baseline ghi init độc lập và actual cost.

## F / original comparison

Đã có bản độc lập + commit trước đọc code gốc. Dùng environment nguyên bản riêng, pin commit, ghi commands/error. Nếu source không chạy được trong máy hiện tại, báo lỗi cụ thể; không đánh dấu passed. So các checkpoint pipeline có số liệu intermediate, không chỉ cuối accuracy. Findings static review phân biệt verified runtime.

## G / empirical và báo cáo

Chạy pilot stress sweep nhỏ trước final. Primary hypotheses/config freeze; mọi số plot trace tới run IDs và config. Aggregation có fixture với mean/sd/delta tính tay; không merge aborted runs vào completed. Units percent vs fraction đúng. Mọi CI/paired diff lấy đúng seed pairing. Confusion matrix dùng mapping đúng, không đảo nhãn sống/chết chỉ vì số 0/1.

## Các mức trạng thái

- `implemented`: code có thật, chưa đủ kiểm chứng.
- `verified_local`: tests + smoke pass trên môi trường được ghi.
- `verified_hosted`: download sạch HF revision/checksum pass.
- `pilot_complete`: có run thật, chưa final benchmark.
- `final_complete`: đủ final seeds hoặc thiếu được giải thích, W&B public synced, bảng đối chiếu/deviation và evidence hoàn chỉnh.

Không dùng tests fixture để chứng nhận real data download hoặc accuracy benchmark. Không nói deterministic tuyệt đối giữa framework/hardware khác.
