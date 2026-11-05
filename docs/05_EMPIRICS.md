# Thực nghiệm kiểm chứng hạn chế

Các mục dưới là **giả thuyết**, chưa có kết quả. Final poster cần hai hạn chế có lập luận và evidence, không chỉ hai biểu đồ hoặc hai bug source.

## E1: khoảng cách đến tâm có thể không đại diện độ khó

Lập luận: lớp nhiều mode có centroid ở vùng ít dữ liệu; thêm đặc trưng không liên quan có thể lấn khoảng cách có ý nghĩa. Dự đoán cần kiểm tra: curriculum Point/Density mất hiệu quả tương đối khi mức vi phạm tăng; không bảo đảm nhất định xấu.

Thiết kế A: tạo synthetic hai lớp nhiều mode có vị trí và Bayes difficulty đã biết, giữ feature scale có kiểm soát; vary khoảng cách mode/overlap, so sáu strategy cùng backbone/init/split/budget. Synthetic là dữ liệu thí nghiệm riêng, không giả làm UCI. Để end-to-end theo yêu cầu thầy, publish release synthetic có seed/config/checksum trên HF trước final runs.

Thiết kế B: thêm 0/5/20/100 chiều nhiễu độc lập trên pilot hosted data, dùng cùng generator scale chuẩn; không dùng test labels để chọn noise. Train/val/test noise độc lập nhưng từ seeds cố định. Vary scale trong một sweep riêng, không gộp nhiều biến mà không biết nguyên nhân.

Đo accuracy/macro-F1 paired deltas, rank correlation score với oracle difficulty trên synthetic, khoảng cách concentration, loss, time, exposures. Đối chứng no-curriculum/no-SMOTE và geometry standardized. Kết luận giới hạn ở family/mức stress đã kiểm tra. Future direction: metric học từ train/đa centroid, đánh giá riêng, không tự biến thành đóng góp chính paper.

## E2: ngưỡng theo batch không phải độ khó từng sample

Lập luận: mean batch loss phụ thuộc thành phần batch; batch chứa mẫu dễ+khó có thể bị chọn/bỏ cùng nhau. Loss phụ thuộc calibration và số lớp, nên cùng lambda có thể tuyển tỷ lệ dữ liệu khác nhau.

Vary batch size 8/16/32/64, order đồng nhất theo score hoặc mixed cùng seed, giữ các yếu tố khác. Log selection rate theo lớp, khó/dễ oracle, số lần easy rỗng, số step/exposure. So threshold cố định và quantile-loss selection như replacement baseline (đây là phương pháp nhóm bổ sung, không gắn nhãn Dynamic DDCL gốc). Dùng dữ liệu clean trước; noise label sweep riêng và không làm nhiễu validation/test nếu nghiên cứu train-label noise.

Có hai cách budget: cùng epochs để gần paper, cùng steps/exposures để kiểm soát compute. Báo cả hai, không nói cùng epoch là cùng thời gian. Chọn cấu hình trên validation trước final seed sweep. Future direction: per-sample hoặc class-aware pace, cần kiểm tra tradeoff chi phí.

## E3 tùy chọn: ensemble và ngân sách

Paper so 6 NN với NN+SVM nên chưa cô lập số thành viên/compute. Đây là hạn chế của thiết kế evidence, không tự nó là hạn chế lý thuyết thứ hai để đáp ứng rubric. So 6 NN no-curriculum, 6 NN curriculum, ablation 3 static/3 dynamic, ensemble size 1/2/3/6; cùng test, tuning budget, seed policy. Đo errors disagreement, double-fault, tie rate, inference latency, total train time. Không tuyên bố curriculum là nguồn tăng accuracy nếu comparator khác số models.

## Quy trình kết luận

Preregister hypothesis, primary metric, yếu tố thay đổi, đối chứng, seed list, expected plots trong YAML trước khi chạy. Sau chạy: results có ủng hộ hay không, alternative explanations, robustness, phạm vi. Không sửa hypothesis để phù hợp test rồi gọi xác nhận từ trước. Nếu một hypothesis không được ủng hộ, giữ và báo; chọn hướng kiểm chứng khác dựa trên lập luận mới, không fabricate weakness.

Đạt cuối: ít nhất 3 strengths có evidence, 2 theoretical limitations cụ thể và 2 future directions gắn với hạn chế. Poster chỉ chọn 2 hạn chế đáng chú ý đã có cả lý thuyết và đo đạc; phần dư vào README.
