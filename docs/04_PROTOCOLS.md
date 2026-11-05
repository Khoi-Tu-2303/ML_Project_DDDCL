# Protocol tái lập và tính lặp lại

## Các track phải tách

1. `paper_interpretation_clean_v1`: bản độc lập, làm theo paper đến mức có thể, assumption công khai, split trước mọi học từ dữ liệu. Đây là track chính của nhóm.
2. `author_reference_<commit>`: chạy implementation gốc ở phase F trong môi trường riêng, ghi đúng dữ liệu/split/order/code path thực tế. Không gộp số với track chính.
3. `controlled_*`: thí nghiệm nhóm thêm như six-NN baseline, sample-matched/no-SMOTE, geometry scale, group split. Không gắn nhãn kết quả tác giả.

## Core scope được đề xuất, chưa chạy

- **R1 / headline comparison**: sáu strategy trên ba bộ Haberman/New-Thyroid/Banknote, cùng MLP backbone trong mỗi dataset, cùng split và init pair theo seed, 200 epoch final.
- **R2 / ensemble comparison**: DDCL six-NN, paper NN+SVM, controlled six-NN no curriculum trên các bộ đã xác minh; 600 epoch/member theo paper. Đủ tài nguyên thì mở rộng cả 10, không gọi 3 bộ là full scope.
- **R3 / additional analysis**: sensitivity lambda hoặc batch size, một yếu tố mỗi lần, trên >=2 dataset. Không đếm ba dataset thành ba nhóm thí nghiệm thay cho R1/R2.
- Pilot 3 seeds x 5 epoch chỉ để kiểm tra end-to-end; final 30 seeds. Không khởi động final trước các gate.

Lưu ý Table 3 paper ensemble chỉ báo 6 dataset cũ. Banknote trong core R2 là **mở rộng riêng của nhóm**, không có ô paper Table 3 để so; Haberman/New-Thyroid mới có đối chiếu trực tiếp. Mỗi số paper phải có đúng dataset/method/table/protocol.

## Split và leakage

Tạo split gốc trước centroid/KDE/SMOTE/scaler. Test 10% toàn bộ; val là 20/90 phần còn lại, có stratify và rounding. Save row_ids dưới `outputs/splits/<protocol>/<dataset>/<seed>.json`. Cùng file split mọi phương pháp. Tất cả members cùng test và ID order. Không dùng train resampled size làm mẫu số tỷ lệ test.

Thiết lập này gần tỷ lệ paper, có thể khác code gốc 72/20/8. Không sửa test để match số paper. Official Optical/ISOLET split là track khác, ghi rõ khi so.

## Scaler và feature space

Input scaler (StandardScaler mặc định nhóm) fit train gốc, apply val/test; paper không ghi mọi chi tiết nên ghi assumption. Curriculum geometry raw trước scaler là track chính, standardized geometry là ablation. SMOTE chỉ train. Không normalize nhãn cùng features. So sánh fairness bằng `smote=false` baseline suite rồi reported curriculum suite có policy SMOTE explicit; không gộp hai suite.

## Tuning

Paper nhắc Bayesian optimization nhưng thiếu toàn bộ search space/budget/trials/seeds. Phase D phải tạo search space có căn cứ, chạy Bayesian search bằng thư viện public như Optuna nếu chọn (thêm dependency + lock + justification), HOẶC giữ cấu hình cố định và công bố là reproduction hạn chế. Không nói đã Bayesian tune khi chỉ chọn thủ công.

Chọn backbone theo validation trên tuning split riêng đã cố định; không tune riêng strategy rồi tuyên bố so cùng architecture. Tuning seeds tách final evaluation seeds nếu đủ dữ liệu; mô tả train/val reuse nếu dùng. Sau chọn cấu hình, freeze trước final. Test chỉ evaluate khi cấu hình đã freeze; không chọn epoch theo test.

## Seed, môi trường và thực tế reproducibility

Final run seeds mặc định 0..29; từ SeedSequence derive riêng split/init/order/SMOTE/SPL seeds, lưu các giá trị. Paired strategy cùng base seed có cùng split và initial weights (tải bản initial state giống nhau), trừ ensemble independent member init được khai báo.

Set Python/NumPy/PyTorch CPU/CUDA seeds, deterministic algorithms, disable cuDNN benchmark; mặc định num_workers=0, threads cố định. PYTHONHASHSEED cần set **trước process**. Nếu CUDA cần CUBLAS_WORKSPACE_CONFIG đặt trước khởi tạo. Checkpoint lưu model/optimizer/epoch/pace/current batch order/ RNG Python NumPy Torch CPU CUDA/ scheduler + manifest/split/config/code fingerprint.

Same seed + package lock không bảo đảm giống bit-for-bit giữa OS, device, BLAS, drivers hoặc framework. CPU same host/env phải test repeat; GPU dùng strict deterministic hoặc báo nondeterministic operator, không âm thầm downgrade. Với thay đổi stack/device, tạo track riêng.

## Resume và accounting

Run identity = hash(dataset revision+manifest hash+split+strategy+seed+protocol+resolved config+code version). Completed -> skip idempotent; interrupted -> resume đúng checkpoint, không dùng checkpoint khác fingerprint. Ghi atomic checkpoint lúc epoch kết thúc. Nếu dừng giữa epoch chưa có safe checkpoint, replay từ đầu epoch với RNG trước epoch; không ghi bước chưa commit là xong. W&B resume cùng run id, tránh duplicated epoch records.

Ghi wall-clock preprocessing/scoring/optimization/inference tách riêng; optimizer steps, sample exposures, easy/hard counts, fallback counts, lambda, drop count, peak memory measurement method. Epoch bằng nhau không nghĩa compute bằng nhau vì replay easy/skip hard; phải báo ngân sách thực tế.

## Metrics và so sánh

Accuracy chính; thêm macro-F1, balanced accuracy, confusion matrix, per-class recall. Binary positive class theo mapping semantic; multiclass averaging explicit. Báo n_success/n_failed; mean±sd, min/max, CI và paired deltas theo cùng seed. Convention sd sample ddof=1, thêm population ddof=0 nếu đối chiếu tác giả; không trộn.

Paper-vs-team: percentage-point delta và relative deviation; nếu paper metric=0 relative là NA. CI trên seeds không phải CI cấp bệnh nhân, không overclaim. Report run-level data trước aggregate. Bootstrap CI fixed RNG, mô tả estimator; paired test điều chỉnh multiple comparisons khi có nhiều hypothesis. Không chọn best seed làm kết quả đại diện.
