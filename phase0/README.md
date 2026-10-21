# Phase 0 — điều kiện tiên quyết (bản 3, PG_update 17/10)

Cấu hình khai báo trước: `config.json`, khối `ban3`. Hash của file này được in ở đầu mỗi báo cáo. Chạy từ gốc repo bằng `python -m phase0.<module>`.
**Cách chạy từng bước: [HUONG_DAN_CHAY.md](HUONG_DAN_CHAY.md).**

## Giai đoạn 1 — P0.1 (cổng chặn, gửi thầy trước)

| Bước | Script | Chạy ở |
|---|---|---|
| Xử lý ảnh + QA tự động (V0/V1/V2) | `p01_preprocess.py` | Colab GPU |
| Chọn 50 ảnh phân tầng + ảnh bị cờ để chấm | `p01_qa_sample.py` | Colab |
| Tổng hợp QA, cổng ≥ 90%, danh sách `zone_missing` | `p01_qa_summary.py` | Colab hoặc local |
| Trích embedding V0/V1/V2 × frozen/control | `p01_extract_embeddings.py` | Colab GPU |
| 4 chỉ số: probe lô, AUC trong lô, Δ, LOBO | `p01_batch_noise.py` | Colab hoặc local (CPU) |
| Báo cáo gửi thầy `output/p01_bao_cao_ban3.md` | `p01_decision.py` | Colab hoặc local |

Tất cả nằm trong notebook `colab/phase0_p01.ipynb`. Giai đoạn này dùng file fold hiện có (mục 3).

## Giai đoạn 2 — sau khi thầy duyệt P0.1 (chưa làm)

- P0.5: chia lại fold theo khóa ghép nới lỏng. 1.366/8.030 lượt khám (17%) vắt fold, cao hơn ngưỡng 1%. Nhóm có người ở test thì cả nhóm vào test. Sau đó chạy lại P0.1 trên fold mới (`p01_batch_noise.py --folds`) và giữ kết quả trên fold cũ để đối chiếu.
- P0.4: boxplot điểm L1 OOF của 57 ca (55 người) `bnn=Có`, `ketqua=2`, và phân tích độ nhạy khi bỏ 55 người này. Không sửa nhãn.
- Sổ kiểm toán CSV (mã ẩn danh · vấn đề · bằng chứng · quyết định · quy tắc).
- P0.6 đã đóng: gộp thành 1 node khám thực thể, bỏ E6. Script `p06_exam_survey.py` giữ làm bằng chứng.
- `p05_select_k.py`, `p05_assemble.py`, `colab/phase0_p05_graph.ipynb` (chạy lại GAT/B3b) chuyển sang tuần 2 theo mục 8.

## Ghi chú

- Output của lần chạy bản 2 được lưu ở `output/ban2/`.
- `output/id_province.csv` (id → tỉnh, tạo bởi `make_id_province.py`) không chứa họ tên hay số điện thoại.
- Ảnh đã xử lý và embedding chỉ giữ ở Drive riêng tư, không đưa vào git.
