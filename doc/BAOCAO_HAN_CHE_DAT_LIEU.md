# Báo cáo: Hạn chế của bộ dữ liệu đối với mục tiêu fusion

> ## ⚠️ CẬP NHẬT 27/09/2026 — ĐỌC TRƯỚC
>
> 1. **Nhãn đích của đồ án đã đổi sang `ketqua`** (đọc phim theo chuẩn ILO). Nhãn `bnn`
>    là **tiền sử** đã được công nhận bệnh nghề nghiệp (97/99 ca chẩn đoán trước đợt
>    khám), không phải kết quả khám lần này.
> 2. **Bộ đặc trưng `crossfit` đã bị xoá khỏi dự án.** Vector của 5 fold do 5 mô hình
>    khác nhau sinh ra nên nằm ở **5 không gian vector khác nhau** (chỉ nhìn vector là
>    đoán đúng fold 100%; `control`/`frozen` chỉ ~22%). Mọi con số của `crossfit` trong
>    tài liệu này **không hợp lệ**, kể cả kết luận "nhãn huấn luyện quan trọng hơn miền
>    dữ liệu" vốn dựa trên phép đo đó.
> 3. Số liệu nhánh ảnh còn hiệu lực: `imagefeat/output/eval_ketqua.md` và
>    `doc/fusion_roadmap.md`.


## Tóm tắt vấn đề

Dự án hiện tại **không thể đạt PR-AUC ≥ 0,80** bằng bất kỳ kiến trúc GNN nào, không phải vì lựa chọn mô hình mà vì **giới hạn bộ dữ liệu** và **chất lượng đặc trưng ảnh**.

## Bằng chứng số cụ thể

### 1. **Trần của tín hiệu bảng lâm sàng (139 trường)**
- Model GBM mặc định trên bảng: **ROC-AUC 0,9521 / PR-AUC 0,4953**
- Model ts phức tạp (multi-input Keras + chuỗi + ensemble 5 fold): **ROC-AUC 0,9556 / PR-AUC 0,5133**
- **Chênh lệch: nằm trong nhiễu** của tập test (n=1.148, 31 ca dương)

**Kết luận:** Kiến trúc phức tạp không mua thêm được tín hiệu. Trần nằm ở dữ liệu, không ở mô hình.

### 2. **Đóng góp khiêm tốn của ảnh**
Trên 1.835 bệnh nhân có ảnh (99 ca dương), CV 5-fold nhóm bệnh nhân:
| Đầu vào | PR-AUC |
|---|---|
| Bảng (139 trường) | 0,5234 |
| Bảng + Ảnh (control) | 0,5695 |
| **Chênh lệch** | **+0,0461** (KTC95: [−0,0337; +0,1238]) |

- P(ảnh có giúp) = 89%, nhưng **khoảng tin cậy vẫn cắt qua 0**
- Đây là toàn bộ "ngân sách" mà nhánh ảnh có thể đóng góp

### 3. **Đặc trưng ảnh hiện tại không sủi**
| Bộ | Eff. rank | PC1 | Vấn đề |
|---|---|---|---|
| control | **1,80** | 90,7% | Sụp đổ thành 1 trục, gần như vô dụng với GNN |
| frozen | 14,79 | — | Chưa học miền, PR-AUC chỉ 0,2183 |
| crossfit | 5,56 | — | Fine-tune sai nhãn (`ketqua` ≠ `bnn`) |

**Chưa có bộ vừa fine-tune đúng `bnn`, vừa đúng miền, vừa không sụp đổ biểu diễn.**

### 4. **Kích thước tập test quá nhỏ**
- Test: 1.148 hàng, **31 ca dương / 1.117 ca âm** (nền 2,70%)
- Để PR-AUC 0,80 cần recall 80% + precision 80%: bắt được 25/31 ca bệnh đúng trong khi chỉ báo động giả 6 người (FPR 0,54%)
- Hiện tại: recall 58% (18/31), precision 42,86% → 24 báo động giả (FPR 2,1%)
- Cần GNN đạt ROC-AUC ≈ **0,99** để có thể. Con số này không khả dĩ với dữ liệu này.

## Ước lượng thực tế kết quả

| Kịch bản | PR-AUC dự kiến | Có thể giải thích |
|---|---|---|
| **Tốt nhất (không sửa gì thêm)** | 0,50–0,52 | Ngang baseline ts |
| **Sửa ảnh + GNN tối ưu** | 0,55–0,60 | GNN truyền ảnh được ~75% lợi ích tới 77% bệnh nhân không ảnh |
| **PR-AUC ≥ 0,80** | Không đạt được | Đòi hỏi ROC-AUC ≈ 0,99, vượt quá khả năng của dữ liệu |

## Nơi vấn đề thực sự nằm

1. **112/211 ca bệnh (53%) không có ảnh X-quang** → chỉ có thể "học ké" qua đồ thị, rất hạn chế
2. **Chỉ 99 ca dương trong 1.835 ảnh** → không đủ để fine-tune mô hình ảnh đúng `bnn`
3. **Đặc trưng ảnh `control` sụp đổ hoàn toàn** (rank 1,8) → GNN không có đa dạng thông tin từ hàng xóm

## Khuyến nghị

1. **Nếu tiêu chí là "con số đẹp để marketing":** dữ liệu hiện tại không cho phép. Cần tăng ca có ảnh, đặc biệt là những ca bệnh.

2. **Nếu tiêu chí là "sản phẩm sàng lọc dùng được":** mô hình ts hiện tại **đã đủ dùng** — recall 58% + precision 43% ở nền 2,6% là sàng lọc 16 lần so với random. Đó là thành tựu của nhánh ts, không phải fusion.

3. **Để cải thiện fusion:** Ưu tiên **làm lại nhánh ảnh** (fine-tune trên `bnn`, cùng miền, giữ rank ≥10) thay vì thay đổi GNN backbone.

4. **Về mặt khoa học:** Dự án vẫn có giá trị làm **ablation study** — chứng minh định lượng những gì ảnh + đồ thị có / không có thể đóng góp với bộ dữ liệu này.
