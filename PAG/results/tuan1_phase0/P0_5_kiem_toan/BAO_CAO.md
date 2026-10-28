# P0.5 — Kiểm toán dữ liệu (ảnh trùng, người bị tách mã)

### Kết quả kiểm toán

| muc | gia_tri |
|---|---|
| Ảnh trùng SHA giữa các lượt khám | 0 |
| Khóa lỏng (tên chuẩn hóa + năm sinh + giới + tỉnh) ứng với >1 patient_uid | 804 |
| … số patient_uid liên quan | 1615 |
| … khóa trải qua cả dev và test (nguy cơ rò rỉ) | 190 |
| … khóa có ít nhất một lượt ketqua=1 | 190 |
| Sau gộp: số người | 7024 → 6210 |
| Lượt dev bị loại vì cùng người với test | 199 (dương 29, có ảnh 71) |
| Dev mới | 6683 lượt; có ảnh 1500 (dương 367); không ảnh 5183 (dương 427) |
| Fold dev cũ: khóa người nằm ở >1 fold | 484 → 0 sau khi chia lại 5 fold theo người đã gộp |
| Chọn lại k của đồ thị kNN chỉ trên dev | làm trong tuần 2 (mô hình so sánh GAT/B3b dựng lại, k chọn bằng validation nội bộ) |

_Tệp: `kiem_toan.csv`_

**Quyết định:** từ đây mọi phân tích dùng mã người đã gộp, dev mới và 5 fold mới. Tập test giữ nguyên và vẫn khóa. Kết quả bản 1 bị ảnh hưởng bởi lỗi này.

*Ghép khóa lỏng chạy cục bộ trên dữ liệu có PII. Chỉ số đếm được ghi ra; không có tên hay mã nào rời máy.*
