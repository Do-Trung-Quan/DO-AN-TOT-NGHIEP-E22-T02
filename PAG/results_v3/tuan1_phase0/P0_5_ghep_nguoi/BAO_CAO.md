# P0.5 — Ghép người bị tách mã và chia lại fold

Khóa ghép lỏng: họ tên chuẩn hóa (bỏ dấu, chữ thường) + năm sinh + giới + tỉnh, gộp với `patient_uid` bằng union-find (hai lượt chung mã hoặc chung khóa → cùng người). Khóa chặt thêm phân xưởng, chỉ để thấy biên dao động. Khóa chỉ dùng cục bộ, không lưu. Luật PG_update §2: lượt thuộc nhóm vắt fold hoặc vắt dev/test ≥ 1% → chia lại fold. Phương án đã chọn: **giữ nguyên test**, bỏ khỏi dev các lượt của người đã có trong test, chia lại 5 fold dev theo người đã ghép (StratifiedGroupKFold, phân tầng theo nhãn × có ảnh, seed 42).

### Kiểm tra vắt fold — trước và sau khi chia lại

| khoa | thoi_diem | luot_xet | luot_vat_fold | pct | nhom_vat | luot_vat_dev_test | luot_co_anh_vat | pct_co_anh |
|---|---|---|---|---|---|---|---|---|
| Khóa lỏng (họ tên + năm sinh + giới + tỉnh) — DÙNG ĐỂ CHIA | Trước (fold gốc) | 8030 | 1372 | 17.090 | 670 | 392 | 190 | 10.350 |
| Khóa lỏng (họ tên + năm sinh + giới + tỉnh) — DÙNG ĐỂ CHIA | Sau (fold mới, đã bỏ lượt dev trùng test) | 7831 | 0 | 0.000 | 0 | 0 | 0 | 0.000 |
| Khóa chặt (khóa lỏng + phân xưởng) — chỉ để so | Trước (fold gốc) | 8030 | 567 | 7.060 | 282 | 136 | 80 | 4.360 |
| Khóa chặt (khóa lỏng + phân xưởng) — chỉ để so | Sau (fold mới, đã bỏ lượt dev trùng test) | 7831 | 0 | 0.000 | 0 | 0 | 0 | 0.000 |

_Tệp: `1_vat_fold_truoc_sau.csv`_

*`luot_vat_fold`: lượt thuộc nhóm "cùng người" có lượt nằm ở ≥ 2 fold khác nhau (test tính là một fold). `luot_vat_dev_test`: phần trong đó vắt giữa dev và test. `luot_co_anh_vat`: chỉ tính lượt có ảnh. Khóa lỏng gộp nhiều hơn khóa chặt (có thể gộp nhầm người trùng tên), nên con số theo khóa lỏng là mức trên.*

### Tóm tắt

| muc | gia_tri |
|---|---|
| Lượt khám (dev / test) | 8030 (6882 / 1148) |
| Người theo patient_uid gốc | 7024 |
| Người sau ghép khóa lỏng (uid2) / khóa chặt | 6210 / 6689 |
| Nhóm gộp ≥ 2 patient_uid — khóa lỏng / khóa chặt | 801 / 334 |
| Lượt dev bị bỏ (người đã có trong test) | 199 (có ảnh: 71; dương: 29) |
| Tập test | giữ nguyên, không mở (chỉ đếm) |
| dev mới (dev2) | 6683 lượt, 794 dương, 5214 người |
| dev mới có ảnh (tập dùng cho P0.1 / việc (1)) | 1500 lượt, 367 dương, 1345 người, 18 lô |

_Tệp: `2_tom_tat.csv`_

### Cân bằng 5 fold mới (dev2)

| fold2 | luot | duong | co_anh | duong_co_anh | nguoi | ti_le_duong | ti_le_duong_co_anh |
|---|---|---|---|---|---|---|---|
| 0 | 1337 | 160 | 299 | 61 | 1043 | 0.120 | 0.204 |
| 1 | 1330 | 155 | 283 | 72 | 1044 | 0.117 | 0.254 |
| 2 | 1336 | 168 | 287 | 71 | 1043 | 0.126 | 0.247 |
| 3 | 1345 | 144 | 332 | 76 | 1040 | 0.107 | 0.229 |
| 4 | 1335 | 167 | 299 | 87 | 1044 | 0.125 | 0.291 |

_Tệp: `3_fold_moi.csv`_

### Nhóm bnn=Có & ketqua=2 (dùng cho phân tích độ nhạy) sau ghép người

| muc | gia_tri |
|---|---|
| Nhóm bnn=Có & ketqua=2 có ảnh: lượt / người theo patient_uid / người theo uid2 | 57 / 55 / 53 |
|   lượt của nhóm này trong dev mới có ảnh | 45 |

_Tệp: `4_nhom_55_nguoi.csv`_
