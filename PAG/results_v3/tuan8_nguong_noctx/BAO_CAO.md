# Chọn ngưỡng PG (PG-noctx) trên dev

- Nguồn: `run_PG-noctx_nghe10.npz` (khoá `oof_cal`, mã băm `a9379284e305afc0`). **Không dùng tập test**, không đọc `test_cal`.
- Mỗi nhóm tính riêng. Bootstrap 1000 lần theo người (seed 42). "Có nguy cơ" khi xác suất ≥ ngưỡng.
- **Độ nhạy / độ đặc hiệu "lồng nhau"** là ước lượng trung thực: ngưỡng của mỗi fold được chọn từ 4 fold còn lại, rồi đo trên fold đó. Các cột `dev` là đo trên chính dữ liệu dùng để chọn ngưỡng nên lạc quan hơn.
- Điểm `oof_cal` là của một mô hình fold, còn app trả trung bình 5 mô hình fold nên phân bố hẹp hơn một chút; ngưỡng chỉ gần đúng.

- **Quy tắc chính là cân bằng độ nhạy và độ đặc hiệu (Youden), theo chỉ đạo của thầy.** Quy tắc độ nhạy ≥ 0,80 chỉ để tham khảo.

## Ngưỡng đề xuất

| nhom | quy_tac | nguong | do_nhay_dev | do_dac_hieu_dev | ti_le_bao_dev | do_nhay_long_nhau | do_nhay_ktc | do_dac_hieu_long_nhau |
|---|---|---|---|---|---|---|---|---|
| Có phim | **CHÍNH** — cân bằng (Youden) | 0.163 | 0.798 | 0.591 | 0.504 | 0.678 | [0.629–0.731] | 0.684 |
| Có phim | tham khảo — độ nhạy ≥ 0,80 (có dự phòng) | 0.149 | 0.837 | 0.545 | 0.548 | 0.834 | [0.795–0.873] | 0.533 |
| Không phim | **CHÍNH** — cân bằng (Youden) | 0.063 | 0.756 | 0.439 | 0.577 | 0.597 | [0.549–0.643] | 0.522 |
| Không phim | tham khảo — độ nhạy ≥ 0,80 (có dự phòng) | 0.052 | 0.841 | 0.316 | 0.697 | 0.831 | [0.796–0.869] | 0.322 |

## Cách đọc

- `nguong`: ngưỡng đưa vào web. `do_nhay_dev`: tỉ lệ ca bệnh dev có điểm ≥ ngưỡng. `do_dac_hieu_dev`: tỉ lệ người không bệnh có điểm < ngưỡng.
- `ti_le_bao_dev`: tỉ lệ **mọi người** bị gán "có nguy cơ", là cái giá thật của ngưỡng.

### Có phim — n=1500, 367 ca bệnh (24.5%), AUC OOF 0.760

- **CHÍNH — Cân bằng (Youden):** ngưỡng **0.1627** (dao động 95% 0.1494–0.2660); độ nhạy lồng nhau 0.678 [0.629–0.731], độ đặc hiệu lồng nhau 0.684 [0.654–0.713]; bị báo "có nguy cơ" 50.4% (dev, độ nhạy dev 0.798, độ đặc hiệu dev 0.591).
- *Tham khảo — nếu ưu tiên bắt ≥ 80% ca bệnh:* nâng mục tiêu lên s* = 0.835 (đạt) → ngưỡng 0.1494; độ nhạy lồng nhau 0.834 [0.795–0.873]; độ đặc hiệu lồng nhau 0.533; bị báo 54.8% (dev).
- *Tham khảo — nhắm đúng 0,80 không dự phòng:* ngưỡng 0.1619, độ nhạy lồng nhau 0.793 [0.751–0.836].

### Không phim — n=5183, 427 ca bệnh (8.2%), AUC OOF 0.630

- **CHÍNH — Cân bằng (Youden):** ngưỡng **0.0631** (dao động 95% 0.0593–0.0964); độ nhạy lồng nhau 0.597 [0.549–0.643], độ đặc hiệu lồng nhau 0.522 [0.508–0.537]; bị báo "có nguy cơ" 57.7% (dev, độ nhạy dev 0.756, độ đặc hiệu dev 0.439).
- *Tham khảo — nếu ưu tiên bắt ≥ 80% ca bệnh:* nâng mục tiêu lên s* = 0.840 (đạt) → ngưỡng 0.0524; độ nhạy lồng nhau 0.831 [0.796–0.869]; độ đặc hiệu lồng nhau 0.322; bị báo 69.7% (dev).
- *Tham khảo — nhắm đúng 0,80 không dự phòng:* ngưỡng 0.0576, độ nhạy lồng nhau 0.778 [0.740–0.817].

## Bảng đánh đổi (ngưỡng theo độ nhạy trên dev)

| nhom | do_nhay_dev | nguong | do_dac_hieu | ti_le_bi_bao |
|---|---|---|---|---|
| Có phim | 0.700 | 0.207 | 0.675 | 0.417 |
| Có phim | 0.750 | 0.185 | 0.628 | 0.465 |
| Có phim | 0.800 | 0.162 | 0.585 | 0.509 |
| Có phim | 0.850 | 0.135 | 0.499 | 0.587 |
| Có phim | 0.900 | 0.108 | 0.424 | 0.656 |
| Không phim | 0.700 | 0.068 | 0.477 | 0.538 |
| Không phim | 0.750 | 0.063 | 0.440 | 0.576 |
| Không phim | 0.800 | 0.058 | 0.382 | 0.633 |
| Không phim | 0.850 | 0.050 | 0.290 | 0.722 |
| Không phim | 0.900 | 0.045 | 0.231 | 0.780 |

## Theo tỉnh (kiểm tra; độ nhạy chỉ tính khi ≥ 20 ca bệnh)

| nhom | tinh | n | ca_benh | nhay_do_nhay_0.80 | dac_hieu_do_nhay_0.80 | ti_le_bao_do_nhay_0.80 | nhay_youden | dac_hieu_youden | ti_le_bao_youden |
|---|---|---|---|---|---|---|---|---|---|
| Có phim | Binh Dinh | 512 | 131 | 0.916 | 0.609 | 0.525 | 0.908 | 0.633 | 0.506 |
| Có phim | Hai Duong | 352 | 76 | 0.684 | 0.649 | 0.423 | 0.618 | 0.696 | 0.372 |
| Có phim | Phu Yen | 3 | 1 | — | — | 0.667 | — | — | 0.333 |
| Có phim | Thai Nguyen | 633 | 159 | 0.843 | 0.435 | 0.635 | 0.799 | 0.498 | 0.577 |
| Không phim | Binh Dinh | 538 | 3 | — | — | 0.546 | — | — | 0.428 |
| Không phim | Dong Nai | 1205 | 110 | 0.836 | 0.289 | 0.722 | 0.773 | 0.417 | 0.600 |
| Không phim | Hai Duong | 943 | 49 | 0.735 | 0.390 | 0.616 | 0.633 | 0.530 | 0.478 |
| Không phim | Phu Yen | 540 | 88 | 0.852 | 0.378 | 0.659 | 0.784 | 0.469 | 0.572 |
| Không phim | Thai Nguyen | 1957 | 177 | 0.864 | 0.236 | 0.773 | 0.763 | 0.358 | 0.653 |
