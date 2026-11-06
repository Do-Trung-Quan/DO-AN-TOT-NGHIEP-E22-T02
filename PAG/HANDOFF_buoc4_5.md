# Bàn giao: chạy bước 4 (huấn luyện PG) và bước 5 (kiểm định H1–H3) trên máy có GPU

Cập nhật: 06/11/2026 · Nhánh git: `PG` · Đề cương gốc: `PAG/PG_update.pdf` (mục 6 và 9)

> **Việc của bạn:** chạy đúng các lệnh ở mục 4, theo đúng thứ tự, rồi gửi lại các file ở mục 5.
> Không cần sửa code. Không sửa file khai báo trước. Không chấm điểm trên tập test.

---

## 1. Bối cảnh trong 1 phút

- **Bài toán:** đoán công nhân có bụi phổi silic không (`ketqua` = 1) từ phim X-quang (nếu có) và hồ sơ khám.
- **PG (Patient Graph):** mỗi lượt khám là một **đồ thị nhỏ**. Mỗi **nút** là một nhóm thông tin (vùng phổi trên phim, nhóm lâm sàng, tỉnh), các **cạnh** nối nút với nhau. Một mô hình *graph transformer* cho các nút "trao đổi" theo cạnh rồi đưa ra xác suất bệnh.
- **Đã xong (bước 0–3):** dữ liệu, fold, bảng biến → nút, dựng đồ thị. QA đạt, xem `PAG/results_v3/tuan2_do_thi_PG/BAO_CAO.md`.
- **Việc của bạn:**
  - **Bước 4:** huấn luyện **PG-global** (1 nút ảnh), **PG** (6 nút vùng phổi) và **PG-shuffle**. PG-shuffle là PG có cạnh giữa các vùng bị xáo trộn, dùng để kiểm tra H2.
  - Chạy lại các **mô hình mốc** (baseline) với cùng cách mã hóa mới.
  - **Bước 5:** chạy kiểm định theo file khai báo trước.

**Điểm mới so với kết quả cũ của Minh:** nút "Phơi nhiễm/nghề" giờ dùng **10 cấp nghề theo mức bụi** thay cho tên công việc dạng chữ thô. Vì vậy các baseline có dùng lâm sàng (L0, L2, X0/X2, T0/T2, B3b/GAT) phải chạy lại để so công bằng.

Mọi file kết quả mới mang đuôi **`_nghe10`**, nên **không ghi đè** kết quả cũ.

---

## 2. Chuẩn bị máy

### 2.1. Lấy code
```bash
git fetch
git checkout PG
git pull
```
Kiểm tra có đủ các file sau trong `PAG/scripts_v3/`: `v3_run_pg.py`, `v3_report_pg.py`, `v3_base.py`, `v3_graph_lib.py`, `v3_baselines.py`, `v3_ftt.py`.
Thiếu file nào thì người gửi chưa push, hãy hỏi lại.

### 2.2. Thư viện Python
Cần: `torch` (bản có CUDA), `numpy`, `pandas`, `scikit-learn`, `scipy`, `xgboost`.
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # phải in True
python -c "import xgboost, sklearn, pandas; print('ok')"
```
`torch.cuda.is_available()` ra `False` thì máy chưa dùng được GPU. Dừng lại, cài đúng bản torch CUDA trước.

### 2.3. Dữ liệu (nhận riêng, KHÔNG có trên git)
Chép **đúng vị trí** dưới đây, tính từ thư mục gốc repo:

```
DO-AN-TOT-NGHIEP-E22-T02/
├── cache_pg/clin_raw.pkl          bảng lâm sàng thô (8030 hàng x 121 cột)   — NẰM NGOÀI thư mục PAG
└── PAG/
    ├── cache_v2/core.pkl          nhãn, mã người, fold, lô, tỉnh
    ├── cache_v2/img_v2.npz        ảnh bản 2 (không dùng nhưng code vẫn đọc — thiếu sẽ lỗi)
    ├── cache_v3/img_v3.npz        đặc trưng 6 vùng phổi BioViL-T (ảnh V2)
    ├── cache_v3/p05_folds.npz     fold chuẩn duy nhất
    └── cache_v3/graphs/           (TUỲ CHỌN, ~609 MB) đồ thị đã dựng — chỉ để đối chiếu, thiếu vẫn chạy được
```
Nếu bạn là Minh: bạn đã có các file này. Chỉ cần kiểm tra `cache_pg/clin_raw.pkl` đúng là file 121 cột đã gửi.

**Quy tắc dữ liệu (bắt buộc):**
- Cache chứa mã lượt khám và mã người, nên **không commit, không chia sẻ tiếp, không đưa lên dịch vụ ngoài** (Drive công khai, chatbot…). `.gitignore` đã chặn các thư mục `cache*`.
- **Không mở / chấm điểm tập test.** Code tự lưu dự đoán test nhưng không chấm. Đừng tự tính AUC trên test.

### 2.4. Kiểm tra dữ liệu (1 phút)
Mọi lệnh từ đây trở đi chạy **trong thư mục `PAG/scripts_v3`**:
```bash
cd PAG/scripts_v3
python -c "from v3_base import load_base_v3; B = load_base_v3(); print(int(B['dev'].sum()), int((B['dev'] & B['has_img']).sum()), int(B['test'].sum()), B['raw'].shape)"
```
Phải ra đúng: `6683 1500 1148 (8030, 126)`. Nghĩa là:
- 6.683 lượt dev, trong đó 1.500 có phim;
- 1.148 lượt test;
- 126 cột: 121 cột của `clin_raw.pkl`, bỏ `nam`, thêm `bmi` và `bnn`, rồi thêm 4 cột cấp nghề tự tạo.
Lệch số nào thì **dừng lại** và báo người gửi.

---

## 3. Chạy thử nhanh trên GPU (2 phút)
```bash
python v3_run_pg.py PG-global --search --smoke
```
Kết quả đúng phải có:
- dòng `thiet bi cuda` (nếu ra `cpu` thì không dùng được GPU, xem lại mục 2.2);
- dòng `doi chieu pg_global_fold0_ngoai.pt: KHOP`, hoặc `bo qua` nếu bạn không chép thư mục `graphs`. Ra **`LECH`** thì dừng lại báo;
- dòng cuối có `XONG`.

Số AUC của bản chạy thử **vô nghĩa** (chỉ 1 fold, 2 epoch). Không dùng.

---

## 4. Chạy thật, theo đúng thứ tự

Thời gian chạy trên GPU **chưa đo**. Hãy ghi lại thời gian thực tế của từng lệnh và gửi kèm kết quả.

| # | Lệnh | Làm gì | Tạo ra (trong `PAG/cache_v3/`) |
|---|---|---|---|
| 1 | `python v3_baselines.py` | Chạy lại các mốc L0, C, L2, X0/X2, B3b, GAT với 10 cấp nghề | `run_baselines_v3_nghe10.npz`, `chon_baselines_v3_nghe10.csv` |
| 2 | `python v3_ftt.py T2` | FT-Transformer có ảnh, mốc cho H3 | `run_T2_nghe10.npz`, `selected_T2_nghe10.json`, `log_T2_nghe10.csv` |
| 3 | `python v3_ftt.py T0` | FT-Transformer chỉ lâm sàng, để tham khảo | `run_T0_nghe10.npz`, … |
| 4 | `python v3_run_pg.py PG-global --search` | **PG-global**: 1 nút ảnh. Làm trước (thầy chốt) | `run_PG-global_nghe10.npz`, `selected_PG-global_nghe10.json`, `log_…` |
| 5 | `python v3_run_pg.py PG --search` | **PG 6 vùng**: mô hình chính | `run_PG_nghe10.npz`, `selected_PG_nghe10.json`, `log_…` |
| 6 | `python v3_run_pg.py PG-shuffle` | PG-shuffle cho H2. **Phải chạy sau lệnh 5** vì dùng lại cấu hình PG đã chọn | `run_PG-shuffle_nghe10.npz`, `log_…` |
| 7 | `python v3_report_pg.py --buoc 4` | Bảng kết quả **mô tả** (chưa kiểm định) | `PAG/results_v3/tuan3_PG_nghe10/BAO_CAO.md` |
| 8 | `python v3_report_pg.py --buoc 5` | **Kiểm định H1–H3** theo file khai báo trước | `PAG/results_v3/tuan4_so_sanh_chinh_nghe10/BAO_CAO.md` |

**Mỗi lệnh 4–6 làm gì bên trong** (để hiểu log in ra):
- Có 5 fold. Với mỗi fold k, mô hình được thử **30 bộ tham số**: học trên 3 fold, chấm trên 1 fold "validation nội bộ". Bộ tốt nhất được giữ lại, và log ghi `fold k: cau hinh … epoch …`.
- Sau đó mô hình được học lại trên 4 fold với **3 seed** (3 lần khởi tạo ngẫu nhiên), lấy trung bình, rồi dự đoán fold k.
- Xác suất được **hiệu chuẩn Platt** trong fold, riêng nhóm có ảnh và không ảnh. Mô hình học với trọng số lớp để không bỏ sót ca bệnh hiếm, nên xác suất thô bị đẩy cao; hiệu chuẩn sửa lại cho khớp tần suất thật.
- Dòng cuối `XONG … co anh AUC … | khong anh AUC … | trong lo …` là kết quả thô để xem nhanh. Số chính thức nằm trong báo cáo ở lệnh 7 và 8.

**Nếu bị dừng giữa chừng** (mất điện, hết RAM GPU…): chạy lại đúng lệnh đó từ đầu. Mỗi lệnh ghi file khi chạy xong. Không cần chạy lại các lệnh đã `XONG` trước đó.

**Lỗi thường gặp:**

| Thông báo | Cách xử lý |
|---|---|
| `ModuleNotFoundError: v3_base` / `pag_lib` | Bạn chưa đứng trong `PAG/scripts_v3`. Chạy `cd PAG/scripts_v3` |
| `FileNotFoundError … selected_PG_nghe10.json` (khi chạy PG-shuffle) | Chưa chạy xong lệnh 5 |
| `No module named xgboost` | `pip install xgboost` |
| `CUDA out of memory` | Đóng chương trình khác dùng GPU rồi chạy lại. Vẫn lỗi thì báo người gửi, **đừng tự đổi tham số** |
| `doi chieu … LECH` | Dừng lại, báo người gửi (dữ liệu khác với lúc dựng đồ thị) |
| Báo cáo bước 5 ghi "Chưa đủ dữ liệu để kiểm định" | Còn thiếu file của lệnh 1, 2, 5 hoặc 6. Xem dòng "Thiếu dự đoán" cuối báo cáo |

---

## 5. Gửi lại những gì

**A. Được commit lên nhánh `PG`** (chỉ chứa số liệu gộp, không có mã người):
- `PAG/results_v3/tuan3_PG_nghe10/` (báo cáo bước 4)
- `PAG/results_v3/tuan4_so_sanh_chinh_nghe10/` (báo cáo bước 5)
- `PAG/results_v3/tuan1_phase0/viec2_baseline_fold_moi_nghe10/` (run_config của baseline mới)

```bash
git add PAG/results_v3/tuan3_PG_nghe10 PAG/results_v3/tuan4_so_sanh_chinh_nghe10 PAG/results_v3/tuan1_phase0/viec2_baseline_fold_moi_nghe10
git commit -m "Buoc 4-5: PG-global, PG, PG-shuffle + baseline nghe10 + kiem dinh H1-H3"
git push
```

**B. Gửi riêng, KHÔNG commit** (dự đoán theo từng lượt khám): mọi file `*_nghe10*` trong `PAG/cache_v3/`:
`run_*_nghe10.npz`, `selected_*_nghe10.json`, `log_*_nghe10.csv`, `chon_baselines_v3_nghe10.csv`.

**C. Nhắn kèm:**
- GPU đã dùng;
- thời gian chạy của từng lệnh;
- dòng `XONG …` cuối mỗi lệnh 4–6;
- mọi cảnh báo hoặc lỗi lạ.

---

## 6. Những điều KHÔNG được làm

1. **Không sửa** `PAG/results_v3/tuan4_so_sanh_chinh_nghe10/khai_bao_truoc_v3.json`. File này ghi luật kiểm định **trước khi có kết quả**. Sửa sau khi đã thấy số là làm hỏng giá trị của bước 5.
2. Không chấm điểm, không xem nhãn tập test.
3. Không đặt `PG_NGHE10=0`. Biến này chỉ để tái hiện số cũ.
4. Không đổi tham số trong code (số cấu hình, seed, epoch…). Có vấn đề thì báo người gửi.
5. Không commit thư mục `cache_*`, không gửi dữ liệu lên dịch vụ bên ngoài.
6. Không thêm `--smoke` khi chạy thật.

---

## 7. Đọc kết quả bước 5 (để tự kiểm tra trước khi gửi)

- **Δ** = điểm của PG trừ điểm mô hình so sánh, tính trên cùng các lượt khám.
- **KTC 95%** là khoảng tin cậy, lấy bằng cách bốc lại người ngẫu nhiên 1.000 lần.
- **"Không kém"** (H1, H3): đạt khi cận dưới KTC > −0,02, tức dù xấu nhất PG cũng chỉ kém tối đa 0,02 AUC.
- **"Hơn"** (H2): đạt khi cả KTC nằm trên 0.
- **Holm:** siết ngưỡng vì so nhiều cặp cùng lúc. Cột `dat_sau_Holm` là kết luận chính thức.
- **`P_delta_duong`** chỉ là tỉ lệ lần bốc lại mà PG hơn, **không phải p-value**.
- **Điểm duyệt 2:** nếu báo cáo ghi H1 và H2 đều KHÔNG ĐẠT thì theo đề cương phải dừng và báo thầy. Đây là kết quả hợp lệ, không phải lỗi chạy.
