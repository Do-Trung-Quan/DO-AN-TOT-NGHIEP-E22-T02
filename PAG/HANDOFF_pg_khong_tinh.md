# Bàn giao: huấn luyện PG không dùng tỉnh và tính ngưỡng cho web (máy có GPU)

Cập nhật: 26/11/2026 · Nhánh git: `PG` · Người nhận: Minh (hoặc agent chạy trên máy của Minh)

> **Việc của bạn:** chạy đúng các lệnh ở mục 4, theo đúng thứ tự, rồi gửi lại các file ở mục 5.
> Không sửa code. Không sửa tham số. Không chấm điểm trên tập test. Không tự chọn ngưỡng.

---

## 1. Vì sao phải làm (1 phút)

- Thầy yêu cầu: **bỏ thông tin tỉnh khỏi mô hình**. Công nhân đều ở Thái Nguyên nên tỉnh dễ thành "đường tắt" (mô hình đoán theo nơi thu thập chứ không theo bệnh nhân).
- Thầy đồng ý **chọn ngưỡng trên tập dev**, theo cách **cân bằng độ nhạy và độ đặc hiệu** (Youden). Không dùng test để chọn ngưỡng.
- Ngưỡng phụ thuộc vào mô hình cuối cùng. Vì vậy phải **huấn luyện lại PG không có tỉnh trước**, rồi mới tính ngưỡng từ dự đoán của chính mô hình đó.
- Tên biến thể mới: **`PG-noctx`** (cùng PG 6 vùng, nhưng nút "tỉnh" bị che hoàn toàn, thông tin không chảy vào mô hình).
- Đây là biến thể đã khai báo trong code từ trước (`use_ctx=False`), không phải mô hình mới. Phải dò lại 30 cấu hình vì bỏ nút tỉnh thì cấu hình tốt nhất có thể đổi.

## 2. Chuẩn bị

### 2.1. Lấy code
```bash
git fetch
git checkout PG
git pull
```
Phải có các file: `PAG/scripts_v3/v3_chon_nguong.py` và trong `v3_run_pg.py` dòng `choices=["PG-global","PG","PG-shuffle","PG-noctx"]`.
Thiếu thì người gửi chưa push. **Dừng và hỏi lại**, đừng tự viết lại.

### 2.2. Thư viện
Giống bước 4–5 trước. Kiểm tra:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # phải in True
```
`False` thì **dừng**, không chạy trên CPU.

### 2.3. Dữ liệu
Dùng đúng bộ dữ liệu đã dùng cho bước 4–5 (`cache_pg/clin_raw.pkl`, `PAG/cache_v2/*`, `PAG/cache_v3/img_v3.npz`, `p05_folds.npz`). Không tải thêm, không thay.

**Quy tắc dữ liệu (bắt buộc):**
- Thư mục `cache*` chứa mã lượt khám và mã người: **không commit, không gửi lên dịch vụ ngoài** (Drive công khai, chatbot…).
- **Không mở, không chấm, không in nhãn hay AUC của tập test.**

### 2.4. Kiểm tra dữ liệu (1 phút)
```bash
cd PAG/scripts_v3
python -c "from v3_base import load_base_v3; B = load_base_v3(); print(int(B['dev'].sum()), int((B['dev'] & B['has_img']).sum()), int(B['test'].sum()), B['raw'].shape)"
```
Phải ra đúng: `6683 1500 1148 (8030, 126)`. Lệch thì **dừng và báo**.

---

## 3. Chạy thử nhanh (2 phút)
```bash
python v3_run_pg.py PG-noctx --search --smoke
```
Đúng khi: có dòng `thiet bi cuda`, có dòng cuối `XONG`. Số AUC của bản thử vô nghĩa, bỏ qua.
Bản thử ghi file có đuôi `_smoke`. **Xóa các file `_smoke` trước khi chạy thật** (hoặc để yên, chúng không ảnh hưởng).

---

## 4. Chạy thật, theo đúng thứ tự

Chạy trong `PAG/scripts_v3`. **Ghi lại thời gian thực tế của từng lệnh** (chưa ai đo) và gửi kèm.

| # | Lệnh | Làm gì | Tạo ra |
|---|---|---|---|
| 1 | `python v3_run_pg.py PG-noctx --search` | Huấn luyện PG không tỉnh: 5 fold x 30 cấu hình, rồi 3 seed. Lệnh lâu nhất | `cache_v3/run_PG-noctx_nghe10.npz`, `selected_PG-noctx_nghe10.json`, `log_PG-noctx_nghe10.csv` |
| 2 | `python v3_xuat_pg.py --ten PG-noctx` | Học lại và lưu trọng số 5 fold cho web, kiểm tra tái tạo điểm | `cache_v3/mo_hinh_PG_noctx_web/` và `results_v3/tuan8_xuat_web_noctx/` |
| 3 | `python ..\xuat_app_pg\kiem_tra.py --ten PG-noctx --bo-anh` | Kiểm tra mô hình xuất ra chạy đúng, và gửi tỉnh lạ không làm đổi kết quả | dòng cuối phải là `... -> DAT` (không phải `KHONG DAT`) |
| 4 | `python v3_chon_nguong.py` | Tính ngưỡng trên **dev** cho 2 nhóm (có phim, không phim) | `results_v3/tuan8_nguong_noctx/` |

Lệnh 3: nếu đường dẫn tương đối không chạy, `cd ..\xuat_app_pg` rồi chạy `python kiem_tra.py --ten PG-noctx --bo-anh`.

**Dấu hiệu đúng:**
- Lệnh 1: mỗi fold in `fold k: cau hinh … epoch …`; dòng cuối `XONG … co anh AUC … | khong anh AUC …`. Không có dòng `LECH`.
- Lệnh 2: mỗi fold in `... (khop) | lech OOF max ...`. Phải là `khop` (không phải `LECH`), `lech OOF max` rất nhỏ (cỡ 1e-7 như bản PG có tỉnh). Lệch lớn (>1e-3) hoặc `LECH` thì **dừng và báo**, kèm nguyên văn dòng in ra. Dòng cuối là `[xuat PG] XONG ...`.
- Lệnh 3: dòng cuối `[kiem] lech diem lon nhat ... -> DAT`; có dòng `ban khong tinh: ... DUNG`.
- Lệnh 4: tạo `BAO_CAO.md`, `nguong.csv`, `danh_doi.csv`, `theo_tinh.csv`, `nguong_pg.json`, `run_config.json`.

**Nếu bị dừng giữa chừng:** chạy lại đúng lệnh đó từ đầu (mỗi lệnh chỉ ghi file khi xong).

**Lỗi thường gặp:**

| Thông báo | Cách xử lý |
|---|---|
| `ModuleNotFoundError: v3_base` | Chưa đứng trong `PAG/scripts_v3` |
| `invalid choice: 'PG-noctx'` | Chưa `git pull` bản mới. Dừng và báo |
| `FileNotFoundError … run_PG-noctx_nghe10.npz` ở lệnh 2/4 | Lệnh 1 chưa xong |
| `CUDA out of memory` | Đóng chương trình khác dùng GPU rồi chạy lại. Vẫn lỗi thì báo, **đừng đổi tham số** |

---

## 5. Gửi lại những gì

**A. Được commit lên nhánh `PG`** (chỉ số liệu gộp, không có mã người) — **chỉ làm khi chủ repo cho phép**; mặc định chỉ chạy xong rồi báo, người gửi tự commit:
- `PAG/results_v3/tuan8_xuat_web_noctx/`
- `PAG/results_v3/tuan8_nguong_noctx/`
- `PAG/results_v3/tuan3_PG_nghe10/PG-noctx/` (nếu có)

**B. Gửi riêng, KHÔNG commit** (dự đoán theo từng lượt khám hoặc trọng số mô hình):
- `PAG/cache_v3/run_PG-noctx_nghe10.npz`
- `PAG/cache_v3/selected_PG-noctx_nghe10.json`
- `PAG/cache_v3/log_PG-noctx_nghe10.csv`
- thư mục `PAG/cache_v3/mo_hinh_PG_noctx_web/` (~23 MB, mô hình dùng cho web, giữ riêng tư)

**C. Gửi kèm trong tin nhắn:** thời gian từng lệnh, và dán nguyên văn dòng cuối (`XONG …`) của lệnh 1. Gặp lỗi thì dán **nguyên văn** thông báo lỗi, đừng tóm tắt.

---

## 6. Prompt cho agent (dán nguyên khối này vào agent bên máy Minh)

```
Bạn là trợ lý chạy lệnh trên máy có GPU. Đọc file PAG/HANDOFF_pg_khong_tinh.md rồi làm đúng từng bước ở mục 2 đến 4. Bạn CHỈ chạy lệnh, KHÔNG phát triển.

ĐƯỢC LÀM:
- git fetch, git checkout PG, git pull.
- Chạy các lệnh trong mục 2.2, 2.4, 3 và 4 theo đúng thứ tự, trong thư mục PAG/scripts_v3.
- Đọc log và báo lại cho tôi.

TUYỆT ĐỐI KHÔNG:
1. Không sửa bất kỳ file code nào, không đổi tham số, không đổi số seed, số epoch, số cấu hình, không thêm cờ lạ vào lệnh.
2. Không sửa hoặc xóa: PAG/results_v3/tuan4_so_sanh_chinh_nghe10/khai_bao_truoc_v3.json, mọi file khoa_cau_hinh.json, mọi file trong results_v3/tuan6_test_mot_lan.
3. Không chấm, không in, không phân tích nhãn hay điểm của TẬP TEST. Không đọc khóa test_cal / test của file .npz.
4. Không tự chọn ngưỡng. Lệnh python v3_chon_nguong.py tự tính; bạn chỉ chạy và báo lại đúng những gì nó in ra.
5. Không chạy các lệnh khác ngoài danh sách (không chạy lại PG, PG-global, PG-shuffle, baseline, báo cáo H1-H3).
6. Không git add / commit / push bất cứ thứ gì nếu tôi chưa nói rõ. Không bao giờ commit thư mục cache_v2, cache_v3, cache_pg hoặc file .npz, .pkl, .pt.
7. Không tải dữ liệu hay file lên dịch vụ bên ngoài (Drive, chatbot, pastebin…). Không gửi mã lượt khám, mã người, tên, số điện thoại.
8. Không đọc file .env.
9. Không "sửa giúp" khi gặp lỗi. Khi lỗi, DỪNG, dán nguyên văn thông báo lỗi và lệnh vừa chạy, chờ tôi.

ĐIỀU KIỆN DỪNG (dừng ngay và báo, không tự xử lý):
- torch.cuda.is_available() ra False.
- Kiểm tra dữ liệu mục 2.4 không ra đúng "6683 1500 1148 (8030, 126)".
- Có dòng chứa chữ LECH (viết hoa), hoặc 'lech OOF max' ở lệnh 2 lớn hơn 1e-3, hoặc lệnh 3 ra KHONG DAT / LOI.
- Bất kỳ lệnh nào lỗi, hoặc chạy quá 12 giờ không có tiến triển.

KHI XONG, báo lại đúng các mục sau:
(a) thời gian từng lệnh; (b) nguyên văn dòng cuối 'XONG' của lệnh 1; (c) nguyên văn dòng tóm tắt của lệnh 2 và lệnh 3;
(d) danh sách file đã tạo ở PAG/cache_v3 và PAG/results_v3; (e) nội dung file PAG/results_v3/tuan8_nguong_noctx/BAO_CAO.md.
Không giải thích thêm, không đề xuất sửa đổi.
```

---

## 7. Sau khi nhận file (việc của người gửi, không phải của Minh)

1. Chép `run_PG-noctx_nghe10.npz`, `selected_...json` vào `PAG/cache_v3/` của máy mình, chạy lại `v3_chon_nguong.py` để chắc kết quả trùng.
2. Đọc `BAO_CAO.md`: lấy ngưỡng Youden cho 2 nhóm (quy tắc chính). Bảng 80% độ nhạy chỉ để tham khảo/cho thầy xem cái giá.
3. So AUC dev của PG-noctx với PG có tỉnh, báo thầy mức giảm (nếu có).
4. Ghi ngưỡng vào `cau_hinh_chung.json` của mô hình web, dạng:
   `"nguong": {"co_phim": {"nguong": x}, "khong_phim": {"nguong": y}}`
   rồi cập nhật backend/web: bỏ tỉnh khỏi form và bộ kiểm tra, thêm `nguong`, `bao_nguy_co`, `nhom`, hiển thị "có nguy cơ / nguy cơ thấp".
