# Bàn giao: dựng đồ thị bệnh nhân PG (Patient-as-Graph) — Phase 0 bản 3

Cập nhật: 28/10/2026. Đặc tả gốc: `PG_update.pdf` (mục 4 — cấu trúc đồ thị). Lời nhắn của thầy ngày 27/10 là phạm vi công việc.

> **Việc của bạn: chỉ DỰNG đồ thị + kiểm tra chất lượng (QA). Chưa huấn luyện mô hình PG.**

---

## 1. Chuẩn bị

### 1.1. Code
Có sẵn trong repo, thư mục `PAG/`:

| Thư mục | Nội dung |
|---|---|
| `scripts_v2/` | `v2_lib.py` (nạp bảng cốt lõi), `pag_lib.py` (nhóm biến, mã hoá trong fold, loại cạnh — code đồ thị của bản 2) |
| `scripts_v3/` | `v3_lib.py` (đường dẫn cache_v3, `run_config`), `v3_base.py` (`load_base_v3()` — điểm vào dữ liệu), các script baseline |
| `results_v2/`, `results_v3/` | Báo cáo đã nộp (BAO_CAO.md + CSV) |

Môi trường Python: `numpy`, `pandas`, `torch`, `scikit-learn`.

### 1.2. Dữ liệu (nhận riêng, KHÔNG có trên git)
5 file cache, chép đúng vị trí dưới đây (tính từ gốc repo):

```
DO-AN-TOT-NGHIEP-E22-T02/
├── cache_pg/clin_raw.pkl            lâm sàng thô, 121 cột (nằm NGOÀI PAG)
└── PAG/
    ├── cache_v2/core.pkl            id, nhãn y, uid2, dev2, fold2, has_img, lô, tỉnh
    ├── cache_v2/img_v2.npz          ảnh bản 2 — không dùng, nhưng load_base() vẫn đọc
    ├── cache_v3/p05_folds.npz       fold chuẩn (id, uid2, dev2, fold2)
    └── cache_v3/img_v3.npz          img_id, regf_biovil_V2 (6 vùng × 512), area (6 vùng)
```

Code tính đường dẫn theo vị trí script (`PAG/..` được coi là thư mục dữ liệu gốc), nên phải đặt đúng như trên.
Nếu thiếu `core.pkl`, code sẽ đi tìm file Excel gốc và báo lỗi.

Kiểm tra nhanh:
```python
import sys; sys.path.insert(0, "PAG/scripts_v3")
from v3_base import load_base_v3
B = load_base_v3()          # tự assert fold/dev/uid khớp p05_folds.npz
print(B["dev"].sum(), (B["dev"] & B["has_img"]).sum())   # mong đợi: 6683, 1500
```

### 1.3. Quy tắc dữ liệu (bắt buộc)
- Cache chứa id lượt khám và mã người → **không commit, không chia sẻ tiếp, không gửi lên dịch vụ ngoài**. `.gitignore` đã chặn `cache*/`.
- `core.pkl` có các cột đọc phim (`matdo`, `kichthuoc`, `chatluong`) và `nam_bnn` → **cấm dùng làm đặc trưng** (rò nhãn).
- Biến cấm khác: xem `NOT_USED` trong `scripts_v2/pag_lib.py`.
- **Fold chuẩn duy nhất:** `dev2` / `fold2` / `uid2` trong `cache_v3/p05_folds.npz` (5 fold theo người, đã ghép người trùng mã).
- **Test khoá:** chỉ được đếm số lượt, không chấm, không nhìn nhãn.
- Mọi bước học từ dữ liệu (điền thiếu, chuẩn hoá, one-hot, PCA) **fit trong từng fold**, chỉ trên phần train.
- Báo cáo chỉ ghi số liệu gộp, không ghi id.

---

## 2. Đã làm (Phase 0 bản 3)

| Bước | Kết quả chính | Báo cáo |
|---|---|---|
| P0.1 | Đo nhiễu theo lô trên ảnh V1/V2. Chọn **BioViL-T đóng băng trên ảnh V2** (che chữ + xoá ngoài phổi). | `results_v3/tuan1_phase0/P0_1_nhieu_lo_V1_V2/` |
| P0.2 | Xử lý ảnh bản 3 (phân đoạn phổi, OCR che chữ, cắt theo phổi) + cờ QA. | `.../P0_2_xu_ly_anh_QA/` |
| P0.5 | Ghép người bị tách mã, chia lại fold → `p05_folds.npz`. Dev = 6683 lượt (1500 có ảnh, 367 dương), test giữ nguyên. | `.../P0_5_ghep_nguoi/` |
| Việc (1) | Vùng so với toàn ảnh, tinh chỉnh C bằng inner CV + stacking. **Chốt C = trung bình 6 vector vùng**: AUC trong lô 0,740. Stacking ngang C (S − C = +0,000 [−0,024; +0,024]). Nối 6 vùng / PCA theo vùng đều kém hơn. | `.../viec1b_vung_tinh_chinh_C_stacking_V2/` |
| Việc (2) | Chạy lại toàn bộ baseline trên fold mới: L0, C, L2, X0/X2, B3b, GAT, T0/T2. | `.../viec2_baseline_fold_moi/` |
| Việc (3) | Giải thích LOBO 0,742 so với 0,764: cùng một tập dự đoán. 0,764 là AUC gộp, 0,741 là AUC trong lô. Từ nay ghi cả hai. | `.../viec3_LOBO/` |

Kết quả việc (2), nhóm có ảnh (1500 lượt):

| Mô hình | ROC-AUC [KTC 95%] | AUC trong lô |
|---|---|---|
| L0 — LR lâm sàng | 0,639 [0,605–0,673] | 0,568 |
| **C — LR ảnh** | **0,752** [0,724–0,778] | **0,729** |
| L2 — LR lâm sàng + ảnh | 0,752 [0,724–0,780] | 0,701 |
| X2 — XGBoost lâm sàng + ảnh | 0,723 [0,697–0,750] | 0,681 |
| B3b — kNN | 0,728 [0,701–0,755] | 0,696 |
| GAT | 0,726 [0,700–0,754] | 0,687 |
| T2 — FT-Transformer + 6 token vùng | 0,725 [0,694–0,753] | 0,656 |

Giải thích chỉ số:
- **ROC-AUC:** xác suất mô hình xếp một ca dương cao hơn một ca âm; 0,5 là đoán bừa.
- **AUC trong lô:** ROC-AUC tính riêng trong từng lô chụp, rồi lấy trung bình theo cỡ lô. Chỉ số này loại phần mô hình "ăn" nhờ chênh tỉ lệ bệnh giữa các lô.

Nhận xét liên quan đến PG:
- Mốc PG cần vượt là **C: AUC trong lô 0,729**.
- Thêm lâm sàng vào ảnh làm giảm AUC trong lô (L2 − C = −0,028 [−0,051; −0,007]). Phần lâm sàng ở nhóm có ảnh có thể mang thông tin lô chụp.
- Ở nhóm không ảnh, chưa mô hình nào vượt LR/XGBoost lâm sàng (ROC-AUC khoảng 0,70–0,71).

---

## 3. Việc cần làm: dựng đồ thị PG

### 3.1. Node (tối đa 14)

| Khe | Node | Đặc trưng |
|---|---|---|
| 0–5 | 6 vùng phổi RU, RM, RL, LU, LM, LL | `regf_biovil_V2` (512 chiều/vùng) + `area`. **Chỉ có khi lượt khám có phim.** |
| 6–11 | 6 nhóm lâm sàng: Nhân khẩu & thể trạng · Phơi nhiễm/nghề · Hút thuốc · Triệu chứng & tiền sử · Khám thực thể · Hô hấp ký | Biến theo `GROUPS` trong `pag_lib.py` |
| 12 | Bối cảnh | `tinh`. **Không dùng `nam`** (năm công nhận BNN → rò nhãn) |
| 13 | Bệnh nhân ảo | Nối tới mọi node |

- Lượt **không có phim**: bỏ 6 node vùng, không điền ảnh giả → còn 8 node.
- **Node vùng dùng projection chung:** một phép chiếu dùng chung cho cả 6 vùng (ví dụ một PCA fit trong fold trên 6 vùng gộp lại, như `ImgEncoder` trong `pag_lib.py`), không chiếu riêng từng vùng.

### 3.2. Cạnh

| Loại | Nối | Số cạnh |
|---|---|---|
| E1 | Vùng kề dọc trong cùng một phổi: RU–RM, RM–RL, LU–LM, LM–LL | 4 |
| E2 | Vùng đối xứng trái–phải: RU–LU, RM–LM, RL–LL | 3 |
| E3 | Lâm sàng ↔ lâm sàng | 15 |
| E4 | Lâm sàng ↔ vùng | 36 |
| E5 | Bệnh nhân ảo ↔ mọi node | 13 |

- Tổng: **71 cạnh khi có phim**; **8 node / 22 cạnh khi không phim** (không có E1/E2/E4).
- **Cờ `noi_vung` (bật/tắt E1 + E2), mặc định BẬT.** Thầy dặn "không nối vùng" nhưng câu này còn hiểu được hai cách, nên làm cả hai qua cờ. Khi tắt: bỏ 7 cạnh E1/E2 → 64 cạnh khi có phim.
- Code có sẵn để tham khảo: `base_types()`, `E1_PAIRS`, `E2_PAIRS`, `CLIN_SLOTS` trong `pag_lib.py`. Lưu ý: bản 2 để khám thực thể ở khe 11 làm "dự phòng".

### 3.3. Giá trị thiếu
- Điền trong fold: số → trung vị + cờ thiếu + z-score; danh mục → one-hot (tần suất ≥ 20, tối đa 15 mức, NA là một mức).
- Dùng lại `ClinEncoder` / `encode_fold()` trong `pag_lib.py`.
- Node khám thực thể thiếu khoảng 88% mỗi biến (bảng bản 2) → ghi rõ trong QA.

### 3.4. Bảng "biến → node" — phải nộp thầy duyệt trước khi chốt
- Bản 2 có sẵn ở `results_v2/tuan2_du_lieu_do_thi/bien_vao_node.csv`.
- Cập nhật theo 6 nhóm lâm sàng ở trên (khám thực thể thành node chính thức, không còn là dự phòng).
- Mỗi biến ghi: node, số cột sau mã hoá, % thiếu.

### 3.5. Sản phẩm cần nộp
Ghi vào `PAG/results_v3/tuan2_do_thi_PG/` (hoặc tên tương tự):

1. **`patient_graphs_fold{k}.pt`**, một file cho mỗi fold vì mã hoá fit theo fold. Mỗi đồ thị gồm: đặc trưng node, danh sách cạnh + loại cạnh, mặt nạ node có mặt, nhãn, cờ has_img, chỉ số hàng. **Không lưu id.** File `.pt` có dữ liệu thật → để trong `cache_v3/`, không commit.
2. **Bảng QA**, chỉ số gộp:
   - Số đồ thị theo nhóm: dev có ảnh 1500 / không ảnh 5183. Test chỉ đếm.
   - Phân bố số node / số cạnh: phải đúng 14/71 và 8/22 khi cờ bật; 14/64 và 8/22 khi tắt.
   - % thiếu theo node.
   - Kiểm tra không có id / không có biến cấm trong đặc trưng.
   - Kiểm tra mã hoá fit chỉ trên phần train của fold.
3. **`run_config.json`**: dùng `run_config()` trong `v3_lib.py`. Nó tự ghi hash script + hash dữ liệu.
4. **Bảng "biến → node"** (mục 3.4).

### 3.6. Báo cáo khi có kết quả sau này
- Tách nhóm có ảnh / không ảnh.
- Tách Đồng Nai.
- Ghi cả ROC-AUC gộp và AUC trong lô.

---

## 4. Chưa làm / ngoài phạm vi
- **Chưa huấn luyện PG.** Chỉ làm khi thầy duyệt bảng biến → node và đồ thị.
- Không đưa CheXFound vào các báo cáo này.
- Mọi việc ngoài lời nhắn mới nhất của thầy: hỏi trước khi làm.
