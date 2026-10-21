# Hướng dẫn chạy P0.1 (bản 3) — xử lý ảnh, QA, 4 chỉ số

Mục tiêu theo đúng lời thầy: **"làm xử lý ảnh + QA và chạy lại P0.1 rồi gửi thầy xem"**.
Khi xong, file gửi thầy là `phase0/output/p01_bao_cao_ban3.md`, kèm tỉ lệ QA và bảng 4 chỉ số.

Tổng thời gian: khoảng 2–3 giờ, phần lớn là máy tự chạy. Phần bạn phải làm tay là chấm khoảng 100–150 ảnh, mất 20–30 phút.

| Bước | Làm ở đâu | Thời gian | Kết quả |
|---|---|---|---|
| 0. Chuẩn bị | máy bạn | 10 phút | mã trên GitHub, ảnh + trọng số trên Drive |
| 1. Thử 30 ảnh | Colab | 3 phút | lưới 30 ảnh để xem bằng mắt |
| 2. Xử lý 1.835 ảnh | Colab GPU | 1–1,5 giờ | `p01_processed.npz`, `p01_mask_qa.csv` |
| 3. QA trực quan | Colab + bạn chấm | 30 phút | `p01_qa_cham_tay.csv`, `p01_qa_tom_tat.json`, `p01_zone_missing.csv` |
| 4. Trích embedding | Colab GPU | 10 phút | `image_features_{frozen,control}_{V0,V1,V2}.parquet` |
| 5. Bốn chỉ số | Colab CPU | 20–30 phút | `p01_{A_goc,V1,V2}_{control,frozen}.json/md` |
| 6. Báo cáo | Colab → máy bạn | 5 phút | `p01_bao_cao_ban3.md` |

Các tên viết tắt:
- **V1**: ảnh đã cắt theo hai phổi và đã che chữ.
- **V2**: V1, đồng thời xóa mọi pixel ngoài phổi.
- **V0**: toàn bộ ảnh đã che chữ. Chỉ dùng làm vector dự phòng cho ảnh bị chấm "hỏng".
- **A_goc**: đặc trưng cũ (ảnh chưa che chữ, cắt kiểu CenterCrop). Chạy lại để so sánh.

---

## Bước 0. Chuẩn bị (máy bạn)

1. **Đẩy mã lên GitHub.** Colab lấy mã bằng cách clone nhánh `PAG`.
   ```
   git add phase0 colab/phase0_p01.ipynb fusion/baselines.py fusion/build_graph.py fusion/gnn.py
   git commit -m "Phase 0 ban 3: xu ly anh + QA + P0.1"
   git push origin PAG
   ```
   Kiểm tra: trên GitHub, nhánh `PAG` có file `phase0/p01_qa_sample.py`.
   Phải commit cả `fusion/baselines.py`, vì phase0 dùng hàm `sk_fold` mới nằm trong đó. Nếu thiếu, Colab báo `ImportError`.

2. **Đóng gói ảnh gốc và trọng số** (chạy ở thư mục gốc repo):
   ```
   python -m phase0.make_colab_bundle --out-dir D:/colab_bundle
   ```
   Kết quả: `D:/colab_bundle/p01_images.zip` (1.835 ảnh gốc, khoảng vài GB) và thư mục `D:/colab_bundle/p01_weights/` (4 file).
   Nếu script báo "con trỏ Git LFS" nghĩa là file trọng số chưa được tải thật: chạy `git lfs pull` rồi chạy lại.

3. **Upload lên Google Drive cá nhân**: `p01_images.zip` đặt vào `MyDrive/`, thư mục `p01_weights/` đặt vào `MyDrive/p01_weights/`.
   **Không chia sẻ** các file này, vì tên file ảnh chứa họ tên bệnh nhân.

## Bước 1. Mở notebook, thử 30 ảnh

1. Mở `colab/phase0_p01.ipynb` trên Colab (File → Open notebook → GitHub → nhánh PAG).
2. Runtime → Change runtime type → **T4 GPU**.
3. Chạy lần lượt 2 cell đầu: cell setup và cell chép ảnh.
   - Phải thấy `so anh (ky vong 1835): 1835`.
   - Phải thấy tên GPU là `Tesla T4`.
4. Chạy cell `--limit 30`, rồi chạy cell vẽ lưới.

**Đọc lưới thế nào**: mỗi ảnh có 2 ô.
- **Ô trái là V1**, có viền đỏ là mask phổi. Kiểm tra:
  - hai phổi nằm trọn trong khung, đỉnh phổi và góc sườn hoành không bị cắt mất;
  - viền đỏ ôm sát phổi;
  - không còn chữ hay ngày tháng.
- **Ô phải là V2**: chỉ còn hai lá phổi, xung quanh đen hết.

Nếu thấy trên 3 ảnh trong 30 ảnh bị cắt mất phổi, hoặc viền đỏ lệch hẳn: **dừng lại**, chụp lại mô tả lỗi (không chụp ảnh bệnh nhân) để sửa script.

## Bước 2. Xử lý toàn bộ 1.835 ảnh

Chạy cell `p01_preprocess ... --out-dir {PROC}`. Kết quả ghi thẳng vào `MyDrive/p01_processed/`.
Nếu Colab bị ngắt: chạy lại 2 cell đầu rồi chạy lại cell này. Nó sẽ bỏ qua các shard (mỗi shard 100 ảnh) đã xong.

Cuối cell in **bảng QA tự động**, ví dụ:
```
  co_dao_am                   :     0  (0.0%)
  co_khong_du_2_phoi          :    12  (0.7%)
  ...
  co_bat_ky                   :    85  (4.6%)
```
Ý nghĩa từng cờ (ảnh bị gắn cờ sẽ được đưa vào danh sách chấm tay):

| Cờ | Ý nghĩa |
|---|---|
| `dao_am` | Ảnh bị lưu ngược màu (xương đen, phổi trắng). Đã tự lật lại. |
| `khong_du_2_phoi` | Không tìm được đủ 2 lá phổi. |
| `ty_le_trai_phai` | Lấy diện tích phổi bên trái ảnh chia cho bên phải, kết quả nằm ngoài [0,6; 1,6], nghĩa là một bên to bất thường. |
| `cham_mep` | Mask phổi chạm biên ảnh: phim có thể bị cắt thiếu, hoặc mask tràn ra ngoài. |
| `con_chu` | Sau khi tô đen vẫn phát hiện ra chữ. |
| `dien_tich_bat_thuong` | Diện tích phổi quá nhỏ hoặc quá lớn so với các ảnh khác (ngoài trung vị ± 3 MAD; MAD là độ lệch tuyệt đối trung vị, một thước đo độ phân tán). |

→ **Chép bảng này vào báo cáo** (file báo cáo ở bước 6 cũng tự chép lại).

### Xem ảnh đã xử lý ở đâu
- Toàn bộ ảnh đã xử lý nằm trong **`MyDrive/p01_processed/p01_processed.npz`**. Đây là file mảng số của numpy (mỗi ảnh là một mảng 448×448), nên **không mở được bằng trình xem ảnh**.
- Để xem, dùng cell **"Xem ảnh đã xử lý"** ngay sau bước 2: điền `XEM = [img_id, ...]`, nó hiện 3 ô V0 | V1 + viền mask | V2, kèm các cờ của ảnh.
- `img_id` lấy trong `p01_mask_qa.csv` (mỗi dòng một ảnh, chỉ có số, không có tên).
- Ảnh này vẫn là phim X-quang thật: **chỉ xem trên Colab, không tải về máy rồi gửi đi, không đưa vào báo cáo**. Trước khi lưu notebook: Edit → Clear all outputs.

## Bước 3. QA trực quan (bạn chấm)

1. Chạy cell `p01_qa_sample`. Nó in ra, ví dụ, `34 tang (lo x nhan) | 50 anh ngau nhien | 85 anh bi co | tong can cham 130`.
   - **Tầng** là một tổ hợp (lô chụp, dương/âm). Mỗi tầng có ít nhất 1 ảnh trong mẫu, để mẫu rải đều.
2. Chạy cell `TRANG = 0` để vẽ trang 1 (40 ảnh, đánh số `#stt`). Đổi `TRANG = 1, 2, …` để xem tiếp.
3. Chấm mỗi ảnh một mức:
   - `dat`: hai phổi trọn vẹn, mask ôm đúng, không còn chữ;
   - `mot_phan`: dùng được nhưng có lỗi nhỏ, ví dụ cắt hơi rộng hoặc mask hụt một góc nhỏ;
   - `hong`: mất một phần phổi, mask sai hẳn, hoặc không đọc được.
4. Ghi điểm, chọn một trong hai cách:
   - **Cách nhanh**: trong cell `KHONG_DAT = {...}`, chỉ ghi những ảnh không đạt, ví dụ `7: 'mot_phan', 23: 'hong'`. Ảnh nào không ghi được tính là `dat`.
   - Hoặc: mở `MyDrive/p01_processed/p01_qa_cham_tay.csv` bằng Google Sheets, điền cột `diem`, tải về dạng CSV và ghi đè đúng tên cũ. Sau đó đặt `DUNG_CELL_NAY = False`.
5. Chạy cell `p01_qa_summary`. Dòng cuối phải là:
   ```
   => CONG QA: DAT, duoc trich embedding va chay P0.1
   ```
   - **Cổng** QA: trong 50 ảnh ngẫu nhiên, ít nhất 45 ảnh (90%) phải là `dat`. `mot_phan` **không** được tính là đạt.
   - Nếu `KHONG DAT`: dừng lại. Ghi lại các ảnh hỏng thuộc kiểu lỗi nào để sửa `p01_preprocess.py`, rồi chạy lại từ bước 2 (xóa thư mục `MyDrive/p01_processed/` trước).
   - Ảnh chấm `hong` được ghi vào `phase0/output/p01_zone_missing.csv`. Các ảnh này **không bị loại**: chúng dùng vector toàn ảnh (V0) và được gắn cờ `zone_missing`.

→ **Số đưa vào báo cáo**: tỉ lệ đạt (ví dụ 47/50 = 94%), số ảnh bị cờ và kết quả chấm của chúng, số ảnh `zone_missing`.

## Bước 4. Trích embedding

Chạy cell vòng lặp `p01_extract_embeddings` (6 lần: 2 nguồn × 3 biến thể).
- **frozen**: BioViL-T giữ nguyên trọng số.
- **control**: bản đã tinh chỉnh, cùng mô hình với embedding cũ.

Mỗi lần in một dòng `da trich (1835, 512)`. Nếu dòng "chieu hang so" lớn hơn 0 thì chỉ là cảnh báo, không phải lỗi.

## Bước 5. Bốn chỉ số P0.1

Chạy cell vòng lặp `p01_batch_noise`. Kiểm chứng ngay ở lượt đầu (`A_goc`, `control`), phải thấy:
```
[kiem tra] L1 tai hien so voi phase3_oof_control.npz: max|diff| = 0.00e+00
L1 AUC gop = 0.760 [...]
  chinh: 0.719 [0.686; 0.751]
```
Đây là kiểm tra hồi quy: đặc trưng cũ phải cho lại đúng số cũ (0,719 với control, 0,716 với frozen). Nếu khác thì có gì đó sai, **dừng lại**.

Với mỗi lượt, bốn chỉ số đọc như sau:
1. **`[1] Probe lo`**: mô hình nhỏ đoán ảnh thuộc lô nào. So `accuracy` với `lop da so` (khoảng 0,11, tức là đoán bừa lô đông nhất). Ảnh cũ cho 0,43–0,47. Sau khi che chữ và cắt, con số này **nên giảm**: ảnh càng ít "dấu vân tay" của lô càng tốt.
2. **`[2] AUC trong lo`** (dòng `chinh`): AUC của L1 tính riêng trong từng lô rồi lấy trung bình. Đây là **chỉ số quyết định**.
3. **`[3] delta AUC gop`**: thêm ảnh vào mô hình chỉ biết lô thì AUC tăng bao nhiêu. Khoảng `[...]` phải nằm trọn trên 0.
4. **`[4] Leave-batch-out`**: bỏ hẳn từng lô khỏi huấn luyện rồi chấm trên lô đó. Báo cả AUC gộp (thầy trích 0,742 cho ảnh cũ) và AUC trong lô.

## Bước 6. Báo cáo

1. Chạy cell `p01_decision`. Nó in ra và ghi `phase0/output/p01_bao_cao_ban3.md`, gồm:
   - mục 1: xử lý ảnh và QA;
   - mục 2: bảng 4 chỉ số × {A_goc, V1, V2} × {control, frozen};
   - mục 3: V1/V2 so với đặc trưng cũ, ghép cặp trên cùng bệnh nhân;
   - mục 4: đối chiếu bảng quyết định của thầy.
2. Chạy cell cuối để chép kết quả về `MyDrive/phase0_results/`. Tải thư mục đó về và chép vào `phase0/output/` của repo.
3. **Clear all outputs** trước khi lưu notebook.

Bảng quyết định của thầy, đọc ở mục 4 của báo cáo:

| AUC trong lô của L1 | Ý nghĩa |
|---|---|
| ≥ 0,70 và KTC của Δ nằm trọn trên 0 | tiếp tục; AUC trong lô thành chỉ số chính |
| 0,60–0,70 | báo cáo thầy trước khi làm tiếp |
| < 0,60 | dừng nhánh ảnh |

Gửi thầy file `p01_bao_cao_ban3.md`, nói rõ biến thể nào (V1 hay V2) và nguồn nào đang đối chiếu với bảng. Trong báo cáo đã có sẵn một dòng ghi chú về P0.5 (17% lượt khám vắt fold → sẽ chia lại fold sau khi thầy duyệt P0.1).

## Tự kiểm chứng kết quả

- **Tái hiện số cũ**: `A_goc` phải ra 0,760 (AUC gộp L1) và 0,719 / 0,716 (AUC trong lô), giống `phase0/output/ban2/p01_tong_hop.md`.
- **Không chạm test**: mọi script chỉ dùng dev (in `dev co anh: 1571 dong, 393 duong`). Có lệnh `assert` tự dừng nếu có dòng test lọt vào.
- **Cùng bệnh nhân khi so sánh**: mục 3 của báo cáo so V1/V2 với A_goc trên đúng 1.571 dòng đó. Vì vậy Δ là so sánh ghép cặp.
- **Ảnh không bị loại âm thầm**: `zone_missing_dev_co_anh` trong mỗi file `p01_V1_*.json` bằng số ảnh hỏng thuộc dev. Tổng số dòng vẫn là 1.571.
- **Xem lại bất kỳ ảnh nào**: cell "Xem ảnh đã xử lý" với `img_id` lấy từ `p01_mask_qa.csv` hoặc `p01_zone_missing.csv`.
- **Cấu hình khai báo trước**: mọi file kết quả ghi `config_hash`. Hash phải giống nhau giữa các file. Nếu sửa `phase0/config.json` thì hash đổi và phải chạy lại.

## Các file

| File | Ở đâu | Có PII/ảnh? |
|---|---|---|
| `p01_processed.npz` (ảnh V0/V1/V2 + mask) | Drive riêng tư | **ảnh thật**: không đưa vào git/báo cáo |
| `p01_mask_qa.csv`, `p01_qa_cham_tay.csv` | Drive, `phase0_results/` | chỉ img_id + số |
| `image_features_*.parquet` | Drive | vector số |
| `p01_qa_tom_tat.json`, `p01_zone_missing.csv` | `phase0/output/` | chỉ số |
| `p01_*_{control,frozen}.json/md`, `p01_oof_*.npz` | `phase0/output/` | chỉ số |
| `p01_bao_cao_ban3.md` | `phase0/output/` | **file gửi thầy** |
| output bản 2 cũ | `phase0/output/ban2/` | lưu trữ để đối chiếu |
