# Chi tiết Thực nghiệm & Giải thích Kiến trúc Pipeline Timeseries (`Timeseries`)

> Tài liệu này mô tả chi tiết **quy trình thực nghiệm, lý do lựa chọn giải pháp kĩ thuật, và các kết quả đạt được** trên nhánh lâm sàng (`Timeseries`). Tài liệu được cập nhật dựa trên phiên bản mã nguồn mới nhất của notebook [`ĐỒ_ÁN.ipynb`](../ĐỒ_ÁN.ipynb).

---

## 1. Tổng quan Kiến trúc Pipeline & Quá trình Phát triển

Mục tiêu chính của nhánh Timeseries là chuyển đổi phiếu khám lâm sàng thô (8,030 bệnh nhân) thành **1 vector đặc trưng nén 32 chiều (32-D)** cho mỗi bệnh nhân, chuẩn bị sẵn sàng cho bước **GNN Multimodal Fusion** với nhánh ảnh X-quang.

### Các nâng cấp kĩ thuật cốt lõi trong phiên bản hiện tại:
1. **Hợp nhất dữ liệu mới (Supplemental Dataset Integration):** Nạp `data/Main_data_Merged_1835_Images.xlsx` được gộp từ bộ 8,030 bệnh nhân ban đầu và `new data.csv`. Tăng số lượng mẫu có ảnh X-quang khớp chuẩn xác 1-1 từ **433 mẫu (29 ca bệnh)** lên **1,835 mẫu (99 ca bệnh)**.
2. **Loại bỏ Data Leakage:** Di chuyển toàn bộ các bước `SimpleImputer` và `StandardScaler` vào bên trong từng Fold của Stratified K-Fold. Đảm bảo dữ liệu tập Validation và Test hoàn toàn độc lập với quá trình tính mean/std.
3. **Chiến lược Huấn luyện Stratified 5-Fold Cross Validation:** Thay thế việc chia cứng static train/val bằng quy trình 5-Fold Ensemble. Mỗi Fold huấn luyện một mô hình độc lập với khởi tạo trọng số ngẫu nhiên mới và được lưu trữ lại.
4. **Cân bằng lớp trực tiếp (Class Weights) & ReduceLROnPlateau:** Áp dụng Class Weights tự động điều chỉnh loss function theo tỷ lệ mất cân bằng nghiêm trọng (1:37), phối hợp cùng callback tự động giảm Learning Rate khi Loss bão hòa.
5. **Tối ưu hóa ngưỡng chẩn đoán tự động bằng F2-Score:** Quét ngưỡng quyết định (Threshold Sweep) từ 0.05 đến 0.95 trên toàn bộ dự báo Out-Of-Fold (OOF). Chọn ngưỡng tối ưu F2-score (`0.77`) ưu tiên nâng cao độ nhạy lâm sàng (Recall).
6. **Vector Trích xuất Ensemble 32D:** Vector đặc trưng nén tại lớp `fusion_dense` của 8,030 bệnh nhân được tạo bằng cách tính **trung bình cộng (Average Pooling)** qua 5 mô hình fold, xuất ra file `output/timeseries_features.parquet` và `output/timeseries_features.npy`.

---

## 2. Tiền xử lý & Mã hóa Dữ liệu (Feature Engineering & Preprocessing)

### 2.1 Loại bỏ Cột Định danh & Rò rỉ Nhãn
* **Cột định danh:** `id`, `tinh`, `hoten`, `sdt`, `sobh`, `bnncuthe`, `nam`.
* **Cột rò rỉ nhánh ảnh X-quang:** `chatluongphim`, `ketqua`, `matdotonthuong`, `tonthuongkhac`, `kichthuoctt`, và các cột mã định dạng đọc phim (`B...`, `C...`, `D...`, `E...`, `F1-F3`, `F6-F13`).
* **Quản lý tên file ảnh:** Tách cột `file_name` chứa tên file ảnh X-quang `.jpg` (1,835 mẫu có ảnh; 6,195 mẫu mang giá trị `"khong"`).

### 2.2 Xử lý Dữ liệu Thiếu (Imputation) & Outliers
* Sửa lỗi nhập liệu số: Giá trị bất thường `F4gang = 6.0` được quy đổi về mode.
* Điền thiếu chuỗi: Chuỗi thiếu điền `"không"`.
* Điền thiếu số: Thực hiện trong từng Fold bằng `SimpleImputer(strategy='median')`.

### 2.3 Phân loại Nghề nghiệp 10 Cấp (Rule-based Job Encoding)
* 4 cột nghề nghiệp (`cviec`, `pxuong`, `cviec1`, `cviec2`) được chuyển đổi thành mã ID nguyên từ 0 đến 9 dựa trên phân cấp mức độ tiếp xúc bụi độc hại (silica, mỏ đá, đúc kim loại, may mặc, hành chính,...).
* Nhóm 4 ID nghề nghiệp này được đưa qua lớp **Embedding(input_dim=10, output_dim=4)** và làm phẳng bằng **Flatten()** để bảo toàn đặc trưng của từng vị trí công việc.

---

## 3. Kiến trúc Mạng Đa Nhánh (Multi-Input Keras Network)

Mô hình bao gồm 3 nhánh đầu vào song song hợp nhất tại lớp Dense nén 32-D:

```
Nhánh Job (4 ID nghề) ──────> Embedding(10, 4) ──> Flatten(16) ──┐
                                                                 │
Nhánh Sequential (6 vùng) ──> Masking ──> Bidirectional LSTM ───┼─> Concatenate(48) ─> Dense(32) ─> Dropout(0.3) ─> Output(1)
                                                                 │   [fusion_dense]
Nhánh Numerical ───────────> Dense(32, Swish) ──> Dense(16) ────┘
```

1. **Job Branch:** `Input(4)` ➔ `Embedding(10, 4)` ➔ `BatchNormalization()` ➔ `Flatten()` (16 chiều).
2. **Sequential Branch (LSTM 3D):** Tensor `[N, 6 vùng phổi, k đặc trưng]` ➔ `Masking(0.0)` ➔ `Bidirectional(LSTM(32, return_sequences=False))` ➔ `BatchNormalization()` ➔ `Dense(16, Swish)` (16 chiều).
3. **Numerical Branch:** `Input(M)` ➔ `Dense(32, Swish)` ➔ `BatchNormalization()` ➔ `Dropout(0.2)` ➔ `Dense(16, Swish)` (16 chiều).
4. **Fusion Dense Layer:** Hợp nhất 48 chiều ➔ `Dense(32, Swish)` (được đặt tên là `fusion_dense`). Đây chính là lớp trích xuất vector đặc trưng cho nhánh Fusion.
5. **Output Layer:** `BatchNormalization()` ➔ `Dropout(0.3)` ➔ `Dense(1, Sigmoid)`.

---

## 4. Quy trình Thực nghiệm & Đánh giá Model

### 4.0 Chia dữ liệu theo NHÓM BỆNH NHÂN (sửa lỗi rò rỉ — 09/2026)

Bản trước dùng `train_test_split` ngẫu nhiên. Kiểm toán ở bước 0.2 phát hiện đây là **rò rỉ nghiêm trọng**: bộ dữ liệu 8.030 lượt khám chỉ thuộc về **6.266 bệnh nhân** — 1.694 người khám nhiều lần. Chia ngẫu nhiên khiến cùng một người xuất hiện ở cả tập huấn luyện lẫn tập kiểm thử:

| | Cách chia CŨ (ngẫu nhiên) | Cách chia MỚI (theo nhóm) |
|---|---|---|
| Bệnh nhân nằm ở cả 2 tập | **440** | **0** |
| Node test bị ảnh hưởng | **444/1.205 (36,8%)** | **0/1.148 (0,0%)** |
| Ca bệnh trong test bị ảnh hưởng | **14/32 (43,8%)** | **0/31 (0,0%)** |

Gần một nửa số ca bệnh của tập test cũ là người mà mô hình đã thấy một lần khám khác của chính họ khi huấn luyện. Mọi chỉ số bản cũ vì vậy **lạc quan hơn thực tế**.

Cách chia hiện tại — đồng chuẩn với nhánh ảnh (`imagefeat/crossfit_finetune.py`):
* `patient_group = sha256(normalize_name(hoten) + "|" + namsinh)[:16]`, dùng **đúng công thức** của `imagefeat/build_index.py`; đã kiểm chứng khớp **1835/1835** với `image_index.parquet`.
* **Dev/Test:** `StratifiedGroupKFold(n_splits=7, groups=patient_group)`, lấy 1 fold làm test → **Dev 6.882 (85,7%) / Test 1.148 (14,3%)**, ca bệnh 180 / 31.
* **5 fold trong Dev:** `StratifiedGroupKFold(n_splits=5, groups=patient_group)`, `fold_id` được lưu ra file để phase fusion dùng lại nguyên vẹn.
* **Assert cứng** trong notebook: 0 bệnh nhân bắc cầu dev/test và giữa 5 fold.

### 4.1 Kết quả Huấn luyện 5-Fold Cross Validation
Mỗi Fold huấn luyện với **Class Weights**, `Adam(lr=0.001)`, `ReduceLROnPlateau(patience=6, factor=0.5)` + `EarlyStopping(monitor="val_pr_auc", patience=12)`.

| Fold | Best Epoch | Val ROC-AUC | Val PR-AUC |
|---|---|---|---|
| **Fold 1** | Epoch 5 | 0.9542 | 0.5054 |
| **Fold 2** | Epoch 13 | 0.9124 | 0.4896 |
| **Fold 3** | Epoch 6 | 0.8999 | 0.3324 |
| **Fold 4** | Epoch 28 | 0.8773 | 0.4543 |
| **Fold 5** | Epoch 18 | 0.9325 | 0.4305 |
| **Trung bình 5 Folds** | **-** | **0.9153** | **0.4424** |
| *(bản cũ, split ngẫu nhiên)* | *-* | *0.9354* | *0.4680* |

### 4.2 Tối ưu hóa Ngưỡng quyết định (Out-of-Fold F2-Score Threshold Sweep)
Gộp toàn bộ dự báo OOF của 6.882 mẫu để quét ngưỡng tối đa hóa F2-Score:
* **OOF ROC-AUC:** `0.8851`  |  **OOF PR-AUC:** `0.4033`
* **Ngưỡng tối ưu:** **`0.80`** (F2 = 0.5483) — bản cũ là `0.77`

### 4.3 Kết quả Ensemble trên Tập Test Hold-out (1.148 mẫu, 31 ca bệnh)
Ensemble Average của 5 mô hình fold, tại ngưỡng `0.80`:

* **Ensemble Test ROC-AUC:** **`0.9585`**
* **Ensemble Test PR-AUC:** **`0.5618`**
* **Test Accuracy:** **`97.39%`**

#### Confusion Matrix Tập Test (Rows: Actual, Cols: Predicted):
```
             Dự đoán Khỏe (0)    Dự đoán Bệnh (1)
Thực tế Khỏe (0)        1102                15
Thực tế Bệnh (1)          15                16
```

#### Detailed Metrics:
* **Precision (Bệnh):** **`51.61%`** (16 / 31)
* **Recall (Bệnh):** **`51.61%`** (16 / 31 ca bệnh được phát hiện chính xác)
* **F1-Score (Bệnh):** **`51.61%`**

### 4.4 Đọc kết quả cho đúng ⚠️

So với bản cũ, **OOF giảm** (PR-AUC trung bình 5 fold `0.4680 → 0.4424`; OOF gộp `0.4033`) đúng như dự đoán khi gỡ rò rỉ — chia theo nhóm bệnh nhân là bài toán khó hơn.

Nhưng **chỉ số trên tập test lại tăng** (PR-AUC `0.4198 → 0.5618`, Recall `46,88% → 51,61%`). **Không được diễn giải đây là mô hình tốt lên.** Lý do:
* Tập test đã **khác hoàn toàn** (1.148 mẫu / 31 ca bệnh, thay vì 1.205 / 32) — hai con số không so sánh trực tiếp được.
* Với **31 ca dương**, sai số lấy mẫu rất lớn: lệch 1 ca đã làm Recall đổi ~3,2 điểm phần trăm.
* **OOF là ước lượng đáng tin hơn** vì tính trên 6.882 mẫu với 180 ca dương.

Kết luận trung thực: *các chỉ số hiện tại đáng tin cậy hơn bản cũ vì không còn rò rỉ; phần tăng trên tập test nằm trong nhiễu lấy mẫu và không phải bằng chứng cải thiện.* Khi báo cáo nên kèm khoảng tin cậy bootstrap.

---

## 5. Trích xuất & Định dạng Vector Đặc trưng (Feature Extraction Specification)

Đặc trưng nén lâm sàng 32-D được trích xuất từ 5 mô hình Fold cho toàn bộ 8.030 bệnh nhân: nạp trọng số từng fold, tạo sub-model tới lớp `fusion_dense`, dự báo rồi lấy trung bình cộng.

### Đổi khóa nối: `file_name` → `id` + `has_image`

Bản cũ dùng `file_name` làm khóa nối sang nhánh ảnh. Bản này bỏ hẳn vì hai lý do:
1. **PII** — tên file chứa họ tên bệnh nhân (`001_TO_VAN_DIEU_20181219.jpg`). Nhánh ảnh cũng đã chuyển sang `img_id` + `file_hash` vì lý do này.
2. **Tái lập được** — `id` khớp `img_id` của `image_index.parquet` **1835/1835**, còn tên file thì phải quét thư mục ảnh mới có.

### File Đầu Ra:
1. **`output/timeseries_features.parquet`** — `(8030, 34)`: `id`, `has_image`, `ts_feat_0..31`
2. **`output/timeseries_features.npy`** — `(8030, 32)` float32
3. **`output/fusion_node_meta.parquet`** — `(8030, 12)`: `row_id, id, has_image, bnn, split, fold_id, patient_group, tuoi, gioitinh, cviec, pxuong, tuoinghe`
   * `split` / `fold_id` **phải được phase fusion dùng lại nguyên vẹn** — tự chia lại là vô hiệu hóa toàn bộ cơ chế chống rò rỉ.
   * Phân bố: có ảnh 1.835 BN (99 ca bệnh, 5,40%) · không ảnh 6.195 BN (112 ca bệnh, 1,81%).

---

## 6. Sẵn sàng cho GNN Multimodal Fusion Phase

Nhánh Timeseries hiện tại đã **hoàn thành 100%** và sẵn sàng kết nối với nhánh ảnh:
* Tập dữ liệu đồ thị có **1,835 nút (nodes) đa phương thức** hoàn chỉnh (chứa cả vector đặc trưng ảnh từ BioViL-T + vector 32D lâm sàng từ Timeseries).
* Ma trận đặc trưng 32D đã được ổn định hóa bằng ensemble 5-fold, loại bỏ hoàn toàn các nguy cơ rò rỉ dữ liệu và đáp ứng tốt tiêu chuẩn đầu vào cho các thuật toán đồ thị như GCN/GAT/DGCNN.
