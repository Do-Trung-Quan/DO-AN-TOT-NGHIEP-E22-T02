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

### 4.0 Xác định danh tính bệnh nhân & chia dữ liệu theo nhóm

**Vấn đề.** 8.030 dòng dữ liệu là 8.030 **lượt khám**, không phải 8.030 **người**. Chia ngẫu nhiên theo dòng khiến cùng một người có lượt khám ở cả tập huấn luyện lẫn tập kiểm thử — mô hình chỉ cần "nhớ mặt" là trả lời đúng.

**Luật xác định "cùng một người"** (cài trong [`scripts/patient_identity.py`](../scripts/patient_identity.py)):

1. Trùng **số điện thoại** → cùng một người.
2. Nếu ít nhất một bên **thiếu** số điện thoại → phải trùng **cả 6 trường**: họ tên, năm sinh, tỉnh, giới tính, cấp độ nghề `cviec`, cấp độ nghề `pxuong`.
3. Còn lại → hai người khác nhau.

Cấp độ nghề dùng đúng hàm `classify_job_10_levels()` nên `"Van hanh may"` và `"van hanh may moc"` quy về cùng một cấp. Xung đột giữa luật 1 và luật 2 (dạng `p1(sđt A) ~ m(thiếu sđt) ~ p2(sđt B)`) được cắt lại theo số điện thoại; trên bộ dữ liệu này **không xảy ra trường hợp nào**.

```
Lượt khám                    : 8.030
Có số điện thoại hợp lệ      : 5.724 (71,3%)
Luật 1 (trùng sđt)           : gộp 734 cặp
Luật 2 (thiếu sđt + 6 trường): gộp 272 cặp
=> SỐ NGƯỜI                  : 7.024
=> Người khám nhiều hơn 1 lần: 1.002  (chiếm 2.008 dòng = 25,0% dữ liệu)
```

**Hai cột nhãn nhóm, hai việc khác nhau** — cả hai đều là *nhãn nhóm*, nhiều `id` khác nhau có thể mang cùng một giá trị; chỉ `id` mới duy nhất trên từng dòng:

| Cột | Công thức | Dùng để |
|---|---|---|
| `patient_uid` | 3 luật trên | **Chia dữ liệu** (dev/test và 5 fold) |
| `patient_group` | `hash(họ tên + năm sinh)` | Chỉ để **đối chiếu chéo** với `image_index.parquet` của nhánh ảnh (assert khớp 1835/1835) |

**Kiểm toán rò rỉ — đo bằng `patient_uid`:**

| Cách chia | Người ở cả 2 tập | Node test bị ảnh hưởng | Ca bệnh bị ảnh hưởng |
|---|---|---|---|
| `train_test_split` ngẫu nhiên | 241 | **241/1.205 (20,0%)** | **5/32 (15,6%)** |
| Nhóm theo `patient_group` | 39 | 39/1.148 (3,4%) | 0/31 (0,0%) |
| **Nhóm theo `patient_uid`** ← đang dùng | **0** | **0/1.148 (0,0%)** | **0/31 (0,0%)** |

Cách nhóm bằng `họ tên + năm sinh` tốt hơn chia ngẫu nhiên nhưng **vẫn bỏ sót** những trường hợp gõ nhầm tên. Ví dụ thật trong dữ liệu:

```
id  734  Ngo Thi Thuy   1984  Nu  Hai Duong  0347247670   → rơi vào tập huấn luyện
id 3693  ngo thi thuye  1984  Nu  Hai Duong  0347247670   → rơi vào tập kiểm thử
```

Cùng số điện thoại, cùng năm sinh, cùng tỉnh, cùng giới tính — một người, lệch đúng một chữ cái ở tên. Luật số điện thoại nhận ra ngay.

**Cách chia hiện tại:**
* **Dev/Test:** `StratifiedGroupKFold(n_splits=7, groups=patient_uid)`, lấy 1 fold làm test → **Dev 6.882 (85,7%) / Test 1.148 (14,3%)**, ca bệnh **180 / 31**.
* **5 fold trong Dev:** `StratifiedGroupKFold(n_splits=5, groups=patient_uid)`; `fold_id` lưu ra file để phase fusion dùng lại nguyên vẹn.
* **Assert cứng** trong notebook: 0 người bắc cầu dev/test và giữa 5 fold.

### 4.1 Kết quả Huấn luyện 5-Fold Cross Validation

Mỗi Fold huấn luyện với **Class Weights**, `Adam(lr=0.001)`, `ReduceLROnPlateau(patience=6, factor=0.5)` + `EarlyStopping(monitor="val_pr_auc", patience=12)`.

| Fold | Best Epoch | Val ROC-AUC | Val PR-AUC |
|---|---|---|---|
| **Fold 1** | Epoch 6 | 0.9344 | 0.6270 |
| **Fold 2** | Epoch 10 | 0.9315 | 0.4107 |
| **Fold 3** | Epoch 5 | 0.9341 | 0.4460 |
| **Fold 4** | Epoch 18 | 0.9420 | 0.3742 |
| **Fold 5** | Epoch 27 | 0.9176 | 0.3745 |
| **Trung bình 5 Folds** | **-** | **0.9319** | **0.4465** |
| *(bản cũ — chia ngẫu nhiên)* | *-* | *0.9354* | *0.4680* |

### 4.2 Tối ưu hóa Ngưỡng quyết định (Out-of-Fold F2-Score Threshold Sweep)

Gộp toàn bộ dự báo OOF của 6.882 mẫu để quét ngưỡng tối đa hóa F2-Score:
* **OOF ROC-AUC:** `0.9141`  |  **OOF PR-AUC:** `0.3957`
* **Ngưỡng tối ưu:** **`0.75`** (F2 = 0.5243) — bản cũ là `0.77`

### 4.3 Kết quả Ensemble trên Tập Test Hold-out (1.148 mẫu, 31 ca bệnh)

Ensemble Average của 5 mô hình fold, tại ngưỡng `0.75`:

* **Ensemble Test ROC-AUC:** **`0.9556`**
* **Ensemble Test PR-AUC:** **`0.5133`**
* **Test Accuracy:** **`96.78%`**

#### Confusion Matrix Tập Test (Rows: Actual, Cols: Predicted):
```
             Dự đoán Khỏe (0)    Dự đoán Bệnh (1)
Thực tế Khỏe (0)        1093                24
Thực tế Bệnh (1)          13                18
```

#### Detailed Metrics:
* **Precision (Bệnh):** **`42.86%`** (18 / 42)
* **Recall (Bệnh):** **`58.06%`** (18 / 31 ca bệnh được phát hiện chính xác)
* **F1-Score (Bệnh):** **`49.32%`**

### 4.4 Đọc kết quả cho đúng ⚠️

So với bản cũ (chia ngẫu nhiên), **OOF giảm** — PR-AUC trung bình 5 fold `0.4680 → 0.4465`, OOF gộp `0.3957` — đúng như dự đoán khi gỡ rò rỉ: chia theo nhóm bệnh nhân là bài toán khó hơn.

Chỉ số trên **tập test lại tăng** (PR-AUC `0.4198 → 0.5133`, Recall `46,88% → 58,06%`). **Không được diễn giải đây là mô hình tốt lên.** Lý do:

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
3. **`output/fusion_node_meta.parquet`** — `(8030, 13)`: `row_id, id, has_image, bnn, split, fold_id, patient_uid, patient_group, tuoi, gioitinh, cviec, pxuong, tuoinghe`
   * `split` / `fold_id` **phải được phase fusion dùng lại nguyên vẹn** — tự chia lại là vô hiệu hóa toàn bộ cơ chế chống rò rỉ.
   * Phân bố: có ảnh 1.835 BN (99 ca bệnh, 5,40%) · không ảnh 6.195 BN (112 ca bệnh, 1,81%).
   * Tập test có 286 lượt khám kèm ảnh X-quang.
4. **`output/patient_identity.parquet`** — `(8030, 2)`: `id`, `patient_uid`
   * **Dành cho nhánh ảnh dùng chung** để hai nhánh có cùng một định nghĩa "ai là ai".
   * Không chứa họ tên hay số điện thoại → commit lên git được.

---

## 6. Sẵn sàng cho GNN Multimodal Fusion Phase

Nhánh Timeseries hiện tại đã **hoàn thành 100%** và sẵn sàng kết nối với nhánh ảnh:
* Tập dữ liệu đồ thị có **1,835 nút (nodes) đa phương thức** hoàn chỉnh (chứa cả vector đặc trưng ảnh từ BioViL-T + vector 32D lâm sàng từ Timeseries).
* Ma trận đặc trưng 32D đã được ổn định hóa bằng ensemble 5-fold, loại bỏ hoàn toàn các nguy cơ rò rỉ dữ liệu và đáp ứng tốt tiêu chuẩn đầu vào cho các thuật toán đồ thị như GCN/GAT/DGCNN.
