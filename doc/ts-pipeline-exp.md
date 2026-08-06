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

Dữ liệu tổng số 8,030 bệnh nhân được tách làm 2 tập độc lập:
* **Dev Set (85% ~ 6,825 mẫu):** Dùng để huấn luyện và đánh giá Out-Of-Fold qua **Stratified 5-Fold Cross Validation**.
* **Hold-out Test Set (15% ~ 1,205 mẫu):** Tập kiểm thử tĩnh không tham gia vào bất kỳ quá trình chọn mô hình hay tìm ngưỡng nào.

### 4.1 Kết quả Huấn luyện 5-Fold Cross Validation
Trong mỗi Fold, mô hình được huấn luyện với **Class Weights** cân bằng loss, optimizer `Adam(learning_rate=0.001)`, và callback `ReduceLROnPlateau(patience=5, factor=0.5)` + `EarlyStopping(patience=12)`.

| Fold | Best Epoch | Val ROC-AUC | Val PR-AUC |
|---|---|---|---|
| **Fold 1** | Epoch 12 | 0.9485 | 0.5308 |
| **Fold 2** | Epoch 14 | 0.9496 | 0.4913 |
| **Fold 3** | Epoch 7 | 0.9230 | 0.3947 |
| **Fold 4** | Epoch 16 | 0.9231 | 0.4724 |
| **Fold 5** | Epoch 13 | 0.9328 | 0.4510 |
| **Trung bình 5 Folds** | **-** | **0.9354** | **0.4680** |

### 4.2 Tối ưu hóa Ngưỡng quyết định (Out-of-Fold F2-Score Threshold Sweep)
Sau khi hoàn thành 5 Folds, toàn bộ xác suất dự báo OOF của 6,825 mẫu được tập hợp lại để quét ngưỡng tìm F2-Score cao nhất (ưu tiên Recall gấp 2 lần Precision):
* **Ngưỡng tối ưu được tìm thấy:** **`0.77`**

### 4.3 Kết quả Ensemble Đánh giá trên Tập Test Hold-out (1,205 mẫu)
Sử dụng phương pháp **Ensemble Average** (trung bình xác suất của 5 mô hình fold) đánh giá trên 1,205 mẫu Test độc lập tại ngưỡng `0.77`:

* **Ensemble Test ROC-AUC:** **`0.9255`**
* **Ensemble Test PR-AUC:** **`0.4198`**
* **Test Accuracy:** **`95.77%`**

#### Confusion Matrix Tập Test (Rows: Actual, Cols: Predicted):
```
             Dự đoán Khỏe (0)    Dự đoán Bệnh (1)
Thực tế Khỏe (0)        1139                34
Thực tế Bệnh (1)          17                15
```

#### Detailed Metrics:
* **Precision (Bệnh):** **`30.61%`** (15 / 49)
* **Recall (Bệnh):** **`46.88%`** (15 / 32 ca bệnh được phát hiện chính xác)
* **F1-Score (Bệnh):** **`37.04%`**

---

## 5. Trích xuất & Định dạng Vector Đặc trưng (Feature Extraction Specification)

Đặc trưng nén lâm sàng 32-D được trích xuất từ 5 mô hình Fold cho toàn bộ 8,030 bệnh nhân bằng cách nạp trọng số từng fold, tạo sub-model tới lớp `fusion_dense`, dự báo đặc trưng và lấy trung bình cộng.

### File Đầu Ra:
1. **`output/timeseries_features.parquet`**:
   * **Shape:** `(8030, 33)` (Cột 0: `file_name`, Cột 1..32: `ts_feat_0` ... `ts_feat_31`).
   * **Số lượng mẫu có file_name hợp lệ (.jpg):** **1,835 bệnh nhân** (với 99 ca mắc bệnh `bnn=1`).
   * **Số lượng mẫu không có ảnh:** 6,195 bệnh nhân (giá trị `file_name = "khong"`).
2. **`output/timeseries_features.npy`**:
   * **Shape:** `(8030, 32)` (Ma trận float32 thuần túy).

---

## 6. Sẵn sàng cho GNN Multimodal Fusion Phase

Nhánh Timeseries hiện tại đã **hoàn thành 100%** và sẵn sàng kết nối với nhánh ảnh:
* Tập dữ liệu đồ thị có **1,835 nút (nodes) đa phương thức** hoàn chỉnh (chứa cả vector đặc trưng ảnh từ BioViL-T + vector 32D lâm sàng từ Timeseries).
* Ma trận đặc trưng 32D đã được ổn định hóa bằng ensemble 5-fold, loại bỏ hoàn toàn các nguy cơ rò rỉ dữ liệu và đáp ứng tốt tiêu chuẩn đầu vào cho các thuật toán đồ thị như GCN/GAT/DGCNN.
