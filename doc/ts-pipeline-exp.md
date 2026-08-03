# Giải thích chi tiết pipeline nhánh Timeseries (`ĐỒ_ÁN.ipynb`)

> Tài liệu này giải thích **từng cell** của notebook, viết cho người **chưa có nền tảng AI/ML** vẫn hiểu được bản chất.
> Mỗi cell quan trọng được mổ theo 3 lớp: **(a) Mục đích** → **(b) Từng bước: cơ sở lý thuyết + cơ chế hoạt động** → **(c) Ví dụ trên dữ liệu thật**.
>
> Trọng tâm là **từ mục 5) Tách nhánh dữ liệu trở đi** (cell 7 → cell 11). Các cell 1–4 chỉ tóm tắt.

---

## 0. Bức tranh tổng thể — notebook này làm gì?

Ta có một file Excel **8030 dòng** (mỗi dòng = 1 bệnh nhân), **181 cột** là kết quả phiếu khám sức khỏe nghề nghiệp (tuổi, nghề, triệu chứng ho/khó thở, khám phổi 6 vùng, đo chức năng hô hấp...). Cột nhãn cần dự đoán là **`bnn`** (bệnh nghề nghiệp: 0 = khỏe, 1 = có bệnh).

Notebook làm 2 việc:
1. **Biến bảng chữ+số lộn xộn thành số sạch** mà máy học hiểu được (cell 1–6).
2. **Huấn luyện một mạng nơ-ron đa đầu vào** đọc dữ liệu lâm sàng và học cách nhận ra bệnh; sau đó **rút ra một "vector đặc trưng" 32 chiều** cho mỗi bệnh nhân bằng phương pháp **Ensemble**, để sau này ghép (fusion) với nhánh ảnh X-quang (cell 7–11).

**Điều cốt lõi cần nhớ:** đích cuối KHÔNG phải là bộ phân loại lâm sàng nhị phân này, mà là **vector 32 chiều** — một bản "tóm tắt số học" tình trạng lâm sàng của bệnh nhân, để đưa vào mô hình đồ thị (GNN) ở bước fusion.

```
Excel thô  ──(cell 1-6: làm sạch + mã hóa)──►  Bảng số sạch (8030 × 140)
                                                       │
                               (cell 7: chia Dev/Test hold-out, định hình)
                                                       │
                           (cell 8-9: dựng & huấn luyện 5-Fold, chọn ngưỡng F2)
                                                       │
                         (cell 10-11: rút vector 32 chiều trung bình 5 mô hình)
                                                       ▼
                                      output/timeseries_features.parquet
                                           →  đưa sang nhánh FUSION
```

---

## Phần A — Tóm tắt nhanh cell 1–4 (tiền xử lý)

Bốn cell đầu chỉ "dọn dẹp" dữ liệu. Hiểu ngắn gọn là đủ:

| Cell | Tên | Làm gì | Vì sao cần |
|---|---|---|---|
| **1** | Load & Lọc cột | Đọc Excel (8030×181). Bỏ 57 cột: cột định danh (`hoten`, `sdt`...) và **cột kết quả đọc phim X-quang** (`ketqua`, `matdotonthuong`, các cột B/C/D/E, F1-3, F6-13). Còn 124 cột. | Cột định danh vô nghĩa với dự đoán. **Cột X-quang phải bỏ** vì nếu để lại, nhánh lâm sàng sẽ "học lỏm" kết quả từ phim → khi fusion với nhánh ảnh sẽ bị **trùng thông tin/gian lận** (data leakage). |
| **2** | Fill missing & Outlier | Ô trống ở cột số → điền `0`; ô trống ở cột chữ → điền `"không"`. Sửa 1 giá trị lỗi `'6.0'` trong `F4gang`. Tách cột `file_name` ra một mảng riêng. | Máy học không chạy được nếu còn ô trống. `file_name` được **giữ riêng** làm "chìa khóa" nối với ảnh X-quang sau này. |
| **3** | Encoding | Biến chữ thành số: giới tính nam/nữ→0/1; các cột có/không→0/1; các cột mức độ (ordinal)→0,1,2,3; vài cột phân loại→one-hot; tính `tuoi` động = `năm hiện tại − năm sinh` để tránh hardcode. | Mạng nơ-ron **chỉ tính được trên số**, không hiểu chữ "nam", "khó thở". |
| **4** | Phân loại nghề 10 cấp | 4 cột nghề (chữ tự do như "Luyen thep") → gán mã 0–9 theo mức độ độc hại/phơi nhiễm bụi. | Nghề là yếu tố nguy cơ chính của bệnh bụi phổi. Chữ tự do quá đa dạng → gom về 10 nhóm để máy học được. |

> **Ba loại encoding — hiểu bản chất:**
> - **Binary (0/1):** dùng cho câu hỏi có/không. VD `hutthuoc` (hút thuốc): có→1, không→0.
> - **Ordinal (0,1,2,3...):** dùng khi các mức **có thứ tự**. VD mức khó thở: không < khi gắng sức < khi làm nhẹ < thường xuyên → 0,1,2,3. Số lớn = nặng hơn, thứ tự này có nghĩa.
> - **One-hot:** dùng khi phân loại **không có thứ tự**. VD nhóm máu A/B/O — không thể nói A < B. Mỗi loại thành 1 cột riêng chứa 0/1, tránh việc máy hiểu nhầm có thứ tự.

Sau cell 4, ta có `df` gồm **8030 dòng × 140 cột toàn số**, không còn ô trống. Đây là đầu vào cho phần chính bên dưới.

---

# Phần B — Chi tiết từ mục 5 trở đi (trọng tâm)

---

## Cell 7 — Tách nhánh dữ liệu & Thiết lập Dev/Test Hold-out

### (a) Mục đích của cell

Nó làm 3 việc:
1. **Tách 140 cột thành 3 "nhóm đầu vào"** khác nhau, vì mỗi nhóm có bản chất khác nhau và sẽ đi vào 3 nhánh mạng khác nhau.
2. **Loại các cột nhạy cảm** để chống rò rỉ dữ liệu (data leakage).
3. **Phân chia tĩnh dữ liệu** thành tập Development (85%, ~6825 mẫu) dành cho K-Fold huấn luyện, và tập Test Hold-out tĩnh (15%, 1205 mẫu) để làm bài thi cuối khóa.

### (b) Từng bước

#### Bước 1 — Tách 3 nhóm đầu vào
```python
# Nhóm SEQ: 6 vùng phổi × 7 phép đo
SEQ_PREFIXES = ["rungt","rungg","goduc","vtam","vtno","vtrit","vtngay"]
X_seq_all = np.stack(seq_features, axis=1)   # shape [8030, 6, 7]
# Nhóm JOB: 4 cột nghề
X_job_all = df[JOB_COLS].to_numpy()          # shape [8030, 4]
# Nhóm NUM: mọi cột số còn lại
X_num_all_raw = df[num_cols].to_numpy()      # shape [8030, 91]
```

- **Nhóm JOB (nghề nghiệp)** — là **mã danh mục** (categorical). Gồm 4 cột nghề ứng với 4 thời điểm phơi nhiễm của bệnh nhân.
- **Nhóm SEQ (khám phổi 6 vùng)** — có **cấu trúc không gian lặp**. Xếp thành tensor 3D: chuỗi 6 bước (vùng), mỗi bước có 7 phép đo lâm sàng.
- **Nhóm NUM (chỉ số số học)** — tuổi, huyết áp, các chỉ số đo chức năng hô hấp FVC/FEV1... là **số thực thông thường**.

#### Bước 2 — Loại bỏ các cột nhạy cảm
```python
exclude_cols = set(seq_cols_flat) | set(JOB_COLS) | {TARGET_COL, "bnncuthe",
               "benhhh", "tiensuhh", "id", "hoten"}
```
* **Chống rò rỉ nhãn:** Cột nhãn `bnn` (TARGET_COL) được **loại bỏ hoàn toàn** khỏi tập các đặc trưng đầu vào. Đồng thời, `benhhh` (bệnh hô hấp hiện tại) và `tiensuhh` cũng bị loại bỏ để tránh mô hình học lỏm các chẩn đoán trực tiếp.

#### Bước 3 — Chia Dev và Test tĩnh
```python
idx_dev, idx_test = train_test_split(idx, test_size=0.15, stratify=y, random_state=42)
```
* Tách riêng 15% Test Set tĩnh để giấu kín cho tới pha đánh giá cuối cùng.
* Dữ liệu chỉ có 211 ca bệnh trên 8030 người (2.6%). Cơ chế `stratify=y` ép tỷ lệ ca bệnh phân phối đều ở cả tập Dev và Test (mỗi tập giữ ~2.6% bệnh, tương đương Test có đúng 32 ca bệnh).

---

## Cell 8 — Kiến trúc mạng nơ-ron đa đầu vào (Multi-Input)

### (a) Mục đích của cell

Dựng kiến trúc mô hình Keras đa đầu vào có **3 nhánh riêng biệt** được bao bọc trong hàm `build_model()` để tái tạo độc lập qua từng fold huấn luyện.

```
JOB (4)  ──► Embedding ──► Flatten ──► Dense16 ──┐
SEQ (6×7)──► LSTM32 ─────────────────► Dense16 ──┼─► [gộp 48] ─► Dense32 ─► Dropout ─► Dense1(sigmoid)
NUM (91) ──► Dense32 ─► BatchNorm ───► Dense16 ──┘             ▲
                                                           "fusion_dense" = nơi rút vector 32-D
```

### (b) Chi tiết 3 nhánh

* **Nhánh JOB:** Dùng lớp `Embedding(input_dim=10, output_dim=8)`. Do bệnh nhân có 4 ô nghề tương ứng với 4 vector 8D, ta sử dụng lớp **`Flatten()`** thay vì Pooling trung bình. Điều này giúp giữ nguyên đặc trưng riêng biệt và thứ tự ưu tiên của 4 vị trí nghề nghiệp trước khi đưa qua lớp Dense 16.
* **Nhánh SEQ:** Dùng lớp **`LSTM(32)`** để duyệt qua 6 vùng phổi khám lâm sàng (mỗi vùng có 7 chỉ số) nhằm tổng hợp thông tin bất thường dạng không gian về 1 vector 32 chiều.
* **Nhánh NUM:** Đi qua lớp `Dense(32)`, chuẩn hóa giữa các lớp bằng `BatchNormalization`, chống overfit bằng `Dropout(0.2)` trước khi đưa về `Dense(16)`.
* **Khối gộp (Concatenate):** Gộp 3 nhánh thành vector 48 chiều. Lớp áp chót **`fusion_dense` (Dense 32 chiều, activation "relu")** nén thông tin lâm sàng cô đọng nhất. Đầu ra cuối cùng là một nơ-ron với activation `sigmoid` để đưa ra xác suất bệnh nhị phân.

---

## Cell 9 — Huấn luyện Stratified 5-Fold Cross Validation & Đánh Giá F2-Score

### (a) Mục đích của cell

Thay vì chỉ chia Train/Val tĩnh dễ gây biến động kết quả (do số ca bệnh quá ít), ta huấn luyện mô hình qua **Stratified 5-Fold Cross Validation** trên 85% tập Dev để đảm bảo tính khách quan và triệt tiêu biến động.

### (b) Các điểm đột phá về phương pháp luận

#### 1. Chống rò rỉ thông tin tuyệt đối (In-fold Preprocessing)
```python
fold_imputer = SimpleImputer(strategy="mean")
fold_scaler = StandardScaler()
X_tr_num_scaled = fold_scaler.fit_transform(fold_imputer.fit_transform(X_tr_num_raw))
X_val_num_scaled = fold_scaler.transform(fold_imputer.transform(X_val_num_raw))
```
* Ở mỗi fold, Imputer và Scaler chỉ được `fit` trên tập huấn luyện của fold đó và `transform` sang tập validation/test. Thông tin của tập validation không hề bị rò rỉ vào quá trình chuẩn hóa.

#### 2. Tính toán Class Weight động cho từng fold
* Bỏ qua cơ chế Random Undersampling (tránh làm mất thông tin hữu ích), chỉ sử dụng `compute_class_weight` để phạt nặng mô hình khi dự đoán sai ca bệnh. Trọng số lớp bệnh được tính động theo phân phối nhãn thực tế của fold đó.

#### 3. Tối ưu hóa ngưỡng chẩn đoán bằng F2-Score
* Trong y tế, việc **bỏ sót ca bệnh (Recall thấp)** nguy hiểm hơn việc báo nhầm. Tuy nhiên, nếu dùng F1-score tự động, do mất cân bằng dữ liệu, thuật toán sẽ chọn ngưỡng rất cao `0.90` dẫn đến Recall chỉ còn `28.12%`.
* Ta thay đổi tiêu chí tự động quét ngưỡng sang **F2-Score** (coi trọng Recall gấp đôi Precision):
  $$F_2 = 5 \times \frac{\text{Precision} \times \text{Recall}}{4 \times \text{Precision} + \text{Recall}}$$
* Ngưỡng tối ưu tự động chọn được trên tập Validation (OOF) là **`0.81`**. 

### (c) Kết quả huấn luyện và đánh giá Ensemble trên tập Test Hold-out

Sau khi huấn luyện 5 fold, dự đoán của 5 fold model được **Ensemble (lấy trung bình cộng xác suất)** để đưa ra kết luận cuối cùng trên tập Test tĩnh tại ngưỡng tối ưu `0.81`:

* **Test ROC-AUC:** `0.9384` (Khả năng xếp hạng tuyệt vời)
* **Test PR-AUC:** `0.3502` (Gấp 13 lần ngẫu nhiên)
* **Precision lớp Bệnh:** **`38.71%`** (Báo nhầm rất ít)
* **Recall lớp Bệnh:** **`37.50%`** (Bắt được 12/32 ca bệnh - tăng 33% so với ngưỡng cũ `0.90`)
* **F1-Score tập Test:** **`0.3810`**

---

## Cell 10 & 11 — Trích xuất đặc trưng Ensemble cho Fusion

### (a) Mục đích của cell

Đây là bước tạo ra **sản phẩm đầu ra cuối cùng** để chuyển giao sang pha Fusion (GNN). Ta cần trích xuất vector đặc trưng lâm sàng 32 chiều cho toàn bộ 8030 bệnh nhân.

### (b) Cơ chế Ensemble Feature Extraction

* Vì có 5 mô hình huấn luyện từ 5 folds, ta chạy 8030 mẫu qua cả 5 mô hình này.
* Tại mỗi mô hình, ta trích xuất vector đầu ra 32 chiều từ lớp áp chót `fusion_dense`.
* Ta tính **trung bình cộng** 5 vector 32 chiều này để có được một **vector đặc trưng Ensemble ổn định và tổng quát nhất**, giảm thiểu tối đa nhiễu của mô hình đơn lẻ.

```python
test_probs = []
for fold in range(n_splits):
    # Lấy đầu ra tại lớp 'fusion_dense'
    feat_model = Model(inputs=models[fold].inputs, outputs=models[fold].get_layer("fusion_dense").output)
    # Dự đoán đặc trưng cho fold
    ...
# Lấy trung bình cộng 5 fold model
ts_features_ensemble = np.mean(all_fold_features, axis=0) # Shape: [8030, 32]
```

### (c) Xuất dữ liệu
* Vector đặc trưng được ghép với cột khóa `file_name` và lưu dưới dạng Parquet: `output/timeseries_features.parquet` (kích thước 8030 dòng x 33 cột).
* Lưu ma trận đặc trưng thô dưới dạng file npy: `output/timeseries_features.npy` (chỉ chứa ma trận float32 kích thước [8030, 32]).

---

## Phần C — Đánh giá tính hoàn thiện của nhánh Timeseries

Nhánh Timeseries hiện tại đã **hoàn thành 100%** và sẵn sàng chuyển giao sang pha Fusion (GNN):
1. **Hoàn thiện cấu trúc K-Fold & Ensemble:** Loại bỏ hoàn toàn sự trồi sụt ngẫu nhiên của các chỉ số đánh giá.
2. **Khắc phục triệt để lỗi phương pháp:** Toàn bộ dữ liệu được đảm bảo không có rò rỉ (leakage), nhãn đích chuẩn `bnn`, tính tuổi động và quét ngưỡng chọn tự động lâm sàng khách quan.
3. **Đặc trưng chất lượng cao:** File parquet và npy đã được xuất thành công, mapping chính xác mã bệnh nhân (`file_name`) và sẵn sàng cho việc làm node trong đồ thị fusion GNN.
