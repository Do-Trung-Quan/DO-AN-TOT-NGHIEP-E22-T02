# Giải thích chi tiết pipeline nhánh Timeseries (`ĐỒ_ÁN.ipynb`)

> Tài liệu này giải thích **từng cell** của notebook, viết cho người **chưa có nền tảng AI/ML** vẫn hiểu được bản chất.
> Mỗi cell quan trọng được mổ theo 3 lớp: **(a) Mục đích** → **(b) Từng bước: cơ sở lý thuyết + cơ chế hoạt động** → **(c) Ví dụ trên dữ liệu thật**.
>
> Trọng tâm là **từ mục 5) Tách nhánh dữ liệu trở đi** (cell 7 → cell 10). Các cell 1–4 chỉ tóm tắt.

---

## 0. Bức tranh tổng thể — notebook này làm gì?

Ta có một file Excel **8030 dòng** (mỗi dòng = 1 bệnh nhân), **181 cột** là kết quả phiếu khám sức khỏe nghề nghiệp (tuổi, nghề, triệu chứng ho/khó thở, khám phổi 6 vùng, đo chức năng hô hấp...). Cột nhãn cần dự đoán là **`bnn`** (bệnh nghề nghiệp: 0 = khỏe, 1 = có bệnh).

Notebook làm 2 việc:
1. **Biến bảng chữ+số lộn xộn thành số sạch** mà máy học hiểu được (cell 1–6).
2. **Huấn luyện một mạng nơ-ron** đọc dữ liệu lâm sàng và học cách nhận ra bệnh; sau đó **rút ra một "vector đặc trưng" 32 chiều** cho mỗi bệnh nhân, để sau này ghép (fusion) với nhánh ảnh X-quang (cell 7–10).

**Điều cốt lõi cần nhớ:** đích cuối KHÔNG phải là bộ phân loại lâm sàng này, mà là **vector 32 chiều** — một bản "tóm tắt số học" tình trạng lâm sàng của bệnh nhân, để đưa vào mô hình đồ thị (GNN) ở bước fusion.

```
Excel thô  ──(cell 1-6: làm sạch + mã hóa)──►  Bảng số sạch (8030 × 140)
                                                       │
                              (cell 7: chia train/val/test, chuẩn hóa)
                                                       │
                              (cell 8-9: dựng & huấn luyện mạng nơ-ron)
                                                       │
                              (cell 10: rút vector 32 chiều mỗi bệnh nhân)
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
| **3** | Encoding | Biến chữ thành số: giới tính nam/nữ→0/1; các cột có/không→0/1; các cột mức độ (ordinal)→0,1,2,3; vài cột phân loại→one-hot; tính `tuoi` = 2026 − năm sinh. | Mạng nơ-ron **chỉ tính được trên số**, không hiểu chữ "nam", "khó thở". |
| **4** | Phân loại nghề 10 cấp | 4 cột nghề (chữ tự do như "Luyen thep") → gán mã 0–9 theo mức độ độc hại/phơi nhiễm bụi. | Nghề là yếu tố nguy cơ chính của bệnh bụi phổi. Chữ tự do quá đa dạng → gom về 10 nhóm để máy học được. |

> **Ba loại encoding — hiểu bản chất:**
> - **Binary (0/1):** dùng cho câu hỏi có/không. VD `hutthuoc` (hút thuốc): có→1, không→0.
> - **Ordinal (0,1,2,3...):** dùng khi các mức **có thứ tự**. VD mức khó thở: không < khi gắng sức < khi làm nhẹ < thường xuyên → 0,1,2,3. Số lớn = nặng hơn, thứ tự này có nghĩa.
> - **One-hot:** dùng khi phân loại **không có thứ tự**. VD nhóm máu A/B/O — không thể nói A < B. Mỗi loại thành 1 cột riêng chứa 0/1, tránh việc máy hiểu nhầm có thứ tự.

Sau cell 4, ta có `df` gồm **8030 dòng × 140 cột toàn số**, không còn ô trống. Đây là đầu vào cho phần chính bên dưới.

---

# Phần B — Chi tiết từ mục 5 trở đi (trọng tâm)

---

## Cell 7 — Tách nhánh dữ liệu, chia Train/Val/Test, chống rò rỉ

### (a) Mục đích của cell

Đây là cell **quan trọng và tinh tế nhất** về mặt phương pháp. Nó làm 4 việc:
1. **Tách 140 cột thành 3 "nhóm đầu vào"** khác nhau, vì mỗi nhóm có bản chất khác nhau và sẽ đi vào 3 nhánh mạng khác nhau.
2. **Chia dữ liệu** thành 3 tập: học (train), điều chỉnh (validation), thi cuối (test).
3. **Chuẩn hóa số liệu** (đưa về cùng thang đo) một cách **chống rò rỉ dữ liệu**.
4. **Xử lý mất cân bằng** (99% khỏe, chỉ ~2.6% bệnh) bằng undersampling + class weight.

### (b) Từng bước

---

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

**Cơ sở lý thuyết — vì sao phải tách?**
Không phải dữ liệu nào cũng "cùng loại". Ba nhóm này có **cấu trúc khác nhau**, nên cần 3 kiểu xử lý khác nhau:

- **Nhóm JOB (nghề nghiệp)** — là **mã danh mục** (categorical). Con số "3" ở đây không có nghĩa "gấp 3 lần số 1", nó chỉ là **tên nhóm nghề**. → cần xử lý bằng *Embedding* (giải thích ở cell 8).
- **Nhóm SEQ (khám phổi 6 vùng)** — có **cấu trúc lặp theo không gian**. Bác sĩ khám 6 vùng phổi (đỉnh trái, đỉnh phải, giữa, đáy...), mỗi vùng ghi 7 chỉ số (rung thanh, gõ đục, ran ẩm, ran nổ...). → xếp thành "chuỗi 6 bước, mỗi bước 7 số" để đưa vào *LSTM* — mạng chuyên đọc chuỗi.
- **Nhóm NUM (chỉ số số học)** — tuổi, huyết áp, chức năng hô hấp FVC/FEV1... là **số thực thông thường**. → đưa vào lớp *Dense* bình thường.

**Cơ chế — hình dung khối SEQ 3 chiều:**
`X_seq_all` có shape `[8030, 6, 7]` nghĩa là: 8030 bệnh nhân, mỗi người một **ma trận 6×7**:

```
          rungt rungg goduc vtam vtno vtrit vtngay   ← 7 phép đo
Vùng 1  [   0     0     0     0    0    0     0   ]
Vùng 2  [   0     1     0     0    0    0     0   ]
Vùng 3  [   1     1     0     0    1    0     0   ]
...
Vùng 6  [   0     0     0     0    0    0     0   ]
```

**(c) Ví dụ dữ liệu thật:**
Vùng 1 gồm 7 cột thật: `rungt1, rungg1, goduc1, vtam1, vtno1, vtrit1, vtngay1`. Bệnh nhân ở dòng 5 có toàn bộ 7 giá trị này trống trong Excel gốc → đã được điền 0 ở cell 2, nghĩa là "vùng phổi 1 khám bình thường, không có dấu hiệu bất thường".

---

#### Bước 2 — Chọn cột NUM và loại các cột không được dùng

```python
exclude_cols = set(seq_cols_flat) | set(JOB_COLS) | {TARGET_COL, "bnncuthe",
               "benhhh", "tiensuhh", "id", "hoten"}
num_cols = [c for c in df.columns if c not in exclude_cols and is_numeric_dtype(df[c])]
```

**Cơ sở lý thuyết — vì sao loại vài cột?**
- `bnn` (TARGET_COL) là **đáp án** — tuyệt đối không được đưa vào đầu vào, nếu không mô hình chỉ việc "chép đáp án" (leakage nghiêm trọng nhất).
- `benhhh`, `tiensuhh`, `bnncuthe` là các cột **liên quan trực tiếp tới chẩn đoán bệnh** → loại để tránh rò rỉ.
- Các cột SEQ và JOB đã tách riêng nên cũng loại khỏi NUM để không bị đếm 2 lần.

Kết quả: **91 cột số** vào nhóm NUM.

**(c) Ví dụ thật:** nhóm NUM chứa `tuoi, tuoinghe` (số năm làm nghề), `hatd/hatt` (huyết áp), `fvc, fev1` (chỉ số hô hấp), v.v.

---

#### Bước 3 — Chia Train / Validation / Test (70/15/15) có phân tầng

```python
idx_train_val, idx_test = train_test_split(idx, test_size=0.15, stratify=y)
idx_train, idx_val = train_test_split(idx_train_val, test_size=0.15/0.85, stratify=y[...])
```

**Cơ sở lý thuyết — vì sao chia 3 tập?** Đây là nguyên tắc vàng của ML, ví như việc học thi:
- **Train (70%, ~5620 mẫu)** = sách giáo khoa để **học**. Mạng nhìn dữ liệu này và điều chỉnh tham số.
- **Validation (15%, 1205 mẫu)** = đề thi thử để **canh chỉnh** (chọn lúc nào dừng học, giảm learning rate). Mạng KHÔNG học trực tiếp từ đây.
- **Test (15%, 1205 mẫu)** = đề thi thật, **giấu kín tới phút chót**. Chỉ dùng đúng 1 lần để báo cáo. Nếu mô hình chỉ "học thuộc lòng" train mà không hiểu bản chất, nó sẽ làm tốt train nhưng tệ trên test → gọi là **overfitting**.

**Cơ chế `stratify=y` — vì sao bắt buộc ở đây?**
Dữ liệu này chỉ có **211 ca bệnh / 8030 (2.6%)**. Nếu chia ngẫu nhiên "mù", có thể xui rủi dồn hết ca bệnh vào train, khiến test không còn ca bệnh nào để đánh giá. `stratify=y` ép **tỷ lệ bệnh/khỏe giống nhau ở cả 3 tập** (mỗi tập giữ ~2.6% bệnh). → test có ~32 ca bệnh, val ~32 ca.

**(c) Ví dụ thật:** phân phối gốc `{0: 7819, 1: 211}`. Sau chia: mỗi tập test/val có khoảng 32 ca dương — con số nhỏ này về sau là **giới hạn lớn nhất** của độ tin cậy đánh giá.

---

#### Bước 4 — Chuẩn hóa (StandardScaler) chống rò rỉ ⭐

```python
imputer = SimpleImputer(strategy="mean")
scaler  = StandardScaler()
X_num_train_imp = imputer.fit_transform(X_num_all_raw[idx_train])   # FIT trên train
X_train_num     = scaler.fit_transform(X_num_train_imp)             # FIT trên train
X_val_num       = scaler.transform(imputer.transform(X_num_all_raw[idx_val]))   # chỉ TRANSFORM
X_test_num      = scaler.transform(imputer.transform(X_num_all_raw[idx_test]))  # chỉ TRANSFORM
```

**Cơ sở lý thuyết — vì sao phải chuẩn hóa (standardize)?**
Các cột số có **thang đo rất khác nhau**: tuổi ~20–60, huyết áp ~80–180, FVC ~2–5 (lít). Nếu để nguyên, cột có số lớn (huyết áp) sẽ "át" cột số nhỏ (FVC) chỉ vì độ lớn, dù chưa chắc quan trọng hơn. StandardScaler biến mỗi cột về **trung bình 0, độ lệch chuẩn 1**:

```
giá_trị_mới = (giá_trị_gốc − trung_bình_cột) / độ_lệch_chuẩn_cột
```

→ mọi cột về cùng "sân chơi", mạng học công bằng và hội tụ nhanh hơn.

**Cơ chế `fit` vs `transform` — điểm mấu chốt chống leakage:**
- `fit` = **tính** trung bình & độ lệch chuẩn.
- `transform` = **áp dụng** công thức trên.

Điểm tinh tế: `fit` **chỉ được nhìn tập train**. Vì sao? Tập test đại diện cho "dữ liệu tương lai chưa từng thấy". Nếu ta tính trung bình từ **cả** test, thì thông tin của test đã "rò rỉ" vào quá trình xử lý train → mô hình gián tiếp biết trước phân phối test → **kết quả đánh giá bị thổi phồng, không trung thực**. Code ở đây làm đúng: `fit_transform` trên train, chỉ `transform` cho val/test. `SimpleImputer` (điền ô trống bằng trung bình) cũng tuân thủ nguyên tắc y hệt.

> 💡 Đây là điểm **cộng lớn** của pipeline — rất nhiều đồ án làm sai chỗ này.

---

#### Bước 5 — Xử lý mất cân bằng: Undersampling + Class weight

```python
UNDERSAMPLE_RATIO = 5   # giữ 5 người khỏe cho mỗi 1 người bệnh
# ... chọn ngẫu nhiên bớt người khỏe trong TRAIN ...
weights = compute_class_weight("balanced", classes=[0,1], y=y_train)
# → class_weights_dict = {0: 0.51, 1: 19.1}
```

**Cơ sở lý thuyết — vì sao phải xử lý mất cân bằng?**
Với 97.4% khỏe, một mô hình "lười" chỉ cần **đoán mọi người đều khỏe** đã đạt 97.4% accuracy — nhưng **vô dụng** vì bỏ sót 100% ca bệnh. Ta cần ép mô hình quan tâm tới thiểu số (ca bệnh). Có 2 vũ khí, notebook dùng **cả hai**:

1. **Undersampling (giảm mẫu):** bớt ngẫu nhiên nhóm đa số (người khỏe) trong train, còn tỷ lệ 1 bệnh : 5 khỏe. → mỗi batch huấn luyện có tỷ lệ bệnh cao hơn, mạng "gặp" ca bệnh thường xuyên hơn.
2. **Class weight (trọng số lớp):** phạt nặng hơn khi đoán sai ca bệnh. Trọng số lớp bệnh = 19.1 nghĩa là **sai 1 ca bệnh bị phạt gấp ~19 lần** sai 1 ca khỏe. → mạng "sợ" bỏ sót bệnh.

**Cơ chế — chỉ làm trên TRAIN, KHÔNG đụng Val/Test:**
Đây là chi tiết quan trọng: undersampling **chỉ áp cho train**. Val và test **giữ nguyên phân phối thật** (2.6% bệnh) để đánh giá phản ánh đúng thực tế lâm sàng. Nếu cân bằng cả test, con số đẹp sẽ là giả.

> ⚠️ **Nhận xét sư phạm:** dùng **đồng thời** cả undersampling *và* class weight là "chữa 2 lần" cho cùng một bệnh → mô hình bị đẩy quá mạnh về phía đoán "bệnh", sinh ra nhiều **báo động giả (false positive)**. Đây chính là lý do precision thấp (~0.26) ở kết quả. Thường chỉ nên chọn **một** cơ chế. Không sai về nguyên tắc, nhưng là điểm có thể tinh chỉnh.

**(c) Ví dụ thật:** train sau resample còn ~5620 mẫu; class weight `{0: 0.51, 1: 19.1}`.

---

## Cell 8 — Kiến trúc mạng nơ-ron đa đầu vào (Multi-Input)

### (a) Mục đích của cell

Dựng một mạng nơ-ron có **3 "cửa vào" riêng** cho 3 nhóm dữ liệu (job / seq / num), mỗi cửa xử lý theo cách phù hợp, rồi **gộp lại** (fusion nội bộ) để đưa ra 1 dự đoán bệnh/khỏe. Đồng thời lớp gộp này chính là nơi ta sẽ **rút vector 32 chiều** sau này.

**Hình dung tổng thể:**
```
JOB (4)  ──► Embedding ──► Pooling ──► Dense16 ──┐
SEQ (6×7)──► LSTM32 ──────────────────► Dense16 ──┼─► [gộp 48] ─► Dense32 ─► Dropout ─► Dense1(sigmoid) ─► xác suất bệnh
NUM (91) ──► Dense32 ─► Dropout ──────► Dense16 ──┘             ▲
                                                          "fusion_dense" = nơi rút vector 32-D
```

### (b) Từng bước — giải thích 3 nhánh

---

#### Nhánh 1 — JOB: Embedding cho dữ liệu danh mục

```python
x_job = layers.Embedding(input_dim=10, output_dim=8)(job_input)
x_job = layers.GlobalAveragePooling1D()(x_job)
x_job = layers.Dense(16, activation="relu")(x_job)
```

**Cơ sở lý thuyết — Embedding là gì và vì sao cần?**
Nghề đã được mã 0–9. Nếu đưa thẳng số này vào mạng, mạng sẽ hiểu nhầm "nghề 8 lớn gấp 4 lần nghề 2" — vô nghĩa. **Embedding** giải quyết bằng cách gán cho mỗi mã nghề một **vector 8 số học được** (learnable). Trong quá trình train, các nghề có "hành vi bệnh" giống nhau sẽ tự động có vector gần nhau. Giống như bản đồ: thay vì gán nghề một con số vô hồn, ta đặt nó vào một "không gian 8 chiều" nơi khoảng cách phản ánh mức độ giống nhau.

**Cơ chế:** `input_dim=10` (10 nhóm nghề), `output_dim=8` (mỗi nghề → vector 8-D). Mỗi bệnh nhân có 4 ô nghề → được 4 vector 8-D → `GlobalAveragePooling1D` lấy **trung bình** 4 vector đó thành 1 vector 8-D đại diện "hồ sơ nghề nghiệp tổng hợp" của người đó.

**(c) Ví dụ thật:** bệnh nhân "Luyen thep / Nau thep" → cả 4 ô đều rơi nhóm nghề độc hại cao (mã 0 = lv1) → embedding học ra vector đặc trưng cho "phơi nhiễm bụi/khói kim loại nặng".

---

#### Nhánh 2 — SEQ: LSTM cho chuỗi 6 vùng phổi

```python
x_seq = layers.LSTM(32, return_sequences=False)(seq_input)
```

**Cơ sở lý thuyết — LSTM là gì?**
LSTM (Long Short-Term Memory) là mạng chuyên đọc **chuỗi**, xử lý lần lượt từng "bước" và **ghi nhớ** thông tin từ bước trước để hiểu bước sau. Nó nổi tiếng dùng cho câu chữ (đọc từng từ) hay chuỗi thời gian. Ở đây "chuỗi" là 6 vùng phổi khám lần lượt; LSTM đọc vùng 1→2→...→6 và tổng hợp thành 1 vector 32-D mô tả "toàn cảnh tổn thương phổi".

**Cơ chế:** input `[6, 7]` (6 bước, mỗi bước 7 số), LSTM cho ra 1 vector 32-D (`return_sequences=False` = chỉ lấy kết quả cuối, tức bản tóm tắt sau khi đọc hết 6 vùng).

> 📌 **Lưu ý học thuật:** 6 vùng phổi thực chất là **quan hệ không gian**, không phải **thời gian** thật sự (vùng 3 không "xảy ra sau" vùng 2). Dùng LSTM ở đây là một lựa chọn chấp nhận được để tổng hợp thông tin, nhưng không phải là ứng dụng "kinh điển" của LSTM. Đây là điểm có thể tranh luận/thử thay thế (VD dùng Dense phẳng, hoặc attention), nhưng chưa cần lúc này.

---

#### Nhánh 3 — NUM: Dense + BatchNorm + Dropout

```python
x_num = layers.Dense(32, activation="relu")(num_input)
x_num = layers.BatchNormalization()(x_num)
x_num = layers.Dropout(0.2)(x_num)
x_num = layers.Dense(16, activation="relu")(x_num)
```

**Cơ sở lý thuyết — 3 khái niệm:**
- **Dense (fully-connected):** lớp nơ-ron cơ bản, mỗi đầu ra là tổ hợp có trọng số của mọi đầu vào. `activation="relu"` (giữ số dương, cắt số âm về 0) cho phép mạng học quan hệ **phi tuyến** (không chỉ đường thẳng).
- **BatchNormalization:** chuẩn hóa lại số liệu **giữa các lớp** trong lúc train → giúp học **ổn định và nhanh** hơn.
- **Dropout(0.2):** trong lúc train, **tắt ngẫu nhiên 20% nơ-ron** mỗi lần. Nghe lạ nhưng đây là mẹo **chống overfitting** kinh điển: ép mạng không được phụ thuộc vào vài nơ-ron cá biệt, phải học đặc trưng "vững" hơn. Khi dự đoán thật thì Dropout tự tắt.

---

#### Bước gộp (Fusion nội bộ) & đầu ra

```python
merged = layers.Concatenate()([x_job, x_seq, x_num])   # 16+16+16 = 48
x = layers.Dense(32, activation="relu", name="fusion_dense")(merged)  # ⭐ 32-D
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation="sigmoid")(x)      # xác suất 0..1
```

**Cơ sở lý thuyết:**
- `Concatenate`: nối 3 vector 16-D thành 1 vector 48-D — "gộp mọi khía cạnh của bệnh nhân".
- `fusion_dense` (32-D): lớp này **nén 48 → 32**, tạo ra **bản tóm tắt số học cô đọng nhất** về bệnh nhân. **Chính lớp này sẽ được rút ra làm vector fusion** ở cell 10.
- `output` với `sigmoid`: ép kết quả về khoảng **0–1**, đọc như **xác suất mắc bệnh** (0.9 = rất khả năng bệnh; 0.1 = khả năng khỏe).

**(c) Ví dụ thật:** mạng chỉ ~11.000 tham số — rất nhỏ (phù hợp dữ liệu ít). Tổng cộng 3 đầu vào shape `(4)`, `(6,7)`, `(91)` → 1 đầu ra `(1)`.

---

## Cell 9 — Huấn luyện & Đánh giá đa ngưỡng

### (a) Mục đích của cell

Cho mạng **học** từ tập train, **canh chỉnh** bằng val (dừng đúng lúc), rồi **thi cuối** trên test hold-out. Vì bài toán mất cân bằng, cell còn **quét nhiều ngưỡng quyết định** để chọn điểm cân bằng phù hợp.

### (b) Từng bước

---

#### Bước 1 — Callbacks: EarlyStopping & ReduceLROnPlateau

```python
EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True)
ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=5, min_lr=1e-5)
```

**Cơ sở lý thuyết:**
- **EarlyStopping (dừng sớm):** theo dõi `val_auc` (điểm trên tập val). Nếu **12 epoch liền không cải thiện**, dừng học và **khôi phục lại trọng số tốt nhất**. → tránh học quá đà thành overfitting. Ví như "ôn thi tới lúc điểm thi thử không tăng nữa thì nghỉ".
- **ReduceLROnPlateau (giảm tốc độ học):** khi val_auc chững lại 5 epoch, **giảm learning rate một nửa**. Learning rate = "bước chân" khi học: đầu bước dài đi nhanh, về sau bước ngắn để dò chính xác điểm tối ưu.

**(c) Ví dụ thật:** lần train gần nhất dừng sớm quanh epoch 14, val_auc chững ở ~0.91 ngay từ những epoch đầu.

---

#### Bước 2 — Huấn luyện

```python
history = model.fit(x=[X_train_job, X_train_seq, X_train_num], y=y_train,
                    validation_data=(...), epochs=60, batch_size=64,
                    class_weight=class_weights_dict, callbacks=callbacks)
```

**Cơ chế:** mạng đi qua dữ liệu tối đa 60 vòng (epoch), mỗi lần nuốt 64 mẫu (batch). `class_weight` áp trọng số phạt lệch như đã nói. Cặp learning-curve (Loss & AUC theo epoch) được vẽ ra để **chẩn đoán**: nếu train tốt mà val tệ dần → overfit; nếu cả hai còn giảm → có thể học thêm.

---

#### Bước 3 — Đánh giá: ROC-AUC vs PR-AUC ⭐

```python
test_auc    = roc_auc_score(y_test, y_test_prob)          # ~0.91
test_pr_auc = average_precision_score(y_test, y_test_prob) # ~0.40
```

**Cơ sở lý thuyết — vì sao 2 chỉ số, và đâu là chỉ số ĐÚNG?**
Mạng cho ra **xác suất** (0–1). Để đo chất lượng không phụ thuộc ngưỡng, ta dùng:
- **ROC-AUC:** đo khả năng xếp hạng đúng (ca bệnh có xác suất cao hơn ca khỏe). Nhưng với dữ liệu **mất cân bằng nặng**, ROC-AUC **lạc quan giả** — dễ đạt 0.9+ dù mô hình còn yếu, vì nó "được thưởng" nhờ đoán đúng vô số ca khỏe.
- **PR-AUC (Precision-Recall AUC):** chỉ tập trung vào lớp hiếm (bệnh). Đây mới là **thước đo trung thực** cho bài toán này.

**Cách đọc PR-AUC 0.40:** với tỷ lệ bệnh 2.6%, một mô hình đoán mò chỉ đạt PR-AUC ≈ 0.027. Đạt 0.40 tức **gấp ~15 lần đoán mò** → mô hình **thực sự học được tín hiệu**, dù giá trị tuyệt đối còn khiêm tốn.

> 📌 **Khi báo cáo hội đồng: nêu PR-AUC là chính**, đừng khoe ROC-AUC 0.91 một mình (dễ gây hiểu nhầm là mô hình đã rất tốt).

---

#### Bước 4 — Threshold sweep (quét ngưỡng)

**Cơ sở lý thuyết — ngưỡng (threshold) là gì?**
Mạng cho xác suất, nhưng cuối cùng phải quyết "bệnh hay khỏe". Ta cần một **ngưỡng cắt**: xác suất ≥ ngưỡng → gọi là bệnh. Mặc định 0.5, nhưng với dữ liệu lệch, ngưỡng khác thường tốt hơn. Có sự **đánh đổi**:
- **Ngưỡng thấp (0.5):** bắt được nhiều bệnh (**recall cao**) nhưng báo nhầm nhiều người khỏe (**precision thấp**).
- **Ngưỡng cao (0.9):** báo nhầm ít, nhưng bỏ sót nhiều bệnh.

**(c) Ví dụ thật (kết quả lần train gần nhất):**

| Ngưỡng | Precision | Recall | F1 | Bắt đúng bệnh | Báo nhầm (FP) |
|---|---|---|---|---|---|
| 0.50 | 0.209 | 0.750 | 0.327 | 24/32 | 91 |
| 0.70 | 0.263 | 0.656 | 0.375 | 21/32 | 59 |
| 0.85 | 0.339 | 0.594 | **0.432** | 19/32 | 37 |
| 0.90 | 0.346 | 0.563 | 0.429 | 18/32 | 34 |

Đọc bảng: ở ngưỡng 0.70, mô hình bắt được 21/32 ca bệnh nhưng báo nhầm 59 người khỏe. F1 (điểm cân bằng precision & recall) cao nhất ~0.43 ở ngưỡng 0.85.

> ⚠️ **Lỗi phương pháp cần sửa:** ngưỡng nên được **chọn trên tập VAL**, rồi mới áp cố định lên test. Chọn ngưỡng bằng cách nhìn bảng kết quả **test** rồi lấy ngưỡng đẹp nhất là một dạng **rò rỉ nhẹ** (peeking) — làm số liệu test tốt hơn thực tế.

---

## Cell 10 — Trích xuất vector đặc trưng 32 chiều cho Fusion ⭐

### (a) Mục đích của cell

Đây là **sản phẩm cuối cùng** của cả nhánh Timeseries. Ta không cần cái đầu ra "bệnh/khỏe" nữa — ta cần **bản tóm tắt 32 số** của mỗi bệnh nhân (lấy từ lớp `fusion_dense`) để đưa sang nhánh fusion GNN.

### (b) Từng bước

```python
feature_extractor = Model(inputs=model.inputs,
                          outputs=model.get_layer("fusion_dense").output)  # cắt tại 32-D
ts_vectors = feature_extractor.predict([X_job_all, X_seq_all, X_num_all])  # [8030, 32]
```

**Cơ sở lý thuyết — "cắt đầu" mô hình (feature extraction) là gì?**
Một mạng nơ-ron học **theo tầng**: các lớp đầu học đặc trưng thô, các lớp sau tổng hợp thành đặc trưng cô đọng, lớp cuối ra quyết định. Kỹ thuật kinh điển là **bỏ lớp quyết định cuối, giữ lại lớp áp chót** — vốn là "bản mô tả giàu thông tin nhất" mà mạng đã học về mỗi mẫu. Ở đây ta cắt tại `fusion_dense` (32-D). Vector này **cô đọng cả 3 khía cạnh** (nghề + phổi + chỉ số) thành 32 con số.

**Cơ chế:** tạo một model mới dùng chung đầu vào nhưng đầu ra là `fusion_dense`, rồi chạy `predict` cho **toàn bộ 8030 bệnh nhân** (không chỉ test) → ma trận `[8030, 32]`.

**Lưu kèm `file_name`:**
```python
df_vectors.insert(0, "file_name", file_names)
df_vectors.to_parquet("output/timeseries_features.parquet")
```
`file_name` là **chìa khóa** để nhánh fusion biết "vector timeseries này của bệnh nhân nào, khớp với ảnh X-quang nào".

**(c) Ví dụ thật:** file xuất ra `output/timeseries_features.parquet` shape **(8030, 33)** = 1 cột `file_name` + 32 cột `ts_feat_0..31`. Mỗi dòng là một bệnh nhân.

> ⚠️ **Sự thật quan trọng về dữ liệu (phát hiện khi kiểm tra file):** trong 8030 bệnh nhân, **chỉ 433 người có ảnh X-quang** (`file_name` thật, đuôi `.jpg`); 7597 người còn lại `file_name = "không"`. Và trong 433 người đó, **chỉ 29 người mắc bệnh** (bnn=1). Nghĩa là bước fusion (cần cả ảnh + lâm sàng) chỉ chạy được trên **433 mẫu / 29 ca bệnh**. Đây là **ràng buộc quyết định** cho toàn bộ chiến lược fusion — xem phần C.

---

## Phần C — Tổng kết & những điểm cần nhớ

**Điểm mạnh của pipeline:**
- ✅ Chống rò rỉ dữ liệu bài bản (scaler/imputer fit chỉ trên train; loại cột X-quang; undersampling chỉ trên train).
- ✅ Kiến trúc đa đầu vào hợp lý (embedding cho nghề, LSTM cho vùng phổi, dense cho chỉ số).
- ✅ Đánh giá đúng bằng PR-AUC + threshold sweep, có callbacks chống overfit.

**Điểm cần tinh chỉnh (không gấp):**
- ⚠️ Dùng đồng thời undersampling + class weight → nhiều báo động giả. Nên chọn 1.
- ⚠️ Ngưỡng nên chốt trên val, không phải test.
- ⚠️ Chỉ 32 ca bệnh trong test → nên K-Fold để số liệu ổn định.

**Ràng buộc lớn nhất cho bước tiếp theo:**
- 🔴 Fusion chỉ có **433 mẫu / 29 ca bệnh**. Đây là con số rất nhỏ, định hình toàn bộ chiến lược fusion (xem tài liệu/thảo luận riêng về fusion).
