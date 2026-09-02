# Roadmap — Phase Fusion (GNN đa phương thức)

> Lộ trình chi tiết từ trạng thái mã nguồn hiện tại tới mô hình chẩn đoán bệnh nghề nghiệp cuối cùng.
> Mỗi bước có **mục tiêu · việc làm · tiêu chí nghiệm thu**. Không bước nào được bỏ qua gate của nó.
>
> Tài liệu liên quan: [ts-overview.md](ts-overview.md) · [ts-pipeline-exp.md](ts-pipeline-exp.md) · [X-ray-overview.md](X-ray-overview.md)

---

## 1. Mục tiêu

Dựng **một đồ thị dân số duy nhất** gồm 8.030 node (mỗi node = 1 bệnh nhân, đặc trưng = vector timeseries ⊕ vector ảnh nếu có), rồi so sánh 3 backbone GNN để chọn mô hình chẩn đoán `bnn` tốt nhất.

**Giả thuyết nghiên cứu cần kiểm chứng:**

> 6.195/8.030 bệnh nhân **không có ảnh X-quang**. Nếu đồ thị cho phép họ nhận thông tin từ hàng xóm **có ảnh**, mô hình sẽ tốt hơn. Cơ chế truyền tin càng linh hoạt thì hiệu quả càng cao:
> **Graph Transformer** (attention toàn cục — nối được tới mọi bệnh nhân) > **GAT** (kNN tĩnh — chỉ nối trong nhóm hàng xóm cố định) > **DGCNN** (kNN động — dễ tự phân cụm thành 2 nhóm cô lập có-ảnh / không-ảnh).

Giả thuyết này **phải được chứng minh bằng số**, không phải bằng lập luận. Phase 6 thiết kế riêng cho việc đó.

### 1.1. Lệch có chủ đích so với bài báo tham khảo

| | Bài báo `Paper_128-...pdf` | Đồ án này |
|---|---|---|
| Đơn vị | **8.030 đồ thị**, mỗi bệnh nhân = 1 đồ thị (node = trường thuộc tính) | **1 đồ thị**, 8.030 node = 8.030 bệnh nhân |
| Bài toán | Phân loại đồ thị (DGCNN + SortPooling) | Phân loại node (transductive) |
| Thông tin chảy giữa | Các thuộc tính trong 1 bệnh nhân | **Các bệnh nhân với nhau** |
| Người không có ảnh | Không nhận được gì từ nhánh ảnh | **Học ké từ hàng xóm có ảnh** |

**Lý do lệch (dùng khi bảo vệ):** bài báo gốc chỉ có dữ liệu bảng nên đồ thị per-patient là hợp lý. Đồ án này có thêm phương thức ảnh nhưng chỉ phủ 22,9% bệnh nhân — **population graph là công thức duy nhất khai thác được phần dữ liệu ảnh cho toàn bộ quần thể**. Tham chiếu học thuật: Parisot et al., *"Spectral Graph Convolutions for Population-based Disease Prediction"*, MICCAI 2017.

---

## 2. Trạng thái đầu vào

| Nguồn | File | Trạng thái |
|---|---|---|
| Timeseries | `output/timeseries_features.parquet` (8030 × 33) | ✅ Có — **sẽ tạo lại ở Phase 0.5** |
| Ảnh — `control` | `imagefeat/output/image_features_control.parquet` (1835 × 259) | ✅ Sẵn sàng |
| Ảnh — chỉ mục | `imagefeat/output/image_index.parquet` (1835 × 6) | ✅ Sẵn sàng |
| Ảnh — `frozen` | *chưa có* | ⬜ Phase 0.3 (~2 phút GPU) |
| Ảnh — `crossfit` | `imagefeat/output/image_features.parquet` (109 hàng) | 🟡 **Mới là smoke test** — Phase 0.4 |
| Metadata node | `output/fusion_node_meta.parquet` | ⬜ Phase 0.5.6 |

### 2.1. Số liệu nền cần thuộc lòng

```
Tổng bệnh nhân          : 8.030          Có ảnh X-quang     : 1.835 (22,85%)
Ca bệnh (bnn=1)         :   211 (2,63%)  Ca bệnh có ảnh     :    99 ( 5,40%)
Mất cân bằng            : 1:37           Ca bệnh không ảnh  :   112 ( 1,81%)
Vector timeseries       : 32-D           Vector ảnh         : 256-D (đã bỏ 256 chiều chết)
```

⚠️ **Nhóm có ảnh có tỷ lệ bệnh cao gấp 3 lần nhóm không ảnh.** Đây là *shortcut* nguy hiểm: model có thể học "có ảnh ⇒ nguy cơ cao" mà không cần nhìn nội dung phim. Phase 4.1 có 3 lớp phòng vệ cho việc này.

### 2.2. Ba nguồn đặc trưng ảnh khác nhau ở đâu

| Nguồn | Backbone fine-tune trên | Rò rỉ | Đặc điểm |
|---|---|---|---|
| `control` | SetA (2.129 ảnh **khác hoàn toàn**, overlap = 0 đã verify) | Không, theo cấu tạo | Đã đo: PC1 90,7% · effective rank 1,80 → **collapse mạnh** |
| `frozen` | **Không fine-tune** | Không, theo cấu tạo | Đặc trưng tổng quát, **ít collapse hơn** |
| `crossfit` | 1.835 ảnh này, chia 5 fold theo bệnh nhân | Không (out-of-fold) | Thích nghi đúng miền dữ liệu |

Vì sao cần cả 3: nhãn huấn luyện nhánh ảnh là `ketqua` (tổn thương trên phim, 25,2%) **khác** nhãn đích `bnn` (5,4%). Đo trên bộ `control`: giữ 8 hướng biến thiên mạnh nhất → `bnn` ROC-AUC 0,832; giữ 32 hướng → **0,877**. Các hướng thứ 9–32 chỉ chiếm ~3% phương sai nhưng nâng PR-AUC từ 0,208 lên 0,282 (**+36%**). Tức là **thông tin `bnn` cần đang nằm ở những chi tiết mà fine-tune xóa bỏ** — nên phải có bộ `frozen` để đối chứng.

---

## 3. Thiết kế thí nghiệm — 19 cấu hình

| Nhóm | Cấu hình | Số ô | Trả lời câu hỏi |
|---|---|---|---|
| Baseline chung | **B0** — TS-only MLP (không ảnh) | 1 | Mốc xuất phát |
| Baseline / nguồn | **B1** — Image-only probe | 3 | Nguồn ảnh nào mạnh nhất |
| Baseline / nguồn | **B2** — TS⊕IMG MLP, **không đồ thị** | 3 | **Then chốt:** đồ thị có đóng góp gì không |
| Đối chứng | **B3** — GCN thường | 3 | Attention có hơn tích chập đơn giản không |
| **Chính** | **GAT · DGCNN · GraphTransformer** | **9** | **Bảng so sánh chính của đồ án** |

Mỗi ô = 5 fold × 3 seed = 15 lần train. GNN trên 8.030 node rất nhẹ → toàn bộ nằm trong vài giờ Colab.

### 3.1. Hai nguyên tắc bất di bất dịch

1. **Trục chính là backbone.** Nguồn ảnh là trục ablation độ bền. Bảng headline dùng **`control`**; 2 nguồn kia vào bảng phụ. Quyết định này chốt **trước** khi chạy, không đổi sau khi thấy kết quả.
2. **Chọn mô hình bằng OOF, chạm test đúng 1 lần.** Test hold-out chỉ có ~32 ca dương; so 19 ô trên test là cầm chắc overfit vào nhiễu.

### 3.2. Metric

| Vai trò | Metric | Ghi chú |
|---|---|---|
| **Chính** | **PR-AUC** (average precision) | Chuẩn cho dữ liệu lệch 1:37 |
| Phụ | ROC-AUC | Dễ so với tài liệu khác |
| Lâm sàng | Recall @ ngưỡng F2 | Ngưỡng quét trên OOF, ưu tiên độ nhạy |
| Bổ trợ | Precision, F1, F2, confusion matrix | |
| **Bắt buộc kèm** | **Bootstrap 95% CI** | Không có CI thì chênh 0,02 không kết luận được gì |

---

## 4. Cấu trúc thư mục sẽ tạo

```
fusion/
  build_dataset.py    # P1 — join 3 nguồn → căn 8030 hàng → chuẩn hóa → .pt
  build_graph.py      # P2 — kNN + cạnh thuộc tính + cầu nối + chẩn đoán đồ thị
  models.py           # P4 — ModalityEncoder chung + GAT/DGCNN/GT + baseline MLP/GCN
  train.py            # P3-5 — giao thức train chung, CLI --backbone --source --seed
  evaluate.py         # P5-6 — gộp kết quả, bootstrap CI, bảng 3×3, metric tách nhóm
  qc_report.py        # kiểm định tổng hợp (mirror imagefeat/qc_report.py)
  output/             # *.pt, *_results.json, *_manifest.json, bảng .md, hình .png
FUSION.ipynb          # notebook điều phối, chạy được cả Colab lẫn local
```

**Môi trường:** `torch` và `torch_geometric` chưa cài ở local → **mặc định chạy Colab T4**. Notebook viết theo kiểu dual-env như [`ĐỒ_ÁN.ipynb`](../ĐỒ_ÁN.ipynb).

```python
# Cell đầu FUSION.ipynb — cài đặt cho Colab
!pip install -q torch_geometric
# KHÔNG cài torch-cluster (hay lỗi biên dịch) — DGCNN sẽ dùng kNN tự cài, xem 4.3
```

---

## 5. Bảng theo dõi tiến độ

| Phase | Nội dung | Ai làm | Chặn? | Trạng thái |
|---|---|---|---|---|
| 0.1–0.2 | Tính `patient_group`, đo rò rỉ split hiện tại | TS | ✅ | ⬜ |
| 0.3 | Trích đặc trưng `frozen` (~2 phút) | Ảnh | ⬜ | ⬜ |
| 0.4 | Cross-fit đầy đủ (thay bản smoke) | Ảnh | ⬜ | ⬜ |
| **0.5** | **Re-split ts theo nhóm bệnh nhân + trích lại vector** | TS | ✅ | ⬜ |
| 1 | Dựng bộ dữ liệu node | Fusion | ✅ | ⬜ |
| 2 | Dựng đồ thị + chẩn đoán | Fusion | ✅ | ⬜ |
| 3 | Baselines B0/B1/B2 | Fusion | ✅ | ⬜ |
| 4 | 3 backbone trên nguồn `control` | Fusion | ✅ | ⬜ |
| 5 | Ma trận đầy đủ 3×3 | Fusion | ⬜ | ⬜ |
| 6 | Ablation + bằng chứng giả thuyết | Fusion | ⬜ | ⬜ |
| 7 | Chốt mô hình + bàn giao | Fusion | ⬜ | ⬜ |

> **Chặn = ✅** nghĩa là phase sau không bắt đầu được nếu phase này chưa xong.
> Phase 0.3 và 0.4 **chạy song song**, không nằm trên đường găng — bộ `control` đã đủ để chạy hết pipeline.

---

# PHASE 0 — Chuẩn bị đầu vào

## 0.1. Tính `patient_group` cho cả 8.030 bệnh nhân

**Mục tiêu:** có khóa nhóm bệnh nhân **nhất quán giữa 2 nhánh**, để chia dữ liệu không bị cùng một người nằm ở cả tập train lẫn test.

**Việc làm:** thêm 1 cell vào [`ĐỒ_ÁN.ipynb`](../ĐỒ_ÁN.ipynb), dùng **đúng công thức** của [`imagefeat/build_index.py`](../imagefeat/build_index.py) — sai một ký tự là hash lệch, hai nhánh mất đồng bộ.

```python
import hashlib, re, unicodedata

def strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", str(value))
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")

def normalize_name(value) -> str:
    """'Tô Văn Diệu' -> 'TO_VAN_DIEU'"""
    return re.sub(r"[^A-Z]+", "_", strip_accents(value).upper()).strip("_")

def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]

def normalize_namsinh(value) -> str:
    """CHỐNG LỆCH HASH: '1970', 1970, 1970.0 phải cho ra cùng một chuỗi."""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value).strip()

patient_group = (
    df_raw["hoten"].map(normalize_name) + "|" + df_raw["namsinh"].map(normalize_namsinh)
).map(stable_hash).values
```

⚠️ **Lấy `hoten`/`namsinh` từ `df_raw`**, không phải `df` — cell 1 của notebook đã drop 2 cột này.

⚠️ **`normalize_namsinh` là bắt buộc.** `info.csv` (nhánh ảnh) và Excel (nhánh ts) có thể đọc `namsinh` ra kiểu khác nhau (`1970` vs `1970.0`), làm hash lệch hoàn toàn. Bước 0.2 sẽ kiểm chứng.

**Nghiệm thu:**
- `len(np.unique(patient_group)) < 8030` (phải có người khám nhiều lần, nếu bằng đúng 8030 là công thức sai)

## 0.2. Kiểm chứng hash đồng bộ + đo rò rỉ của split hiện tại

**Mục tiêu:** (a) chứng minh `patient_group` hai nhánh khớp nhau; (b) lấy con số rò rỉ để đưa vào báo cáo phần "phát hiện và khắc phục".

```python
# (a) Đối chiếu với chỉ mục nhánh ảnh trên 1.835 bệnh nhân chung
img_index = pd.read_parquet("imagefeat/output/image_index.parquet")
ts_side = pd.DataFrame({"id": df_raw["id"].values, "pg_ts": patient_group})
chk = img_index.merge(ts_side, left_on="img_id", right_on="id", how="inner")
assert len(chk) == 1835, f"Join id thất bại: {len(chk)}/1835"
match = (chk["patient_group"] == chk["pg_ts"]).mean()
print(f"patient_group khớp giữa 2 nhánh: {match:.1%}")
assert match == 1.0, "Hash lệch — kiểm tra lại normalize_namsinh()"

# (b) Đo rò rỉ của split NGẪU NHIÊN hiện tại (trước khi sửa)
pg_dev, pg_test = set(patient_group[idx_dev]), set(patient_group[idx_test])
overlap = pg_dev & pg_test
n_leak_nodes = int(np.isin(patient_group[idx_test], list(overlap)).sum())
print(f"Bệnh nhân nằm ở CẢ dev và test : {len(overlap)}")
print(f"Node test bị ảnh hưởng          : {n_leak_nodes}/{len(idx_test)} "
      f"({100*n_leak_nodes/len(idx_test):.1f}%)")
```

**Nghiệm thu:** hash khớp **100%**. Ghi lại 2 con số rò rỉ vào [ts-pipeline-exp.md](ts-pipeline-exp.md).

## 0.3. Trích bộ đặc trưng `frozen` *(song song — không chặn)*

**Mục tiêu:** bộ đặc trưng thứ 3, ít bị collapse, để đối chứng. Chi phí ~2 phút GPU.

**Việc làm:** dùng [`extract_image_features.py`](../extract_image_features.py) với 3 thay đổi:
1. Đổi checkpoint sang backbone gốc `Pre-train BioViL-T/biovil_t_image_model_proj_size_128.pt` — **không load** `best_model_ce.pth`, không có head phân loại.
2. **Chỉ giữ 256 chiều đầu** (`X[:, :256]`) — 256 chiều sau là hằng số do BioViL-T là mô hình temporal chỉ nhận 1 ảnh.
3. Xuất theo **đúng schema** `image_features_control.parquet`: `img_id, has_image, fold_id(=-1), img_feat_0..255`, khóa theo `img_id` (không dùng tên file — PII).

**Nghiệm thu:** 1835 hàng · 256 chiều · **`(X.astype(np.float64).std(0) == 0).sum() == 0`** · 0 NaN · kèm manifest + QC như 2 bộ kia.

> ⚠️ Kiểm chiều hằng **phải dùng `float64`**. Ở `float32` phép kiểm này chỉ phát hiện 1/256 chiều chết — đúng cái bẫy đã bỏ lọt ở vòng bàn giao trước.

## 0.4. Cross-fit đầy đủ *(song song — không chặn)*

**Mục tiêu:** thay bản smoke 109 mẫu bằng bản đủ 1.835 mẫu.

**Vì sao bản smoke không dùng được:**

```
Phủ sóng đồ thị :  109/8030 = 1,36%   (bản đủ: 22,85%)
k=10 → số hàng xóm có ảnh trung bình: 0,14   (bản đủ: 2,29)
→ ~87% node không-ảnh có ĐÚNG 0 hàng xóm có ảnh → cơ chế "học ké" KHÔNG TỒN TẠI
Ca bnn dương trong nhóm có ảnh: 10 (bản đủ: 99) → không đo được gì có ý nghĩa thống kê
```

**Vì sao lần chạy trước máy nóng nhiều giờ** — 3 nguyên nhân trong mã, đều sửa được:

| # | Nguyên nhân | Sửa |
|---|---|---|
| 1 | [`crossfit_finetune.py`](../imagefeat/crossfit_finetune.py) đọc thẳng JPEG ~9 MP mỗi epoch (script cũ có bộ đệm `_cache512`, bản mới **làm mất**) | Resize sẵn 1835 ảnh về cạnh ngắn 512 **một lần**, train trên bản đã resize → nhanh hơn 15–20× |
| 2 | `--workers 0` → giải nén trên luồng chính, GPU đứng chờ CPU | `--workers 4` |
| 3 | `--batch-size 4` ở 448² → GPU chạy dưới 20% công suất | `--batch-size 16` (giảm còn 8 nếu hết VRAM) |

**Kiểm tra đầu tiên:** log phải in `Device=cuda`. Nếu là `Device=cpu` thì đó là toàn bộ nguyên nhân — fine-tune ResNet50 ở 448² trên CPU mất cả ngày.

```bash
# Chạy trên Colab T4, ~35-45 phút cho 5 fold
python imagefeat/crossfit_finetune.py --workers 4 --batch-size 16 --epochs 15
```

Nên xin thêm tùy chọn `--only-fold k` để chạy từng fold (~7 phút/lần), tránh mất trắng khi Colab ngắt phiên.

**Nghiệm thu:** manifest ghi `"samples": 1835`, `"patients": 1647`, `StratifiedGroupKFold(5)`; QC report có dòng "Bệnh nhân nằm ở >1 fold = 0".

---

# PHASE 0.5 — Re-split nhánh timeseries theo nhóm bệnh nhân ⚠️ TRÊN ĐƯỜNG GĂNG

**Vì sao phải làm:** [`ĐỒ_ÁN.ipynb`](../ĐỒ_ÁN.ipynb) hiện chia dev/test bằng `train_test_split` ngẫu nhiên và chia fold bằng `StratifiedKFold` — **cả hai đều không nhóm theo bệnh nhân**. Riêng trong 1.835 ảnh đã có 187 người khám nhiều lần. Nhánh ảnh đã chuyển sang `StratifiedGroupKFold`; nhánh ts phải đồng chuẩn, nếu không toàn bộ chỉ số fusion sẽ kế thừa lạc quan từ nhánh ts.

> **Chuẩn bị tinh thần:** split theo nhóm **khó hơn** split ngẫu nhiên. ROC-AUC/PR-AUC nhánh ts **nhiều khả năng giảm** so với 0,9255 / 0,4198. Đó là con số **thật hơn**, không phải mô hình kém đi. Cần viết rõ điều này trong báo cáo để không bị hiểu nhầm.

## 0.5.1–0.5.3. Đổi cách chia + assert cứng

Sửa **cell 5** của notebook:

```python
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_STATE = 42
groups = patient_group          # từ bước 0.1

# --- Dev/Test: lấy 1 trong 7 fold làm test (≈14,3% ≈ 15% cũ) ---
sgkf_outer = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=RANDOM_STATE)
idx_dev, idx_test = next(sgkf_outer.split(np.arange(len(df)), y, groups=groups))

# --- 5 fold bên trong Dev ---
sgkf_inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_id = np.full(len(df), -1, dtype=np.int8)
for k, (_, val_in) in enumerate(sgkf_inner.split(idx_dev, y[idx_dev], groups=groups[idx_dev])):
    fold_id[idx_dev[val_in]] = k

# --- ASSERT CỨNG: không có bệnh nhân nào bắc cầu giữa các tập ---
assert not (set(groups[idx_dev]) & set(groups[idx_test])), "Bệnh nhân bắc cầu dev/test!"
for k in range(5):
    in_k = set(groups[idx_dev][fold_id[idx_dev] == k])
    out_k = set(groups[idx_dev][fold_id[idx_dev] != k])
    assert not (in_k & out_k), f"Bệnh nhân bắc cầu fold {k}!"

print(f"Dev {len(idx_dev)} | Test {len(idx_test)} ({100*len(idx_test)/len(df):.1f}%)")
print(f"Ca bệnh — Dev: {y[idx_dev].sum()} | Test: {y[idx_test].sum()}")
```

**Nghiệm thu:** 2 assert pass · tỷ lệ test trong khoảng 13–16% · tập test có ít nhất 25 ca dương (nếu ít hơn, đổi `n_splits` và ghi lại lý do).

## 0.5.4–0.5.5. Train lại và trích lại vector

Thay `skf.split(idx_dev, y_dev)` bằng vòng lặp theo `fold_id` đã tính. **Giữ nguyên** mọi thứ khác: kiến trúc, class weights, `ReduceLROnPlateau`, `EarlyStopping(monitor="val_pr_auc")`, imputer/scaler fit trong fold, quét ngưỡng F2 trên OOF, trung bình 5 fold khi trích đặc trưng.

Model chỉ 11.809 tham số → chạy lại 5 fold mất vài phút.

**Nghiệm thu:** ghi đè `output/timeseries_features.parquet` (8030 × 33) + `.npy` (8030 × 32) · in bảng so sánh **số cũ vs số mới** để đưa vào báo cáo.

## 0.5.6. Xuất `fusion_node_meta.parquet`

**Đây là file bắt buộc để phase fusion bắt đầu.**

```python
split = np.array(["dev"] * len(df), dtype=object)
split[idx_test] = "test"

pd.DataFrame({
    "row_id"       : np.arange(len(df)),          # thứ tự hàng, khớp timeseries_features.parquet
    "id"           : df_raw["id"].values,         # KHÓA JOIN sang img_id của nhánh ảnh
    "file_name"    : file_names,
    "has_image"    : (pd.Series(file_names).str.lower() != "khong").astype(np.int8),
    "bnn"          : y.astype(np.int8),           # nhãn đích
    "split"        : split,                       # 'dev' / 'test'
    "fold_id"      : fold_id,                     # 0..4 cho dev, -1 cho test
    "patient_group": patient_group,               # để fusion kiểm tra lại + gom nhóm
    "tuoi"         : df["tuoi"].values,           # dựng cạnh thuộc tính
    "gioitinh"     : df["gioitinh"].values,
    "cviec"        : df["cviec"].values,
    "pxuong"       : df["pxuong"].values,
    "tuoinghe"     : df["tuoinghe"].values,
}).to_parquet("output/fusion_node_meta.parquet", index=False)
```

**Khóa join đã được kiểm chứng:** `extract_id(file_name)` (5 pattern trong [`build_index.py`](../imagefeat/build_index.py)) khớp cột `id` **433/433 = 100%**; nhãn `bnn` và `ketqua` giữa 2 nguồn khớp **433/433**.

**Nghiệm thu:** 8030 hàng · `has_image.sum() == 1835` · `(split=='test').sum()` khớp `len(idx_test)` · `fold_id` phủ đủ 0–4 trên dev.

## 0.5.7. Cập nhật tài liệu

Cập nhật [ts-overview.md](ts-overview.md) và [ts-pipeline-exp.md](ts-pipeline-exp.md) với số liệu mới, kèm đoạn giải thích vì sao đổi cách chia.

Nhân tiện sửa luôn các chỗ tài liệu đang **lệch với mã nguồn thật**:

| Tài liệu ghi | Mã nguồn thật |
|---|---|
| Bidirectional LSTM + Masking | `LSTM(32)` thường, không Masking |
| Activation Swish | `relu` |
| `Embedding(10, 4)` → Flatten 16 | `Embedding(10, 8)` → Flatten 32 |
| Vector ảnh ~2048-D | 512-D → **256 chiều sống** |

**GATE PHASE 0.5** — không qua thì không sang Phase 1:
- [ ] 2 assert nhóm bệnh nhân pass
- [ ] `output/fusion_node_meta.parquet` tồn tại, đúng schema
- [ ] `output/timeseries_features.parquet` đã tạo lại từ split mới
- [ ] Bảng so sánh số cũ/mới đã ghi vào tài liệu

---

# PHASE 1 — Dựng bộ dữ liệu node (`fusion/build_dataset.py`)

## 1.1. Join 4 nguồn

```python
ts    = pd.read_parquet("output/timeseries_features.parquet")          # 8030 × 33
meta  = pd.read_parquet("output/fusion_node_meta.parquet")             # 8030 × 13
index = pd.read_parquet("imagefeat/output/image_index.parquet")        # 1835 × 6
img   = pd.read_parquet(f"imagefeat/output/image_features_{source}.parquet")  # 1835 × 259

# TRÁNH ĐỤNG TÊN: fold_id của nhánh ảnh ≠ fold_id của nhánh ts
img = img.rename(columns={"fold_id": "img_fold_id", "has_image": "_img_has"})

node = meta.merge(index, left_on="id", right_on="img_id", how="left") \
           .merge(img,   left_on="id", right_on="img_id", how="left")
```

## 1.2. Assertions cứng — thà chết sớm còn hơn sai âm thầm

```python
N = 8030
assert len(ts) == len(meta) == N,                    "Lệch số hàng"
assert (ts["file_name"].values == meta["file_name"].values).all(), "Lệch thứ tự hàng"
assert meta["has_image"].sum() == 1835,              "Sai số node có ảnh"
assert node["img_id"].notna().sum() == 1835,         "Join id thất bại"

# Nhãn từ 2 nhánh phải khớp trên 1835 hàng chung
m = node["img_id"].notna()
assert (node.loc[m, "bnn"] == node.loc[m, "label_bnn"]).all(), "Nhãn bnn 2 nhánh lệch nhau!"

# patient_group 2 nhánh phải khớp (kiểm lại lần 2, phòng khi file được tạo lại)
assert (node.loc[m, "patient_group_x"] == node.loc[m, "patient_group_y"]).all(), "Hash lệch!"

# Đặc trưng ảnh phải sạch
F = node.filter(like="img_feat_").to_numpy(np.float64)
assert not np.isnan(F[m.values]).any() and not np.isinf(F[m.values]).any(), "NaN/Inf ở hàng có ảnh"
assert (F[m.values].std(0) == 0).sum() == 0, "Còn chiều hằng số"   # NHỚ: float64

# Split phải nhất quán
assert not (set(node.loc[node.split=="dev", "patient_group_x"]) &
            set(node.loc[node.split=="test","patient_group_x"])), "Bệnh nhân bắc cầu dev/test"
```

## 1.3. Chuẩn hóa từng modality

Hai vector **lệch nhau cả thang đo lẫn số chiều** — đã đo: ts là 32-D đầu ra ReLU nằm trong [0; 4,73]; ảnh là 256-D có cả âm lẫn dương, per-dim std chỉ 0,04.

```python
def prepare_features(node, train_mask):
    """Fit scaler CHỈ trên node train của fold hiện tại. Trả về ts, img đã chuẩn hóa."""
    # --- Timeseries: standardize ---
    ts_scaler = StandardScaler().fit(X_ts[train_mask])
    X_ts_z = ts_scaler.transform(X_ts)

    # --- Ảnh: MEAN-CENTER TRƯỚC, rồi standardize ---
    has = node["has_image"].values.astype(bool)
    fit_rows = train_mask & has                      # chỉ node train VÀ có ảnh
    img_mean = X_img[fit_rows].mean(0)
    X_img_c  = X_img - img_mean                      # bước bắt buộc, xem cảnh báo dưới
    img_scaler = StandardScaler().fit(X_img_c[fit_rows])
    X_img_z = img_scaler.transform(X_img_c)
    X_img_z[~has] = np.nan                           # KHÔNG điền 0 — để model xử lý bằng missing-token
    return X_ts_z, X_img_z
```

> ⚠️ **Mean-center là bắt buộc, không phải tùy chọn.** QC nhánh ảnh đo được cosine trung bình giữa các ảnh là **0,98** ở dạng thô, chỉ còn **0,002** sau khi mean-center. Bỏ qua bước này thì mọi ảnh "giống nhau 98%" và **đồ thị kNN trở nên vô nghĩa**.

> ⚠️ **Không điền 0 cho node thiếu ảnh.** Điền 0 tạo ra một điểm giả ở gốc tọa độ mà 6.195 node cùng chia sẻ. Để `NaN` thì nếu ai đó quên xử lý, code sẽ **crash ngay** thay vì cho ra kết quả sai âm thầm.

## 1.4. Xuất dataset

`fusion/output/fusion_dataset_{source}.pt` chứa: `X_ts` (8030×32 thô), `X_img` (8030×256 thô), `has_image`, `y`, `split`, `fold_id`, `patient_group`, `batch_date`, cùng các cột thuộc tính để dựng cạnh. **Chuẩn hóa làm trong vòng lặp fold**, không lưu sẵn bản đã chuẩn hóa.

Kèm `fusion_dataset_{source}_manifest.json` theo format của `imagefeat/`.

**GATE PHASE 1:**
- [ ] Toàn bộ assertion ở 1.2 pass
- [ ] Bảng thống kê: tỷ lệ bệnh nhóm có ảnh (kỳ vọng 5,40%) vs không ảnh (1,81%)
- [ ] Phân bố `batch_date` (18 lô) đã in ra

---

# PHASE 2 — Dựng đồ thị (`fusion/build_graph.py`)

## 2.1–2.2. Cạnh kNN theo tương đồng lâm sàng

```python
def knn_edges(X_ts_z, k=10):
    """kNN cosine trên vector TIMESERIES đã chuẩn hóa."""
    Xn = X_ts_z / np.linalg.norm(X_ts_z, axis=1, keepdims=True)
    sim = Xn @ Xn.T                              # 8030² fp32 ≈ 258 MB — thoải mái trên T4
    np.fill_diagonal(sim, -np.inf)               # bỏ self-loop
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    src = np.repeat(np.arange(len(Xn)), k)
    dst = idx.reshape(-1)
    w   = sim[src, dst]
    # Đối xứng hóa (union) + khử trùng lặp
    e = np.unique(np.sort(np.stack([src, dst]), axis=0), axis=1)
    return e, w
```

> ⚠️ **Dựng cạnh trên timeseries, KHÔNG trên vector ghép.** 6.195 node không ảnh có nửa vector giống hệt nhau (cùng missing-token) → nếu tính khoảng cách trên vector ghép, chúng **tự dính chùm thành một cụm**. Đó chính là hiện tượng ta muốn *quan sát ở DGCNN*; nếu đưa nó vào đồ thị tĩnh thì GAT và Graph Transformer cũng nhiễm, và phép so sánh 3 backbone mất ý nghĩa.

Quét `k ∈ {5, 10, 20}` ở Phase 6.

## 2.3. Cạnh thuộc tính *(cờ bật/tắt)*

Theo tinh thần bài báo tham khảo — đưa tri thức nghiệp vụ vào đồ thị:

```python
# Nối 2 bệnh nhân nếu CÙNG cấp độ nghề VÀ chênh lệch tuổi ≤ 5
same_job = cviec[:, None] == cviec[None, :]
close_age = np.abs(tuoi[:, None] - tuoi[None, :]) <= 5
attr_edges = np.argwhere(np.triu(same_job & close_age, k=1)).T
```

Số cạnh có thể rất lớn → giới hạn số cạnh thuộc tính mỗi node (ví dụ tối đa 5, chọn ngẫu nhiên có seed).

## 2.4. Cầu nối modality *(cờ bật/tắt — đóng góp riêng của đồ án)*

```python
def bridge_edges(X_ts_z, has_image):
    """Mỗi node KHÔNG ảnh nối thêm 1 cạnh tới node CÓ ảnh gần nhất.
    Đảm bảo 100% node không-ảnh chạm được nguồn thông tin ảnh."""
    Xn = X_ts_z / np.linalg.norm(X_ts_z, axis=1, keepdims=True)
    imaged = np.where(has_image)[0]
    sim = Xn[~has_image] @ Xn[imaged].T
    nearest = imaged[sim.argmax(1)]
    return np.stack([np.where(~has_image)[0], nearest])
```

Đây là can thiệp trực tiếp vào vấn đề trung tâm: mặc định mỗi node không-ảnh chỉ có **trung bình 2,29** hàng xóm có ảnh (k=10) và một phần bị bỏ rơi hoàn toàn; cầu nối nâng con số đó lên **100% phủ sóng**. Bật/tắt cầu nối là một ablation rất mạnh cho phần bảo vệ.

## 2.5. Nguyên tắc tuyệt đối

> **Cạnh chỉ được dựng từ ĐẶC TRƯNG, tuyệt đối không từ NHÃN.** Không bao giờ nối 2 node vì cùng `bnn`. Đây là dạng rò rỉ nghiêm trọng nhất trong GNN y tế và rất khó phát hiện sau khi đã xảy ra.

## 2.6. Chẩn đoán đồ thị — 4 chỉ số bắt buộc

```python
def diagnose(edge_index, y, has_image):
    src, dst = edge_index
    p = y.mean()

    homophily = (y[src] == y[dst]).mean()
    baseline  = p**2 + (1-p)**2                 # mốc ngẫu nhiên
    cross_mod = (has_image[src] != has_image[dst]).mean()

    # Phủ sóng: bao nhiêu % node không-ảnh có ≥1 hàng xóm CÓ ảnh
    cnt = np.zeros(len(y))
    np.add.at(cnt, dst, has_image[src]); np.add.at(cnt, src, has_image[dst])
    coverage = (cnt[~has_image] > 0).mean()

    return {"homophily": homophily, "homophily_baseline": baseline,
            "cross_modality_edge_ratio": cross_mod,
            "isolated_node_coverage": coverage,
            "mean_degree": 2*len(src)/len(y)}
```

| Chỉ số | Ý nghĩa | Ngưỡng |
|---|---|---|
| **Label homophily** | % cạnh nối 2 node cùng nhãn | Phải **> mốc ngẫu nhiên** |
| **Cross-modality edge ratio** | % cạnh nối có-ảnh ↔ không-ảnh | Báo cáo; với DGCNN đo **từng layer** |
| **Isolated-node coverage** | % node không-ảnh có ≥1 hàng xóm có ảnh | Càng cao càng tốt; bật cầu nối → 100% |
| **Mean degree** | Bậc trung bình | Kiểm tra đồ thị không quá thưa/dày |

**GATE PHASE 2 — quan trọng nhất của cả roadmap:**
- [ ] `homophily > homophily_baseline`

> Nếu homophily **không** cao hơn mốc ngẫu nhiên thì **dừng lại ngay**. Điều đó có nghĩa các bệnh nhân "giống nhau về lâm sàng" không hề có xu hướng cùng nhãn — GNN sẽ **không thể** vượt MLP dù dùng backbone nào. Phải dựng lại cạnh (đổi k, đổi không gian tương đồng, thêm cạnh thuộc tính) trước khi train. Bước kiểm tra này tốn 1 phút và có thể cứu bạn nhiều ngày.

---

# PHASE 3 — Baselines (chạy TRƯỚC khi đụng tới GNN)

| Mã | Mô hình | Số ô | Mục đích |
|---|---|---|---|
| **B0** | MLP chỉ dùng ts 32-D | 1 | Tái lập kết quả nhánh ts → xác nhận pipeline fusion không làm hỏng gì |
| **B1** | Hồi quy logistic trên đặc trưng ảnh (chỉ 1835 node có ảnh) | 3 | Nguồn ảnh nào mạnh nhất, đo độc lập với đồ thị |
| **B2** | MLP trên [ts ⊕ ảnh], **không đồ thị** | 3 | **Ô then chốt** |

Dùng **đúng** `fold_id` và `split` từ `fusion_node_meta.parquet` — không tự chia lại.

**GATE PHASE 3:**
- [ ] B0 ra PR-AUC xấp xỉ kết quả nhánh ts sau re-split (lệch nhiều ⇒ pipeline fusion có bug)
- [ ] **B2 > B0** — nếu không, ảnh **không** đóng góp gì; dừng và điều tra trước khi train GNN

> B2 là ô mà cả 3 backbone GNN **bắt buộc phải vượt**. Nếu GNN không vượt được B2, kết luận trung thực là *"đồ thị không đóng góp trên bộ dữ liệu này"* — vẫn là một kết luận khoa học hợp lệ và đáng báo cáo, nhưng phải biết sớm.

---

# PHASE 4 — Ba backbone trên nguồn chính (`control`)

## 4.1. `ModalityEncoder` dùng chung

Cả 3 backbone dùng **chung** encoder và head — khác biệt duy nhất là cơ chế truyền tin. Đó là điều kiện để phép so sánh công bằng.

```python
class ModalityEncoder(nn.Module):
    """ts 32-D + ảnh 256-D (có thể thiếu) -> 129-D"""
    def __init__(self, d_ts=32, d_img=256, d=64, p_drop=0.2):
        super().__init__()
        self.proj_ts  = nn.Sequential(nn.Linear(d_ts,  d), nn.BatchNorm1d(d), nn.SiLU())
        self.proj_img = nn.Sequential(nn.Linear(d_img, d), nn.BatchNorm1d(d), nn.SiLU())
        self.missing_token = nn.Parameter(torch.zeros(d))   # HỌC ĐƯỢC, thay cho vector 0
        self.p_drop = p_drop

    def forward(self, x_ts, x_img, has_img):
        h_ts = self.proj_ts(x_ts)
        h_img = self.proj_img(torch.nan_to_num(x_img))      # NaN -> 0 trước khi qua Linear
        m = has_img.float()
        if self.training and self.p_drop > 0:               # MODALITY DROPOUT
            m = m * (torch.rand_like(m) > self.p_drop).float()
        h_img = m[:, None] * h_img + (1 - m[:, None]) * self.missing_token
        return torch.cat([h_ts, h_img, m[:, None]], dim=1)  # 64 + 64 + 1 = 129
```

Ba lớp phòng vệ chống shortcut `has_image` (nhóm có ảnh có tỷ lệ bệnh gấp 3 lần):

1. **Missing-token học được** — model biểu diễn "thiếu ảnh" như một trạng thái có nghĩa, thay vì bịa ra một điểm giả ở gốc tọa độ.
2. **Modality dropout `p=0.2`** — ngẫu nhiên "giấu" ảnh của node có ảnh khi train, model không được phép chỉ dựa vào *sự tồn tại* của ảnh.
3. **Cờ `has_image` đưa vào công khai** — để model không phải suy đoán ngầm, và để ta có thể **tắt nó đi** trong ablation. Nếu tắt mà điểm không tụt ⇒ model đang học nội dung ảnh thật.

## 4.2. GAT — đồ thị tĩnh

```python
from torch_geometric.nn import GATv2Conv
# GATv2 chứ không phải GAT gốc: attention của GAT gốc là "tĩnh", đã được chứng minh
# hạn chế về khả năng biểu diễn (Brody et al., "How Attentive are GATs?", ICLR 2022)
self.convs = nn.ModuleList([GATv2Conv(129, 32, heads=4, dropout=0.2),
                            GATv2Conv(128, 32, heads=4, dropout=0.2)])
```

## 4.3. DGCNN — kNN động, tự cài để né `torch-cluster`

```python
def knn_graph_dense(x, k, chunk=1024):
    """kNN trong không gian đặc trưng, tính lại mỗi layer.
    Tự cài bằng torch.cdist để KHÔNG phụ thuộc torch-cluster
    (thư viện này hay lỗi biên dịch trên Colab)."""
    N = x.size(0)
    idx = []
    for s in range(0, N, chunk):
        d = torch.cdist(x[s:s+chunk], x)
        rows = torch.arange(d.size(0), device=x.device)
        d[rows, torch.arange(s, min(s+chunk, N), device=x.device)] = float("inf")
        idx.append(d.topk(k, largest=False).indices)
    idx = torch.cat(idx, 0)
    row = torch.arange(N, device=x.device).repeat_interleave(k)
    return torch.stack([idx.reshape(-1), row], dim=0)      # chiều truyền tin: col -> row
```

**Bắt buộc log ở mỗi layer:** `cross_modality_edge_ratio` của đồ thị động.

> Đây là **bằng chứng định lượng cho giả thuyết của bạn**. Nếu tỷ lệ này **tụt dần qua các layer**, bạn đã chứng minh bằng số hiện tượng "DGCNN tự phân cụm thành 2 nhóm cô lập có-ảnh / không-ảnh" — biến một phỏng đoán thành một phát hiện có dữ liệu.

## 4.4. Graph Transformer — attention toàn cục

```python
from torch_geometric.nn import GPSConv, GINEConv
# Attention đầy đủ 8030² × 4 head ở fp32 ≈ 1 GB chỉ riêng ma trận attention
# -> backward sẽ OOM trên T4. Dùng linear attention (Performer): O(N) thay vì O(N²),
#    vẫn giữ đúng tính chất "mọi node nối được tới mọi node".
GPSConv(channels=128, conv=GINEConv(...), heads=4,
        attn_type="performer", attn_kwargs={"dim_head": 32})
```

Nếu phiên bản PyG không có `attn_type="performer"`: dự phòng bằng `TransformerConv` trên đồ thị thưa + một nhóm nhỏ **virtual node** (8–16 node ảo nối tới mọi node) để mô phỏng đường đi toàn cục với chi phí O(N).

## 4.5. Giao thức huấn luyện chung

| Thành phần | Thiết lập |
|---|---|
| Optimizer | AdamW, lr 1e-3, weight decay 1e-4, cosine schedule |
| Loss | Focal Loss (γ=2) hoặc weighted BCE — chọn 1 và dùng chung cho mọi ô |
| Early stopping | theo **val PR-AUC**, patience 30 |
| Chia dữ liệu | dùng đúng `fold_id`/`split` từ `fusion_node_meta.parquet` |
| Seed | 3 seed: 42, 1337, 2024 |
| Ngưỡng quyết định | quét **F2 trên OOF** (nhất quán với nhánh ts) |
| Transductive | forward **toàn bộ** 8030 node + mọi cạnh; chỉ **che nhãn** val/test khi tính loss |

> Transductive không phải là rò rỉ: đặc trưng của mọi node đều có sẵn từ đầu (không phụ thuộc nhãn), chỉ nhãn bị che. Đây là chuẩn của population graph.

**GATE PHASE 4:**
- [ ] Cả 3 backbone ≥ B2 trên OOF PR-AUC
- [ ] Nếu không: vấn đề nằm ở **đồ thị hoặc đặc trưng**, không phải ở backbone — quay lại Phase 2

---

# PHASE 5 — Ma trận đầy đủ 3 nguồn × 3 backbone

Lặp Phase 4 cho `frozen` và `crossfit`. Bảng kết quả chính:

| Backbone | `control` | `frozen` | `crossfit` |
|---|---|---|---|
| GAT | PR-AUC ± std | | |
| DGCNN | | | |
| Graph Transformer | | | |
| *(B2 — MLP không đồ thị)* | | | |

Mỗi ô: mean ± std qua 3 seed, đo trên **OOF**. Cột test chỉ điền ở Phase 7.

---

# PHASE 6 — Ablation & bằng chứng cho giả thuyết

## 6.1. Ablation (chạy trên cấu hình tốt nhất của Phase 5)

| Yếu tố | Giá trị quét |
|---|---|
| k của kNN | 5 · 10 · 20 |
| Cạnh thuộc tính | bật / tắt |
| **Cầu nối modality** | bật / tắt |
| Xử lý thiếu ảnh | missing-token / zero-padding |
| Modality dropout | bật / tắt |
| Cờ `has_image` | có / không |
| Số layer | 2 · 3 · 4 |

## 6.2. Bảng chứng minh giả thuyết ⭐

**Đây là bảng quan trọng nhất của cả đồ án:**

| Backbone | PR-AUC nhóm **có ảnh** | PR-AUC nhóm **không ảnh** |
|---|---|---|
| B2 (không đồ thị) | | |
| GAT | | |
| DGCNN | | |
| Graph Transformer | | |

**Cách đọc:** nếu Graph Transformer thắng chủ yếu nhờ nâng điểm ở **cột phải (nhóm không ảnh)** thì cơ chế *"học ké từ mọi bệnh nhân có ảnh"* được **xác nhận bằng dữ liệu**. Nếu nó chỉ thắng ở cột trái thì giả thuyết **bị bác bỏ** — và đó cũng là một kết quả khoa học hợp lệ, phải báo cáo trung thực.

## 6.3. Metric tách theo lô chụp

Tỷ lệ dương lệch **8 lần** giữa các lô (`20191115`: 5,5% → `20181223`: 44,8%), và embedding ảnh đoán được lô chụp với độ chính xác **44,7%** (ngẫu nhiên 7,7%). Cần báo cáo metric tách theo `batch_date` để chứng minh mô hình không chỉ đang nhận diện máy chụp.

---

# PHASE 7 — Chốt mô hình & bàn giao

## 7.1. Đánh giá cuối trên test — chạm đúng 1 lần

```python
def bootstrap_ci(y, p, metric, n=2000, seed=42):
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(metric(y[i], p[i]))
    return np.percentile(vals, [2.5, 97.5])
```

Quy trình: chọn cấu hình bằng **OOF PR-AUC** → ensemble 5 fold model → dự đoán test **một lần** → báo cáo kèm **CI 95%**.

## 7.2. Chọn mô hình cuối — không chỉ nhìn điểm số

| Backbone | Chi phí suy luận 1 ca mới |
|---|---|
| GAT | O(k) — chỉ cần k hàng xóm |
| DGCNN | O(N) — tính lại kNN mỗi layer |
| Graph Transformer | O(N) — cần **toàn bộ quần thể** trong bộ nhớ |

> Nếu Graph Transformer chỉ hơn GAT **< 0,02 PR-AUC** (nằm trong CI), thì với tư duy sản phẩm **GAT là lựa chọn nên chốt**. Một kết luận cân nhắc cả chất lượng lẫn chi phí triển khai có giá trị hơn nhiều so với việc chọn mô hình đứng đầu bảng.

## 7.3. Hàm suy luận cho ca mới

Population graph là transductive → cần lưu lại "quần thể tham chiếu":

```
Đầu vào: ảnh X-quang mới (nếu có) + phiếu khám mới
  1. Phiếu khám  -> pipeline ts        -> vector 32-D
  2. Ảnh         -> BioViL-T           -> vector 256-D   (không ảnh -> missing-token)
  3. Tính kNN từ vector 32-D tới ma trận 8030 × 32 đã lưu
  4. Chèn node mới + k cạnh vào đồ thị -> forward -> xác suất
  5. So với ngưỡng F2 đã chốt          -> kết luận
```

Cần lưu kèm checkpoint: ma trận ts 8030×32, các scaler, `img_mean`, ngưỡng quyết định, và cấu hình đồ thị.

## 7.4. Hoàn tất tài liệu

- [ ] Cập nhật [X-ray-overview.md](X-ray-overview.md): vector ảnh là **512-D → 256 chiều sống**, không phải 2048-D; backbone chốt và nhãn huấn luyện là `ketqua`, không phải `bnn`
- [ ] Ghi rõ trong báo cáo: **không được đặt AUC nhánh ảnh cạnh AUC nhánh ts trong cùng một bảng** — hai bài toán khác nhau (`ketqua` 25,2% vs `bnn` 5,4%)
- [ ] Viết mục "lệch có chủ đích so với bài báo tham khảo" (mục 1.1)

---

# 8. Rủi ro & nguyên tắc bất di bất dịch

| # | Rủi ro | Cách chặn |
|---|---|---|
| 1 | **Rò rỉ xuyên giai đoạn** | Fusion dùng đúng `split`/`fold_id` từ `fusion_node_meta.parquet`, **không tự chia lại** |
| 2 | Bệnh nhân bắc cầu train/test | Phase 0.5 re-split theo `patient_group` + assert cứng |
| 3 | Shortcut `has_image` | Missing-token + modality dropout + ablation tắt cờ |
| 4 | **Anisotropy đặc trưng ảnh** (cosine thô 0,98) | **Bắt buộc mean-center trước kNN** |
| 5 | Collapse đặc trưng ảnh (PC1 90,7%, eff. rank 1,80) | Có bộ `frozen` đối chứng; không nén xuống < 32 chiều |
| 6 | Nhiễu loạn lô chụp | Báo cáo metric tách theo `batch_date` |
| 7 | Đồ thị chất lượng kém | Gate Phase 2: homophily phải > mốc ngẫu nhiên |
| 8 | Overfit test do so 19 ô | Chọn bằng OOF, chạm test 1 lần, kèm bootstrap CI |
| 9 | Kiểm chiều hằng bằng `float32` | **Luôn dùng `float64`** — `float32` chỉ phát hiện 1/256 chiều chết |
| 10 | Nhánh `origin/fusion` đang chạy sai bài toán | 1475 node / 512 chiều còn chiều chết / nhãn `ketqua` — **thống nhất với nhóm**, không dùng số liệu đó |

---

# 9. Phụ lục — Checklist nghiệm thu nhanh

```
PHASE 0.5  □ 2 assert nhóm bệnh nhân pass
           □ fusion_node_meta.parquet đúng schema (có cột `id`!)
           □ timeseries_features.parquet đã tạo lại
PHASE 1    □ Mọi assertion join/nhãn/NaN/chiều-hằng pass
           □ Bảng tỷ lệ bệnh theo nhóm có/không ảnh
PHASE 2    □ homophily > mốc ngẫu nhiên          ← GATE QUAN TRỌNG NHẤT
           □ 4 chỉ số chẩn đoán đã in
PHASE 3    □ B0 ≈ kết quả nhánh ts
           □ B2 > B0                              ← GATE
PHASE 4    □ Cả 3 backbone ≥ B2
PHASE 5    □ Bảng 3×3 đầy đủ, mean ± std 3 seed
PHASE 6    □ Bảng tách nhóm có ảnh / không ảnh    ← BẰNG CHỨNG GIẢ THUYẾT
PHASE 7    □ Test chạm đúng 1 lần, kèm CI 95%
           □ Hàm inference + checkpoint + tài liệu
```
