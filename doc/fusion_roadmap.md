# Roadmap Fusion v2 — nhãn `ketqua` (cập nhật 27/09/2026)

> Bản v1 (nhãn `bnn`, 3 nguồn ảnh) đã lỗi thời: nhãn đổi, bộ `crossfit` bị loại.
> Xem lịch sử git nếu cần đối chiếu.
>
> **Ưu tiên của bản này: đi tới kết quả fusion nhanh nhất, bằng đường ngắn nhất
> mà vẫn trung thực.** Không mở rộng phạm vi trước khi có một kết quả hoàn chỉnh.

---

## 1. Những gì đã chốt

| Hạng mục | Chốt | Ghi chú |
|---|---|---|
| **Nhãn đích** | `ketqua` (1 = bụi phổi theo đọc phim ILO) | 961/8.030 ca dương (12,0%); riêng nhóm có ảnh 462/1.835 (25,2%) |
| **Nhãn cũ `bnn`** | Chỉ để đối chiếu | Là **tiền sử** đã được công nhận: 97/99 ca chẩn đoán trước đợt khám |
| **Nguồn đặc trưng ảnh** | `control`, `frozen` | `crossfit` đã xoá: 5 fold nằm ở 5 không gian vector khác nhau |
| **Quần thể** | 8.030 node, 1.835 có ảnh | `ketqua` có đủ cho cả 8.030 người |
| **Chia dữ liệu** | Theo `patient_uid`, test 1.148 người | Cố định, phân tầng theo `bnn` để hai nhãn dùng chung tập test |
| **Chỉ số** | ROC-AUC + PR-AUC, kèm bootstrap KTC 95% theo bệnh nhân | PR-AUC phụ thuộc tỷ lệ bệnh nên luôn in kèm tỷ lệ nền |

### Đầu vào đã sẵn sàng

```
output/ketqua/timeseries_features.parquet   (8030, 34)   ĐỒ_ÁN.ipynb, TARGET="ketqua"
output/ketqua/fusion_node_meta.parquet      (8030, 13)   id, ketqua, split, fold_id, patient_uid
output/ketqua/ts_predictions.parquet        (8030, 7)    điểm ngoài mẫu của nhánh ts
imagefeat/output/image_features_control.parquet  (1835, 259)
imagefeat/output/image_features_frozen.parquet   (1835, 259)
fusion/output/fusion_dataset_ketqua_{control,frozen}.npz    9/9 kiểm định đạt
fusion/output/tabular_ketqua.npz            (8030, 139)  đặc trưng lâm sàng dạng bảng
fusion/output/graph_ketqua_{k5,k10_bridge,k20_bridge}.npz
```

---

## 2. Mốc phải vượt — đã đo, không phải ước lượng

### 2.1. Trên 1.835 người **có ảnh** (CV 5 fold nhóm bệnh nhân, tỷ lệ bệnh 0,252)

| Mô hình | ROC-AUC | PR-AUC |
|---|---|---|
| Nhánh ts (Keras) | 0,641 [0,608–0,669] | 0,388 |
| Ảnh `control` (probe) | 0,761 [0,736–0,786] | 0,544 |
| Ảnh `frozen` (probe) | 0,759 [0,733–0,784] | 0,533 |
| **Late fusion (ts + ảnh)** | **0,766 [0,739–0,791]** | **0,551** |
| *Mốc chỉ dùng tuổi* | *0,636* | *0,358* |

→ **Mốc GNN phải vượt: ROC-AUC 0,766.** Hồ sơ lâm sàng gần như không thêm gì cho nhóm
này (late fusion chỉ hơn ảnh đơn thuần +0,005).

### 2.2. Trên 6.195 người **không có ảnh** (tỷ lệ bệnh 0,081) ⭐

Đây mới là nơi đồ thị có đất diễn, và **giả thuyết trung tâm của đồ án đã được chứng
minh bằng một phép thử thô sơ**: gán cho mỗi người không ảnh điểm ảnh trung bình của
10 hàng xóm có ảnh gần nhất (theo đặc trưng ts).

| Nguồn thông tin | ROC-AUC | PR-AUC |
|---|---|---|
| Chỉ lâm sàng (nhánh ts) | 0,617 | 0,127 |
| **Chỉ "mượn" điểm ảnh của 10 hàng xóm** | **0,629** | 0,127 |
| **Gộp cả hai** | **0,640** | 0,132 |

**Gộp − chỉ lâm sàng = +0,023 ROC-AUC, KTC 95% [+0,007; +0,041], P = 100%.**

Thông tin ảnh truyền qua hàng xóm **thật sự giúp người không có phim**, và tự nó còn
mạnh hơn hồ sơ lâm sàng của chính họ. Trung bình cộng thô sơ đã làm được vậy — GNN có
attention học được trọng số hàng xóm thì tối thiểu phải bằng, kỳ vọng hơn.
**Đây là câu chuyện chính của đồ án.**

### 2.3. Cảnh báo: đồ thị yếu hơn nhiều so với thời nhãn `bnn`

| | Nhãn `bnn` | Nhãn `ketqua` |
|---|---|---|
| Homophily lớp dương (k10_bridge) | 0,4113 | 0,1501 |
| Tỷ lệ nền | 0,0263 | 0,1197 |
| **Lift** | **15,65×** | **1,25×** |

Lý do: cạnh dựng từ đặc trưng ts, mà ts chỉ đạt ROC 0,64 với `ketqua`. Lift tối đa có
thể đạt cũng chỉ là 1/0,12 ≈ 8,4×. Đã thử các cách dựng cạnh khác trên nhóm có ảnh:

| Cạnh dựng từ | Lift lớp dương |
|---|---|
| ts 32 chiều | 1,46× |
| ảnh 256 chiều | 1,63× |
| **ts + ảnh ghép** | **1,72×** |

*(Số 1,25× đo lại sau khi khử rò rỉ đặc trưng ts; trước đó là 1,73× — phần chênh là ảo.)*

→ Gate Phase 2 vẫn qua ở cả 3 cấu hình, nhưng **đừng kỳ vọng GNN tạo đột phá từ cấu
trúc đồ thị**. Giá trị nằm ở kênh truyền ảnh sang người không ảnh (mục 2.2).

---

## 3. Kế hoạch — 4 bước, mỗi bước có cổng

Nguyên tắc: **một cấu hình chạy được trước, mở rộng sau.** Mỗi bước phải cho ra một con
số báo cáo được, không bước nào chỉ để "chuẩn bị".

### Phase 3 — Baselines chuẩn hoá `[1–2 giờ, chạy được ở local]`

Mục tiêu: đưa cả 4 mốc về **cùng một đường ống, cùng cách chia, cùng cách đo** để so với
GNN cho công bằng. Hiện chúng đang đến từ ba script khác nhau.

| Mã | Mô hình | Đầu vào |
|---|---|---|
| **B0** | MLP | lâm sàng: bảng thô 139 cột + điểm nhánh ts |
| **B1** | MLP | chỉ `X_img` (256 chiều), chỉ 1.835 node có ảnh |
| **B2** | MLP | lâm sàng ⊕ ảnh, node không ảnh dùng token thiếu học được |
| **B3** | MLP | B2 + đặc trưng hàng xóm gộp sẵn (trung bình 10 hàng xóm) — **không có GNN** |

B3 quan trọng: nó chính là phép thử thô ở mục 2.2, đưa vào đường ống chuẩn. **Nếu GNN
không vượt được B3 thì GNN không đáng dùng** — kết luận đó cũng phải báo trung thực.

**Cổng:** B2 ≥ B1 trên nhóm có ảnh, và B3 > B0 trên nhóm không ảnh.

### Phase 4 — GAT, một backbone duy nhất `[2–3 giờ]`

Chỉ làm GAT trước. GATv2Conv 2 lớp, đồ thị `graph_ketqua_k10_bridge`, node feature =
`X_ts ⊕ X_img` với missing-token, huấn luyện 5 fold × 3 seed trên tập dev.

**Cổng:** GAT > B3 trên nhóm không ảnh (đây là lý do tồn tại của GNN trong đồ án này).

Nếu trượt cổng: dừng, không chạy tiếp 2 backbone kia, báo cáo kết quả âm tính kèm lý do
(homophily 1,73×) — đó là một kết luận khoa học hợp lệ và giải thích được.

### Phase 5 — Mở rộng, chỉ khi Phase 4 qua cổng `[3–4 giờ]`

- DGCNN và Graph Transformer, cùng giao thức.
- Ablation gọn: `frozen` thay `control`; bỏ cạnh cầu nối; bỏ cờ `has_image`.
- Thử đồ thị cạnh lai (ảnh–ảnh dùng tương đồng ảnh) — lift 1,72× so với 1,46×.

### Phase 6 — Chốt `[1 giờ]`

- Chọn cấu hình tốt nhất **bằng OOF**, sau đó **chạm test đúng một lần**.
- Bảng bằng chứng chính: kết quả **tách riêng nhóm có ảnh / không ảnh**, kèm KTC 95%.
- Hàm suy luận cho ca mới + tài liệu bàn giao.

### Nơi chạy — **Colab GPU** (máy local yếu)

Phase 3 trở đi chạy trên Colab. Để Colab chỉ cần `git clone` là có đủ dữ liệu, các
artifact dẫn xuất sau được **commit vào repo** (tổng ~7 MB, không chứa thông tin cá
nhân — `patient_uid` là mã băm 16 ký tự, không có tên hay số điện thoại):

```
output/ketqua/timeseries_features.parquet · fusion_node_meta.parquet · ts_predictions.parquet
fusion/output/fusion_dataset_ketqua_{control,frozen}.npz
fusion/output/graph_ketqua_*.npz  +  *_manifest.json
```

Cell đầu notebook Colab:

```python
!git clone -b fusion_gnn https://github.com/<user>/DO-AN-TOT-NGHIEP-E22-T02.git repo
%cd repo
!pip install -q torch_geometric          # torch đã có sẵn trên Colab
```

Không cài `torch-scatter` / `torch-cluster` (hay lỗi biên dịch): GATv2Conv không cần,
còn DGCNN dùng hàm kNN tự viết bằng `torch.cdist`.

| Notebook | Nội dung | Thời gian ước tính trên T4 |
|---|---|---|
| `colab/fusion_phase3_baselines.ipynb` | B0–B3, 5 fold × 3 seed | 10–15 phút |
| `colab/fusion_phase4_gat.ipynb` | GAT, 5 fold × 3 seed | 15–25 phút |
| `colab/fusion_phase5_more.ipynb` | DGCNN + Graph Transformer + ablation | 45–90 phút |

Mỗi notebook ghi kết quả ra `fusion/output/*.json` + `*.md` để tải về và commit.
**Không `git push` từ Colab** (rủi ro lộ token) — tải file kết quả về rồi commit ở local.

---

## 4. Kỳ vọng trung thực

| Chỉ số | Hiện tại | Kỳ vọng sau GNN |
|---|---|---|
| ROC-AUC, nhóm có ảnh | 0,766 | 0,77 – 0,79 |
| ROC-AUC, nhóm không ảnh | 0,617 (ts) → 0,640 (mượn ảnh thô) | 0,64 – 0,68 |

Phần tăng ở nhóm **có ảnh** sẽ nhỏ, vì ảnh đã nói gần hết những gì nó biết và homophily
chỉ 1,73×. Phần tăng đáng kể hơn nằm ở nhóm **không ảnh** — và đó đúng là đóng góp mà
đồ án tuyên bố. Nếu con số cuối rơi vào khoảng trên thì đồ án có một kết quả mạch lạc,
giải thích được, dù không phải con số gây choáng.

---

## 5. Rủi ro còn treo

| # | Rủi ro | Trạng thái |
|---|---|---|
| 1 | ~~Đặc trưng ts "biết nhãn" ở dev~~ | **ĐÃ XỬ LÝ 27/09.** Vector 32 chiều không dùng được theo cả hai cách: trung bình 5 model thì rò rỉ (MLP đạt ROC 0,752 so với 0,641 thật), lấy out-of-fold thì 5 fold nằm ở 5 không gian (probe còn 0,487). **Thay bằng bảng thô 139 cột + điểm dự đoán nhánh ts** — sạch tuyệt đối và mạnh hơn (0,663) |
| 2 | Shortcut `has_image` — nhóm có ảnh có tỷ lệ bệnh 25,2% so với 8,1% | Missing-token + modality dropout + ablation tắt cờ ở Phase 5 |
| 3 | Homophily chỉ 1,73× | Đã đo, đã ghi vào kỳ vọng. Cổng Phase 4 chặn việc đầu tư tiếp nếu không hiệu quả |
| 4 | Test chỉ 147 ca dương | Chọn bằng OOF, chạm test một lần, luôn kèm KTC 95% |
| 5 | Khoảng cách với bài SiCLIP (ROC 0,76 so với 0,96–0,98) | Chưa giải thích được. Ba khả năng: 202 ca âm họ loại, họ fine-tune CNN end-to-end, hoặc số của họ lạc quan. Việc tái lập DenseNet201 là tuỳ chọn, **không chặn** roadmap này |

---

## 6. Checklist

```
CHỐT       ☑ Nhãn ketqua, 2 nguồn ảnh, đã xoá crossfit
           ☑ Phase 1 — dataset ketqua × {control, frozen}, 9/9 kiểm định
           ☑ Phase 2 — đồ thị ketqua, qua gate (lift 1,73×)
           ☑ Baseline đã đo: có ảnh 0,766 | không ảnh 0,617 → 0,640
PHASE 3    □ B0/B1/B2/B3 cùng một đường ống       ← cổng: B3 > B0 ở nhóm không ảnh
PHASE 4    □ GAT                                   ← CỔNG CHÍNH: GAT > B3
PHASE 5    □ DGCNN, Graph Transformer, ablation
PHASE 6    □ Chạm test 1 lần, bảng tách nhóm, bàn giao
```
