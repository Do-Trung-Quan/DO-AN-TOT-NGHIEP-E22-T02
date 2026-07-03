# Hướng dẫn: Thí nghiệm hàm Loss (Ablation Loss) — nhánh ảnh BioViL-T

> Mục tiêu (theo yêu cầu thầy): **giữ nguyên backbone**, chỉ **thay đổi hàm loss** để xem kết quả có thay đổi không. Đây là "ablation study": đổi **đúng 1 yếu tố** (loss), giữ **mọi thứ khác giống hệt** → chênh lệch nào cũng là do loss.

---

## 1. Nguyên tắc VÀNG

Mỗi loss = **1 folder + 1 script riêng**, nhưng **copy y hệt** một script mẫu, **chỉ đổi 2 chỗ**:
1. `OUTPUT_DIR` → trỏ tới folder mới
2. Dòng `criterion = ...` → đổi sang loss cần thử

**KHÔNG được đổi** gì khác (backbone, data, split seed 42, sampler, augmentation, learning rate, số epoch, cách đánh giá) — nếu đổi thì so sánh mất công bằng.

Backbone = **BioViL-T**, weight đọc từ `Pre-train BioViL-T/biovil_t_image_model_proj_size_128.pt`.

---

## 2. Bảng kết quả (cập nhật dần)

| # | Loss | Folder | ROC-AUC | Sens | Spec | Trạng thái |
|---|---|---|---|---|---|---|
| 1 | CrossEntropy (mặc định) | `Finetune CrossEntropy` | **0.913** | 0.847 | 0.861 | ✅ xong |
| 2 | Weighted CE | `Finetune WeightedCE` | 0.891 | 0.837 | 0.839 | ✅ xong |
| 3 | Focal γ=2 | `Finetune 4` | 0.902 | 0.906 | 0.781 | ✅ xong |
| 4 | Focal γ=1 | `Finetune FocalG1` | ? | ? | ? | ⬜ cần làm |
| 5 | Focal γ=5 | `Finetune FocalG5` | ? | ? | ? | ⬜ cần làm |
| 6 | Label smoothing | `Finetune LabelSmooth` | ? | ? | ? | ⬜ cần làm |

---

## 3. Cách làm từng loss còn lại

### 3a. Focal γ=1 và γ=5 (dễ nhất: copy Finetune 4)

**Base:** copy file `Finetune 4/finetune_biovilt_setA.py` (nó đã có sẵn class `FocalLoss` + `weights_tensor`).

Đổi:
- `OUTPUT_DIR` → `r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune FocalG5"` (hoặc FocalG1)
- Tìm dòng:
  ```python
  criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
  ```
  đổi `gamma`:
  ```python
  criterion = FocalLoss(alpha=weights_tensor, gamma=5.0)   # hoac gamma=1.0
  ```

### 3b. Label smoothing (copy Finetune CrossEntropy)

**Base:** copy file `Finetune CrossEntropy/finetune_biovilt_ce.py`.

Đổi:
- `OUTPUT_DIR` → `r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune LabelSmooth"`
- Tìm dòng:
  ```python
  criterion = nn.CrossEntropyLoss()
  ```
  đổi thành:
  ```python
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  ```

### (Tùy chọn) đổi tên file output cho gọn
Trong script có vài chỗ đặt tên file (`best_model_ce.pth`, `evaluation_ce.png`, `confusion_matrix_ce.csv`). **Không bắt buộc đổi** — kết quả vẫn nằm trong folder mới. Nếu muốn gọn thì đổi hậu tố `_ce`/`_wce` cho khớp.

---

## 4. Bảng tra hàm loss (paste sẵn)

| Loss | Dòng `criterion` |
|---|---|
| CrossEntropy | `criterion = nn.CrossEntropyLoss()` |
| Weighted CE | `criterion = nn.CrossEntropyLoss(weight=weights_tensor)` |
| Label smoothing | `criterion = nn.CrossEntropyLoss(label_smoothing=0.1)` |
| Label smooth + weight | `criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.1)` |
| Focal γ=1 | `criterion = FocalLoss(alpha=weights_tensor, gamma=1.0)` |
| Focal γ=2 | `criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)` |
| Focal γ=5 | `criterion = FocalLoss(alpha=weights_tensor, gamma=5.0)` |

> Lưu ý: loss dùng `weights_tensor` (Weighted CE, Focal) thì script phải có block tính `weights_tensor` (Finetune 4 và WeightedCE đã có; bản CrossEntropy thuần thì không có).

### Class FocalLoss (nếu base chưa có, paste vào đầu script)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
    def forward(self, outputs, targets):
        ce = self.ce(outputs, targets)
        pt = torch.exp(-ce)
        fl = (1 - pt) ** self.gamma * ce
        return fl.mean() if self.reduction == "mean" else (fl.sum() if self.reduction == "sum" else fl)
```

---

## 5. Cách chạy

Từ thư mục gốc `Data DGCNN` (nhớ `deactivate` nếu thấy `(myvenv)`, để nguyên dấu nháy vì tên folder có dấu cách):

```powershell
# test nhanh truoc (vai phut, kiem tra khong loi):
C:\Python313\python.exe "Finetune FocalG5\finetune_biovilt_focalg5.py" --smoke

# chay that (~1-1.5h moi loss, co early stopping nen thuong dung som ~ep 15-18):
C:\Python313\python.exe "Finetune FocalG5\finetune_biovilt_focalg5.py"
```

Mỗi loss chạy full ~1-1.5 giờ (GPU RTX 3060). Có early stopping nên thực tế dừng sớm.

---

## 6. Đọc kết quả ở đâu

Sau khi chạy xong, mở file `train_log.txt` trong folder tương ứng, xem dòng cuối:
```
DONE — ... | AUC=0.xxxx | Sens=0.xxx | Spec=0.xxx
```
Lấy 3 số **ROC-AUC / Sensitivity / Specificity** điền vào bảng ở Mục 2. Ngoài ra có sẵn `evaluation_*.png` (ROC + confusion matrix) để đưa vào báo cáo.

---

## 7. Kết luận sơ bộ (từ 3 loss đã chạy)

1. **Đổi loss gần như KHÔNG làm tăng ROC-AUC** (cả 3 nằm 0.891–0.913, chênh ~0.02 = trong nhiễu của 339 ảnh test). CrossEntropy mặc định thậm chí cao nhất.
2. **Loss chủ yếu dịch điểm cân bằng Sensitivity ↔ Specificity:**
   - CrossEntropy / Weighted CE → cân bằng (~0.84 / 0.84–0.86)
   - Focal γ=2 → đẩy **Sensitivity cao** (0.906) nhưng Specificity thấp (0.781)
3. **Cú đẩy Sensitivity đến từ GAMMA (focal focus), không phải trọng số lớp** — vì Weighted CE (có trọng số, không gamma) vẫn cân bằng như CE.

→ **Ý nghĩa:** chọn loss = chọn ưu tiên (AUC/cân bằng → CE; bắt nhiều bệnh → Focal γ cao), **không phải để "tăng điểm"**.

### ⚠️ Lưu ý trung thực (nên ghi trong báo cáo)
- Đây là kết quả trên **1 lần split** (339 ảnh test). Chênh lệch ~0.02 AUC có thể là **nhiễu**. Muốn kết luận chắc chắn nên chạy **k-fold CV / nhiều seed**.
- **Không** cân bằng tập test hay resample tập test để "làm đẹp số" — test phải giữ phân bố thật.

---

## 8. Bối cảnh (để hiểu vì sao làm cái này)

- **Backbone đã chọn xong:** so sánh 4 backbone (BYOL 0.832, ImageNet 0.891, CheXNet 0.847, **BioViL-T 0.902** — tốt nhất). Thầy dặn **giữ nguyên backbone**, chuyển sang thử **loss**.
- Đây **chỉ là nhánh ảnh**. Nhánh lâm sàng (Timeseries 2, boosting, target `bnn`) và bước **fusion** (ghép ảnh + lâm sàng) là phần riêng, không dính thí nghiệm loss này.
