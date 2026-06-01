# Handoff: Pretrain → Fine-tune

Tài liệu bàn giao kết quả bước **pretrain BYOL** cho người làm bước **fine-tune silicosis**.

---

## 1. Tổng quan workflow

```
[NIH ChestX-ray14 ~112k ảnh KHÔNG nhãn]
              │
              │ Pretrain BYOL (self-supervised) — ĐÃ XONG
              │ 60 epoch · 10.4h · RTX 3060 Laptop
              ▼
   ┌──────────────────────────────────┐
   │ xray_byol_backbone.pth (42.7 MB) │  ← Đây là sản phẩm bàn giao
   │ ResNet18 backbone weights        │
   └──────────────────────────────────┘
              │
              │ Fine-tune (supervised) — VIỆC CỦA BẠN
              │ 433 ảnh silicosis có nhãn (29 Co / 404 Khong)
              ▼
     [Mô hình phân loại silicosis]
```

## 1.1. Dữ liệu fine-tune (đã verify)

| Resource | Đường dẫn | Ghi chú |
|---|---|---|
| **Ảnh X-quang** | `sillicosis/*.jpg` | 535 file JPG (chỉ 433 file có nhãn) |
| **File nhãn** | `Main_data_fixed_Not_Encode_Mapping_New.xlsx` | 8030 rows phiếu khảo sát |
| **Cột nhãn** | `bnn` | Values: `"Co"` (mắc bệnh) / `"Khong"` (không) |
| **Cột mapping** | `file_name` | Tên file ảnh khớp 1-1 với folder `sillicosis/` |

**Quan trọng**: Excel có 8030 rows nhưng **chỉ 433 rows có `file_name`** (các bệnh nhân khác chưa được chụp ảnh hoặc thiếu). Phải filter trước khi train.

### Phân phối nhãn (đã kiểm tra trên 433 mẫu khớp)

| Class | Số mẫu | Tỷ lệ |
|---|---|---|
| `Khong` (không silicosis) | 404 | 93.3% |
| `Co` (có silicosis) | 29 | 6.7% |
| **Imbalance ratio** | **~14:1** | ⚠️ rất mất cân |

⚠️ Đây là **highly imbalanced binary classification**. Cần kỹ thuật chống imbalance (xem mục 5).

---

## 2. File cần dùng (1 file duy nhất)

### `Pre-train/xray_byol_backbone.pth`

- **Kích thước**: 42.7 MB
- **Nội dung**: `state_dict` của **ResNet18** (chỉ phần backbone, không có FC layer)
- **Format**: tương thích `torchvision.models.resnet18()` — load thẳng được
- **Đầu vào kỳ vọng**: ảnh tensor `[B, 3, 224, 224]`, đã normalize ImageNet mean/std
- **Đầu ra**: feature vector `[B, 512]` sau Global Average Pool

---

## 3. Cách load backbone

```python
import torch
import torch.nn as nn
from torchvision import models

PRETRAINED = "Pre-train/xray_byol_backbone.pth"

# 1. Tạo ResNet18 không có pretrain (sẽ load từ file ta tự pretrain)
model = models.resnet18(weights=None)

# 2. Load state_dict, bỏ các key của FC layer (vì FC sẽ thay theo task)
state = torch.load(PRETRAINED, map_location="cpu")
state = {k: v for k, v in state.items() if not k.startswith("fc.")}
missing, unexpected = model.load_state_dict(state, strict=False)
print(f"missing={len(missing)} unexpected={len(unexpected)}")
# Kỳ vọng: missing=2 (fc.weight, fc.bias), unexpected=0
# Nếu khác → có vấn đề, kiểm tra lại

# 3. Thay FC head cho task binary silicosis
model.fc = nn.Linear(model.fc.in_features, 2)  # 2 classes: Co / Khong

# 4. Bây giờ model sẵn sàng fine-tune
```

### Lưu ý quan trọng

- **`strict=False`** trong `load_state_dict` là cần thiết vì ta đã bỏ FC keys.
- **`map_location="cpu"`** cho phép load file trên máy không có GPU. Sau đó `model.to("cuda")` nếu có GPU.
- **`weights=None`** tránh download ImageNet pretrain (ta đã có pretrain riêng).

---

## 3.5. Load dataset từ Excel + folder ảnh

```python
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN")
EXCEL = ROOT / "Main_data_fixed_Not_Encode_Mapping_New.xlsx"
IMG_DIR = ROOT / "sillicosis"

def build_dataframe():
    """Lay 433 mau co ca anh va nhan."""
    df = pd.read_excel(EXCEL)
    files_on_disk = set(os.listdir(IMG_DIR))

    # Filter: phai co file_name HOP LE va co tren disk
    df = df[df["file_name"].notna()].copy()
    df = df[df["file_name"].isin(files_on_disk)]
    df = df[df["bnn"].isin(["Co", "Khong"])].copy()

    # Map text -> int: Co = 1 (positive), Khong = 0 (negative)
    df["label"] = (df["bnn"] == "Co").astype(int)
    df = df[["file_name", "label"]].reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    # Expected: {0: 404, 1: 29}
    return df


class SilicosisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row["file_name"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"])
```

---

## 4. Preprocessing (BẮT BUỘC dùng đúng để khớp pretrain)

```python
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Train transform (có augmentation)
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(7),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Eval transform (không augmentation)
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
```

### Ràng buộc bắt buộc

| Tham số | Giá trị | Tại sao |
|---|---|---|
| Image size | **224×224** | Pretrain dùng 224, đổi sẽ làm BN statistics sai |
| Channels | **RGB (3 kênh)** | Convert ảnh xám → RGB bằng `Image.open(p).convert("RGB")` |
| Normalize | **ImageNet mean/std** | Pretrain dùng giá trị này |
| Pixel range | **[0, 1] rồi normalize** | Dùng `ToTensor()` chuẩn |

---

## 5. Đề xuất chiến lược fine-tune

### Stage 1 — Linear probe (epoch 1-10)

Mục đích: **kiểm tra chất lượng feature** từ pretrain.

```python
# Freeze backbone, chỉ train FC
for name, p in model.named_parameters():
    p.requires_grad = name.startswith("fc.")

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
```

- LR: `1e-3` (FC có thể học nhanh)
- Epochs: 10
- Augmentation: chỉ flip + crop nhẹ (vì backbone đã frozen)

**Kỳ vọng sau Stage 1**: AUC ≥ 0.70 trên val set. Nếu thấp hơn → feature không đủ tốt, cần xem lại preprocessing.

### Stage 2 — Full fine-tune (epoch 11-40)

Mục đích: tinh chỉnh toàn bộ network cho silicosis.

```python
# Unfreeze tất cả
for p in model.parameters():
    p.requires_grad = True

# LR khác nhau cho backbone vs FC
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if not n.startswith("fc.")],
     "lr": 1e-4},   # backbone: LR thấp (giữ feature đã pretrain)
    {"params": model.fc.parameters(),
     "lr": 1e-3},   # FC: LR cao (học mạnh hơn)
], weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
```

- LR backbone: `1e-4` (thấp để không phá vỡ feature pretrain)
- LR head: `1e-3` (cao để học task-specific)
- Epochs: 30
- AMP (mixed precision): bật để tiết kiệm VRAM

### Xử lý imbalance ⚠️ QUAN TRỌNG — data của bạn imbalance 14:1

Với 29 positives / 404 negatives, **bắt buộc** phải dùng kỹ thuật chống imbalance, nếu không model sẽ học trick "predict luôn Khong" → accuracy 93% nhưng F1/AUC tệ.

**Khuyến nghị: kết hợp CẢ HAI** (Class weight + WeightedSampler):

```python
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

# A. Class weight trong loss (penalty cho việc dự đoán sai positive)
counts = train_df["label"].value_counts().sort_index().values.astype(float)
# counts = [n_khong, n_co] ví dụ [283, 20] sau khi split
class_weights = counts.sum() / (len(counts) * counts)
# Ket qua: ~ [0.54, 7.6] — class Co duoc weight cao gap ~14 lan
weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# B. WeightedRandomSampler (over-sample minority trong DataLoader)
sample_weights = train_df["label"].map(lambda y: 1.0 / counts[y]).values
sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)
train_loader = DataLoader(
    train_ds, batch_size=32, sampler=sampler,  # KHONG dung shuffle khi co sampler
    num_workers=2, pin_memory=True,
)
```

### Augmentation MẠNH cho minority class

Vì chỉ có 29 mẫu Co, mỗi mẫu cần được "khai thác" nhiều lần với augmentation mạnh:

```python
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),                    # dao hon pretrain (7)
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # dich nho
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # cutout
])
```

### Khuyến nghị Stratified K-Fold thay vì single split (vì positives ít)

Với chỉ 29 positives, single 70/15/15 split → val có ~4 positives, test có ~5 positives → metrics không stable.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    train_df = df.iloc[tr_idx]
    val_df = df.iloc[val_idx]
    # train 1 model cho moi fold, average metrics cuoi cung
```

→ Báo cáo "AUC = X ± Y" qua 5 folds, chuyên nghiệp hơn nhiều cho paper.

---

## 6. Data split

### Cách A — Single split (nhanh, dùng cho test/debug)

```python
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    df, test_size=0.15, stratify=df["label"], random_state=42)
train_df, val_df = train_test_split(
    train_df, test_size=0.1765, stratify=train_df["label"], random_state=42)
# 0.1765 ≈ 0.15 / (1 - 0.15) để val = 15% tổng
```

Phân phối dự kiến với 433 mẫu (29 positives):

| Tập | Số ảnh | Số positives (`Co`) |
|---|---|---|
| Train (70%) | ~303 | ~20 |
| Val (15%) | ~65 | ~4 |
| Test (15%) | ~65 | ~5 |

⚠️ Val/test chỉ có ~4-5 positives → metric AUC dao động lớn giữa các seed.

### Cách B — Stratified 5-Fold CV (KHUYẾN NGHỊ cho paper)

Với data nhỏ và imbalance, single split không ổn định. Dùng 5-fold:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_metrics = []
for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df["label"])):
    train_df_fold = df.iloc[tr_idx]   # ~346 mẫu, ~23 positives
    val_df_fold = df.iloc[val_idx]    # ~87 mẫu, ~6 positives
    # Train + eval fold
    auc = train_and_evaluate(train_df_fold, val_df_fold)
    fold_metrics.append(auc)

print(f"5-fold AUC: {np.mean(fold_metrics):.3f} ± {np.std(fold_metrics):.3f}")
```

→ Báo cáo trong paper: `"AUC = 0.82 ± 0.05"` chuyên nghiệp hơn `"AUC = 0.82"` nhiều.

---

## 7. Metrics để báo cáo

| Metric | Lý do |
|---|---|
| **Accuracy** | Tiêu chuẩn, dễ hiểu |
| **F1 score (binary)** | Quan trọng khi imbalance |
| **F1 macro** | Trung bình F1 của cả 2 class |
| **ROC-AUC** | Đánh giá khả năng phân biệt, không phụ thuộc threshold |
| **Confusion matrix** | Xem chi tiết FP/FN |

```python
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
```

Báo cáo paper nên có **AUC** ở dòng chính, kèm confusion matrix trong appendix.

---

## 8. Thông tin pretrain (cho phần Methodology của paper)

| Tham số | Giá trị |
|---|---|
| Phương pháp | BYOL (self-supervised) |
| Backbone | ResNet18 |
| Dataset | NIH ChestX-ray14 |
| Số ảnh pretrain | 112,120 |
| Image size | 224×224 RGB |
| Epochs | 60 |
| Batch size | 64 |
| Optimizer | AdamW, lr=3e-4, wd=1e-6 |
| Scheduler | CosineAnnealingLR |
| Momentum (EMA target) | 0.996 → 1.0 (cosine schedule) |
| Mixed precision | Có (AMP fp16) |
| Augmentation | RandomResizedCrop(0.7-1.0), HFlip, Rotation(±7°), ColorJitter(brightness/contrast), GaussianBlur(kernel=9) |
| Loss cuối | 0.0143 |
| Thời gian train | 10.4 giờ trên RTX 3060 Laptop |

### Chất lượng feature đã verify

- Mean per-dim std: **0.28** (healthy, không collapse)
- Mean pairwise cosine similarity: **0.76** (giảm từ 0.88 sau augmentation fix → feature đã tách)
- Output feature: vector 512-d sau Global Average Pool

---

## 9. File phụ trợ trong `Pre-train/`

| File | Vai trò |
|---|---|
| `xray_byol_backbone.pth` | **File chính** — backbone weights |
| `pretrain_loss.csv` | Loss 60 epoch (epoch, loss) — vẽ chart cho paper |
| `pretrain_loss.png` | Biểu đồ loss đã render sẵn |
| `pretrain_log.txt` | Log đầy đủ training |
| `pretrain_config.json` | Toàn bộ hyperparameters dạng JSON |
| `pretrain_byol.py` | Script đã dùng pretrain |
| `check_collapse.py` | Script kiểm tra chất lượng feature |
| `resize_images.py` | Script preprocess ảnh (nếu cần chạy lại) |
| `images_256/` | Folder ảnh đã pre-resize 256×256 (~3 GB) |

---

## 10. Checklist cho người fine-tune

Trước khi bắt đầu:
- [ ] Đã có file `Pre-train/xray_byol_backbone.pth` (42.7 MB)
- [ ] Đã có folder `sillicosis/` chứa 535 ảnh JPG
- [ ] Đã có file `Main_data_fixed_Not_Encode_Mapping_New.xlsx` (cột `file_name` + `bnn`)
- [ ] Filter Excel + folder → ra **433 samples** (404 Khong + 29 Co)
- [ ] Đã cài: `torch`, `torchvision`, `pandas`, `scikit-learn`, `Pillow`, `openpyxl`

Khi load backbone:
- [ ] In ra `missing=2, unexpected=0` (chỉ FC keys thiếu)
- [ ] Forward thử 1 ảnh, kiểm tra output shape `[1, 2]` (logits)

Khi train:
- [ ] Stage 1 linear probe đạt AUC ≥ 0.70 trên val
- [ ] Stage 2 full fine-tune cải thiện thêm
- [ ] Lưu best checkpoint theo val AUC

Khi báo cáo:
- [ ] AUC, F1, accuracy trên TEST set (không phải val)
- [ ] Confusion matrix
- [ ] Loss curve fine-tune (Stage 1 + Stage 2)

---

## 11. Liên hệ / Ghi chú

Có thắc mắc về backbone hoặc preprocessing → kiểm tra:
1. `Pre-train/pretrain_config.json` để xem hyperparameters
2. `Pre-train/pretrain_byol.py` (mục `make_transform`) để xem augmentation
3. `Pre-train/check_collapse.py` để hiểu cách feature được verify

Nếu fine-tune cho AUC quá thấp (<0.65) trên Stage 1 linear probe:
- Vấn đề có thể ở **preprocessing không khớp pretrain** → kiểm tra mục 4 phía trên
- Hoặc **data nhãn có vấn đề** → kiểm tra phân phối Co/Khong, ảnh có khớp file_name không

---

*Tài liệu này được sinh ra sau khi pretrain hoàn thành. Mọi file kèm theo đều ở thư mục `Pre-train/`.*
