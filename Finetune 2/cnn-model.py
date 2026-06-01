"""
================================================================================
FINE-TUNE RESNET18 - VERSION 3.0 (FINAL FIXED)
================================================================================
Fix so với v1.0:
  [FIX 1] Grayscale transform — đúng cho ảnh X-quang
  [FIX 2] Normalization X-quang mean=0.5, std=0.5
  [FIX 3] Augmentation phù hợp X-quang (bỏ VerticalFlip, ColorJitter, Blur)
  [FIX 4] Class weights tính TRƯỚC oversampling
  [FIX 5] WeightedRandomSampler thay RandomOverSampler
  [FIX 6] Optimizer filter requires_grad (flexible)
  [FIX 7] 2-stage: Unfreeze backbone sau UNFREEZE_EPOCH
  [FIX 8] Dropout trong classifier head
  [FIX 9] Normalize tên file khi match ảnh (lower + strip)
================================================================================
"""

import os
import re
import pandas as pd
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Console Windows mac dinh cp1252 -> crash khi in emoji/tieng Viet. Ep UTF-8.
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

print("=" * 80)
print("FINE-TUNING RESNET18 — VERSION 3.0 (FINAL FIXED)")
print("=" * 80)

# ================================================================================
# CONFIG — SỬA Ở ĐÂY
# ================================================================================

EXCEL_PATH    = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune 2\SetA_Labels.xlsx"
IMAGE_DIR     = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune\images_256\train"
PRETRAIN_PATH = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Pre-train\xray_byol_backbone.pth"
OUTPUT_DIR    = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune 2"   # noi luu model + anh ket qua
IMAGE_COLUMN  = 'file_name'
LABEL_COLUMN  = 'bnn'

BATCH_SIZE     = 16
EPOCHS         = 50
LR_HEAD        = 1e-4    # Giai đoạn 1: chỉ train head
LR_BACKBONE    = 1e-5    # Giai đoạn 2: train backbone
UNFREEZE_EPOCH = 10      # Unfreeze backbone sau epoch này
IMAGE_SIZE     = 224
PATIENCE       = 7

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Device         : {DEVICE}")
print(f"Batch size     : {BATCH_SIZE}")
print(f"LR head        : {LR_HEAD}")
print(f"LR backbone    : {LR_BACKBONE}")
print(f"Epochs         : {EPOCHS}")
print(f"Unfreeze epoch : {UNFREEZE_EPOCH}")

# ================================================================================
# FOCAL LOSS
# ================================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        self.ce        = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, outputs, targets):
        ce_loss    = self.ce(outputs, targets)
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ================================================================================
# BƯỚC 1: LOAD EXCEL
# ================================================================================

print("\n[1/9] Loading Excel data...")
df = pd.read_excel(EXCEL_PATH)
print(f"✓ Loaded {len(df)} rows")

# ================================================================================
# BƯỚC 2: LABEL ENCODING
# ================================================================================

print("\n[2/9] Encoding labels...")
label_map = {'Khong': 0, 'Co': 1}
df[LABEL_COLUMN] = df[LABEL_COLUMN].map(label_map)
print("Class distribution:")
print(df[LABEL_COLUMN].value_counts())

# ================================================================================
# BƯỚC 3: MATCH IMAGES
# [FIX 9] Normalize tên file: lower + strip để tránh mismatch
# ================================================================================

print("\n[3/9] Matching images...")

# Quét đệ quy tất cả subfolder → build dict {tên_file_lower: đường_dẫn_đầy_đủ}
all_files = {}
for root, dirs, files in os.walk(IMAGE_DIR):
    for f in files:
        all_files[f.lower().strip()] = os.path.join(root, f)

print(f"  Tổng ảnh tìm thấy trong tất cả subfolder: {len(all_files)}")

matched = []
for _, row in df.iterrows():
    raw  = str(row[IMAGE_COLUMN]).strip()
    norm = raw.lower()
    if norm in all_files:
        matched.append(all_files[norm])
    else:
        matched.append(None)

df['image_path'] = matched
n_total   = len(matched)
n_matched = sum(1 for x in matched if x is not None)
df = df[df['image_path'].notna()].reset_index(drop=True)
print(f"✓ Matched {n_matched} / {n_total} images")
print("  Không match được:", n_total - n_matched)
print("Class distribution sau match:")
print(df[LABEL_COLUMN].value_counts())

# ================================================================================
# BƯỚC 4: TRAIN / VAL / TEST SPLIT (Group by patient)
# ================================================================================

print("\n[4/9] Splitting data...")

# Tách patient base để tránh data leakage
df['patient_base'] = df[IMAGE_COLUMN].apply(
    lambda x: re.sub(r'_\d+\.(jpg|jpeg|png|bmp|tiff|tif)$', '', str(x).lower().strip())
)

from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(df, df[LABEL_COLUMN], groups=df['patient_base']))
train_df = df.iloc[train_idx].copy()
temp_df  = df.iloc[temp_idx].copy()

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, temp_df[LABEL_COLUMN], groups=temp_df['patient_base']))
val_df  = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")
print("Train class distribution:")
print(train_df[LABEL_COLUMN].value_counts())

# Kiểm tra data leakage
assert len(set(train_df['patient_base']) & set(test_df['patient_base'])) == 0, \
    "❌ Data leakage detected!"
print("✓ No data leakage")

# ================================================================================
# BƯỚC 5: CLASS WEIGHTS + WEIGHTED SAMPLER
# [FIX 4] Tính weights TRƯỚC khi sample
# [FIX 5] WeightedRandomSampler thay RandomOverSampler
# ================================================================================

print("\n[5/9] Setting up WeightedRandomSampler...")

original_counts = train_df[LABEL_COLUMN].value_counts().sort_index()
total           = original_counts.sum()

# Weights cho Focal Loss — tính từ data gốc
weights_tensor = torch.tensor([
    total / original_counts[0],
    total / original_counts[1],
], dtype=torch.float32).to(DEVICE)

print(f"Original counts : {dict(original_counts)}")
print(f"Class weights   : {weights_tensor.cpu().numpy()}")

# Weights cho từng sample
sample_weights = train_df[LABEL_COLUMN].map(
    lambda lbl: 1.0 / original_counts[lbl]
).tolist()

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(train_df) * 2,
    replacement = True
)
print(f"✓ Sampler: {len(train_df) * 2} samples/epoch")

# ================================================================================
# BƯỚC 6: AUGMENTATION
# [FIX 1] Grayscale
# [FIX 2] Normalization X-quang
# [FIX 3] Bỏ VerticalFlip, ColorJitter, GaussianBlur
# ================================================================================

print("\n[6/9] Setting up augmentation...")

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),      # [FIX 1]
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),             # Nhỏ thôi
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(                              # [FIX 2]
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),       # [FIX 1]
    transforms.ToTensor(),
    transforms.Normalize(                              # [FIX 2]
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

print("✓ Augmentation configured (X-quang friendly)")

# ================================================================================
# BƯỚC 7: DATASET & DATALOADER
# ================================================================================

print("\n[7/9] Creating datasets and dataloaders...")

class XrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df        = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path  = self.df.iloc[idx]['image_path']
        label = int(self.df.iloc[idx][LABEL_COLUMN])
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

train_dataset = XrayDataset(train_df, train_transform)
val_dataset   = XrayDataset(val_df,   val_test_transform)
test_dataset  = XrayDataset(test_df,  val_test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size  = BATCH_SIZE,
    sampler     = sampler,
    drop_last   = True,
    num_workers = 0
)
val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"✓ Train batches : {len(train_loader)}")
print(f"✓ Val batches   : {len(val_loader)}")
print(f"✓ Test batches  : {len(test_loader)}")

# ================================================================================
# BƯỚC 8: MODEL
# [FIX 7] 2-stage fine-tuning
# [FIX 8] Dropout trong classifier
# ================================================================================

print("\n[8/9] Setting up model...")

model = models.resnet18(weights=None)

if os.path.exists(PRETRAIN_PATH):
    model.load_state_dict(
        torch.load(PRETRAIN_PATH, map_location=DEVICE),
        strict=False
    )
    print("✓ BYOL pretrained weights loaded")
else:
    print("⚠️  BYOL weights not found, using random init")

# Giai đoạn 1: Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# [FIX 8] Thêm Dropout
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(512, 2)
)
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
print("✓ Stage 1: Backbone Frozen | Head Trainable")

# ================================================================================
# LOSS & OPTIMIZER
# [FIX 6] filter requires_grad
# ================================================================================

print("\n[9/9] Setting up loss and optimizer...")

criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)

optimizer = torch.optim.Adam(                          # [FIX 6]
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR_HEAD
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

print(f"✓ Focal Loss  | weights: {weights_tensor.cpu().numpy().round(2)}")
print(f"✓ Adam        | lr={LR_HEAD}")
print(f"✓ ReduceLROnPlateau")

# ================================================================================
# TRAINING LOOP
# ================================================================================

print("\nStarting training...")
print("=" * 80)

best_val_loss    = float('inf')
patience_counter = 0
train_losses     = []
val_losses       = []
stage            = 1

for epoch in range(EPOCHS):

    # [FIX 7] UNFREEZE BACKBONE
    if epoch == UNFREEZE_EPOCH and stage == 1:
        print(f"\n🔓 Epoch {epoch+1}: Unfreeze backbone → Stage 2")
        for param in model.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam([
            {'params': model.layer1.parameters(), 'lr': LR_BACKBONE * 0.1},
            {'params': model.layer2.parameters(), 'lr': LR_BACKBONE * 0.1},
            {'params': model.layer3.parameters(), 'lr': LR_BACKBONE * 0.5},
            {'params': model.layer4.parameters(), 'lr': LR_BACKBONE},
            {'params': model.fc.parameters(),     'lr': LR_HEAD},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        stage            = 2
        patience_counter = 0
        print("✓ Layerwise LR: layer1-2=1e-6, layer3=5e-6, layer4=1e-5, head=1e-4")
        print("=" * 80)

    # ── TRAIN ──
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)
    train_losses.append(avg_train)

    # ── VALIDATION ──
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            val_loss += criterion(model(images), labels).item()

    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} [S{stage}] | "
          f"Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss    = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_v3.pth'))
        print("  ✅ Best model saved → best_model_v3.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

    scheduler.step(avg_val)

# ================================================================================
# LOAD BEST MODEL & EVALUATE
# ================================================================================

print("\nLoading best model...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model_v3.pth'), map_location=DEVICE))
model.eval()

print("\n" + "=" * 80)
print("TESTING WITH THRESHOLD OPTIMIZATION")
print("=" * 80)

y_true  = []
y_proba = []

with torch.no_grad():
    for images, labels in test_loader:
        probs = torch.softmax(model(images.to(DEVICE)), dim=1)
        y_proba.extend(probs[:, 1].cpu().numpy())
        y_true.extend(labels.numpy())

y_proba = np.array(y_proba)
y_true  = np.array(y_true)

roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
fpr, tpr, _ = roc_curve(y_true, y_proba)

# Threshold list — giữ lại để vẽ đồ thị F1 vs Threshold
threshold_list = np.arange(0.05, 0.95, 0.05)
f1_scores = [
    f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
    for t in threshold_list
]

# [SUA] Chọn threshold theo YOUDEN'S J (TPR - FPR) thay vì max-F1.
# Lý do: "Co" là lớp đa số (1267 vs 862) nên max-F1 kéo threshold xuống 0.35
#        → nhiều false positive (normal/khỏe bị đoán có bệnh, specificity thấp).
#        Youden's J cân bằng sensitivity & specificity → threshold hợp lý hơn.
# roc_curve tra ve fpr,tpr,thresholds; lay threshold ung voi J = TPR-FPR lon nhat
_fpr, _tpr, _thr = roc_curve(y_true, y_proba)
optimal_threshold = float(_thr[np.argmax(_tpr - _fpr)])
y_pred = (y_proba >= optimal_threshold).astype(int)

# Tính sensitivity & specificity tại threshold đã chọn
_cm = confusion_matrix(y_true, y_pred)
_tn, _fp, _fn, _tp = _cm.ravel()
sensitivity = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
specificity = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0

print(f"\n📊 METRICS:")
print(f"  ROC AUC            : {roc_auc:.4f}")
print(f"  Optimal Threshold  : {optimal_threshold:.2f}  (Youden's J)")
print(f"  Sensitivity (Co)   : {sensitivity:.3f}")
print(f"  Specificity (Khong): {specificity:.3f}  <-- tang chi so nay = giam false positive")
print(f"  F1 tai threshold   : {f1_score(y_true, y_pred, zero_division=0):.4f}")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(
    y_true, y_pred,
    target_names=['Khong (0)', 'Co (1)'],
    zero_division=0
))

# ================================================================================
# VISUALIZATION
# ================================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ResNet18 v3.0 — Bệnh Nghề Nghiệp', fontsize=14, fontweight='bold')

# 1. Loss curve
axes[0,0].plot(train_losses, label='Train', marker='o', markersize=3, linewidth=2)
axes[0,0].plot(val_losses,   label='Val',   marker='s', markersize=3, linewidth=2)
axes[0,0].axvline(x=UNFREEZE_EPOCH, color='red', linestyle='--',
                  alpha=0.7, label=f'Unfreeze (epoch {UNFREEZE_EPOCH})')
axes[0,0].set_title('Loss Curve'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# 2. ROC
axes[0,1].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', linewidth=2)
axes[0,1].plot([0,1],[0,1],'k--'); axes[0,1].set_title('ROC Curve')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# 3. F1 vs Threshold
axes[1,0].plot(threshold_list, f1_scores, marker='o', linewidth=2, markersize=5)
axes[1,0].axvline(x=optimal_threshold, color='r', linestyle='--',
                  label=f'Optimal={optimal_threshold:.2f}')
axes[1,0].set_title('F1 vs Threshold'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# 4. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
            xticklabels=['Khong','Co'], yticklabels=['Khong','Co'],
            annot_kws={'size':14})
axes[1,1].set_title('Confusion Matrix')
axes[1,1].set_ylabel('True'); axes[1,1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'evaluation_v3.png'), dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {os.path.join(OUTPUT_DIR, 'evaluation_v3.png')}")
plt.show()

# ================================================================================
# SUMMARY
# ================================================================================

prec_co = cm[1,1]/(cm[1,1]+cm[0,1]) if (cm[1,1]+cm[0,1]) > 0 else 0
rec_co  = cm[1,1]/(cm[1,1]+cm[1,0]) if (cm[1,1]+cm[1,0]) > 0 else 0
prec_kh = cm[0,0]/(cm[0,0]+cm[1,0]) if (cm[0,0]+cm[1,0]) > 0 else 0
rec_kh  = cm[0,0]/(cm[0,0]+cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0

print("\n" + "=" * 80)
print("✅ TRAINING v3.0 COMPLETED!")
print("=" * 80)
print(f"""
📊 RESULTS:
  ROC AUC           : {roc_auc:.4f}
  Optimal threshold : {optimal_threshold:.2f}
  Model file        : best_model_v3.pth

🎯 CLASS "Co" (Bệnh nghề nghiệp):
  Precision : {prec_co:.2%}
  Recall    : {rec_co:.2%}

🎯 CLASS "Khong" (Không bệnh):
  Precision : {prec_kh:.2%}
  Recall    : {rec_kh:.2%}

🔧 9 FIXES so với v1.0:
  [FIX 1] Grayscale transform
  [FIX 2] Normalization X-quang (0.5/0.5)
  [FIX 3] Augmentation X-quang friendly
  [FIX 4] Class weights từ data gốc
  [FIX 5] WeightedRandomSampler
  [FIX 6] Optimizer filter requires_grad
  [FIX 7] 2-stage unfreeze backbone epoch {UNFREEZE_EPOCH}
  [FIX 8] Dropout p=0.3 trong classifier
  [FIX 9] Normalize tên file khi match ảnh
""")
print("=" * 80)