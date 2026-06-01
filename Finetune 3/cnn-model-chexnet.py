"""
================================================================================
FINE-TUNE DENSENET121 (CheXNet pretrained) - phien ban CheXNet
================================================================================
Bien the cua cnn-model.py: thay backbone BYOL-ResNet18 -> CheXNet-DenseNet121.
Giu nguyen cnn-model.py (BYOL, Finetune 2) de SO SANH 2 backbone.

Khac biet so voi cnn-model.py:
  - Backbone: DenseNet121 (thay ResNet18)
  - Pretrain: CheXNet (co giam sat tren NIH 14 benh) thay BYOL (tu giam sat)
  - Load weights arnoweng: remap key (bo 'module.densenet121.', sua norm.1->norm1)
  - Normalization: ImageNet (CheXNet train voi ImageNet norm) thay 0.5/0.5
  - Head: model.classifier (1024->2) thay model.fc

Output (vao Finetune 3/):
  - best_model_chexnet.pth
  - evaluation_chexnet.png

CHAY:
  python "Finetune 3/cnn-model-chexnet.py"
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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---- Luu TOAN BO output ra log file trong Finetune 3 ----
OUTPUT_DIR = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune 3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class _Tee:
    """Ghi dong thoi ra console + file log."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

_logfile = open(os.path.join(OUTPUT_DIR, "train_log.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _logfile)

print("=" * 80)
print("FINE-TUNING DENSENET121 — CheXNet pretrained")
print("=" * 80)

# ================================================================================
# CONFIG
# ================================================================================
EXCEL_PATH    = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune 2\SetA_Labels.xlsx"
IMAGE_DIR     = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune\images_256\train"
CHEXNET_PATH  = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Pre-train CheXNET\model.pth"
OUTPUT_DIR    = r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN\Finetune 3"
IMAGE_COLUMN  = 'file_name'
LABEL_COLUMN  = 'bnn'

BATCH_SIZE     = 16
EPOCHS         = 50
LR_HEAD        = 1e-4
LR_BACKBONE    = 1e-5
UNFREEZE_EPOCH = 10
IMAGE_SIZE     = 224
PATIENCE       = 7

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device         : {DEVICE}")
print(f"Backbone       : DenseNet121 (CheXNet)")
print(f"Batch size     : {BATCH_SIZE} | Epochs: {EPOCHS} | Unfreeze: {UNFREEZE_EPOCH}")

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
# LOAD CHEXNET BACKBONE (remap key arnoweng -> torchvision densenet121)
# ================================================================================
def remap_chexnet(state_dict):
    """arnoweng/CheXNet checkpoint -> torchvision densenet121 keys.
       - bo prefix 'module.densenet121.'
       - sua 'norm.1'->'norm1', 'conv.1'->'conv1' (naming cu)
       - bo classifier 14 lop (se thay bang head nhi phan)
    """
    out = {}
    for k, v in state_dict.items():
        nk = k.replace('module.densenet121.', '')
        nk = re.sub(r'norm\.(\d)', r'norm\1', nk)
        nk = re.sub(r'conv\.(\d)', r'conv\1', nk)
        if nk.startswith('classifier'):
            continue
        out[nk] = v
    return out

# ================================================================================
# BƯỚC 1-2: LOAD EXCEL + ENCODE
# ================================================================================
print("\n[1/9] Loading Excel...")
df = pd.read_excel(EXCEL_PATH)
print(f"✓ Loaded {len(df)} rows")

print("\n[2/9] Encoding labels...")
label_map = {'Khong': 0, 'Co': 1}
df[LABEL_COLUMN] = df[LABEL_COLUMN].map(label_map)
print(df[LABEL_COLUMN].value_counts())

# ================================================================================
# BƯỚC 3: MATCH IMAGES
# ================================================================================
print("\n[3/9] Matching images...")
all_files = {}
for root, dirs, files in os.walk(IMAGE_DIR):
    for f in files:
        all_files[f.lower().strip()] = os.path.join(root, f)
print(f"  Tong anh tim thay: {len(all_files)}")

matched = []
for _, row in df.iterrows():
    norm = str(row[IMAGE_COLUMN]).strip().lower()
    matched.append(all_files.get(norm))
df['image_path'] = matched
n_total = len(matched)
df = df[df['image_path'].notna()].reset_index(drop=True)
print(f"✓ Matched {len(df)} / {n_total}")

# ================================================================================
# BƯỚC 4: SPLIT (group by patient, CUNG seed 42 -> CUNG tap test voi ban BYOL)
# ================================================================================
print("\n[4/9] Splitting (group by patient)...")
df['patient_base'] = df[IMAGE_COLUMN].apply(
    lambda x: re.sub(r'_\d+\.(jpg|jpeg|png|bmp|tiff|tif)$', '', str(x).lower().strip()))

gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(df, df[LABEL_COLUMN], groups=df['patient_base']))
train_df = df.iloc[train_idx].copy()
temp_df  = df.iloc[temp_idx].copy()
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(temp_df, temp_df[LABEL_COLUMN], groups=temp_df['patient_base']))
val_df  = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
assert len(set(train_df['patient_base']) & set(test_df['patient_base'])) == 0, "Leakage!"
print("✓ No data leakage")

# ================================================================================
# BƯỚC 5: CLASS WEIGHTS + SAMPLER
# ================================================================================
print("\n[5/9] WeightedRandomSampler...")
original_counts = train_df[LABEL_COLUMN].value_counts().sort_index()
total = original_counts.sum()
weights_tensor = torch.tensor([total/original_counts[0], total/original_counts[1]],
                              dtype=torch.float32).to(DEVICE)
print(f"Counts: {dict(original_counts)} | Weights: {weights_tensor.cpu().numpy().round(2)}")
sample_weights = train_df[LABEL_COLUMN].map(lambda l: 1.0/original_counts[l]).tolist()
sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_df)*2, replacement=True)

# ================================================================================
# BƯỚC 6: AUGMENTATION — ImageNet norm (CheXNet yeu cau)
# ================================================================================
print("\n[6/9] Augmentation (ImageNet norm)...")
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor(),
    transforms.Normalize(IM_MEAN, IM_STD),
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IM_MEAN, IM_STD),
])

# ================================================================================
# BƯỚC 7: DATASET / DATALOADER
# ================================================================================
print("\n[7/9] Datasets & loaders...")
class XrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
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

train_loader = DataLoader(XrayDataset(train_df, train_transform),
                          batch_size=BATCH_SIZE, sampler=sampler, drop_last=True, num_workers=0)
val_loader   = DataLoader(XrayDataset(val_df, val_test_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(XrayDataset(test_df, val_test_transform),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================================================================================
# BƯỚC 8: MODEL — DenseNet121 + CheXNet weights
# ================================================================================
print("\n[8/9] Building model (DenseNet121 + CheXNet)...")
model = models.densenet121(weights=None)

if os.path.exists(CHEXNET_PATH):
    ckpt = torch.load(CHEXNET_PATH, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    remapped = remap_chexnet(sd)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    n_miss = len([k for k in missing if 'num_batches' not in k and not k.startswith('classifier')])
    print(f"✓ CheXNet loaded | unexpected={len(unexpected)} | missing(real)={n_miss}")
else:
    print(f"⚠️  Khong thay {CHEXNET_PATH} — dung init ngau nhien!")

# Thay classifier 14 lop -> nhi phan + Dropout
model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(1024, 2))

# Stage 1: freeze features, chi train classifier
for p in model.features.parameters():
    p.requires_grad = False
for p in model.classifier.parameters():
    p.requires_grad = True
model = model.to(DEVICE)
print("✓ Stage 1: Features frozen | Classifier trainable")

# ================================================================================
# LOSS & OPTIMIZER
# ================================================================================
print("\n[9/9] Loss & optimizer...")
criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
print(f"✓ Focal Loss | Adam lr={LR_HEAD} | ReduceLROnPlateau")


def make_layerwise_groups(model, lr_backbone, lr_head):
    """Layerwise LR cho DenseNet: block sau LR cao hon, block dau LR thap."""
    g_low, g_mid, g_high, g_head = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith('classifier'):
            g_head.append(p)
        elif 'denseblock4' in n or 'norm5' in n:
            g_high.append(p)
        elif 'denseblock3' in n or 'transition3' in n:
            g_mid.append(p)
        else:
            g_low.append(p)
    return [
        {'params': g_low,  'lr': lr_backbone * 0.1},
        {'params': g_mid,  'lr': lr_backbone * 0.5},
        {'params': g_high, 'lr': lr_backbone},
        {'params': g_head, 'lr': lr_head},
    ]

# ================================================================================
# TRAINING LOOP (2-stage)
# ================================================================================
print("\nStarting training...")
print("=" * 80)
best_val_loss = float('inf')
patience_counter = 0
train_losses, val_losses = [], []
stage = 1

for epoch in range(EPOCHS):
    if epoch == UNFREEZE_EPOCH and stage == 1:
        print(f"\n🔓 Epoch {epoch+1}: Unfreeze features → Stage 2")
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.Adam(make_layerwise_groups(model, LR_BACKBONE, LR_HEAD))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
        stage = 2
        patience_counter = 0
        print("✓ Layerwise LR: block1-2=1e-6, block3=5e-6, block4=1e-5, head=1e-4")
        print("=" * 80)

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

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            val_loss += criterion(model(images), labels).item()
    avg_val = val_loss / len(val_loader)
    val_losses.append(avg_val)

    print(f"Epoch {epoch+1:3d}/{EPOCHS} [S{stage}] | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model_chexnet.pth'))
        print("  ✅ Best model saved → best_model_chexnet.pth")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    scheduler.step(avg_val)

# ================================================================================
# EVALUATE
# ================================================================================
print("\nLoading best model...")
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_model_chexnet.pth'), map_location=DEVICE))
model.eval()

print("\n" + "=" * 80)
print("TESTING (Youden's J threshold)")
print("=" * 80)
y_true, y_proba = [], []
with torch.no_grad():
    for images, labels in test_loader:
        probs = torch.softmax(model(images.to(DEVICE)), dim=1)
        y_proba.extend(probs[:, 1].cpu().numpy())
        y_true.extend(labels.numpy())
y_proba = np.array(y_proba); y_true = np.array(y_true)

roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
fpr, tpr, _ = roc_curve(y_true, y_proba)

threshold_list = np.arange(0.05, 0.95, 0.05)
f1_scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in threshold_list]

# Youden's J
_fpr, _tpr, _thr = roc_curve(y_true, y_proba)
optimal_threshold = float(_thr[np.argmax(_tpr - _fpr)])
y_pred = (y_proba >= optimal_threshold).astype(int)

_cm = confusion_matrix(y_true, y_pred)
_tn, _fp, _fn, _tp = _cm.ravel()
sensitivity = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0
specificity = _tn / (_tn + _fp) if (_tn + _fp) > 0 else 0.0

print(f"\n📊 METRICS:")
print(f"  ROC AUC            : {roc_auc:.4f}")
print(f"  Optimal Threshold  : {optimal_threshold:.2f}  (Youden's J)")
print(f"  Sensitivity (Co)   : {sensitivity:.3f}")
print(f"  Specificity (Khong): {specificity:.3f}")
print(f"  F1 tai threshold   : {f1_score(y_true, y_pred, zero_division=0):.4f}")

print("\n📋 CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=['Khong (0)', 'Co (1)'], zero_division=0))

# ================================================================================
# VISUALIZATION
# ================================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('DenseNet121 (CheXNet) — Benh Nghe Nghiep', fontsize=14, fontweight='bold')

axes[0,0].plot(train_losses, label='Train', marker='o', markersize=3, linewidth=2)
axes[0,0].plot(val_losses, label='Val', marker='s', markersize=3, linewidth=2)
axes[0,0].axvline(x=UNFREEZE_EPOCH, color='red', linestyle='--', alpha=0.7, label=f'Unfreeze (ep {UNFREEZE_EPOCH})')
axes[0,0].set_title('Loss Curve'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

axes[0,1].plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', linewidth=2)
axes[0,1].plot([0,1],[0,1],'k--'); axes[0,1].set_title('ROC Curve')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

axes[1,0].plot(threshold_list, f1_scores, marker='o', linewidth=2, markersize=5)
axes[1,0].axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal={optimal_threshold:.2f}')
axes[1,0].set_title('F1 vs Threshold'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
            xticklabels=['Khong','Co'], yticklabels=['Khong','Co'], annot_kws={'size':14})
axes[1,1].set_title('Confusion Matrix'); axes[1,1].set_ylabel('True'); axes[1,1].set_xlabel('Predicted')

plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, 'evaluation_chexnet.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {out_png}")

# Luu confusion matrix ra CSV (vao Finetune 3)
import csv as _csv
cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix_chexnet.csv')
with open(cm_path, 'w', newline='', encoding='utf-8') as f:
    w = _csv.writer(f)
    w.writerow(['', 'pred_Khong', 'pred_Co'])
    w.writerow(['true_Khong', int(cm[0, 0]), int(cm[0, 1])])
    w.writerow(['true_Co',    int(cm[1, 0]), int(cm[1, 1])])
print(f"✓ Saved: {cm_path}")

print("\n" + "=" * 80)
print("✅ DONE — CheXNet fine-tune")
print(f"  ROC AUC = {roc_auc:.4f} | Sensitivity = {sensitivity:.3f} | Specificity = {specificity:.3f}")
print("=" * 80)

# Dong log file
sys.stdout.flush()
_logfile.close()
