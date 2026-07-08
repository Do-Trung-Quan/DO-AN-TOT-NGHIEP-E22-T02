"""
================================================================================
FINE-TUNE BioViL-T (day du) tren set_A   [Finetune 4]
================================================================================
Bien the cua Finetune 3 (CheXNet): thay backbone -> BioViL-T (Microsoft, vision-language).
Fine-tune DAY DU (mo bang encoder) de SO SANH apples-to-apples voi BYOL/CheXNet.

  - Finetune 2 = BYOL-ResNet18
  - Finetune 3 = CheXNet-DenseNet121
  - Finetune 4 = BioViL-T  (file nay)

Khac biet quan trong so voi CheXNet:
  - Backbone: BioViL-T ImageModel (ResNet50 multi-image, projector 128) tu hi-ml-multimodal
  - Load tu FILE LOCAL trong folder "Pre-train BioViL-T":
        Pre-train BioViL-T/biovil_t_image_model_proj_size_128.pt
    (neu khong thay -> tu dong tai tu HuggingFace)
  - Feature de gan head: out.img_embedding (pooled ~2048-d)
  - Tien xu ly RIENG cua BioViL-T: Resize(512)->Crop(448)->ToTensor(0-1)->ExpandChannels
    (KHONG co ImageNet normalize, anh xam -> 3 kenh bang repeat)
  - AMP (mixed precision) de vua VRAM 6GB; neu OOM giam --batch-size

Output (vao Finetune 4/):
  - best_model_biovilt_finetuned.pth
  - evaluation_biovilt_finetune.png, confusion_matrix_biovilt_finetune.csv, train_log_finetune.txt

DUNG CUNG split (GroupShuffleSplit seed 42, 2 tang) -> cung tap test 339 anh voi BYOL/CheXNet.

CHAY:
  # test nhanh truoc (2 epoch, it anh):
  python "Finetune 4/finetune_biovilt_setA.py" --smoke
  # chay that:
  python "Finetune 4/finetune_biovilt_setA.py"
================================================================================
"""
import os
import re
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import (Compose, Resize, CenterCrop, RandomCrop,
                                     RandomHorizontalFlip, RandomRotation, ToTensor)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ================================================================================
# ARGS
# ================================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="chay nhanh de test (2 epoch, subset)")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--unfreeze-epoch", type=int, default=6)
args = parser.parse_args()

# ================================================================================
# CONFIG  (paths giong Finetune 3; backbone doc tu folder Pre-train BioViL-T)
# ================================================================================
EXCEL_PATH    = r"C:\Users\ASUS\Downloads\SetA_Labels.xlsx"
IMAGE_DIR     = r"C:\Users\ASUS\Downloads\Silicodata_Updated_Feb2025\set_A_folder\train_images_256"
BACKBONE_DIR  = r"C:\Users\ASUS\AndroidStudioProjects\cgv1\New folder (2)\DO-AN-TOT-NGHIEP-E22-T02\Pre-train BioViL-T"
BACKBONE_PATH = os.path.join(BACKBONE_DIR, "biovil_t_image_model_proj_size_128.pt")
IMAGE_COLUMN  = "file_name"
LABEL_COLUMN  = "bnn"

BATCH_SIZE     = args.batch_size
EPOCHS         = 2 if args.smoke else args.epochs
UNFREEZE_EPOCH = 1 if args.smoke else args.unfreeze_epoch
LR_HEAD        = 1e-4
LR_BACKBONE    = 1e-5
RESIZE         = 512
CROP           = 448
PATIENCE       = 7


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

# ---- Tee log ----
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            if getattr(s, "closed", False): continue
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams:
            if getattr(s, "closed", False): continue
            s.flush()

_logname = "train_log_finetune_smoke.txt" if args.smoke else "train_log_finetune.txt"
_logfile = open(_logname, "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _logfile)

print("=" * 80)
print(f"FINE-TUNE BioViL-T — set_A  [Finetune 4]  {'[SMOKE]' if args.smoke else ''}")
print("=" * 80)
print(f"Backbone: {BACKBONE_PATH}")
print(f"Device={DEVICE} | AMP={USE_AMP} | Batch={BATCH_SIZE} | Epochs={EPOCHS} | Unfreeze@{UNFREEZE_EPOCH}")

# ================================================================================
# FOCAL LOSS  (giong CheXNet)
# ================================================================================
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

# ================================================================================
# 1) LOAD EXCEL + MATCH + SPLIT  (giong Finetune 3)
# ================================================================================
print("\n[1] Excel + match + split...")
df = pd.read_excel(EXCEL_PATH)
df[LABEL_COLUMN] = df[LABEL_COLUMN].map({"Khong": 0, "Co": 1})

all_files = {}
for root, _d, files in os.walk(IMAGE_DIR):
    for f in files:
        all_files[f.lower().strip()] = os.path.join(root, f)
df["image_path"] = df[IMAGE_COLUMN].apply(lambda x: all_files.get(str(x).strip().lower()))
df = df[df["image_path"].notna()].reset_index(drop=True)

df["patient_base"] = df[IMAGE_COLUMN].apply(
    lambda x: re.sub(r"_\d+\.(jpg|jpeg|png|bmp|tiff|tif)$", "", str(x).lower().strip()))
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_idx, tmp_idx = next(gss.split(df, df[LABEL_COLUMN], groups=df["patient_base"]))
train_df, temp_df = df.iloc[tr_idx].copy(), df.iloc[tmp_idx].copy()
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
v_idx, te_idx = next(gss2.split(temp_df, temp_df[LABEL_COLUMN], groups=temp_df["patient_base"]))
val_df, test_df = temp_df.iloc[v_idx].copy(), temp_df.iloc[te_idx].copy()

if args.smoke:
    train_df = train_df.sample(n=min(80, len(train_df)), random_state=0).reset_index(drop=True)
    val_df   = val_df.sample(n=min(40, len(val_df)), random_state=0).reset_index(drop=True)
    test_df  = test_df.sample(n=min(40, len(test_df)), random_state=0).reset_index(drop=True)

print(f"  Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
assert len(set(train_df["patient_base"]) & set(test_df["patient_base"])) == 0, "Leakage!"

# ================================================================================
# 2) TRANSFORMS  (DUNG dung pipeline BioViL-T: ToTensor 0-1 + ExpandChannels, KHONG normalize)
# ================================================================================
from health_multimodal.image.data.transforms import ExpandChannels

train_tf = Compose([
    Resize(RESIZE),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    RandomCrop(CROP),
    ToTensor(),
    ExpandChannels(),
])
eval_tf = Compose([
    Resize(RESIZE),
    CenterCrop(CROP),
    ToTensor(),
    ExpandChannels(),
])

class XrayDataset(Dataset):
    def __init__(self, dframe, transform):
        self.df = dframe.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("L")   # 1 kenh -> ExpandChannels can [1,H,W]
        return self.transform(img), int(row[LABEL_COLUMN])

# ================================================================================
# 3) SAMPLER + LOADERS  (giong CheXNet)
# ================================================================================
counts = train_df[LABEL_COLUMN].value_counts().sort_index()
total = counts.sum()
weights_tensor = torch.tensor([total/counts[0], total/counts[1]], dtype=torch.float32).to(DEVICE)
sample_w = train_df[LABEL_COLUMN].map(lambda l: 1.0/counts[l]).tolist()
sampler = WeightedRandomSampler(sample_w, num_samples=len(train_df)*2, replacement=True)
print(f"  Counts {dict(counts)} | class weights {weights_tensor.cpu().numpy().round(2)}")

train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                          sampler=sampler, drop_last=True, num_workers=0)
val_loader   = DataLoader(XrayDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(XrayDataset(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================================================================================
# 4) MODEL = BioViL-T backbone (doc tu Pre-train BioViL-T) + head nhi phan
# ================================================================================
print("\n[2] Building BioViL-T model...")
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.types import ImageEncoderType

def build_backbone():
    if os.path.exists(BACKBONE_PATH):
        print(f"  Load backbone LOCAL: {BACKBONE_PATH}")
        return ImageModel(img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
                          joint_feature_size=128, pretrained_model_path=BACKBONE_PATH)
    print("  Khong thay file local -> tu dong tai tu HuggingFace...")
    from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
    return get_biovil_t_image_encoder()

class BioViLTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(backbone.feature_size, num_classes))
    def forward(self, x):
        feat = self.backbone(x).img_embedding      # [B, feature_size]
        feat = feat.flatten(1)
        return self.head(feat)
    def set_backbone_trainable(self, flag):
        for p in self.backbone.parameters():
            p.requires_grad = flag

model = BioViLTClassifier(build_backbone()).to(DEVICE)
print(f"  feature_size = {model.backbone.feature_size}")

# Stage 1: dong bang backbone, chi train head
model.set_backbone_trainable(False)
for p in model.head.parameters(): p.requires_grad = True
print("  Stage 1: backbone FROZEN | head trainable")

criterion = FocalLoss(alpha=weights_tensor, gamma=5.0)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ================================================================================
# 5) TRAIN (2-stage)
# ================================================================================
print("\n[3] Training...")
print("=" * 80)
best_val = float("inf"); patience_ctr = 0; stage = 1
train_losses, val_losses = [], []
CKPT = "best_model_biovilt_finetuned.pth"

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        if train: optimizer.zero_grad()
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, labels)
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))

for epoch in range(EPOCHS):
    if epoch == UNFREEZE_EPOCH and stage == 1:
        print(f"\n[UNFREEZE] Epoch {epoch+1}: mo bang backbone -> Stage 2")
        model.set_backbone_trainable(True)
        optimizer = torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
            {"params": model.head.parameters(),     "lr": LR_HEAD},
        ])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        stage = 2; patience_ctr = 0

    tr = run_epoch(train_loader, True)
    va = run_epoch(val_loader, False)
    train_losses.append(tr); val_losses.append(va)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} [S{stage}] | Train {tr:.4f} | Val {va:.4f}")

    if va < best_val:
        best_val = va; patience_ctr = 0
        torch.save(model.state_dict(), CKPT)
        print("  -> best saved")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[EARLY STOP] epoch {epoch+1}")
            break
    scheduler.step(va)

# ================================================================================
# 6) EVALUATE (Youden's J — giong CheXNet)
# ================================================================================
print("\nLoad best model...")
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
model.eval()

y_true, y_proba = [], []
with torch.no_grad():
    for images, labels in test_loader:
        with torch.autocast("cuda", enabled=USE_AMP):
            probs = torch.softmax(model(images.to(DEVICE)), dim=1)
        y_proba.extend(probs[:, 1].float().cpu().numpy())
        y_true.extend(labels.numpy())
y_proba = np.array(y_proba); y_true = np.array(y_true)

roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
fpr, tpr, _ = roc_curve(y_true, y_proba)
_fpr, _tpr, _thr = roc_curve(y_true, y_proba)
opt_thr = float(_thr[np.argmax(_tpr - _fpr)])
y_pred = (y_proba >= opt_thr).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
sens = tp/(tp+fn) if (tp+fn) > 0 else 0.0
spec = tn/(tn+fp) if (tn+fp) > 0 else 0.0
acc = (tp+tn)/cm.sum()

print("\n" + "=" * 80)
print("TEST (Youden's J)")
print("=" * 80)
print(f"  ROC AUC            : {roc_auc:.4f}")
print(f"  Accuracy           : {acc:.4f}")
print(f"  Optimal Threshold  : {opt_thr:.3f}")
print(f"  Sensitivity (Co)   : {sens:.3f}")
print(f"  Specificity (Khong): {spec:.3f}")
print(f"  F1 (Co)            : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("\n" + classification_report(y_true, y_pred, target_names=["Khong (0)", "Co (1)"], zero_division=0))

# ================================================================================
# 7) SAVE figure + cm csv  (bo qua neu smoke)
# ================================================================================
if not args.smoke:
    thr_list = np.arange(0.05, 0.95, 0.05)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thr_list]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BioViL-T (fine-tune day du) — Benh Nghe Nghiep", fontsize=14, fontweight="bold")
    ax[0,0].plot(train_losses, label="Train", marker="o", ms=3); ax[0,0].plot(val_losses, label="Val", marker="s", ms=3)
    ax[0,0].axvline(x=UNFREEZE_EPOCH, color="red", ls="--", alpha=0.6, label=f"Unfreeze ep{UNFREEZE_EPOCH}")
    ax[0,0].set_title("Loss"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); ax[0,1].plot([0,1],[0,1],"k--")
    ax[0,1].set_title("ROC"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
    ax[1,0].plot(thr_list, f1s, marker="o"); ax[1,0].axvline(x=opt_thr, color="r", ls="--", label=f"Opt={opt_thr:.2f}")
    ax[1,0].set_title("F1 vs Threshold"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1,1],
                xticklabels=["Khong","Co"], yticklabels=["Khong","Co"], annot_kws={"size":14})
    ax[1,1].set_title("Confusion Matrix"); ax[1,1].set_ylabel("True"); ax[1,1].set_xlabel("Predicted")
    plt.tight_layout()
    out_png = "evaluation_biovilt_finetune.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); print(f"\nSaved -> {out_png}")
    import csv as _csv
    with open("confusion_matrix_biovilt_finetune.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f); w.writerow(["", "pred_Khong", "pred_Co"])
        w.writerow(["true_Khong", int(cm[0,0]), int(cm[0,1])])
        w.writerow(["true_Co",    int(cm[1,0]), int(cm[1,1])])
    print("Saved -> confusion_matrix_biovilt_finetune.csv")

print("\n" + "=" * 80)
print(f"DONE — BioViL-T fine-tune | AUC={roc_auc:.4f} | Sens={sens:.3f} | Spec={spec:.3f}")
print("=" * 80)

sys.stdout.flush()
sys.stdout = sys.__stdout__
_logfile.close()
