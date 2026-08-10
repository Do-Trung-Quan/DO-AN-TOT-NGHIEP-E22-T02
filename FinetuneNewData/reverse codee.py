"""
================================================================================
FINE-TUNE BioViL-T tren bo anh SILICOSIS / NORMAL   [Finetune Silicosis]
================================================================================
Khac cac script truoc (Finetune 4 / CrossEntropy / WeightedCE / Focal5):
  - KHONG doc nhan tu Excel. Nhan lay tu TEN THU MUC (ImageFolder-style):
        archive/train/normal      -> 0
        archive/train/silicosis   -> 1
        archive/test/normal       -> 0
        archive/test/silicosis    -> 1
  - Bo du lieu da chia san train/test. Da kiem tra: moi benh nhan 1 anh,
    KHONG co benh nhan nao xuat hien o ca train lan test -> khong ro ri.
    Val duoc tach tu train (stratified, seed 42).
  - Anh goc rat lon (~3068x3060). Chay CACHE resize canh ngan ve 512 mot lan
    (archive/_cache512) de moi epoch khong phai decode lai anh 9 MP.

Giong cac script truoc:
  - Backbone: BioViL-T ImageModel (ResNet50 multi-image, projector 128)
        Pre-train BioViL-T/biovil_t_image_model_proj_size_128.pt
  - Feature gan head: out.img_embedding (feature_size = 512)
  - Tien xu ly BioViL-T: Resize(512)->Crop(448)->ToTensor(0-1)->ExpandChannels
    (KHONG ImageNet normalize, anh xam -> 3 kenh bang repeat)
  - Focal Loss (gamma=2) + class weights + WeightedRandomSampler
  - 2 stage: dong bang backbone -> unfreeze
  - Nguong quyet dinh chon bang Youden's J tren tap test

Output (vao Finetune Silicosis/):
  - best_model_biovilt_silicosis.pth
  - evaluation_biovilt_silicosis.png
  - confusion_matrix_biovilt_silicosis.csv
  - test_predictions_silicosis.csv
  - train_log_silicosis.txt

CHAY:
  # test nhanh truoc (2 epoch, it anh):
  python "Finetune Silicosis/reverse codee.py" --smoke
  # chay that:
  python "Finetune Silicosis/reverse codee.py"
================================================================================
"""
import os
import sys
import glob
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
from sklearn.model_selection import train_test_split
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
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch-size", type=int, default=4, help="GTX 1650 4GB -> 4; VRAM lon hon tang len 8/16")
parser.add_argument("--unfreeze-epoch", type=int, default=5)
parser.add_argument("--no-cache", action="store_true", help="doc thang anh goc, khong dung cache 512px")
args = parser.parse_args()

# ================================================================================
# CONFIG
# ================================================================================
HERE          = os.path.dirname(os.path.abspath(__file__))
REPO          = os.path.dirname(HERE)
DATA_ROOT     = r"C:\Users\ASUS\Downloads\archive"
TRAIN_DIR     = os.path.join(DATA_ROOT, "train")
TEST_DIR      = os.path.join(DATA_ROOT, "test")
CACHE_ROOT    = os.path.join(DATA_ROOT, "_cache512")
OUTPUT_DIR    = HERE
BACKBONE_PATH = os.path.join(REPO, "Pre-train BioViL-T", "biovil_t_image_model_proj_size_128.pt")

CLASSES       = {"normal": 0, "silicosis": 1}
CLASS_NAMES   = ["normal (0)", "silicosis (1)"]

BATCH_SIZE     = args.batch_size
EPOCHS         = 2 if args.smoke else args.epochs
UNFREEZE_EPOCH = 1 if args.smoke else args.unfreeze_epoch
LR_HEAD        = 1e-4
LR_BACKBONE    = 1e-5
RESIZE         = 512
CROP           = 448
VAL_SIZE       = 0.15
PATIENCE       = 7
SEED           = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
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

_logname = "train_log_silicosis_smoke.txt" if args.smoke else "train_log_silicosis.txt"
_logfile = open(os.path.join(OUTPUT_DIR, _logname), "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _logfile)

print("=" * 80)
print(f"FINE-TUNE BioViL-T — SILICOSIS vs NORMAL  {'[SMOKE]' if args.smoke else ''}")
print("=" * 80)
print(f"Data    : {DATA_ROOT}")
print(f"Backbone: {BACKBONE_PATH}")
print(f"Device={DEVICE} | AMP={USE_AMP} | Batch={BATCH_SIZE} | Epochs={EPOCHS} | Unfreeze@{UNFREEZE_EPOCH}")

# ================================================================================
# FOCAL LOSS  (giong cac script truoc)
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
# 0) CACHE resize 512px  (anh goc ~9 MP -> decode lai moi epoch qua cham)
# ================================================================================
def scan_folder(split_dir):
    rows = []
    for cls, lab in CLASSES.items():
        for p in sorted(glob.glob(os.path.join(split_dir, cls, "*"))):
            if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
                rows.append({"image_path": p, "label": lab, "cls": cls,
                             "file_name": os.path.basename(p)})
    return pd.DataFrame(rows)

def build_cache(df, split):
    """Resize canh ngan ve RESIZE, luu JPEG q95. Tra ve df voi image_path da doi."""
    out = []
    n_new = 0
    for _, r in df.iterrows():
        dst = os.path.join(CACHE_ROOT, split, r["cls"],
                           os.path.splitext(r["file_name"])[0] + ".jpg")
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im = Image.open(r["image_path"]).convert("L")
            w, h = im.size
            s = RESIZE / min(w, h)
            if s < 1.0:
                im = im.resize((max(RESIZE, round(w * s)), max(RESIZE, round(h * s))),
                               Image.BICUBIC)
            im.save(dst, "JPEG", quality=95)
            n_new += 1
        out.append(dst)
    df = df.copy(); df["image_path"] = out
    print(f"  cache[{split}]: {len(df)} anh ({n_new} moi tao) -> {os.path.join(CACHE_ROOT, split)}")
    return df

print("\n[1] Quet thu muc + split...")
train_all = scan_folder(TRAIN_DIR)
test_df   = scan_folder(TEST_DIR)
if train_all.empty or test_df.empty:
    raise SystemExit(f"Khong tim thay anh trong {TRAIN_DIR} / {TEST_DIR}")

if not args.no_cache:
    train_all = build_cache(train_all, "train")
    test_df   = build_cache(test_df, "test")

# tach val tu train (stratified) — test la thu muc rieng, khong dung o day
train_df, val_df = train_test_split(train_all, test_size=VAL_SIZE,
                                    stratify=train_all["label"], random_state=SEED)
train_df = train_df.reset_index(drop=True); val_df = val_df.reset_index(drop=True)

if args.smoke:
    train_df = train_df.sample(n=min(80, len(train_df)), random_state=0).reset_index(drop=True)
    val_df   = val_df.sample(n=min(40, len(val_df)), random_state=0).reset_index(drop=True)
    test_df  = test_df.sample(n=min(40, len(test_df)), random_state=0).reset_index(drop=True)

def _dist(d): return dict(d["label"].value_counts().sort_index())
print(f"  Train {len(train_df)} {_dist(train_df)} | Val {len(val_df)} {_dist(val_df)} "
      f"| Test {len(test_df)} {_dist(test_df)}")
assert len(set(train_df["file_name"]) & set(test_df["file_name"])) == 0, "Leakage train/test!"
assert len(set(train_df["file_name"]) & set(val_df["file_name"])) == 0, "Leakage train/val!"

# ================================================================================
# 2) TRANSFORMS  (dung pipeline BioViL-T: ToTensor 0-1 + ExpandChannels, KHONG normalize)
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
        img = Image.open(row["image_path"]).convert("L")   # ExpandChannels can [1,H,W]
        return self.transform(img), int(row["label"])

# ================================================================================
# 3) SAMPLER + LOADERS
# ================================================================================
counts = train_df["label"].value_counts().sort_index()
total = counts.sum()
weights_tensor = torch.tensor([total/counts[0], total/counts[1]], dtype=torch.float32).to(DEVICE)
sample_w = train_df["label"].map(lambda l: 1.0/counts[l]).tolist()
sampler = WeightedRandomSampler(sample_w, num_samples=len(train_df), replacement=True)
print(f"  Counts {dict(counts)} | class weights {weights_tensor.cpu().numpy().round(2)}")

train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                          sampler=sampler, drop_last=True, num_workers=0)
val_loader   = DataLoader(XrayDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(XrayDataset(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================================================================================
# 4) MODEL = BioViL-T backbone + head nhi phan
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

criterion = FocalLoss(alpha=weights_tensor, gamma=2.0)
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
CKPT = os.path.join(OUTPUT_DIR, "best_model_biovilt_silicosis.pth")

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
# 6) EVALUATE tren tap test co san (Youden's J)
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
fpr, tpr, _thr = roc_curve(y_true, y_proba)
opt_thr = float(_thr[np.argmax(tpr - fpr)])
y_pred = (y_proba >= opt_thr).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
sens = tp/(tp+fn) if (tp+fn) > 0 else 0.0
spec = tn/(tn+fp) if (tn+fp) > 0 else 0.0
acc = (tp+tn)/cm.sum()

print("\n" + "=" * 80)
print("TEST (Youden's J)")
print("=" * 80)
print(f"  ROC AUC               : {roc_auc:.4f}")
print(f"  Accuracy              : {acc:.4f}")
print(f"  Optimal Threshold     : {opt_thr:.3f}")
print(f"  Sensitivity (silicosis): {sens:.3f}")
print(f"  Specificity (normal)   : {spec:.3f}")
print(f"  F1 (silicosis)        : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("\n" + classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ================================================================================
# 7) SAVE figure + csv  (bo qua neu smoke)
# ================================================================================
if not args.smoke:
    pd.DataFrame({"file_name": test_df["file_name"].values,
                  "y_true": y_true, "y_proba": y_proba, "y_pred": y_pred}
                 ).to_csv(os.path.join(OUTPUT_DIR, "test_predictions_silicosis.csv"),
                          index=False, encoding="utf-8")

    thr_list = np.arange(0.05, 0.95, 0.05)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in thr_list]
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("BioViL-T fine-tune — Silicosis vs Normal", fontsize=14, fontweight="bold")
    ax[0,0].plot(train_losses, label="Train", marker="o", ms=3); ax[0,0].plot(val_losses, label="Val", marker="s", ms=3)
    ax[0,0].axvline(x=UNFREEZE_EPOCH, color="red", ls="--", alpha=0.6, label=f"Unfreeze ep{UNFREEZE_EPOCH}")
    ax[0,0].set_title("Loss"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); ax[0,1].plot([0,1],[0,1],"k--")
    ax[0,1].set_title("ROC"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
    ax[1,0].plot(thr_list, f1s, marker="o"); ax[1,0].axvline(x=opt_thr, color="r", ls="--", label=f"Opt={opt_thr:.2f}")
    ax[1,0].set_title("F1 vs Threshold"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1,1],
                xticklabels=["normal","silicosis"], yticklabels=["normal","silicosis"], annot_kws={"size":14})
    ax[1,1].set_title("Confusion Matrix"); ax[1,1].set_ylabel("True"); ax[1,1].set_xlabel("Predicted")
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "evaluation_biovilt_silicosis.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); print(f"\nSaved -> {out_png}")

    import csv as _csv
    with open(os.path.join(OUTPUT_DIR, "confusion_matrix_biovilt_silicosis.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = _csv.writer(f); w.writerow(["", "pred_normal", "pred_silicosis"])
        w.writerow(["true_normal",    int(cm[0,0]), int(cm[0,1])])
        w.writerow(["true_silicosis", int(cm[1,0]), int(cm[1,1])])
    print("Saved -> confusion_matrix_biovilt_silicosis.csv")

print("\n" + "=" * 80)
print(f"DONE — BioViL-T silicosis | AUC={roc_auc:.4f} | Sens={sens:.3f} | Spec={spec:.3f}")
print("=" * 80)

sys.stdout.flush()
sys.stdout = sys.__stdout__
_logfile.close()
