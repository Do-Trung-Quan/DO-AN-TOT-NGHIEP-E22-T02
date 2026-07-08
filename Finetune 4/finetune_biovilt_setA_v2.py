"""
================================================================================
FINE-TUNE BioViL-T (day du) tren set_A  —  V2 (TUNING)   [Finetune 4]
================================================================================
GIU NGUYEN BACKBONE BioViL-T (ImageModel ResNet50 multi-image, proj 128).
Chi thay doi THAM SO + HAM LOSS + AUGMENT + CHIEN LUOC CHON MODEL de xem
ket qua co tang len khong. Split GIU Y HET v1 (GroupShuffleSplit seed 42)
=> CUNG tap test 339 anh => so sanh cong bang voi v1 / BYOL / CheXNet.

KHAC v1 (finetune_biovilt_setA.py) — tat ca deu BAT/TAT duoc qua argparse:
  [1] --weight-decay   : them regularization cho Adam (v1 = 0)  -> giam overfit
  [2] --select-by auc  : chon best model theo VAL AUC thay vi val-loss
                         (metric bao cao la AUC; val-loss overfit som)
  [3] --tta            : test-time augmentation (lat ngang) -> +AUC gan nhu free
  [4] --erasing        : them RandomErasing (cutout) vao augment -> giam overfit
  [5] --color-jitter   : ColorJitter nhe (brightness/contrast) mo phong phoi sang
  [6] --loss {focal,ce}, --gamma, --label-smoothing : thu cac ham loss khac
  [7] --no-class-weight / --no-sampler : tat bot bu mat can bang (v1 bu 2 lan)
  [8] --scheduler {plateau,cosine}     : thu lich giam LR khac
  [9] --dropout        : chinh dropout o head

DUONG DAN: tham so hoa. Mac dinh tro toi o E: nhu v1. Neu data cho khac:
  python "Finetune 4/finetune_biovilt_setA_v2.py" --data-root "D:/path/Data DGCNN"
Hoac chinh tung duong dan: --excel --image-dir --backbone-dir --output-dir

CHAY:
  # kiem tra nhanh:
  python "Finetune 4/finetune_biovilt_setA_v2.py" --smoke
  # cong thuc "cai tien" mac dinh (wd + select-by-auc + tta + erasing):
  python "Finetune 4/finetune_biovilt_setA_v2.py"
  # revert ve dung nhu v1 de doi chung:
  python "Finetune 4/finetune_biovilt_setA_v2.py" --baseline

Xem cuoi file de co GOI Y SWEEP (thu tung thay doi 1 lan de biet cai nao giup).
================================================================================
"""
import os
import re
import sys
import json
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
                                     RandomHorizontalFlip, RandomRotation,
                                     ColorJitter, RandomApply, ToTensor, RandomErasing)
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
parser.add_argument("--baseline", action="store_true",
                    help="Revert TAT CA cai tien -> chay dung nhu v1 (doi chung)")

# --- paths (tham so hoa) ---
parser.add_argument("--data-root", default=r"e:\Hoc hanh\ĐÒ ÁN\Data DGCNN",
                    help="Thu muc goc; cac duong dan con suy ra tu day.")
parser.add_argument("--excel", default=None)
parser.add_argument("--image-dir", default=None)
parser.add_argument("--backbone-dir", default=None)
parser.add_argument("--output-dir", default=None)

# --- train hyperparams ---
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--unfreeze-epoch", type=int, default=6)
parser.add_argument("--lr-head", type=float, default=1e-4)
parser.add_argument("--lr-backbone", type=float, default=1e-5)
parser.add_argument("--weight-decay", type=float, default=1e-4,
                    help="[CAI TIEN] v1 = 0. Them regularization giam overfit.")
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--patience", type=int, default=7)
parser.add_argument("--scheduler", choices=["plateau", "cosine"], default="plateau")

# --- loss ---
parser.add_argument("--loss", choices=["focal", "ce"], default="focal")
parser.add_argument("--gamma", type=float, default=2.0, help="focal gamma")
parser.add_argument("--label-smoothing", type=float, default=0.0,
                    help="chi ap dung cho --loss ce")
parser.add_argument("--no-class-weight", action="store_true",
                    help="Bo alpha (class weight) trong loss")
parser.add_argument("--no-sampler", action="store_true",
                    help="Bo WeightedRandomSampler (dung shuffle thuong)")
parser.add_argument("--sampler-mult", type=float, default=2.0,
                    help="num_samples = len(train) * mult")

# --- augment ---
parser.add_argument("--erasing", dest="erasing", action="store_true", default=True,
                    help="[CAI TIEN] them RandomErasing (mac dinh BAT)")
parser.add_argument("--no-erasing", dest="erasing", action="store_false")
parser.add_argument("--color-jitter", dest="color_jitter", action="store_true", default=True,
                    help="[CAI TIEN] ColorJitter nhe (mac dinh BAT)")
parser.add_argument("--no-color-jitter", dest="color_jitter", action="store_false")

# --- eval / model selection ---
parser.add_argument("--select-by", choices=["loss", "auc"], default="auc",
                    help="[CAI TIEN] chon best model theo VAL AUC thay vi val-loss")
parser.add_argument("--tta", dest="tta", action="store_true", default=True,
                    help="[CAI TIEN] test-time augmentation lat ngang (mac dinh BAT)")
parser.add_argument("--no-tta", dest="tta", action="store_false")

parser.add_argument("--tag", default="v2", help="hau to ten file output de khong de len v1")
args = parser.parse_args()

# --baseline: ep tat ca ve dung v1
if args.baseline:
    args.weight_decay = 0.0
    args.select_by = "loss"
    args.tta = False
    args.erasing = False
    args.color_jitter = False
    args.scheduler = "plateau"
    args.loss = "focal"
    args.gamma = 2.0
    args.label_smoothing = 0.0
    args.no_class_weight = False
    args.no_sampler = False
    args.tag = "baseline"

# ================================================================================
# CONFIG (suy ra tu --data-root, cho phep override tung cai)
# ================================================================================
ROOT = args.data_root
EXCEL_PATH    = args.excel        or os.path.join(ROOT, "Finetune 2", "SetA_Labels.xlsx")
IMAGE_DIR     = args.image_dir    or os.path.join(ROOT, "Finetune", "images_256", "train")
BACKBONE_DIR  = args.backbone_dir or os.path.join(ROOT, "Pre-train BioViL-T")
OUTPUT_DIR    = args.output_dir   or os.path.join(ROOT, "Finetune 4")
BACKBONE_PATH = os.path.join(BACKBONE_DIR, "biovil_t_image_model_proj_size_128.pt")
IMAGE_COLUMN  = "file_name"
LABEL_COLUMN  = "bnn"

BATCH_SIZE     = args.batch_size
EPOCHS         = 2 if args.smoke else args.epochs
UNFREEZE_EPOCH = 1 if args.smoke else args.unfreeze_epoch
LR_HEAD        = args.lr_head
LR_BACKBONE    = args.lr_backbone
RESIZE         = 512
CROP           = 448
PATIENCE       = args.patience

os.makedirs(OUTPUT_DIR, exist_ok=True)
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

_suffix = "_smoke" if args.smoke else ""
_logname = f"train_log_{args.tag}{_suffix}.txt"
_logfile = open(os.path.join(OUTPUT_DIR, _logname), "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _logfile)

print("=" * 80)
print(f"FINE-TUNE BioViL-T V2 (tuning) — set_A  [{args.tag}]  {'[SMOKE]' if args.smoke else ''}")
print("=" * 80)
print(f"Backbone: {BACKBONE_PATH}")
print(f"Device={DEVICE} | AMP={USE_AMP} | Batch={BATCH_SIZE} | Epochs={EPOCHS} | Unfreeze@{UNFREEZE_EPOCH}")
print("[cfg] " + json.dumps({k: v for k, v in vars(args).items()
                             if k not in ("excel", "image_dir", "backbone_dir", "output_dir")},
                            ensure_ascii=False))

# ================================================================================
# LOSS: Focal (co the bo alpha) hoac CrossEntropy (co label smoothing)
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

def build_criterion(class_weights):
    alpha = None if args.no_class_weight else class_weights
    if args.loss == "focal":
        print(f"  Loss = Focal(gamma={args.gamma}, alpha={'None' if alpha is None else 'class_weights'})")
        return FocalLoss(alpha=alpha, gamma=args.gamma)
    print(f"  Loss = CrossEntropy(label_smoothing={args.label_smoothing}, "
          f"weight={'None' if alpha is None else 'class_weights'})")
    return nn.CrossEntropyLoss(weight=alpha, label_smoothing=args.label_smoothing)

# ================================================================================
# 1) LOAD EXCEL + MATCH + SPLIT  (GIU Y HET v1 -> cung test set)
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
# 2) TRANSFORMS (pipeline BioViL-T: ToTensor 0-1 + ExpandChannels, KHONG normalize)
#    [CAI TIEN] them ColorJitter (truoc ToTensor) + RandomErasing (sau ToTensor)
# ================================================================================
from health_multimodal.image.data.transforms import ExpandChannels

_train_ops = [Resize(RESIZE), RandomHorizontalFlip(p=0.5), RandomRotation(degrees=10)]
if args.color_jitter:
    # anh xam -> chi brightness/contrast co y nghia (mo phong khac biet phoi sang may chup)
    _train_ops.append(RandomApply([ColorJitter(brightness=0.2, contrast=0.2)], p=0.5))
_train_ops += [RandomCrop(CROP), ToTensor(), ExpandChannels()]
if args.erasing:
    _train_ops.append(RandomErasing(p=0.25, scale=(0.02, 0.15), value=0.0))
train_tf = Compose(_train_ops)

eval_tf = Compose([Resize(RESIZE), CenterCrop(CROP), ToTensor(), ExpandChannels()])
print(f"  Augment: color_jitter={args.color_jitter} | erasing={args.erasing}")

class XrayDataset(Dataset):
    def __init__(self, dframe, transform):
        self.df = dframe.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["image_path"]).convert("L")
        return self.transform(img), int(row[LABEL_COLUMN])

# ================================================================================
# 3) SAMPLER + LOADERS
# ================================================================================
counts = train_df[LABEL_COLUMN].value_counts().sort_index()
total = counts.sum()
weights_tensor = torch.tensor([total/counts[0], total/counts[1]], dtype=torch.float32).to(DEVICE)
print(f"  Counts {dict(counts)} | class weights {weights_tensor.cpu().numpy().round(2)}")

if args.no_sampler:
    print("  Sampler: OFF (shuffle thuong)")
    train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
else:
    sample_w = train_df[LABEL_COLUMN].map(lambda l: 1.0/counts[l]).tolist()
    sampler = WeightedRandomSampler(sample_w,
                                    num_samples=int(len(train_df)*args.sampler_mult),
                                    replacement=True)
    print(f"  Sampler: WeightedRandomSampler (x{args.sampler_mult})")
    train_loader = DataLoader(XrayDataset(train_df, train_tf), batch_size=BATCH_SIZE,
                              sampler=sampler, drop_last=True, num_workers=0)

val_loader  = DataLoader(XrayDataset(val_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(XrayDataset(test_df, eval_tf), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================================================================================
# 4) MODEL = BioViL-T backbone (GIU NGUYEN) + head nhi phan
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
        feat = self.backbone(x).img_embedding
        feat = feat.flatten(1)
        return self.head(feat)
    def set_backbone_trainable(self, flag):
        for p in self.backbone.parameters():
            p.requires_grad = flag

model = BioViLTClassifier(build_backbone(), dropout=args.dropout).to(DEVICE)
print(f"  feature_size = {model.backbone.feature_size} | dropout = {args.dropout}")

# Stage 1: dong bang backbone
model.set_backbone_trainable(False)
for p in model.head.parameters(): p.requires_grad = True
print("  Stage 1: backbone FROZEN | head trainable")

criterion = build_criterion(weights_tensor)

def make_optimizer(stage):
    if stage == 1:
        return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=LR_HEAD, weight_decay=args.weight_decay)
    return torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(),     "lr": LR_HEAD},
    ], weight_decay=args.weight_decay)

def make_scheduler(opt):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS - UNFREEZE_EPOCH))
    return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

optimizer = make_optimizer(1)
scheduler = make_scheduler(optimizer)
scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

# ================================================================================
# 5) TRAIN (2-stage) + chon best theo val-loss HOAC val-AUC
# ================================================================================
print("\n[3] Training...")
print("=" * 80)
best_metric = float("inf") if args.select_by == "loss" else -1.0
patience_ctr = 0; stage = 1
train_losses, val_losses, val_aucs = [], [], []
CKPT = os.path.join(OUTPUT_DIR, f"best_model_biovilt_{args.tag}.pth")

def train_one_epoch(loader):
    model.train()
    total = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        with torch.autocast("cuda", enabled=USE_AMP):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(loader, tta=False):
    """Tra ve (avg_loss, y_true, y_proba). tta=True -> trung binh softmax voi ban lat ngang."""
    model.eval()
    total = 0.0; ys, ps = [], []
    for images, labels in loader:
        images = images.to(DEVICE); lb = labels.to(DEVICE)
        with torch.autocast("cuda", enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, lb)
            prob = torch.softmax(logits, dim=1)
            if tta:
                prob_f = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)
                prob = 0.5 * (prob + prob_f)
        total += loss.item()
        ps.extend(prob[:, 1].float().cpu().numpy())
        ys.extend(labels.numpy())
    return total / max(1, len(loader)), np.array(ys), np.array(ps)

for epoch in range(EPOCHS):
    if epoch == UNFREEZE_EPOCH and stage == 1:
        print(f"\n[UNFREEZE] Epoch {epoch+1}: mo bang backbone -> Stage 2")
        model.set_backbone_trainable(True)
        optimizer = make_optimizer(2)
        scheduler = make_scheduler(optimizer)
        stage = 2; patience_ctr = 0

    tr = train_one_epoch(train_loader)
    va, vy, vp = evaluate(val_loader, tta=False)
    v_auc = roc_auc_score(vy, vp) if len(np.unique(vy)) > 1 else 0.0
    train_losses.append(tr); val_losses.append(va); val_aucs.append(v_auc)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} [S{stage}] | Train {tr:.4f} | Val {va:.4f} | ValAUC {v_auc:.4f}")

    if args.select_by == "loss":
        improved = va < best_metric
        cur = va
    else:
        improved = v_auc > best_metric
        cur = v_auc
    if improved:
        best_metric = cur; patience_ctr = 0
        torch.save(model.state_dict(), CKPT)
        print(f"  -> best saved ({args.select_by}={cur:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n[EARLY STOP] epoch {epoch+1}")
            break

    if args.scheduler == "plateau":
        scheduler.step(va)
    else:
        if stage == 2:
            scheduler.step()

# ================================================================================
# 6) EVALUATE (Youden's J) + tuy chon TTA
# ================================================================================
print("\nLoad best model...")
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))

_, y_true, y_proba = evaluate(test_loader, tta=args.tta)
print(f"  TTA (test-time flip): {args.tta}")

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
print(f"TEST (Youden's J)  [{args.tag}]")
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
    fig.suptitle(f"BioViL-T V2 [{args.tag}] — Benh Nghe Nghiep", fontsize=14, fontweight="bold")
    ax[0,0].plot(train_losses, label="Train", marker="o", ms=3)
    ax[0,0].plot(val_losses, label="Val", marker="s", ms=3)
    ax[0,0].plot(val_aucs, label="Val AUC", marker="^", ms=3)
    ax[0,0].axvline(x=UNFREEZE_EPOCH, color="red", ls="--", alpha=0.6, label=f"Unfreeze ep{UNFREEZE_EPOCH}")
    ax[0,0].set_title("Loss / ValAUC"); ax[0,0].legend(); ax[0,0].grid(alpha=0.3)
    ax[0,1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}"); ax[0,1].plot([0,1],[0,1],"k--")
    ax[0,1].set_title("ROC"); ax[0,1].legend(); ax[0,1].grid(alpha=0.3)
    ax[1,0].plot(thr_list, f1s, marker="o"); ax[1,0].axvline(x=opt_thr, color="r", ls="--", label=f"Opt={opt_thr:.2f}")
    ax[1,0].set_title("F1 vs Threshold"); ax[1,0].legend(); ax[1,0].grid(alpha=0.3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[1,1],
                xticklabels=["Khong","Co"], yticklabels=["Khong","Co"], annot_kws={"size":14})
    ax[1,1].set_title("Confusion Matrix"); ax[1,1].set_ylabel("True"); ax[1,1].set_xlabel("Predicted")
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, f"evaluation_biovilt_{args.tag}.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight"); print(f"\nSaved -> {out_png}")
    import csv as _csv
    with open(os.path.join(OUTPUT_DIR, f"confusion_matrix_biovilt_{args.tag}.csv"), "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f); w.writerow(["", "pred_Khong", "pred_Co"])
        w.writerow(["true_Khong", int(cm[0,0]), int(cm[0,1])])
        w.writerow(["true_Co",    int(cm[1,0]), int(cm[1,1])])
    print(f"Saved -> confusion_matrix_biovilt_{args.tag}.csv")

print("\n" + "=" * 80)
print(f"DONE — BioViL-T V2 [{args.tag}] | AUC={roc_auc:.4f} | Sens={sens:.3f} | Spec={spec:.3f}")
print("=" * 80)

sys.stdout.flush()
sys.stdout = sys.__stdout__
_logfile.close()

# ================================================================================
# GOI Y SWEEP (chay tung cai, so voi baseline AUC=0.902 / Sens=0.906 / Spec=0.781)
# ------------------------------------------------------------------------------
#  0) Doi chung  :  --baseline
#  1) +regularize:  (mac dinh da bat wd + select-auc + tta + erasing)
#  2) Chi wd     :  --no-tta --no-erasing --no-color-jitter --select-by loss --weight-decay 1e-4
#  3) Chi select-auc: --no-tta --no-erasing --no-color-jitter --weight-decay 0 --select-by auc
#  4) Chi TTA    :  --no-erasing --no-color-jitter --weight-decay 0 --select-by loss --tta
#  5) Loss gamma :  --gamma 1.0   /   --gamma 3.0
#  6) Label smooth: --loss ce --label-smoothing 0.05
#  7) Bo bu 2 lan:  --no-sampler        (giu focal alpha)   -> xem spec/sens doi
#                   --no-class-weight    (giu sampler)
#  8) Cosine LR  :  --scheduler cosine
#  9) Dropout    :  --dropout 0.5
# ------------------------------------------------------------------------------
# Moi lan chay tu dong luu: train_log_<tag>.txt, best_model_biovilt_<tag>.pth,
# evaluation_biovilt_<tag>.png, confusion_matrix_biovilt_<tag>.csv
# Dat --tag rieng cho moi thi nghiem de khong de len nhau, vd:
#   python "Finetune 4/finetune_biovilt_setA_v2.py" --gamma 1.0 --tag g1.0
# ================================================================================
