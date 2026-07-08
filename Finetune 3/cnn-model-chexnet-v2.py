"""
================================================================================
FINE-TUNE DENSENET121 (CheXNet) — V2 (CAI TIEN)   [Finetune 3]
================================================================================
GIU NGUYEN BACKBONE DenseNet121 + trong so CheXNet (arnoweng).
Chi thay doi THAM SO + HAM LOSS + AUGMENT + CHIEN LUOC CHON MODEL de xem
ket qua co tang len khong. Split GIU Y HET ban goc (GroupShuffleSplit seed 42)
=> CUNG tap test => so sanh cong bang voi ban goc.

CAI TIEN so voi cnn-model-chexnet.py (da bake san, khong can co dong lenh):
  [1] weight_decay = 1e-4 cho Adam (goc = 0)     -> giam overfit
  [2] Chon best model theo VAL AUC (goc: val-loss) -> bam dung metric bao cao
  [3] TTA (test-time flip): trung binh softmax anh goc + anh lat ngang -> +AUC
  [4] RandomErasing (cutout) trong augment       -> regularization
  [5] ColorJitter nhe (brightness/contrast)      -> mo phong khac biet phoi sang
  Van giu: Focal Loss (gamma=2), WeightedRandomSampler + class weight,
           layerwise LR cho DenseNet, Youden's J chon nguong.

>>> SUA DUONG DAN o muc CONFIG ben duoi (EXCEL_PATH / IMAGE_DIR / CHEXNET_PATH).
>>> Muon chay thu nhanh: dat SMOKE = True.

CHAY:  python "Finetune 3/cnn-model-chexnet-v2.py"
================================================================================
"""
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ================================================================================
# CONFIG  —  SUA O DAY
# ================================================================================
EXCEL_PATH    = r"C:\Users\ASUS\PycharmProjects\JupyterProject\SetA_Labels.xlsx"
IMAGE_DIR     = r"C:\Users\ASUS\Downloads\Silicodata_Updated_Feb2025\set_A_folder\train_images_256"
CHEXNET_PATH  = r"C:\Users\ASUS\AndroidStudioProjects\cgv1\New folder (2)\DO-AN-TOT-NGHIEP-E22-T02\Pre-train CheXNET\model.pth"
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))   # ghi ket qua ngay canh script
IMAGE_COLUMN  = "file_name"
LABEL_COLUMN  = "bnn"

SMOKE          = True    # True = chay nhanh 2 epoch + subset de test
BATCH_SIZE     = 16
EPOCHS         = 50
LR_HEAD        = 1e-4
LR_BACKBONE    = 1e-5
UNFREEZE_EPOCH = 10
IMAGE_SIZE     = 224
PATIENCE       = 7
WEIGHT_DECAY   = 1e-4    # [CAI TIEN 1] goc = 0
DROPOUT        = 0.3
FOCAL_GAMMA    = 2.0

if SMOKE:
    EPOCHS = 2
    UNFREEZE_EPOCH = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Tee: in ra console + ghi train_log.txt ----
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

_logfile = open(os.path.join(OUTPUT_DIR, "train_log_v2.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _logfile)

print("=" * 80)
print("FINE-TUNING DENSENET121 (CheXNet) — V2 CAI TIEN")
print("=" * 80)
print(f"Device={DEVICE} | Batch={BATCH_SIZE} | Epochs={EPOCHS} | Unfreeze@{UNFREEZE_EPOCH}")
print(f"weight_decay={WEIGHT_DECAY} | dropout={DROPOUT} | focal_gamma={FOCAL_GAMMA}")

# ================================================================================
# FOCAL LOSS
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
# REMAP CHEXNET (arnoweng -> torchvision densenet121)
# ================================================================================
def remap_chexnet(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k.replace("module.densenet121.", "")
        nk = re.sub(r"norm\.(\d)", r"norm\1", nk)
        nk = re.sub(r"conv\.(\d)", r"conv\1", nk)
        if nk.startswith("classifier"):
            continue
        out[nk] = v
    return out

# ================================================================================
# [1] LOAD EXCEL + ENCODE
# ================================================================================
print("\n[1/8] Loading Excel...")
df = pd.read_excel(EXCEL_PATH)
df[LABEL_COLUMN] = df[LABEL_COLUMN].map({"Khong": 0, "Co": 1})
print(f"  {len(df)} rows | {dict(df[LABEL_COLUMN].value_counts())}")

# ================================================================================
# [2] MATCH IMAGES (quet de quy tat ca subfolder)
# ================================================================================
print("\n[2/8] Matching images...")
all_files = {}
for root, _d, files in os.walk(IMAGE_DIR):
    for f in files:
        all_files[f.lower().strip()] = os.path.join(root, f)
df["image_path"] = df[IMAGE_COLUMN].apply(lambda x: all_files.get(str(x).strip().lower()))
n_total = len(df)
df = df[df["image_path"].notna()].reset_index(drop=True)
print(f"  Tong anh tim thay: {len(all_files)} | Matched {len(df)}/{n_total}")

# ================================================================================
# [3] SPLIT (group by patient, seed 42 -> cung test set voi ban goc)
# ================================================================================
print("\n[3/8] Splitting (group by patient)...")
df["patient_base"] = df[IMAGE_COLUMN].apply(
    lambda x: re.sub(r"_\d+\.(jpg|jpeg|png|bmp|tiff|tif)$", "", str(x).lower().strip()))
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_idx, tmp_idx = next(gss.split(df, df[LABEL_COLUMN], groups=df["patient_base"]))
train_df, temp_df = df.iloc[tr_idx].copy(), df.iloc[tmp_idx].copy()
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
v_idx, te_idx = next(gss2.split(temp_df, temp_df[LABEL_COLUMN], groups=temp_df["patient_base"]))
val_df, test_df = temp_df.iloc[v_idx].copy(), temp_df.iloc[te_idx].copy()

if SMOKE:
    train_df = train_df.sample(n=min(120, len(train_df)), random_state=0).reset_index(drop=True)
    val_df   = val_df.sample(n=min(40, len(val_df)), random_state=0).reset_index(drop=True)
    test_df  = test_df.sample(n=min(40, len(test_df)), random_state=0).reset_index(drop=True)

print(f"  Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
assert len(set(train_df["patient_base"]) & set(test_df["patient_base"])) == 0, "Leakage!"

# ================================================================================
# [4] CLASS WEIGHTS + SAMPLER
# ================================================================================
print("\n[4/8] Class weights + sampler...")
counts = train_df[LABEL_COLUMN].value_counts().sort_index()
total = counts.sum()
weights_tensor = torch.tensor([total/counts[0], total/counts[1]], dtype=torch.float32).to(DEVICE)
print(f"  Counts {dict(counts)} | class weights {weights_tensor.cpu().numpy().round(2)}")
sample_w = train_df[LABEL_COLUMN].map(lambda l: 1.0/counts[l]).tolist()
sampler = WeightedRandomSampler(sample_w, num_samples=len(train_df)*2, replacement=True)

# ================================================================================
# [5] AUGMENTATION — ImageNet norm (CheXNet yeu cau)
#     [CAI TIEN 4-5] them ColorJitter (truoc ToTensor) + RandomErasing (sau Normalize)
# ================================================================================
print("\n[5/8] Augmentation...")
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),  # [CAI TIEN 5]
    transforms.ToTensor(),
    transforms.Normalize(IM_MEAN, IM_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0.0),                         # [CAI TIEN 4]
])
val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(IM_MEAN, IM_STD),
])

# ================================================================================
# [6] DATASET / DATALOADER
# ================================================================================
print("\n[6/8] Datasets & loaders...")
class XrayDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform: image = self.transform(image)
        return image, int(row[LABEL_COLUMN])

train_loader = DataLoader(XrayDataset(train_df, train_transform), batch_size=BATCH_SIZE,
                          sampler=sampler, drop_last=True, num_workers=0)
val_loader   = DataLoader(XrayDataset(val_df, val_test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(XrayDataset(test_df, val_test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ================================================================================
# [7] MODEL — DenseNet121 + CheXNet weights (GIU NGUYEN)
# ================================================================================
print("\n[7/8] Building model (DenseNet121 + CheXNet)...")
model = models.densenet121(weights=None)
if os.path.exists(CHEXNET_PATH):
    ckpt = torch.load(CHEXNET_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(remap_chexnet(sd), strict=False)
    n_miss = len([k for k in missing if "num_batches" not in k and not k.startswith("classifier")])
    print(f"  CheXNet loaded | unexpected={len(unexpected)} | missing(real)={n_miss}")
else:
    print(f"  ⚠️  Khong thay {CHEXNET_PATH} — dung init ngau nhien!")

model.classifier = nn.Sequential(nn.Dropout(p=DROPOUT), nn.Linear(1024, 2))
for p in model.features.parameters(): p.requires_grad = False
for p in model.classifier.parameters(): p.requires_grad = True
model = model.to(DEVICE)
print(f"  Stage 1: Features frozen | Classifier trainable | dropout={DROPOUT}")

criterion = FocalLoss(alpha=weights_tensor, gamma=FOCAL_GAMMA)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LR_HEAD, weight_decay=WEIGHT_DECAY)          # [CAI TIEN 1]
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

def make_layerwise_groups(m, lr_backbone, lr_head):
    """Layerwise LR cho DenseNet: block sau LR cao hon, block dau LR thap."""
    g_low, g_mid, g_high, g_head = [], [], [], []
    for n, p in m.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("classifier"): g_head.append(p)
        elif "denseblock4" in n or "norm5" in n: g_high.append(p)
        elif "denseblock3" in n or "transition3" in n: g_mid.append(p)
        else: g_low.append(p)
    return [
        {"params": g_low,  "lr": lr_backbone * 0.1},
        {"params": g_mid,  "lr": lr_backbone * 0.5},
        {"params": g_high, "lr": lr_backbone},
        {"params": g_head, "lr": lr_head},
    ]

# ================================================================================
# [8] TRAIN (2-stage) — [CAI TIEN 2] chon best theo VAL AUC
# ================================================================================
print("\n[8/8] Training...")
print("=" * 80)
best_auc = -1.0
patience_ctr = 0
stage = 1
train_losses, val_losses, val_aucs = [], [], []
CKPT = os.path.join(OUTPUT_DIR, "best_model_chexnet_v2.pth")

def train_one_epoch(loader):
    model.train()
    total = 0.0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

@torch.no_grad()
def evaluate(loader, tta=False):
    """Tra ve (avg_loss, y_true, y_proba). tta=True -> trung binh softmax voi ban lat ngang."""
    model.eval()
    total = 0.0; ys, ps = [], []
    for images, labels in loader:
        images = images.to(DEVICE); lb = labels.to(DEVICE)
        logits = model(images)
        total += criterion(logits, lb).item()
        prob = torch.softmax(logits, dim=1)
        if tta:
            prob_f = torch.softmax(model(torch.flip(images, dims=[3])), dim=1)   # [CAI TIEN 3]
            prob = 0.5 * (prob + prob_f)
        ps.extend(prob[:, 1].cpu().numpy()); ys.extend(labels.numpy())
    return total / max(1, len(loader)), np.array(ys), np.array(ps)

for epoch in range(EPOCHS):
    if epoch == UNFREEZE_EPOCH and stage == 1:
        print(f"\n🔓 Epoch {epoch+1}: Unfreeze features → Stage 2 (layerwise LR)")
        for p in model.parameters(): p.requires_grad = True
        optimizer = torch.optim.Adam(make_layerwise_groups(model, LR_BACKBONE, LR_HEAD),
                                     weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        stage = 2; patience_ctr = 0

    tr = train_one_epoch(train_loader)
    va, vy, vp = evaluate(val_loader, tta=False)
    v_auc = roc_auc_score(vy, vp) if len(np.unique(vy)) > 1 else 0.0
    train_losses.append(tr); val_losses.append(va); val_aucs.append(v_auc)
    print(f"Epoch {epoch+1:3d}/{EPOCHS} [S{stage}] | Train {tr:.4f} | Val {va:.4f} | ValAUC {v_auc:.4f}")

    if v_auc > best_auc:                          # [CAI TIEN 2] chon theo AUC
        best_auc = v_auc; patience_ctr = 0
        torch.save(model.state_dict(), CKPT)
        print(f"  ✅ Best saved (ValAUC={v_auc:.4f})")
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    scheduler.step(va)

# ================================================================================
# EVALUATE (Youden's J) + TTA
# ================================================================================
print("\nLoading best model...")
model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
_, y_true, y_proba = evaluate(test_loader, tta=True)     # [CAI TIEN 3] TTA khi test
print("  TTA (test-time flip): True")

roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else 0.0
fpr, tpr, _ = roc_curve(y_true, y_proba)
threshold_list = np.arange(0.05, 0.95, 0.05)
f1_scores = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0) for t in threshold_list]

# Youden's J
_fpr, _tpr, _thr = roc_curve(y_true, y_proba)
optimal_threshold = float(_thr[np.argmax(_tpr - _fpr)])
y_pred = (y_proba >= optimal_threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp/(tp+fn) if (tp+fn) > 0 else 0.0
specificity = tn/(tn+fp) if (tn+fp) > 0 else 0.0

print("\n" + "=" * 80)
print("TESTING (Youden's J threshold)")
print("=" * 80)
print(f"  ROC AUC            : {roc_auc:.4f}")
print(f"  Optimal Threshold  : {optimal_threshold:.3f}  (Youden's J)")
print(f"  Sensitivity (Co)   : {sensitivity:.3f}")
print(f"  Specificity (Khong): {specificity:.3f}")
print(f"  F1 tai threshold   : {f1_score(y_true, y_pred, zero_division=0):.4f}")
print("\n" + classification_report(y_true, y_pred, target_names=["Khong (0)", "Co (1)"], zero_division=0))

# ================================================================================
# VISUALIZATION
# ================================================================================
if not SMOKE:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("DenseNet121 (CheXNet) V2 — Benh Nghe Nghiep", fontsize=14, fontweight="bold")
    axes[0,0].plot(train_losses, label="Train", marker="o", markersize=3, linewidth=2)
    axes[0,0].plot(val_losses,   label="Val",   marker="s", markersize=3, linewidth=2)
    axes[0,0].plot(val_aucs,     label="Val AUC", marker="^", markersize=3, linewidth=2)
    axes[0,0].axvline(x=UNFREEZE_EPOCH, color="red", linestyle="--", alpha=0.7, label=f"Unfreeze (ep {UNFREEZE_EPOCH})")
    axes[0,0].set_title("Loss / ValAUC"); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(fpr, tpr, label=f"AUC={roc_auc:.3f}", linewidth=2)
    axes[0,1].plot([0,1],[0,1],"k--"); axes[0,1].set_title("ROC Curve")
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(threshold_list, f1_scores, marker="o", linewidth=2, markersize=5)
    axes[1,0].axvline(x=optimal_threshold, color="r", linestyle="--", label=f"Optimal={optimal_threshold:.2f}")
    axes[1,0].set_title("F1 vs Threshold"); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1,1],
                xticklabels=["Khong","Co"], yticklabels=["Khong","Co"], annot_kws={"size":14})
    axes[1,1].set_title("Confusion Matrix"); axes[1,1].set_ylabel("True"); axes[1,1].set_xlabel("Predicted")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "evaluation_chexnet_v2.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved: {out_png}")

    import csv as _csv
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_chexnet_v2.csv")
    with open(cm_path, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["", "pred_Khong", "pred_Co"])
        w.writerow(["true_Khong", int(cm[0,0]), int(cm[0,1])])
        w.writerow(["true_Co",    int(cm[1,0]), int(cm[1,1])])
    print(f"✓ Saved: {cm_path}")

print("\n" + "=" * 80)
print(f"✅ DONE — CheXNet V2 | AUC={roc_auc:.4f} | Sens={sensitivity:.3f} | Spec={specificity:.3f}")
print("=" * 80)

sys.stdout.flush()
_logfile.close()
