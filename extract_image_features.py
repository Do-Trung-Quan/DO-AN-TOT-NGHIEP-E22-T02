"""
================================================================================
SCRIPT TRÍCH XUẤT VECTOR ĐẶC TRƯNG ẢNH X-QUANG CHO GNN FUSION
================================================================================
Script này thực hiện:
1. Đọc file `output/timeseries_features.parquet` chứa 8.030 dòng bệnh nhân.
2. Tìm 1.835 ảnh X-quang khớp chuẩn xác trong thư mục `data/img_train` và `data/img_test`.
3. Load checkpoint `Finetune CrossEntropy/best_model_ce.pth` (Backbone BioViL-T).
4. Forward 1.835 ảnh qua Backbone để trích xuất vector đặc trưng thị giác 512-D.
5. Ghép nối căn chỉnh khớp 1-1 với 8.030 dòng bệnh nhân (bệnh nhân thiếu ảnh điền vector 0).
6. Xuất ra 2 file:
   - `output/image_features.parquet`: Shape (8030, 513) [cột file_name + 512 cột đặc trưng]
   - `output/image_features.npy`: Shape (8030, 512) [ma trận float32]
================================================================================
"""

import os
import re
import sys
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor
from health_multimodal.image.data.transforms import ExpandChannels
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.types import ImageEncoderType

# ================================================================================
# CONFIGURATION & PATHS
# ================================================================================
import torch
torch.set_num_threads(2)  # Giới hạn số thread để không chiếm 100% CPU/RAM

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMESERIES_PARQUET = os.path.join(BASE_DIR, "output", "timeseries_features.parquet")
MODEL_CKPT = os.path.join(BASE_DIR, "Finetune CrossEntropy", "best_model_ce.pth")
BACKBONE_PATH = os.path.join(BASE_DIR, "Pre-train BioViL-T", "biovil_t_image_model_proj_size_128.pt")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_PARQUET = os.path.join(BASE_DIR, "output", "image_features.parquet")
OUTPUT_NPY = os.path.join(BASE_DIR, "output", "image_features.npy")

BATCH_SIZE = 32
NUM_WORKERS = 0  # Đặt = 0 để tránh chiếm RAM dùng chung (/dev/shm) gây ngắt tiến trình
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = (DEVICE == "cuda")

print("=" * 80)
print("TRÍCH XUẤT VECTOR ĐẶC TRƯNG ẢNH X-QUANG (512-D) CHO GNN FUSION")
print(f"Device: {DEVICE} | AMP: {USE_AMP} | Batch Size: {BATCH_SIZE}")
print("=" * 80)

# ================================================================================
# 1. LOAD MODEL BIOVIL-T
# ================================================================================
def build_backbone():
    if os.path.exists(BACKBONE_PATH):
        print(f"[1/5] Loading local backbone: {BACKBONE_PATH}")
        return ImageModel(
            img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
            joint_feature_size=128,
            pretrained_model_path=BACKBONE_PATH
        )
    print("[1/5] Local backbone not found, downloading from HuggingFace...")
    from health_multimodal.image.model.pretrained import get_biovil_t_image_encoder
    return get_biovil_t_image_encoder()

class BioViLTClassifier(nn.Module):
    def __init__(self, backbone, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(backbone.feature_size, num_classes)
        )
    def forward(self, x):
        feat = self.backbone(x).img_embedding
        return self.head(feat.flatten(1))
    
    def extract_features(self, x):
        feat = self.backbone(x).img_embedding
        return feat.flatten(1)

print("[2/5] Building BioViL-T model & loading weights...")
raw_backbone = build_backbone()
model = BioViLTClassifier(raw_backbone).to(DEVICE)

if os.path.exists(MODEL_CKPT):
    print(f"      Loading fine-tuned checkpoint: {MODEL_CKPT}")
    state_dict = torch.load(MODEL_CKPT, map_location=DEVICE)
    model.load_state_dict(state_dict)
else:
    raise FileNotFoundError(f"Checkpoint not found at: {MODEL_CKPT}")

model.eval()
print(f"      Backbone feature size: {model.backbone.feature_size}")

# ================================================================================
# 2. SCAN PHYSICAL IMAGE FILES & MATCH WITH TIMESERIES
# ================================================================================
print("[3/5] Mapping image files with timeseries records...")
if not os.path.exists(TIMESERIES_PARQUET):
    raise FileNotFoundError(f"Timeseries parquet not found at: {TIMESERIES_PARQUET}")

df_ts = pd.read_parquet(TIMESERIES_PARQUET)
print(f"      Loaded timeseries data shape: {df_ts.shape}")

# Find physical images in data/
physical_images = {}
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")):
            physical_images[f.lower().strip()] = os.path.join(root, f)

print(f"      Found {len(physical_images)} physical image files in {DATA_DIR}")

# Match valid image filenames from parquet
parquet_filenames = df_ts[df_ts["file_name"] != "khong"]["file_name"].tolist()
valid_image_items = []
missing_count = 0

for fn in set(parquet_filenames):
    clean_fn = str(fn).lower().strip()
    if clean_fn in physical_images:
        valid_image_items.append((clean_fn, physical_images[clean_fn]))
    else:
        missing_count += 1

print(f"      Valid images to extract: {len(valid_image_items)} | Missing physical files: {missing_count}")

# ================================================================================
# 3. PYTORCH DATASET & DATALOADER FOR FEATURE EXTRACTION
# ================================================================================
eval_tf = Compose([
    Resize(512),
    CenterCrop(448),
    ToTensor(),
    ExpandChannels()
])

class XrayInferenceDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fn_key, img_path = self.items[idx]
        img = Image.open(img_path).convert("L")
        tensor_img = self.transform(img)
        return tensor_img, fn_key

inference_dataset = XrayInferenceDataset(valid_image_items, eval_tf)
inference_loader = DataLoader(
    inference_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE == "cuda")
)

# ================================================================================
# 4. RUN FEATURE EXTRACTION INFERENCE
# ================================================================================
print("[4/5] Extracting 512-D features from images...")
extracted_features = {}

with torch.no_grad():
    for images, fn_keys in tqdm(inference_loader, desc="Extracting"):
        images = images.to(DEVICE)
        with torch.autocast("cuda", enabled=USE_AMP):
            features = model.extract_features(images)
        features_np = features.float().cpu().numpy()
        for fn_key, feat_vec in zip(fn_keys, features_np):
            extracted_features[fn_key] = feat_vec

print(f"      Successfully extracted features for {len(extracted_features)} images.")

# ================================================================================
# 5. ALIGN WITH 8,030 TIMESERIES PATIENTS & SAVE OUTPUT
# ================================================================================
print("[5/5] Aligning features with 8,030 patient records & saving outputs...")
num_patients = len(df_ts)
feature_dim = model.backbone.feature_size
img_feature_matrix = np.zeros((num_patients, feature_dim), dtype=np.float32)

filled_count = 0
zero_count = 0

for i, row in df_ts.iterrows():
    fn_key = str(row["file_name"]).lower().strip()
    if fn_key != "khong" and fn_key in extracted_features:
        img_feature_matrix[i] = extracted_features[fn_key]
        filled_count += 1
    else:
        zero_count += 1

print(f"      Matched patient records: {filled_count} / {num_patients}")
print(f"      Zero-padded patient records (no image): {zero_count} / {num_patients}")

# Construct Parquet DataFrame
feature_cols = [f"img_feat_{j}" for j in range(feature_dim)]
df_out = pd.DataFrame(img_feature_matrix, columns=feature_cols)
df_out.insert(0, "file_name", df_ts["file_name"].values)

os.makedirs(os.path.dirname(OUTPUT_PARQUET), exist_ok=True)

# Save parquet
df_out.to_parquet(OUTPUT_PARQUET, index=False)
print(f"      Saved Parquet -> {OUTPUT_PARQUET} (Shape: {df_out.shape})")

# Save npy
np.save(OUTPUT_NPY, img_feature_matrix)
print(f"      Saved NPY -> {OUTPUT_NPY} (Shape: {img_feature_matrix.shape})")

print("=" * 80)
print("FEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
print("=" * 80)
