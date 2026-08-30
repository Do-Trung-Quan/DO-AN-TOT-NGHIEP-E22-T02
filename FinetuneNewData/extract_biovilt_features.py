"""Extract 512-D BioViL-T image features for fusion."""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

from health_multimodal.image.data.transforms import ExpandChannels
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.types import ImageEncoderType


HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEFAULT_CHECKPOINT = os.path.join(HERE, "biovilt_silicosis_ce_backbone.pth")
DEFAULT_DATA_ROOT = r"C:\Users\ASUS\Downloads\archive"
DEFAULT_OUTPUT = os.path.join(HERE, "biovilt_silicosis_ce_features.pt")
DEFAULT_METADATA = os.path.join(HERE, "biovilt_silicosis_ce_features.csv")
RESIZE = 512
CROP = 448
CLASSES = {"normal": 0, "silicosis": 1}


def scan_folder(folder, split):
    rows = []
    for class_name, label in CLASSES.items():
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        for root, _, files in os.walk(class_dir):
            for file_name in sorted(files):
                if os.path.splitext(file_name)[1].lower() in (
                    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
                ):
                    path = os.path.join(root, file_name)
                    rows.append({
                        "image_path": path,
                        "file_name": os.path.relpath(path, folder),
                        "split": split,
                        "label": label,
                    })
    return rows


class XrayDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("L")
        return self.transform(image), index


def load_state_dict(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Backbone checkpoint must contain a state_dict.")
    return state


def main():
    parser = argparse.ArgumentParser(description="Extract BioViL-T features for fusion.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    rows = scan_folder(os.path.join(args.data_root, "train"), "train")
    rows += scan_folder(os.path.join(args.data_root, "test"), "test")
    if not rows:
        raise SystemExit(f"No images found under {args.data_root!r}.")

    transform = Compose([
        Resize(RESIZE),
        CenterCrop(CROP),
        ToTensor(),
        ExpandChannels(),
    ])
    loader = DataLoader(
        XrayDataset(rows, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    backbone_path = os.path.join(
        REPO, "Pre-train BioViL-T", "biovil_t_image_model_proj_size_128.pt"
    )
    backbone = ImageModel(
        img_encoder_type=ImageEncoderType.RESNET50_MULTI_IMAGE,
        joint_feature_size=128,
        pretrained_model_path=backbone_path,
    )
    backbone.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    backbone.to(device).eval()

    features = [None] * len(rows)
    with torch.inference_mode():
        for images, indices in loader:
            output = backbone(images.to(device)).img_embedding.flatten(1).cpu()
            for feature, index in zip(output, indices.tolist()):
                features[index] = feature

    feature_tensor = torch.stack(features)
    metadata = pd.DataFrame([
        {"file_name": row["file_name"], "split": row["split"], "label": row["label"]}
        for row in rows
    ])
    payload = {
        "features": feature_tensor,
        "file_names": metadata["file_name"].tolist(),
        "splits": metadata["split"].tolist(),
        "labels": torch.tensor(metadata["label"].tolist(), dtype=torch.long),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(payload, args.output)
    metadata.to_csv(args.metadata, index=False, encoding="utf-8")
    print(f"Extracted features: {tuple(feature_tensor.shape)}")
    print(f"Feature file: {args.output}")
    print(f"Metadata file: {args.metadata}")


if __name__ == "__main__":
    main()
