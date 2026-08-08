import os
import json
import random
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

DATASET_DIR = Path(r"E:\Mr Biryo\ian\dataset")
OUTPUT_DIR = Path("outputs/v2_alnet")
MANIFEST_PATH = OUTPUT_DIR / "split_manifest_v2.json"
TRAIN_PCT = 0.70
VAL_PCT = 0.15
TEST_PCT = 0.15
IMG_SIZE = 224
SEED = 42

CLASS_MAP = {
    "monocyte": 0,
    "myeloblast": 1,
}


def collect_file_paths():
    data = []
    for class_name, label in CLASS_MAP.items():
        class_path = DATASET_DIR / class_name
        if not class_path.is_dir():
            continue
        for fname in os.listdir(class_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                data.append({
                    "path": str(class_path / fname),
                    "label": label,
                    "class_name": class_name,
                    "filename": fname,
                })
    return data


def split_data(data):
    np.random.seed(SEED)
    random.seed(SEED)

    paths = [d["path"] for d in data]
    labels = [d["label"] for d in data]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=(1 - TRAIN_PCT), stratify=labels, random_state=SEED
    )
    val_size = VAL_PCT / (VAL_PCT + TEST_PCT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_size), stratify=temp_labels, random_state=SEED
    )

    path_to_meta = {d["path"]: d for d in data}

    def build_split(path_list, label_list, split_name):
        entries = []
        for p, l in zip(path_list, label_list):
            meta = dict(path_to_meta[p])
            meta["split"] = split_name
            entries.append(meta)
        return entries

    return (build_split(train_paths, train_labels, "train"),
            build_split(val_paths, val_labels, "val"),
            build_split(test_paths, test_labels, "test"))


def save_manifest(train, val, test):
    train_pos = sum(1 for x in train if x["label"] == 1)
    train_neg = sum(1 for x in train if x["label"] == 0)
    val_pos = sum(1 for x in val if x["label"] == 1)
    val_neg = sum(1 for x in val if x["label"] == 0)
    test_pos = sum(1 for x in test if x["label"] == 1)
    test_neg = sum(1 for x in test if x["label"] == 0)

    manifest = {
        "train": train, "val": val, "test": test,
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "train_positive": train_pos, "train_negative": train_neg,
            "val_positive": val_pos, "val_negative": val_neg,
            "test_positive": test_pos, "test_negative": test_neg,
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


class AML_Dataset(Dataset):
    def __init__(self, entries, transform=None):
        self.entries = entries
        self.transform = transform

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(entry["label"], dtype=torch.long)
        return img, label


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def create_dataloaders(batch_size=32, num_workers=0):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run with --split first.")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    train_ds = AML_Dataset(manifest["train"], transform=TRAIN_TRANSFORM)
    val_ds = AML_Dataset(manifest["val"], transform=EVAL_TRANSFORM)
    test_ds = AML_Dataset(manifest["test"], transform=EVAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


import torch

def main():
    print("Collecting file paths ...")
    data = collect_file_paths()
    print(f"Total images found: {len(data)}")

    pos = sum(1 for d in data if d["label"] == 1)
    neg = sum(1 for d in data if d["label"] == 0)
    print(f"  myeloblast (AML=1): {pos}")
    print(f"  monocyte   (AML=0): {neg}")
    print(f"  Balance ratio: {max(pos,neg)/min(pos,neg):.2f}:1")

    print("\nSplitting data (70/15/15 stratified) ...")
    train, val, test = split_data(data)
    manifest = save_manifest(train, val, test)

    print(f"\nTrain: {manifest['counts']['train']} "
          f"(pos={manifest['counts']['train_positive']}, neg={manifest['counts']['train_negative']})")
    print(f"Val:   {manifest['counts']['val']} "
          f"(pos={manifest['counts']['val_positive']}, neg={manifest['counts']['val_negative']})")
    print(f"Test:  {manifest['counts']['test']} "
          f"(pos={manifest['counts']['test_positive']}, neg={manifest['counts']['test_negative']})")

    print(f"\nClasses are balanced -- using standard CrossEntropyLoss (no weighted focal loss)")
    print(f"Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
