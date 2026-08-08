import os
import json
import random
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split

DATASET_DIR = Path(r"E:\Mr Biryo\ian\dataset\dataset")
OUTPUT_DIR = Path("outputs")
MANIFEST_PATH = OUTPUT_DIR / "split_manifest.json"
TRAIN_PCT = 0.70
VAL_PCT = 0.15
TEST_PCT = 0.15
IMG_SIZE = 224
SEED = 42


def collect_file_paths():
    data = []
    for label_dir in ["AML positive", "NEGATIVE"]:
        label = 1 if "AML" in label_dir else 0
        class_path = DATASET_DIR / label_dir
        for sub in os.listdir(class_path):
            sub_path = class_path / sub
            if sub_path.is_dir():
                for fname in os.listdir(sub_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                        data.append({
                            "path": str(sub_path / fname),
                            "label": label,
                            "class_name": label_dir,
                            "subtype": sub,
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
    manifest = {
        "train": train, "val": val, "test": test,
        "counts": {
            "train": len(train), "val": len(val), "test": len(test),
            "train_positive": sum(1 for x in train if x["label"] == 1),
            "train_negative": sum(1 for x in train if x["label"] == 0),
            "val_positive": sum(1 for x in val if x["label"] == 1),
            "val_negative": sum(1 for x in val if x["label"] == 0),
            "test_positive": sum(1 for x in test if x["label"] == 1),
            "test_negative": sum(1 for x in test if x["label"] == 0),
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
        return img, label, idx


POSITIVE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

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


class BalancedBatchSampler(Sampler):
    """Creates batches balanced between positive and negative classes."""
    def __init__(self, dataset, batch_size, pos_ratio=0.3):
        self.pos_indices = [i for i, e in enumerate(dataset.entries) if e["label"] == 1]
        self.neg_indices = [i for i, e in enumerate(dataset.entries) if e["label"] == 0]
        self.batch_size = batch_size
        self.pos_per_batch = max(1, int(batch_size * pos_ratio))
        self.neg_per_batch = batch_size - self.pos_per_batch

        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)

    def __iter__(self):
        pos_repeats = math.ceil(self.num_neg / self.neg_per_batch * self.pos_per_batch / max(1, self.num_pos))
        pos_pool = (self.pos_indices * pos_repeats)[:self.num_pos * pos_repeats]
        random.shuffle(pos_pool)

        neg_repeats = math.ceil(self.num_pos / self.pos_per_batch * self.neg_per_batch / max(1, self.num_neg))
        neg_pool = (self.neg_indices * neg_repeats)[:self.num_neg * neg_repeats]
        random.shuffle(neg_pool)

        batches = []
        pos_idx = 0
        neg_idx = 0

        while pos_idx + self.pos_per_batch <= len(pos_pool) and neg_idx + self.neg_per_batch <= len(neg_pool):
            batch = pos_pool[pos_idx:pos_idx + self.pos_per_batch] + neg_pool[neg_idx:neg_idx + self.neg_per_batch]
            random.shuffle(batch)
            batches.append(batch)
            pos_idx += self.pos_per_batch
            neg_idx += self.neg_per_batch

        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return min(
            len(self.pos_indices) * self.batch_size // self.pos_per_batch,
            len(self.neg_indices) * self.batch_size // self.neg_per_batch,
        )


class PerClassTransformDataset(Dataset):
    """Dataset wrapper that applies different transforms per class."""
    def __init__(self, entries, pos_transform=None, neg_transform=None, eval_transform=None, is_train=True):
        self.entries = entries
        self.pos_transform = pos_transform if is_train else eval_transform
        self.neg_transform = neg_transform if is_train else eval_transform
        self.is_train = is_train

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img = Image.open(entry["path"]).convert("RGB")
        if entry["label"] == 1:
            img = self.pos_transform(img)
        else:
            img = self.neg_transform(img)
        return img, torch.tensor(entry["label"], dtype=torch.long)


def create_dataloaders(batch_size=16, num_workers=0, use_balanced=True):
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at {MANIFEST_PATH}. Run with --split first.")

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    if use_balanced:
        train_ds = PerClassTransformDataset(
            manifest["train"],
            pos_transform=POSITIVE_TRANSFORM,
            neg_transform=TRAIN_TRANSFORM,
            eval_transform=EVAL_TRANSFORM,
            is_train=True,
        )
        sampler = BalancedBatchSampler(
            AML_Dataset(manifest["train"]), batch_size, pos_ratio=0.3
        )
        train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers)
    else:
        train_ds = PerClassTransformDataset(
            manifest["train"], pos_transform=POSITIVE_TRANSFORM, neg_transform=TRAIN_TRANSFORM,
            eval_transform=EVAL_TRANSFORM, is_train=True,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = PerClassTransformDataset(
        manifest["val"], pos_transform=EVAL_TRANSFORM, neg_transform=EVAL_TRANSFORM,
        eval_transform=EVAL_TRANSFORM, is_train=False,
    )
    test_ds = PerClassTransformDataset(
        manifest["test"], pos_transform=EVAL_TRANSFORM, neg_transform=EVAL_TRANSFORM,
        eval_transform=EVAL_TRANSFORM, is_train=False,
    )

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def main():
    print("Collecting file paths ...")
    data = collect_file_paths()
    print(f"Total images found: {len(data)}")

    print("Splitting data (70/15/15) ...")
    train, val, test = split_data(data)
    manifest = save_manifest(train, val, test)

    train_pos = manifest['counts']['train_positive']
    train_neg = manifest['counts']['train_negative']

    print(f"Train: {manifest['counts']['train']} "
          f"(pos={train_pos}, neg={train_neg})")
    print(f"Val:   {manifest['counts']['val']} "
          f"(pos={manifest['counts']['val_positive']}, neg={manifest['counts']['val_negative']})")
    print(f"Test:  {manifest['counts']['test']} "
          f"(pos={manifest['counts']['test_positive']}, neg={manifest['counts']['test_negative']})")

    computed_alpha = round(train_neg / (train_pos + train_neg), 4)
    print(f"\nComputed focal alpha (neg_ratio): {computed_alpha}")
    print(f"Manifest saved to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
