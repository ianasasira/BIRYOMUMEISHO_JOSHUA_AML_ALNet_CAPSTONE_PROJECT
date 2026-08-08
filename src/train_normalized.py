#!/usr/bin/env python3
"""
Full pipeline: stain normalization -> shortcut check -> train -> verify -> compare
"""

import sys, os, json, time, random, shutil, hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from stain_norm import ReinhardNormalizer, load_and_pad_array, load_image_safe
from alnet_model import ALNet, WeightedFocalLoss, count_parameters

PROJECT_ROOT = SRC_DIR.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = 400
IMG_SIZE_TRAIN = 224
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "normalized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 24
EPOCHS = 80
LR_INIT = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_ALPHA = 0.75
FOCAL_GAMMA = 2.0
EARLY_STOP_PATIENCE = 12


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# STEP 1: Fit normalizer & normalize both datasets
# ============================================================

def fit_and_normalize():
    print("=" * 60)
    print("STEP 1: STAIN NORMALIZATION")
    print("=" * 60)

    kaggle_dir = PROJECT_ROOT / "dataset"
    original_dir = Path("/media/ianasasiratusiime/PROSCOVIA/Mr Biryo/dataset/dataset")
    norm_kaggle_dir = PROJECT_ROOT / "dataset_normalized"
    norm_original_dir = PROJECT_ROOT / "dataset_normalized_original"

    # Fit normalizer on random samples from both classes
    print("  Fitting normalizer on reference samples...")
    normalizer = ReinhardNormalizer()
    ref_images = []
    for cls in ["monocyte", "myeloblast"]:
        cls_dir = kaggle_dir / cls
        files = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in random.sample(files, min(50, len(files))):
            path = cls_dir / fname
            arr = load_image_safe(str(path), TARGET_SIZE)
            if arr is not None:
                ref_images.append(arr)
    normalizer.fit(ref_images)
    print(f"  Normalizer fitted on {len(ref_images)} reference images")

    # Normalize Kaggle dataset
    print(f"\n  Normalizing Kaggle dataset -> {norm_kaggle_dir}")
    for cls in ["monocyte", "myeloblast"]:
        cls_src = kaggle_dir / cls
        cls_dst = norm_kaggle_dir / cls
        cls_dst.mkdir(parents=True, exist_ok=True)
        files = sorted([f for f in os.listdir(cls_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for i, fname in enumerate(files):
            src_path = cls_src / fname
            dst_path = cls_dst / fname.replace('.jpg', '.png').replace('.jpeg', '.png')
            if dst_path.exists():
                continue
            arr = load_image_safe(str(src_path), TARGET_SIZE)
            if arr is None:
                continue
            try:
                norm = normalizer.transform(arr)
                Image.fromarray(norm).save(str(dst_path))
            except Exception as e:
                pass
            if (i + 1) % 200 == 0:
                print(f"    {cls}: {i + 1}/{len(files)}")
        print(f"    {cls}: {len(files)} images normalized")

    # Normalize original dataset
    print(f"\n  Normalizing original dataset -> {norm_original_dir}")
    for split in ["AML positive", "NEGATIVE"]:
        split_src = original_dir / split
        if not split_src.is_dir():
            continue
        for subtype in os.listdir(split_src):
            subtype_src = split_src / subtype
            if not subtype_src.is_dir():
                continue
            subtype_dst = norm_original_dir / split / subtype
            subtype_dst.mkdir(parents=True, exist_ok=True)
            files = sorted([f for f in os.listdir(subtype_src)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
            for i, fname in enumerate(files):
                dst_path = subtype_dst / (os.path.splitext(fname)[0] + '.png')
                if dst_path.exists():
                    continue
                arr = load_image_safe(str(subtype_src / fname), TARGET_SIZE)
                if arr is None:
                    continue
                try:
                    norm = normalizer.transform(arr)
                    Image.fromarray(norm).save(str(dst_path))
                except Exception as e:
                    pass
                if (i + 1) % 500 == 0:
                    print(f"      {split}/{subtype}: {i + 1}/{len(files)}")
            count = len(files)
            print(f"      {split}/{subtype}: {count} images normalized")

    print("  Normalization complete.\n")
    return normalizer


# ============================================================
# STEP 2: Shortcut detection checks
# ============================================================

def run_shortcut_checks():
    print("=" * 60)
    print("STEP 2: SHORTCUT DETECTION CHECKS")
    print("=" * 60)

    norm_kaggle_dir = PROJECT_ROOT / "dataset_normalized"

    mono_r, mono_g, mono_b, mono_rb = [], [], [], []
    myelo_r, myelo_g, myelo_b, myelo_rb = [], [], [], []
    mono_sizes, myelo_sizes = set(), set()
    mono_bright, myelo_bright = [], []

    for cls, r_list, g_list, b_list, rb_list, sizes, bright in [
        ("monocyte", mono_r, mono_g, mono_b, mono_rb, mono_sizes, mono_bright),
        ("myeloblast", myelo_r, myelo_g, myelo_b, myelo_rb, myelo_sizes, myelo_bright),
    ]:
        cls_dir = norm_kaggle_dir / cls
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = cls_dir / fname
            try:
                img = Image.open(path)
                arr = np.array(img, dtype=np.float32)
                sizes.add(img.size)
                r_list.append(arr[:, :, 0].mean())
                g_list.append(arr[:, :, 1].mean())
                b_list.append(arr[:, :, 2].mean())
                rb_list.append(arr[:, :, 0].mean() / max(1, arr[:, :, 2].mean()))
                bright.append(arr.mean())
            except:
                pass

    print(f"\n  monocyte (n={len(mono_r)}):")
    print(f"    Sizes: {mono_sizes}")
    print(f"    R/B mean={np.mean(mono_rb):.3f}, std={np.std(mono_rb):.4f}")
    print(f"    Brightness mean={np.mean(mono_bright):.1f}, std={np.std(mono_bright):.1f}")

    print(f"\n  myeloblast (n={len(myelo_r)}):")
    print(f"    Sizes: {myelo_sizes}")
    print(f"    R/B mean={np.mean(myelo_rb):.3f}, std={np.std(myelo_rb):.4f}")
    print(f"    Brightness mean={np.mean(myelo_bright):.1f}, std={np.std(myelo_bright):.1f}")

    # Test R/B separability
    all_rb = mono_rb + myelo_rb
    all_labels = [0] * len(mono_rb) + [1] * len(myelo_rb)
    best_acc = 0
    best_thr = 0
    for thr in np.arange(0.5, 2.0, 0.005):
        preds = [1 if x < thr else 0 for x in all_rb]
        acc = sum(1 for p, l in zip(preds, all_labels) if p == l) / len(all_labels)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr

    print(f"\n  R/B alone accuracy: {best_acc:.1%} (threshold={best_thr:.3f})")
    print(f"  (Was 98.2% before normalization)")

    # Test brightness separability
    all_bright = mono_bright + myelo_bright
    best_b_acc = 0
    for thr in np.arange(50, 250, 0.5):
        preds = [1 if x < thr else 0 for x in all_bright]
        acc = sum(1 for p, l in zip(preds, all_labels) if p == l) / len(all_labels)
        if acc > best_b_acc:
            best_b_acc = acc
    print(f"  Brightness alone accuracy: {best_b_acc:.1%}")

    # Size check
    if mono_sizes == myelo_sizes:
        print(f"  Size check: PASS (both classes same sizes: {mono_sizes})")
    else:
        print(f"  Size check: WARNING - sizes differ: mono={mono_sizes}, myelo={myelo_sizes}")

    # Final verdict
    if best_acc < 0.80:
        print(f"\n  VERDICT: Shortcut checks PASSED. R/B separability dropped to {best_acc:.1%}")
        print(f"  Continue to training.")
        return True
    else:
        print(f"\n  VERDICT: SHORTCUT STILL PRESENT. R/B={best_acc:.1%}. STOPPING.")
        return False


# ============================================================
# STEP 3: Data pipeline for normalized Kaggle data
# ============================================================

class NormalizedDataset(Dataset):
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
        return img, torch.tensor(entry["label"], dtype=torch.long)


def create_kaggle_splits():
    norm_dir = PROJECT_ROOT / "dataset_normalized"
    entries = []
    for cls_name, label in [("monocyte", 0), ("myeloblast", 1)]:
        cls_dir = norm_dir / cls_name
        for fname in os.listdir(cls_dir):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            entries.append({
                "path": str(cls_dir / fname),
                "label": label,
                "class_name": cls_name,
            })

    paths = [e["path"] for e in entries]
    labels = [e["label"] for e in entries]

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=SEED
    )

    path_map = {e["path"]: e for e in entries}

    def build(path_list, label_list):
        return [path_map[p] for p in path_list]

    train_entries = build(train_paths, train_labels)
    val_entries = build(val_paths, val_labels)
    test_entries = build(test_paths, test_labels)

    train_pos = sum(1 for e in train_entries if e["label"] == 1)
    train_neg = sum(1 for e in train_entries if e["label"] == 0)
    print(f"\n  Splits: train={len(train_entries)} (pos={train_pos}, neg={train_neg}), "
          f"val={len(val_entries)}, test={len(test_entries)}")
    print(f"  Train pos ratio: {train_pos / len(train_entries):.2%}")

    return train_entries, val_entries, test_entries, train_neg / max(1, train_pos + train_neg)


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE_TRAIN, IMG_SIZE_TRAIN)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE_TRAIN, IMG_SIZE_TRAIN)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_dataloaders_norm(train_entries, val_entries, test_entries):
    train_ds = NormalizedDataset(train_entries, get_train_transform())
    val_ds = NormalizedDataset(val_entries, get_eval_transform())
    test_ds = NormalizedDataset(test_entries, get_eval_transform())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader


# ============================================================
# STEP 4: Training
# ============================================================

def train_model(train_loader, val_loader, alpha):
    set_seed(SEED)
    print(f"\n  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    model = ALNet(num_classes=2).to(DEVICE)
    print(f"  Trainable params: {count_parameters(model):,}")

    criterion = WeightedFocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else torch.amp.GradScaler('cpu')

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    best_model_path = OUTPUT_DIR / "alnet_normalized_best.pt"

    print(f"\n  Training {EPOCHS} epochs, batch={BATCH_SIZE}, focal_alpha={alpha:.3f}")
    print(f"  {'='*50}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast() if DEVICE.type == "cuda" else torch.amp.autocast('cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            _, pred = outputs.max(1)
            train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.cuda.amp.autocast() if DEVICE.type == "cuda" else torch.amp.autocast('cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            marker = " *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, best_model_path)
        else:
            patience_counter += 1

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epoch == 1 or marker:
            print(f"  Epoch {epoch:3d} | TrLoss {train_loss:.4f} | TrAcc {train_acc:.1f}% | "
                  f"VaLoss {val_loss:.4f} | VaAcc {val_acc:.1f}% | LR {current_lr:.2e} | "
                  f"{elapsed:.1f}s{marker}")

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n  Training complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    with open(OUTPUT_DIR / "training_history_norm.json", "w") as f:
        json.dump(history, f, indent=2)

    checkpoint = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, history, best_epoch


# ============================================================
# STEP 5: Full verification
# ============================================================

def verify_model(model, test_loader):
    print("\n--- Test Set Evaluation ---")
    model.eval()
    all_probs, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.cuda.amp.autocast() if DEVICE.type == "cuda" else torch.amp.autocast('cpu'):
                outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

    acc = 100.0 * correct / total
    print(f"  Accuracy: {acc:.2f}%")

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # AUC-ROC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"  AUC-ROC: {auc:.4f}")

    # Best F1 from PR curve
    best_f1, best_thr = 0, 0
    best_r, best_p = 0, 0
    best_tp, best_fp, best_fn = 0, 0, 0

    for thr in np.arange(0.01, 1.0, 0.01):
        preds = (all_probs >= thr).astype(int)
        tp = np.sum((preds == 1) & (all_labels == 1))
        fp = np.sum((preds == 1) & (all_labels == 0))
        fn = np.sum((preds == 0) & (all_labels == 1))
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        f1 = 2 * recall * precision / max(0.001, recall + precision)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_r = recall
            best_p = precision
            best_tp = tp
            best_fp = fp
            best_fn = fn

    print(f"  Best F1={best_f1:.4f} at threshold={best_thr:.2f}")
    print(f"  Recall={best_r:.1%}, Precision={best_p:.1%}")
    print(f"  TP={best_tp}, FP={best_fp}, FN={best_fn}")

    # Confidence check
    errors = []
    all_confidences = []
    for i in range(len(all_labels)):
        prob_pos = all_probs[i]
        pred_label = 1 if prob_pos >= 0.5 else 0
        conf = prob_pos if pred_label == 1 else 1 - prob_pos
        all_confidences.append(conf)
        if pred_label != all_labels[i]:
            errors.append({"idx": i, "true": int(all_labels[i]),
                           "pred": int(pred_label), "prob_pos": float(prob_pos),
                           "confidence": float(conf)})

    fp_errors = [e for e in errors if e["true"] == 0]
    fn_errors = [e for e in errors if e["true"] == 1]

    print(f"\n  Total errors: {len(errors)} (FP={len(fp_errors)}, FN={len(fn_errors)})")
    if errors:
        confs = np.array([e["confidence"] for e in errors])
        print(f"  Error confidence: mean={confs.mean():.4f}, min={confs.min():.4f}")
        high_conf_errors = sum(1 for c in confs if c > 0.95)
        print(f"  High-confidence errors (>0.95): {high_conf_errors}/{len(errors)}")

    return {
        "test_acc": acc, "auc_roc": auc,
        "best_f1": best_f1, "best_threshold": best_thr,
        "recall": best_r, "precision": best_p,
        "tp": best_tp, "fp": best_fp, "fn": best_fn,
        "total_errors": len(errors), "fp_errors": len(fp_errors),
        "fn_errors": len(fn_errors),
    }


def original_dataset_test(model):
    """Test on normalized original dataset."""
    print("\n--- Cross-Dataset Test (Normalized Original Dataset) ---")
    norm_orig_dir = PROJECT_ROOT / "dataset_normalized_original"

    transform = get_eval_transform()
    results = []

    for split, label in [("AML positive", 1), ("NEGATIVE", 0)]:
        split_dir = norm_orig_dir / split
        if not split_dir.is_dir():
            continue
        for subtype in os.listdir(split_dir):
            subtype_dir = split_dir / subtype
            if not subtype_dir.is_dir():
                continue
            for fname in os.listdir(subtype_dir):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                path = subtype_dir / fname
                try:
                    img = Image.open(path).convert("RGB")
                    inp = transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        with torch.cuda.amp.autocast() if DEVICE.type == "cuda" else torch.amp.autocast('cpu'):
                            outputs = model(inp)
                        probs = torch.softmax(outputs, dim=1)
                    results.append({
                        "true_label": label, "subtype": subtype,
                        "prob_pos": float(probs[0, 1]),
                    })
                except:
                    pass

    if not results:
        print("  No results. Check dataset paths.")
        return None

    pos = [r for r in results if r["true_label"] == 1]
    neg = [r for r in results if r["true_label"] == 0]
    print(f"  Total: {len(results)} (pos={len(pos)}, neg={len(neg)})")

    for st in ["MOB", "MYB", "MON", "MYO"]:
        sr = [r for r in results if r["subtype"] == st]
        if sr:
            pos_rate = sum(1 for r in sr if r["prob_pos"] >= 0.5)
            print(f"    {st}: {pos_rate}/{len(sr)} predicted AML ({100*pos_rate/max(1,len(sr)):.1f}%)")

    y_true = [r["true_label"] for r in results]
    y_score = [r["prob_pos"] for r in results]
    auc = roc_auc_score(y_true, y_score)

    best_f1, best_thr = 0, 0
    best_r, best_p = 0, 0
    for thr in np.arange(0.01, 1.0, 0.01):
        preds = [1 if s >= thr else 0 for s in y_score]
        tp = sum(1 for p, t in zip(preds, y_true) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(preds, y_true) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(preds, y_true) if p == 0 and t == 1)
        r = tp / max(1, tp + fn)
        p = tp / max(1, tp + fp)
        f1 = 2 * r * p / max(0.001, r + p)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
            best_r = r
            best_p = p

    print(f"\n  Cross-dataset AUC-ROC: {auc:.4f}")
    print(f"  Best F1={best_f1:.4f} at thr={best_thr:.2f}, R={best_r:.1%}, P={best_p:.1%}")

    return {"auc": auc, "f1": best_f1, "recall": best_r, "precision": best_p}


# ============================================================
# STEP 6: Comparison table
# ============================================================

def print_comparison(old_results, rejected_results, new_own, new_cross):
    print("\n" + "=" * 90)
    print("FINAL COMPARISON TABLE")
    print("=" * 90)
    print(f"{'Metric':<30} {'Old (original 68/5057)':<28} {'Rejected (unnorm Kaggle)':<28} {'New (norm Kaggle)':<28}")
    print("-" * 90)

    rows = [
        ("On Own Test Set", "", "", ""),
        ("  AUC-ROC", old_results.get("auc_roc", "N/A"), rejected_results.get("auc_roc", 1.000),
         new_own.get("auc_roc", "N/A")),
        ("  Best F1", old_results.get("best_f1", "N/A"), rejected_results.get("best_f1", 1.000),
         new_own.get("best_f1", "N/A")),
        ("  Recall", old_results.get("recall", "N/A"), rejected_results.get("recall", 1.0),
         new_own.get("recall", "N/A")),
        ("  Precision", old_results.get("precision", "N/A"), rejected_results.get("precision", 1.0),
         new_own.get("precision", "N/A")),
        ("", "", "", ""),
        ("On Original Dataset", "", "", ""),
        ("  AUC-ROC", "0.968", rejected_results.get("cross_auc", 0.500),
         new_cross.get("auc", "N/A") if new_cross else "N/A"),
        ("  Best F1", "0.385", rejected_results.get("cross_f1", 0.026),
         new_cross.get("f1", "N/A") if new_cross else "N/A"),
        ("  Recall", "0.50", "1.00 (pred all AML)",
         f"{new_cross.get('recall', 'N/A'):.2f}" if new_cross else "N/A"),
        ("  Precision", "0.31", "0.013",
         f"{new_cross.get('precision', 'N/A'):.2f}" if new_cross else "N/A"),
    ]

    for name, old_v, rej_v, new_v in rows:
        if name == "":
            print()
            continue
        if name.startswith("  "):
            def fmt(v):
                if isinstance(v, (int, float)):
                    if isinstance(v, float) and v < 0.01:
                        return f"{v:.4f}"
                    elif isinstance(v, float):
                        return f"{v:.3f}"
                    return str(v)
                return str(v)

            print(f"{name:<32}{fmt(old_v):<28}{fmt(rej_v):<28}{fmt(new_v):<28}")
        else:
            print(f"\n{name}")

    print("=" * 90)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    # Step 1: Normalize
    fit_and_normalize()

    # Step 2: Shortcut checks
    if not run_shortcut_checks():
        print("\nABORT: Shortcut still detectable after normalization.")
        return

    # Step 3: Create splits
    print("\n" + "=" * 60)
    print("STEP 3: DATA SPLITS")
    print("=" * 60)
    train_entries, val_entries, test_entries, neg_ratio = create_kaggle_splits()
    alpha = round(neg_ratio, 4)
    print(f"  Computed focal alpha: {alpha}")

    # Step 4: Train
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING")
    print("=" * 60)
    train_loader, val_loader, test_loader = create_dataloaders_norm(
        train_entries, val_entries, test_entries
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    model, history, best_epoch = train_model(train_loader, val_loader, alpha)

    # Step 5: Verify
    print("\n" + "=" * 60)
    print("STEP 5: VERIFICATION")
    print("=" * 60)
    print(f"\n  Model: epoch {best_epoch}")
    new_own = verify_model(model, test_loader)
    new_cross = original_dataset_test(model)

    # Step 6: Comparison
    old_results = {
        "auc_roc": 0.968, "best_f1": 0.385,
        "recall": 0.50, "precision": 0.31,
    }
    rejected_results = {
        "auc_roc": 1.000, "best_f1": 1.000,
        "recall": 1.0, "precision": 1.0,
        "cross_auc": 0.500, "cross_f1": 0.026,
    }

    print_comparison(old_results, rejected_results, new_own, new_cross)

    # Save final model
    final_path = OUTPUT_DIR / "alnet_normalized_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")
    print(f"All outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
