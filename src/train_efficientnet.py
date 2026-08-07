import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score

from data_pipeline import (
    POSITIVE_TRANSFORM, TRAIN_TRANSFORM, EVAL_TRANSFORM,
    PerClassTransformDataset,
)
from alnet_model import ALNet_EfficientNet, WeightedCrossEntropy

OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS_PHASE1 = 10
EPOCHS_PHASE2 = 25
LR_PHASE1 = 1e-3
LR_PHASE2 = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 12
GRAD_CLIP_NORM = 1.0
SEED = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        with autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return running_loss / len(loader), 100.0 * correct / total, recall, f1


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                scaler, epochs, phase_name):
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "val_recall": [], "val_f1": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"PHASE {phase_name}: {epochs} epochs | LR={scheduler.get_last_lr()[0]:.1e}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc, val_recall, val_f1 = evaluate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        elapsed = time.time() - start_time
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            marker = " *"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_recall": val_recall,
            }, OUTPUT_DIR / "alnet_efficientnet_head_best.pt")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s{marker}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nPhase {phase_name} complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")
    return history


def train():
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Loading data with batch_size={BATCH_SIZE} ...")
    with open(OUTPUT_DIR / "split_manifest.json", "r") as f:
        manifest = json.load(f)
    train_pos = manifest["counts"]["train_positive"]
    train_neg = manifest["counts"]["train_negative"]

    train_ds = PerClassTransformDataset(
        manifest["train"], pos_transform=POSITIVE_TRANSFORM, neg_transform=TRAIN_TRANSFORM,
        eval_transform=EVAL_TRANSFORM, is_train=True,
    )
    labels = [e["label"] for e in manifest["train"]]
    pos_w = 1.0 / train_pos
    neg_w = 1.0 / train_neg
    sample_weights = [pos_w if l == 1 else neg_w for l in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=train_pos * 50, replacement=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

    val_ds = PerClassTransformDataset(
        manifest["val"], pos_transform=EVAL_TRANSFORM, neg_transform=EVAL_TRANSFORM,
        eval_transform=EVAL_TRANSFORM, is_train=False,
    )
    test_ds = PerClassTransformDataset(
        manifest["test"], pos_transform=EVAL_TRANSFORM, neg_transform=EVAL_TRANSFORM,
        eval_transform=EVAL_TRANSFORM, is_train=False,
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {train_pos} pos + {train_neg} neg (sampler: {train_pos * 50} samples/epoch)")
    print(f"Val:   {manifest['counts']['val_positive']} pos + {manifest['counts']['val_negative']} neg")

    print("Building ALNet_EfficientNet (EfficientNet-B0 backbone) ...")
    model = ALNet_EfficientNet(num_classes=2).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    criterion = WeightedCrossEntropy(num_pos=train_pos, num_neg=train_neg)

    # --- Phase 1: Train classifier head only ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE1, eta_min=1e-5)
    scaler = GradScaler('cuda')

    history_p1 = train_phase(model, train_loader, val_loader, criterion,
                             optimizer, scheduler, scaler, EPOCHS_PHASE1, "1 (head only)")

    # --- Phase 2: Unfreeze backbone, fine-tune ---
    model.unfreeze_backbone()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nUnfrozen backbone. Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_PHASE2, eta_min=1e-6)
    scaler = GradScaler('cuda')

    history_p2 = train_phase(model, train_loader, val_loader, criterion,
                             optimizer, scheduler, scaler, EPOCHS_PHASE2, "2 (full fine-tune)")

    # --- Save ---
    history = {}
    for k in history_p1:
        history[k] = history_p1[k] + history_p2[k]

    with open(OUTPUT_DIR / "training_history_efficientnet.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(OUTPUT_DIR / "alnet_efficientnet_best.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    torch.save(model.state_dict(), OUTPUT_DIR / "alnet_efficientnet_final.pt")
    print(f"\nModel saved to outputs/alnet_efficientnet_final.pt")
    return model, history


if __name__ == "__main__":
    train()
