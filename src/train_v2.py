import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torch.amp import GradScaler, autocast

from data_pipeline import create_dataloaders
from alnet_model import ALNet, WeightedFocalLoss, count_parameters

OUTPUT_DIR = Path("outputs")
LOG_DIR = Path("logs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCHS = 120
LR_INIT = 1e-3
WEIGHT_DECAY = 1e-4
FOCAL_GAMMA = 2.0
EARLY_STOP_PATIENCE = 20
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

    return running_loss / len(loader), 100.0 * correct / total


def train():
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"Loading data with batch_size={BATCH_SIZE} ...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE, num_workers=0, use_balanced=True
    )

    with open(OUTPUT_DIR / "split_manifest.json", "r") as f:
        manifest = json.load(f)
    train_pos = manifest["counts"]["train_positive"]
    train_neg = manifest["counts"]["train_negative"]
    focal_alpha = round(train_neg / (train_pos + train_neg), 4)

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Computed focal alpha (class ratio): {focal_alpha}")

    print("Building ALNet ...")
    model = ALNet(num_classes=2).to(DEVICE)
    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,}")

    criterion = WeightedFocalLoss(alpha=focal_alpha, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler('cuda')

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    print(f"\n{'='*60}")
    print(f"Training ALNet v2 | Epochs: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"Focal Loss: alpha={focal_alpha}, gamma={FOCAL_GAMMA}")
    print(f"Optimizer: AdamW, lr={LR_INIT}, wd={WEIGHT_DECAY}")
    print(f"Balanced batches: 30% positive per batch")
    print(f"Stronger augmentation on positive class")
    print(f"{'='*60}\n")

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
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
            }, OUTPUT_DIR / "alnet_v2_best.pt")
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {current_lr:.2e} | {elapsed:.1f}s{marker}"
        )

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f} at epoch {best_epoch}")

    with open(OUTPUT_DIR / "training_history_v2.json", "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(OUTPUT_DIR / "alnet_v2_best.pt", weights_only=False)["model_state_dict"])
    torch.save(model.state_dict(), OUTPUT_DIR / "alnet_v2_final.pt")
    print(f"Model saved to outputs/alnet_v2_final.pt")

    return model, history


if __name__ == "__main__":
    train()
