import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path("outputs")


def plot_training_curves(history_path=None):
    if history_path is None:
        history_path = OUTPUT_DIR / "training_history.json"
    if not history_path.exists():
        print(f"History file not found: {history_path}")
        return

    with open(history_path, "r") as f:
        history = json.load(f)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.5)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Weighted Focal Loss)")
    ax1.set_title("ALNet Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    best_epoch = np.argmin(history["val_loss"]) + 1
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7,
                label=f"Best epoch ({best_epoch})")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy", linewidth=1.5)
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("ALNet Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    save_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def main():
    history_path = OUTPUT_DIR / "training_history.json"
    if history_path.exists():
        plot_training_curves(history_path)
    else:
        print("No training history found. Run train.py first.")


if __name__ == "__main__":
    main()
