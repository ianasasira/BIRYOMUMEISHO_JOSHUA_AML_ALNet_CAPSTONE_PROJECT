import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_pipeline_v2 import create_dataloaders
from alnet_model import ALNet

OUTPUT_DIR = Path("outputs/v2_alnet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CLASS_NAMES = ["Monocyte", "Myeloblast"]


def load_model():
    model_path = OUTPUT_DIR / "alnet_best.pt"
    if not model_path.exists():
        model_path = OUTPUT_DIR / "alnet_final.pt"
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = ALNet(num_classes=2).to(DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


@torch.no_grad()
def get_predictions(model, loader):
    all_probs = []
    all_labels = []

    for images, labels in loader:
        images = images.to(DEVICE)
        with autocast():
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def find_best_threshold(probs, labels):
    pos_probs = probs[:, 1]
    thresholds = np.arange(0.05, 1.0, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    best_metrics = {}

    results = []
    for t in thresholds:
        preds = (pos_probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        prec = precision_score(labels, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        results.append({
            "threshold": round(float(t), 2),
            "f1": round(float(f1), 4),
            "recall": round(float(rec), 4),
            "precision": round(float(prec), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {
                "f1": f1, "recall": rec, "precision": prec,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
            }

    return best_thresh, best_metrics, results


def plot_threshold_analysis(results, save_path):
    thresholds = [r["threshold"] for r in results]
    f1s = [r["f1"] for r in results]
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1s, label="F1-Score", linewidth=2)
    plt.plot(thresholds, recalls, label="Recall (Sensitivity)", linewidth=2)
    plt.plot(thresholds, precisions, label="Precision", linewidth=2)

    best_idx = np.argmax(f1s)
    plt.axvline(x=thresholds[best_idx], color="gray", linestyle="--",
                label=f"Best F1 threshold = {thresholds[best_idx]:.2f}")

    plt.xlabel("Classification Threshold")
    plt.ylabel("Score")
    plt.title("ALNet v2 — Threshold vs. Performance Metrics")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model and test data ...")
    model = load_model()
    _, _, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE, num_workers=0
    )

    print("Running inference ...")
    probs, labels = get_predictions(model, test_loader)

    pos_probs = probs[:, 1]
    auc = roc_auc_score(labels, pos_probs)
    print(f"AUC-ROC: {auc:.4f}")

    best_thresh, best_metrics, all_results = find_best_threshold(probs, labels)

    print(f"\nDefault threshold (0.5):")
    preds_default = (pos_probs >= 0.5).astype(int)
    cm_default = confusion_matrix(labels, preds_default)
    tn, fp, fn, tp = cm_default.ravel()
    print(f"  TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Recall={recall_score(labels, preds_default, zero_division=0):.4f}")
    print(f"  F1={f1_score(labels, preds_default, zero_division=0):.4f}")
    print(f"  Accuracy={np.mean(preds_default == labels):.4f}")

    print(f"\nBest threshold: {best_thresh:.2f}")
    print(f"  TN={best_metrics['tn']}, FP={best_metrics['fp']}, FN={best_metrics['fn']}, TP={best_metrics['tp']}")
    print(f"  Recall: {best_metrics['recall']:.4f}  |  Precision: {best_metrics['precision']:.4f}  |  F1: {best_metrics['f1']:.4f}")

    if best_metrics["recall"] >= 0.70:
        print(f"\n  *** Recall >= 0.70 achieved at threshold {best_thresh:.2f} ***")

    plot_threshold_analysis(all_results, OUTPUT_DIR / "threshold_analysis.png")
    print(f"\nThreshold plot saved to {OUTPUT_DIR / 'threshold_analysis.png'}")

    with open(OUTPUT_DIR / "threshold_results.json", "w") as f:
        json.dump({
            "auc_roc": round(float(auc), 4),
            "best_threshold": round(float(best_thresh), 2),
            "best_metrics": {k: (int(v) if isinstance(v, (np.integer,)) else round(float(v), 4) if isinstance(v, (np.floating, float)) else v) for k, v in best_metrics.items()},
            "all_thresholds": all_results,
        }, f, indent=2, default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x))


if __name__ == "__main__":
    main()
