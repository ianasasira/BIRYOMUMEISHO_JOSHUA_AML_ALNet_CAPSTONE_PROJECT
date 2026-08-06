import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data_pipeline import create_dataloaders
from alnet_model import ALNet

OUTPUT_DIR = Path("outputs")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
CLASS_NAMES = ["Non-AML", "AML"]


def load_model():
    model_path = OUTPUT_DIR / "alnet_best.pt"
    if not model_path.exists():
        model_path = OUTPUT_DIR / "alnet_final.pt"
    if not model_path.exists():
        raise FileNotFoundError("No trained model found. Run train.py first.")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model = ALNet(num_classes=2).to(DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best model from epoch {checkpoint.get('epoch', '?')} "
              f"(val_loss={checkpoint.get('val_loss', '?'):.4f})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict")

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

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.argmax(all_probs, axis=1)

    return all_probs, all_labels, all_preds


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("ALNet Confusion Matrix — Test Set")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")
    return cm


def plot_roc_curve(y_true, y_score, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ALNet (AUC = {auc:.4f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve — ALNet on Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"ROC curve saved to {save_path}")
    return auc


def evaluate():
    print(f"Device: {DEVICE}")
    print("Loading model ...")
    model = load_model()

    print("Loading test data ...")
    _, _, test_loader = create_dataloaders(batch_size=BATCH_SIZE, num_workers=0, use_weighted_sampler=False)

    print("Running inference on test set ...")
    probs, labels, preds = get_predictions(model, test_loader)
    pos_probs = probs[:, 1]

    f1 = f1_score(labels, preds, average="binary")
    precision = precision_score(labels, preds, average="binary", zero_division=0)
    macro_precision = precision_score(labels, preds, average="macro", zero_division=0)
    sensitivity = recall_score(labels, preds, average="binary", zero_division=0)
    specificity = recall_score(labels, preds, pos_label=0, zero_division=0)
    auc = roc_auc_score(labels, pos_probs)
    cm = confusion_matrix(labels, preds)

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    results = {
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "macro_precision": round(float(macro_precision), 4),
        "sensitivity_recall": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "auc_roc": round(float(auc), 4),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "test_size": int(len(labels)),
        "test_positives": int(np.sum(labels)),
        "test_negatives": int(len(labels) - np.sum(labels)),
    }

    print(f"\n{'='*50}")
    print("ALNet Evaluation Results — Test Set")
    print(f"{'='*50}")
    print(f"Test samples: {results['test_size']} "
          f"(AML={results['test_positives']}, Non-AML={results['test_negatives']})")
    print(f"\nConfusion Matrix:")
    print(f"  TN={tn:5d}  FP={fp:5d}")
    print(f"  FN={fn:5d}  TP={tp:5d}")
    print(f"\nMetrics:")
    print(f"  F1-Score:            {f1:.4f}")
    print(f"  Precision (binary):  {precision:.4f}")
    print(f"  Macro Precision:     {macro_precision:.4f}")
    print(f"  Sensitivity/Recall:  {sensitivity:.4f}")
    print(f"  Specificity:         {specificity:.4f}")
    print(f"  AUC-ROC:             {auc:.4f}")

    if sensitivity < 0.70:
        print(f"\n  *** WARNING: Recall ({sensitivity:.4f}) is below 0.70. "
              f"False negatives ({fn}) are HIGH. This is a screening tool — "
              f"missed AML cases are clinically dangerous. ***")

    print(f"\n{'-'*50}")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, zero_division=0))

    cm_path = OUTPUT_DIR / "confusion_matrix.png"
    roc_path = OUTPUT_DIR / "roc_curve.png"
    plot_confusion_matrix(labels, preds, cm_path)
    plot_roc_curve(labels, pos_probs, roc_path)

    results_path = OUTPUT_DIR / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    evaluate()
