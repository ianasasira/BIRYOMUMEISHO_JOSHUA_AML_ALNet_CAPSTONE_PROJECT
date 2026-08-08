"""
Generate ALL visualizations for the capstone report.
No PyTorch dependency — uses only matplotlib, numpy, PIL, and existing JSON data.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps

OUTPUT_DIR = Path("outputs")
SRC_DIR = Path("src")
DATASET_DIR = Path("dataset")
NORM_DIR = Path("dataset_normalized")
FIGS_DIR = OUTPUT_DIR / "report_figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
})


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def add_figure_border(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#cccccc")


# ============================================================
# Figure 1: Sample Images Grid (Monocyte vs Myeloblast, Pre/Post Norm)
# ============================================================
def generate_sample_images_grid():
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    fig.suptitle("Sample Images: Monocyte and Myeloblast Cells\n(Pre- and Post-Reinhard Normalization)",
                 fontsize=14, fontweight="bold", y=1.0)

    raw_mono = sorted((DATASET_DIR / "monocyte").glob("*.jpg"))
    raw_myelo = sorted((DATASET_DIR / "myeloblast").glob("*.jpg"))
    norm_mono = sorted((NORM_DIR / "monocyte").glob("*.png"))
    norm_myelo = sorted((NORM_DIR / "myeloblast").glob("*.png"))

    image_sets = [
        (raw_mono[:2], "Monocyte (Raw)", axes[0, :2]),
        (raw_myelo[:2], "Myeloblast (Raw)", axes[0, 2:]),
        (norm_mono[:2], "Monocyte (Normalized)", axes[1, :2]),
        (norm_myelo[:2], "Myeloblast (Normalized)", axes[1, 2:]),
    ]

    for img_paths, title, axs in image_sets:
        for ax, img_path in zip(axs, img_paths):
            if img_path.exists():
                img = Image.open(img_path).convert("RGB")
                img = ImageOps.expand(img, border=2, fill="black")
                ax.imshow(img)
            ax.axis("off")
        axs[0].set_title(title, fontsize=11, fontweight="bold", loc="left", pad=5)

    plt.tight_layout()
    path = FIGS_DIR / "figure01_sample_images.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 2: Dataset Class Distribution
# ============================================================
def generate_class_distribution():
    inventory = load_json(OUTPUT_DIR / "dataset_inventory.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Dataset Composition and Distribution", fontsize=14, fontweight="bold")

    # Ax1: Original dataset composition (imbalanced)
    cats_orig = list(inventory.get("AML positive", {}).keys())
    cats_orig = [c for c in cats_orig if not c.startswith("_")]
    counts_orig = [inventory["AML positive"][c] for c in cats_orig]
    cats_neg = list(inventory.get("NEGATIVE", {}).keys())
    cats_neg = [c for c in cats_neg if not c.startswith("_")]
    counts_neg = [inventory["NEGATIVE"][c] for c in cats_neg]

    colors1 = ["#d4a0a0", "#c0392b"]
    wedges1, _, autotexts1 = ax1.pie(
        [inventory["_negative"], inventory["_positive"]],
        labels=[f"Negative\n({inventory['_negative']})", f"Positive\n({inventory['_positive']})"],
        autopct="%1.1f%%", colors=["#3498db", "#e74c3c"],
        explode=(0, 0.05), startangle=90
    )
    for t in autotexts1:
        t.set_fontweight("bold")
        t.set_fontsize(10)
    ax1.set_title("Original Dataset\n(5,125 images)", fontsize=12, fontweight="bold")

    # Ax2: Balanced training dataset
    labels2 = ["Monocyte\n(1,000)", "Myeloblast\n(1,000)"]
    colors2 = ["#2ecc71", "#e74c3c"]
    wedges2, _, autotexts2 = ax2.pie(
        [1000, 1000], labels=labels2, autopct="%1.1f%%",
        colors=colors2, startangle=90
    )
    for t in autotexts2:
        t.set_fontweight("bold")
        t.set_fontsize(10)
    ax2.set_title("Training Dataset\n(Balanced, 2,000 images)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = FIGS_DIR / "figure02_class_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 3: Data Split Proportions
# ============================================================
def generate_data_split():
    fig, ax = plt.subplots(figsize=(7, 5))
    splits = {"Training": 1400, "Validation": 300, "Test": 300}
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]

    wedges, texts, autotexts = ax.pie(
        splits.values(), labels=[f"{k}\n({v})" for k, v in splits.items()],
        autopct="%1.0f%%", colors=colors, startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts:
        t.set_fontweight("bold")
        t.set_fontsize(11)
    ax.set_title("Data Split Distribution\n(Stratified 70/15/15)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = FIGS_DIR / "figure03_data_split.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 4: Pre vs Post-Normalization Colour Analysis
# ============================================================
def generate_color_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Stain Normalization Impact: Colour-Based Class Separability",
                 fontsize=13, fontweight="bold")

    metrics = ["Red/Blue\nRatio", "Brightness", "Saturation"]
    pre = [98.2, 91.0, 88.3]
    post = [50.0, 62.9, 55.1]

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax1.bar(x - width/2, pre, width, label="Pre-Normalization", color="#e74c3c", alpha=0.8)
    bars2 = ax1.bar(x + width/2, post, width, label="Post-Normalization", color="#2ecc71", alpha=0.8)
    ax1.set_ylabel("Class Separability (%)")
    ax1.set_title("Single-Feature Class Separability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random chance")
    ax1.set_ylim(0, 110)
    ax1.grid(axis="y", alpha=0.3)
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    shortcut_types = ["Colour Ratio", "Brightness", "Edge Density", "Texture"]
    pre_sep = [98.2, 91.0, 72.5, 64.8]
    post_sep = [50.0, 62.9, 71.8, 63.2]

    y_pos = range(len(shortcut_types))
    ax2.barh([y - 0.15 for y in y_pos], pre_sep, 0.3, label="Pre-Norm", color="#e74c3c", alpha=0.8)
    ax2.barh([y + 0.15 for y in y_pos], post_sep, 0.3, label="Post-Norm", color="#2ecc71", alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(shortcut_types)
    ax2.set_xlabel("Separability (%)")
    ax2.set_title("Shortcut Detection Analysis")
    ax2.axvline(x=50, color="gray", linestyle="--", alpha=0.5)
    ax2.legend()
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = FIGS_DIR / "figure04_color_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 5: Detailed Training Curves (ALNet)
# ============================================================
def generate_detailed_training():
    history = load_json(OUTPUT_DIR / "training_history.json")
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = np.argmin(history["val_loss"]) + 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ALNet Training Dynamics", fontsize=14, fontweight="bold")

    # Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history["train_loss"], "b-", label="Training Loss", linewidth=1.5, alpha=0.8)
    ax1.plot(epochs, history["val_loss"], "r-", label="Validation Loss", linewidth=1.5, alpha=0.8)
    ax1.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.6,
                label=f"Best Epoch = {best_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (Weighted Focal Loss)")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Accuracy curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history["train_acc"], "b-", label="Training Accuracy", linewidth=1.5, alpha=0.8)
    ax2.plot(epochs, history["val_acc"], "r-", label="Validation Accuracy", linewidth=1.5, alpha=0.8)
    ax2.axvline(x=best_epoch, color="gray", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Learning rate schedule
    ax3 = axes[1, 0]
    ax3.plot(epochs, history["lr"], "g-", linewidth=1.5)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Cosine Annealing LR Schedule")
    ax3.grid(alpha=0.3)
    ax3.ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))

    # Loss gap analysis
    ax4 = axes[1, 1]
    loss_gap = np.array(history["train_loss"]) - np.array(history["val_loss"])
    colors_gap = ["#e74c3c" if g > 0 else "#2ecc71" for g in loss_gap]
    ax4.bar(epochs, loss_gap, color=colors_gap, alpha=0.7)
    ax4.axhline(y=0, color="black", linewidth=0.8)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Train Loss − Val Loss")
    ax4.set_title("Overfitting Gap Analysis")
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = FIGS_DIR / "figure05_detailed_training.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 6: Confusion Matrix (Actual Numbers)
# ============================================================
def generate_confusion_matrix():
    results = load_json(OUTPUT_DIR / "evaluation_results.json")
    cm_data = results["confusion_matrix"]
    cm = np.array([[cm_data["TN"], cm_data["FP"]],
                   [cm_data["FN"], cm_data["TP"]]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("ALNet Confusion Matrix — Test Set (N = 769)", fontsize=13, fontweight="bold")

    # Raw counts
    im1 = ax1.matshow(cm, cmap="Blues")
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Predicted\nNon-AML", "Predicted\nAML"])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Actual\nNon-AML", "Actual\nAML"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, fontweight="bold", color=color)
    ax1.set_title("Raw Counts", fontweight="bold")
    fig.colorbar(im1, ax=ax1, shrink=0.8)

    # Normalized by row
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
    im2 = ax2.matshow(cm_norm, cmap="Blues", vmin=0, vmax=100)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Predicted\nNon-AML", "Predicted\nAML"])
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(["Actual\nNon-AML", "Actual\nAML"])
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 50 else "black"
            ax2.text(j, i, f"{cm_norm[i, j]:.1f}%", ha="center", va="center",
                     fontsize=14, fontweight="bold", color=color)
    ax2.set_title("Normalized by Row (%)", fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.8)
    ax2.text(0.5, -0.18,
             f"TN = 748 | FP = 11 | FN = 5 | TP = 5\n"
             f"Sensitivity: 50.0% | Specificity: 98.55%",
             ha="center", transform=ax2.transAxes, fontsize=10, fontstyle="italic")

    plt.tight_layout()
    path = FIGS_DIR / "figure06_confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 7: ROC Curve + PR Curve
# ============================================================
def generate_roc_pr_curves():
    results = load_json(OUTPUT_DIR / "evaluation_results.json")
    thresh = load_json(OUTPUT_DIR / "threshold_results.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("ALNet Classification Performance Curves", fontsize=13, fontweight="bold")

    # ROC Curve (simulated from available data)
    # Since we don't have raw probabilities, we build approximate curves
    all_t = thresh["all_thresholds"]
    thresholds_vals = [r["threshold"] for r in all_t]
    recalls = [r["recall"] for r in all_t]
    f1s = [r["f1"] for r in all_t]
    precisions = [r["precision"] for r in all_t]
    fp_counts = [r["fp"] for r in all_t]
    tp_counts = [r["tp"] for r in all_t]
    tn_counts = [r["tn"] for r in all_t]
    fn_counts = [r["fn"] for r in all_t]

    total_neg = results["test_negatives"]
    total_pos = results["test_positives"]

    fpr_vals = [fp / total_neg for fp in fp_counts]
    tpr_vals = [tp / total_pos for tp in tp_counts]

    auc = results["auc_roc"]

    ax1.plot(fpr_vals, tpr_vals, "b-", linewidth=2,
             label=f"ALNet (AUC = {auc:.4f})")
    ax1.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.5)")
    ax1.fill_between(fpr_vals, tpr_vals, alpha=0.1, color="blue")
    ax1.set_xlabel("False Positive Rate (1 − Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # PR Curve
    ax2.plot(recalls, precisions, "r-", linewidth=2, label="ALNet")
    ax2.fill_between(recalls, precisions, alpha=0.1, color="red")
    baseline = total_pos / (total_pos + total_neg)
    ax2.axhline(y=baseline, color="gray", linestyle="--", alpha=0.7,
                label=f"Baseline ({baseline:.3f})")
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_DIR / "figure07_roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 8: Threshold Analysis Enhanced
# ============================================================
def generate_threshold_analysis_enhanced():
    thresh = load_json(OUTPUT_DIR / "threshold_results.json")
    all_t = thresh["all_thresholds"]

    thresholds_vals = [r["threshold"] for r in all_t]
    f1s = [r["f1"] for r in all_t]
    recalls = [r["recall"] for r in all_t]
    precisions = [r["precision"] for r in all_t]
    fps = [r["fp"] for r in all_t]
    fns = [r["fn"] for r in all_t]
    tps = [r["tp"] for r in all_t]

    best_idx = np.argmax(f1s)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("Classification Threshold Analysis", fontsize=13, fontweight="bold")

    # Top: F1, Recall, Precision
    ax1.plot(thresholds_vals, f1s, "g-", linewidth=2.5, label="F1-Score")
    ax1.plot(thresholds_vals, recalls, "b-", linewidth=2, label="Recall (Sensitivity)")
    ax1.plot(thresholds_vals, precisions, "r-", linewidth=2, label="Precision")
    ax1.axvline(x=thresholds_vals[best_idx], color="gray", linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"Best F1 = {thresholds_vals[best_idx]:.2f}")
    ax1.axvline(x=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5, label="Default (0.50)")
    ax1.set_ylabel("Score")
    ax1.set_title("F1 Score, Recall, and Precision by Threshold")
    ax1.legend(loc="center right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Bottom: Error counts
    ax2.fill_between(thresholds_vals, fps, alpha=0.3, color="#e74c3c", label="False Positives")
    ax2.plot(thresholds_vals, fps, "r-", linewidth=1.5, alpha=0.7)
    ax2.fill_between(thresholds_vals, fns, alpha=0.3, color="#f39c12", label="False Negatives")
    ax2.plot(thresholds_vals, fns, "orange", linewidth=1.5, alpha=0.7)
    ax2.axvline(x=thresholds_vals[best_idx], color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_xlabel("Classification Threshold")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Analysis — False Positives and False Negatives by Threshold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # Annotation boxes
    ax2.annotate("Clinical Preference:\nLow FN (missed AML)\nat cost of higher FP",
                 xy=(0.25, 65), fontsize=9, fontstyle="italic",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    path = FIGS_DIR / "figure08_threshold_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 9: Model Performance Radar Chart
# ============================================================
def generate_metrics_radar():
    results = load_json(OUTPUT_DIR / "evaluation_results.json")

    metrics_names = ["F1-Score", "Precision", "Recall\n(Sensitivity)", "Specificity", "AUC-ROC"][::-1]
    metrics_values = [
        results["f1_score"],
        results["precision"],
        results["sensitivity_recall"],
        results["specificity"],
        results["auc_roc"],
    ][::-1]

    N = len(metrics_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values = metrics_values + metrics_values[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, values, alpha=0.25, color="#3498db")
    ax.plot(angles, values, "b-", linewidth=2, label="ALNet")
    ax.fill(angles, [0.5] * (N + 1), alpha=0.1, color="gray")
    ax.plot(angles, [0.5] * (N + 1), "gray", linestyle="--", linewidth=1, label="Random Baseline")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.set_title("ALNet Performance Metrics\n(Radar Chart)", fontsize=13, fontweight="bold", pad=20)

    plt.tight_layout()
    path = FIGS_DIR / "figure09_metrics_radar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 10: Model Complexity Analysis
# ============================================================
def generate_model_complexity():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("ALNet Model Complexity and Efficiency", fontsize=13, fontweight="bold")

    models = ["ALNet", "EfficientNet-B0", "DenseNet121", "ResNet50", "ViT-B/16"]
    params = [27393, 5289000, 7979000, 25560000, 86000000]
    sizes_mb = [0.136, 20.2, 30.5, 97.5, 330.0]

    bars1 = ax1.barh(models, params, color=["#2ecc71", "#3498db", "#3498db", "#3498db", "#e74c3c"])
    ax1.set_xlabel("Number of Parameters")
    ax1.set_title("Model Parameters Comparison")
    ax1.set_xscale("log")
    for bar, val in zip(bars1, params):
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height()/2,
                 f"  {val:,}", va="center", fontsize=9, fontweight="bold")

    bars2 = ax2.bar(models, sizes_mb, color=["#2ecc71", "#3498db", "#3498db", "#3498db", "#e74c3c"])
    ax2.set_ylabel("Disk Size (MB)")
    ax2.set_title("Model Size on Disk")
    ax2.set_yscale("log")
    for bar, val in zip(bars2, sizes_mb):
        if val < 1:
            lbl = f"  {val*1024:.0f} KB"
        else:
            lbl = f"  {val:.1f} MB"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.05,
                 lbl, ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    path = FIGS_DIR / "figure10_model_complexity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 11: Training Convergence Comparison (ALNet vs EfficientNet)
# ============================================================
def generate_training_comparison():
    hist_alnet = load_json(OUTPUT_DIR / "training_history.json")
    hist_eff = load_json(OUTPUT_DIR / "training_history_efficientnet.json")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Training Comparison: ALNet vs EfficientNet-B0", fontsize=13, fontweight="bold")

    # ALNet loss
    ax1 = axes[0, 0]
    e1 = range(1, len(hist_alnet["train_loss"]) + 1)
    ax1.plot(e1, hist_alnet["train_loss"], "b-", label="Train", linewidth=1, alpha=0.7)
    ax1.plot(e1, hist_alnet["val_loss"], "r-", label="Val", linewidth=1, alpha=0.7)
    ax1.set_title("ALNet — Loss (49 epochs)")
    ax1.legend(); ax1.grid(alpha=0.3)

    # EfficientNet loss
    ax2 = axes[0, 1]
    e2 = range(1, len(hist_eff["train_loss"]) + 1)
    ax2.plot(e2, hist_eff["train_loss"], "b-", label="Train", linewidth=1, alpha=0.7)
    ax2.plot(e2, hist_eff["val_loss"], "r-", label="Val", linewidth=1, alpha=0.7)
    ax2.set_title("EfficientNet-B0 — Loss (28 epochs)")
    ax2.legend(); ax2.grid(alpha=0.3)

    # ALNet accuracy
    ax3 = axes[1, 0]
    ax3.plot(e1, hist_alnet["train_acc"], "b-", label="Train", linewidth=1, alpha=0.7)
    ax3.plot(e1, hist_alnet["val_acc"], "r-", label="Val", linewidth=1, alpha=0.7)
    ax3.set_title("ALNet — Accuracy")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("%")
    ax3.legend(); ax3.grid(alpha=0.3)

    # EfficientNet accuracy
    ax4 = axes[1, 1]
    ax4.plot(e2, hist_eff["train_acc"], "b-", label="Train", linewidth=1, alpha=0.7)
    ax4.plot(e2, hist_eff["val_acc"], "r-", label="Val", linewidth=1, alpha=0.7)
    ax4.set_title("EfficientNet-B0 — Accuracy")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("%")
    ax4.legend(); ax4.grid(alpha=0.3)

    plt.tight_layout()
    path = FIGS_DIR / "figure11_training_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 12: Training Progress Timeline
# ============================================================
def generate_training_progress():
    history = load_json(OUTPUT_DIR / "training_history.json")
    epochs = range(1, len(history["train_loss"]) + 1)
    best_epoch = np.argmin(history["val_loss"]) + 1

    fig, ax1 = plt.subplots(figsize=(12, 5))

    c1 = ax1.plot(epochs, history["train_acc"], "o-", color="#3498db", markersize=4,
                  linewidth=1.5, label="Train Accuracy", alpha=0.7)
    c2 = ax1.plot(epochs, history["val_acc"], "s-", color="#e74c3c", markersize=4,
                  linewidth=1.5, label="Val Accuracy", alpha=0.7)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlabel("Epoch")
    ax1.axvline(x=best_epoch, color="green", linestyle="--", linewidth=2,
                alpha=0.6, label=f"Best Epoch = {best_epoch}")

    ax2 = ax1.twinx()
    c3 = ax2.plot(epochs, history["train_loss"], "v-", color="#95a5a6", markersize=4,
                  linewidth=1, label="Train Loss", alpha=0.5)
    c4 = ax2.plot(epochs, history["val_loss"], "^-", color="#e67e22", markersize=4,
                  linewidth=1, label="Val Loss", alpha=0.5)
    ax2.set_ylabel("Loss")

    lines1 = c1 + c2
    lines2 = c3 + c4
    labels1 = [l.get_label() for l in lines1]
    labels2 = [l.get_label() for l in lines2]
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)
    ax1.grid(alpha=0.3)

    ax1.set_title("ALNet Training Progress Timeline", fontsize=13, fontweight="bold")
    ax1.annotate(f"Best: Val Loss={history['val_loss'][best_epoch-1]:.6f}\n"
                 f"Val Acc={history['val_acc'][best_epoch-1]:.2f}%",
                 xy=(best_epoch, history['val_acc'][best_epoch-1]),
                 xytext=(best_epoch + 5, history['val_acc'][best_epoch-1] + 3),
                 arrowprops=dict(arrowstyle="->", color="green", lw=1.5),
                 fontsize=9, fontstyle="italic",
                 bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    plt.tight_layout()
    path = FIGS_DIR / "figure12_training_progress.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 13: Architecture Detail Diagram (Simplified Flow)
# ============================================================
def generate_architecture_flow():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    def block(x, y, w, h, text, color="#4A90D9", subtext=""):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor="#2C5F8A", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
        if subtext:
            ax.text(x, y - h/2 - 0.25, subtext, ha="center", va="top", fontsize=7,
                    color="#2C3E50", fontstyle="italic")

    def arrow(x1, y1, x2, y2, color="#555", lw=1.5):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=lw))

    # Title
    ax.text(5, 11.5, "ALNet Architecture — Data Flow", ha="center", va="center",
            fontsize=14, fontweight="bold")

    # Input
    block(5, 10.5, 2.2, 0.6, "Input Image", color="#E67E22", subtext="224×224×3 (RGB)")
    arrow(5, 10.2, 5, 9.7)

    # Conv Block 1
    block(5, 9.3, 2.5, 0.5, "Conv Block 1", color="#2980B9", subtext="2× Depthwise SepConv + Attention")
    arrow(5, 9.05, 5, 8.5)

    # Conv Block 2
    block(5, 8.15, 2.5, 0.5, "Conv Block 2", color="#2980B9", subtext="2× Depthwise SepConv + Attention")
    arrow(5, 7.9, 5, 7.35)

    # Pooling
    block(5, 7.0, 2.0, 0.5, "Max Pool + GAP", color="#27AE60", subtext="2×2 stride 2 + Global Avg Pool")
    arrow(5, 6.75, 5, 6.2)

    # Dense
    block(5, 5.85, 2.0, 0.5, "Dense 128 + ReLU", color="#8E44AD", subtext="Dropout 0.5")
    arrow(5, 5.6, 5, 5.05)

    block(5, 4.7, 2.0, 0.5, "Dense 64 + ReLU", color="#8E44AD", subtext="Dropout 0.3")
    arrow(5, 4.45, 5, 3.9)

    # Output
    block(5, 3.55, 2.0, 0.5, "Softmax (2 units)", color="#C0392B")
    arrow(5, 3.3, 5, 2.6)

    block(2.5, 2.1, 2.0, 0.6, "Non-AML", color="#2ECC71")
    block(7.5, 2.1, 2.0, 0.6, "AML Detected", color="#E74C3C")
    arrow(5, 2.3, 5, 2.1)
    arrow(5, 2.1, 3.5, 2.1)
    arrow(5, 2.1, 6.5, 2.1)

    # Stats box
    ax.text(8.2, 10.5, "Parameters: 27,393\nSize: 136 KB\nAUC-ROC: 0.9675",
            fontsize=8, fontstyle="italic", color="#7F8C8D",
            bbox=dict(boxstyle="round", facecolor="#ECF0F1", alpha=0.7))

    plt.tight_layout()
    path = FIGS_DIR / "figure13_architecture_flow.png"
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 14: Error Type Breakdown & Clinical Impact
# ============================================================
def generate_error_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle("Error Analysis and Clinical Implications", fontsize=13, fontweight="bold")

    outcomes = ["True\nNegative", "False\nPositive", "False\nNegative", "True\nPositive"]
    counts = [748, 11, 5, 5]
    colors_err = ["#2ecc71", "#f1c40f", "#e74c3c", "#2ecc71"]
    total = sum(counts)

    bars = ax1.bar(outcomes, counts, color=colors_err, edgecolor="white", linewidth=1.5)
    ax1.set_ylabel("Number of Cases")
    ax1.set_title("Classification Outcomes (N = 769)")
    ax1.grid(axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 f"{c}\n({c/total*100:.1f}%)", ha="center", fontsize=10, fontweight="bold")

    # Clinical impact matrix
    categories = [("False Negatives\n(Missed AML)", "5 cases\n50.0% recall", "#e74c3c"),
                  ("False Positives\n(Over-referral)", "11 cases\n98.55% specificity", "#f39c12"),
                  ("Correct Diagnoses", "753 cases\n97.9% accuracy", "#2ecc71")]
    y_pos = [2, 1, 0]
    for i, (label, detail, color) in enumerate(categories):
        ax2.barh(y_pos[i], 1, height=0.7, color=color, alpha=0.7, edgecolor="white")
        ax2.text(0.05, y_pos[i], f"{label}: {detail}", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 1.5)
    ax2.set_ylim(-0.7, 2.7)
    ax2.axis("off")
    ax2.set_title("Clinical Impact Assessment", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = FIGS_DIR / "figure14_error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 15: Performance Metrics Summary Dashboard
# ============================================================
def generate_performance_dashboard():
    results = load_json(OUTPUT_DIR / "evaluation_results.json")
    thresh = load_json(OUTPUT_DIR / "threshold_results.json")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.4)

    metric_data = {
        (0, 0): ("Accuracy", f"{97.9:.1f}%", "#3498db"),
        (0, 1): ("AUC-ROC", f"{results['auc_roc']:.4f}", "#9b59b6"),
        (0, 2): ("F1-Score", f"{results['f1_score']:.4f}", "#2ecc71"),
        (0, 3): ("Sensitivity", f"{results['sensitivity_recall']*100:.1f}%", "#e74c3c"),
        (1, 0): ("Specificity", f"{results['specificity']*100:.2f}%", "#3498db"),
        (1, 1): ("Precision", f"{results['precision']:.4f}", "#9b59b6"),
        (1, 2): ("Best Threshold", f"{thresh['best_threshold']:.2f}", "#e67e22"),
        (1, 3): ("Test Size", f"{results['test_size']}", "#1abc9c"),
    }

    for (r, c), (label, value, color) in metric_data.items():
        ax = fig.add_subplot(gs[r, c])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        rect = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.05",
                              facecolor=color, edgecolor="#2C3E5A", linewidth=2, alpha=0.85)
        ax.add_patch(rect)
        ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=22,
                fontweight="bold", color="white")
        ax.text(0.5, 0.28, label, ha="center", va="center", fontsize=11,
                color="white", alpha=0.9)

    # Confusion matrix small
    ax_cm = fig.add_subplot(gs[2:, :])
    ax_cm.axis("off")
    ax_cm.set_title("ALNet Performance Summary Dashboard", fontsize=14, fontweight="bold", y=1.05)

    cm_text = (
        f"Confusion Matrix at Default Threshold (0.50):\n"
        f"      TN = 748 (classify Non-AML correctly)       FP = 11 (Non-AML flagged as AML)\n"
        f"      FN = 5 (AML missed — clinically critical)    TP = 5 (AML correctly identified)"
    )
    ax_cm.text(0.5, 0.5, cm_text, ha="center", va="center", fontsize=11,
               fontfamily="monospace",
               bbox=dict(boxstyle="round", facecolor="#ECF0F1", edgecolor="#BDC3C7", alpha=0.9))

    path = FIGS_DIR / "figure15_performance_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 16: Stain Normalization Pipeline Diagram
# ============================================================
def generate_normalization_pipeline():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(6, 5.6, "Reinhard Stain Normalization Pipeline", ha="center", fontsize=14,
            fontweight="bold")

    steps = [
        (1.5, 4, "Load Raw\nImage", "#E67E22", "Wright-Giemsa\nstained .jpg"),
        (3.5, 4, "RGB → LAB\nConversion", "#3498db", "CIE L*a*b*\ncolour space"),
        (5.5, 4, "Compute\nStatistics", "#9b59b6", "μ, σ per\nchannel"),
        (7.5, 4, "Match to\nReference", "#e74c3c", "Scale to target\ndistribution"),
        (9.5, 4, "LAB → RGB\nConversion", "#27ae60", "Normalized\noutput image"),
        (11, 4, "Resize\n400×400", "#2c3e50", "Uniform\nsize"),
    ]

    for i, (x, y, text, color, sub) in enumerate(steps):
        rect = FancyBboxPatch((x - 0.8, y - 0.7), 1.6, 1.4,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor="#2C3E5A", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, text, ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
        ax.text(x, y - 0.35, sub, ha="center", va="center", color="white",
                fontsize=7, alpha=0.9)
        if i < len(steps) - 2:
            next_x = steps[i + 1][0]
            ax.annotate("", xy=(next_x - 0.85, 4), xytext=(x + 0.85, 4),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=2))
        elif i == len(steps) - 2:
            ax.annotate("", xy=(steps[-1][0] - 0.85, 4), xytext=(x + 0.85, 4),
                        arrowprops=dict(arrowstyle="->", color="#555", lw=2))

    # Reference box
    ref_box = FancyBboxPatch((6.5, 2.0), 2.0, 1.0, boxstyle="round,pad=0.1",
                              facecolor="#ecf0f1", edgecolor="#bdc3c7", linewidth=1)
    ax.add_patch(ref_box)
    ax.text(7.5, 2.7, "Reference Distribution", ha="center", fontsize=9, fontweight="bold")
    ax.text(7.5, 2.35, "100 randomly sampled\nimages from both classes", ha="center", fontsize=8)

    ax.annotate("", xy=(7.5, 3.0), xytext=(7.5, 3.6),
                arrowprops=dict(arrowstyle="->", color="#9b59b6", lw=1.5, linestyle="dashed"))

    # Validation box
    val_box = FancyBboxPatch((0.3, 0.5), 11.4, 1.2, boxstyle="round,pad=0.1",
                              facecolor="#eafaf1", edgecolor="#2ecc71", linewidth=1)
    ax.add_patch(val_box)
    ax.text(6, 1.35, "Validation checks:", ha="center", fontsize=10, fontweight="bold")
    ax.text(6, 0.9,
            "Red/Blue ratio separability: 98.2% → 50.0% (random)    "
            "Brightness separability: 91.0% → 62.9%    "
            "Image sizes: Variable → Uniform 400×400",
            ha="center", fontsize=9, fontstyle="italic")

    plt.tight_layout()
    path = FIGS_DIR / "figure16_normalization_pipeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# Figure 17: Desktop Application Architecture
# ============================================================
def generate_app_architecture():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5.5, 6.7, "ALNet Screening Tool — Desktop Application Architecture",
            ha="center", fontsize=13, fontweight="bold")

    layers = [
        ("Presentation Layer", 5.8, "#E74C3C", [
            "CustomTkinter GUI", "Image Preview Panel",
            "Classification Results", "Confidence Display",
            "History Viewer (SQLite)"
        ]),
        ("Inference Engine", 4.2, "#3498DB", [
            "ALNet Model (27,393 params)", "Softmax Probability",
            "Threshold Decision", "Stain Normalization"
        ]),
        ("Data Layer", 2.6, "#27AE60", [
            "Image Loader (PIL/OpenCV)", "Preprocessing Pipeline",
            "SQLite Session Logger", "Export Formats (PNG, CSV)"
        ]),
        ("Deployment", 1.0, "#8E44AD", [
            "PyInstaller Bundle", "Standalone .exe (850 MB)",
            "No Python Required", "Windows 10/11 Compatible"
        ]),
    ]

    for label, y, color, items in layers:
        rect = FancyBboxPatch((0.3, y - 0.5), 10.4, 1.3,
                              boxstyle="round,pad=0.1", facecolor=color,
                              edgecolor="white", linewidth=2, alpha=0.85)
        ax.add_patch(rect)

        ax.text(5.5, y + 0.9, label, ha="center", fontsize=11, fontweight="bold", color="white")

        cols = 5
        total_w = cols * 2.0 + (cols - 1) * 0.1
        start_x = 5.5 - total_w / 2 + 1.0

        for j, item in enumerate(items):
            item_x = start_x + j * 2.1
            item_rect = FancyBboxPatch((item_x - 0.9, y - 0.3), 1.8, 0.55,
                                        boxstyle="round,pad=0.05", facecolor="white",
                                        edgecolor="none", alpha=0.9)
            ax.add_patch(item_rect)
            ax.text(item_x, y - 0.02, item, ha="center", va="center",
                    fontsize=7, color="#2C3E50", fontweight="bold")

    # Arrows between layers
    for y_from, y_to in [(5.3, 4.8), (3.7, 3.2), (2.1, 1.6)]:
        ax.annotate("", xy=(5.5, y_to), xytext=(5.5, y_from),
                    arrowprops=dict(arrowstyle="->", color="white", lw=3))

    plt.tight_layout()
    path = FIGS_DIR / "figure17_app_architecture.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")
    return path


# ============================================================
# MAIN
# ============================================================
def main():
    print("Generating all report figures...")
    print("=" * 50)

    generate_sample_images_grid()
    generate_class_distribution()
    generate_data_split()
    generate_color_analysis()
    generate_detailed_training()
    generate_confusion_matrix()
    generate_roc_pr_curves()
    generate_threshold_analysis_enhanced()
    generate_metrics_radar()
    generate_model_complexity()
    generate_training_comparison()
    generate_training_progress()
    generate_architecture_flow()
    generate_error_analysis()
    generate_performance_dashboard()
    generate_normalization_pipeline()
    generate_app_architecture()

    print("=" * 50)
    print("All figures generated successfully!")


if __name__ == "__main__":
    main()
