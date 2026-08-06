import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTPUT_DIR = Path("outputs")


def draw_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def add_block(x, y, w, h, text, color="#4A90D9", text_color="white", fontsize=10, bold=False):
        rect = FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="#2C5F8A", linewidth=1.5,
        )
        ax.add_patch(rect)
        weight = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", color=text_color,
                fontsize=fontsize, fontweight=weight)

    def add_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=2))

    def add_section_label(x, y, text):
        ax.text(x, y, text, ha="left", va="center", fontsize=11,
                fontweight="bold", color="#2C3E50")

    # Title
    ax.text(6, 9.5, "ALNet Architecture for AML Detection",
            ha="center", va="center", fontsize=16, fontweight="bold", color="#1A1A2E")

    # Input
    add_block(6, 8.6, 2.2, 0.7, "Input\n224x224x3", color="#E67E22", fontsize=9)
    add_arrow(6, 8.25, 6, 7.8)

    # Section: Convolutional Feature Extraction
    add_section_label(0.2, 7.35, "Conv Block 1")

    # Conv Block 1
    add_block(2.2, 7.1, 2.0, 0.55, "Depthwise SepConv\n32 filters", fontsize=8)
    add_block(4.6, 7.1, 2.0, 0.55, "Depthwise SepConv\n32 filters", fontsize=8)
    add_arrow(2.2, 6.825, 4.6, 6.825)

    # Attention Block
    add_block(7.2, 7.1, 2.6, 0.55, "Localized Sparse\nMulti-Head Attention", color="#8E44AD", fontsize=8)
    add_arrow(4.6, 6.825, 7.2, 6.825)

    add_arrow(7.2, 6.825, 6, 6.25)

    # Section: Conv Block 2
    add_section_label(0.2, 6.05, "Conv Block 2")

    add_block(2.2, 5.8, 2.0, 0.55, "Depthwise SepConv\n64 filters", fontsize=8)
    add_block(4.6, 5.8, 2.0, 0.55, "Depthwise SepConv\n64 filters", fontsize=8)
    add_arrow(2.2, 5.525, 4.6, 5.525)

    add_block(7.2, 5.8, 2.6, 0.55, "Localized Sparse\nMulti-Head Attention", color="#8E44AD", fontsize=8)
    add_arrow(4.6, 5.525, 7.2, 5.525)

    add_arrow(7.2, 5.525, 6, 4.95)

    # Transition Block
    add_section_label(0.2, 4.65, "Transition")
    add_block(6, 4.6, 2.5, 0.7, "Max Pooling\n2x2, stride 2", color="#27AE60", fontsize=9)
    add_arrow(6, 4.25, 6, 3.75)

    # Dense Block
    add_section_label(0.2, 3.45, "Dense Block")
    add_block(6, 3.5, 2.0, 0.5, "Dense 128 + ReLU", fontsize=8)
    add_arrow(6, 3.25, 6, 2.95)
    add_block(6, 2.8, 2.0, 0.35, "Dropout (0.5)", color="#7F8C8D", fontsize=8)
    add_arrow(6, 2.625, 6, 2.35)
    add_block(6, 2.2, 2.0, 0.5, "Dense 64 + ReLU", fontsize=8)
    add_arrow(6, 1.95, 6, 1.65)
    add_block(6, 1.5, 2.0, 0.35, "Dropout (0.3)", color="#7F8C8D", fontsize=8)

    add_arrow(6, 1.325, 6, 1.05)

    # Output
    add_block(3.2, 0.6, 2.4, 0.6, "Non AML", color="#E74C3C", fontsize=11, bold=True)
    add_block(8.8, 0.6, 2.4, 0.6, "AML", color="#C0392B", fontsize=11, bold=True)

    add_arrow(6, 0.85, 6, 0.6)
    add_arrow(6, 0.6, (3.2 + 2.4/2), 0.6)
    add_arrow(6, 0.6, (8.8 - 2.4/2), 0.6)

    ax.text(6, 0.9, "Softmax (2 units)", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#2C3E50")

    # Legend
    legend_items = [
        ("Depthwise SepConv", "#4A90D9"),
        ("Attention Block", "#8E44AD"),
        ("Pooling", "#27AE60"),
        ("Dropout", "#7F8C8D"),
        ("Output", "#E74C3C"),
    ]
    legend_x = 9.2
    legend_y = 1.6
    ax.text(legend_x + 0.3, legend_y + 1.1, "Legend", fontsize=9, fontweight="bold")
    for i, (label, color) in enumerate(legend_items):
        y_pos = legend_y + 0.8 - i * 0.3
        rect = mpatches.Rectangle((legend_x, y_pos - 0.08), 0.35, 0.16,
                                   facecolor=color, edgecolor="#2C5F8A", linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, y_pos, label, va="center", fontsize=8)

    # Weighted Focal Loss annotation
    ax.text(6, 0.15, "Loss: Weighted Focal Loss (α=0.75, γ=2.0)  |  Optimizer: AdamW  |  LR: Cosine Annealing",
            ha="center", va="center", fontsize=8, color="#7F8C8D", style="italic")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "alnet_architecture.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Architecture diagram saved to {save_path}")


if __name__ == "__main__":
    draw_architecture()
