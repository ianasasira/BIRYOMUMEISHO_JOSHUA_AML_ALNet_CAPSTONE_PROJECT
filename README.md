# ALNet — Automatic Detection of Acute Myeloblastic Leukemia

**MSc AI & Data Science Capstone Project**  
**Student:** BIRYOMUMEISHO JOSHUA (JAN26/MAIDS/0819U)  
**Supervisor:** DR. KASSIM KALINAKI (PhD)  
**Institution:** Universal Technology and Management University (UTAMU)

---

## Overview

ALNet (Acute Leukemia Network) is a lightweight deep learning model for automated AML screening from peripheral blood smear images. The model uses depthwise separable convolutions and localized sparse multi-head self-attention to extract morphological features while maintaining a compact 27,393-parameter footprint — suitable for deployment on standard laboratory hardware.

The project includes a **standalone Windows desktop application** (`ALNet_Screening_Tool.exe`) that bundles the trained model for one-click deployment by lab technicians — no Python, TensorFlow, or any dependency installation required.

### Key Features

- **ALNet Architecture**: Dual-branch hybrid model — depthwise separable convs + sparse attention  
- **Weighted Focal Loss**: Engineered for 74:1 class imbalance in AML screening  
- **Desktop App**: CustomTkinter GUI with drag-drop image input, real-time inference, confidence scores, SQLite audit logging  
- **Standalone .exe**: PyInstaller-packaged — runs on any Windows machine with one click  
- **Metrics**: AUC-ROC 0.9675 | 27,393 params | < 50ms inference on GPU  

---

## Quick Start

### Option 1: Run the Desktop App (.exe) — Recommended

1. Download `ALNet_Screening_Tool.exe` from [Releases](https://github.com/ianasasira/BIRYOMUMEISHO_JOSHUA_AML_ALNet_CAPSTONE_PROJECT/releases)  
2. Double-click to launch  
3. Load a blood smear image (PNG, JPG, BMP, TIFF)  
4. Click **Run Analysis**  
5. View prediction (AML / Non-AML) with confidence scores  

> **Note:** The .exe is ~2.5 GB because it bundles PyTorch, CUDA, and all dependencies. No installations needed.

### Option 2: Run from Source

```powershell
# 1. Clone the repo
git clone https://github.com/ianasasira/BIRYOMUMEISHO_JOSHUA_AML_ALNet_CAPSTONE_PROJECT.git
cd BIRYOMUMEISHO_JOSHUA_AML_ALNet_CAPSTONE_PROJECT

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 3. Run the desktop app
python src/desktop_app.py

# 4. (Optional) Build the standalone .exe
pip install pyinstaller
python src/build_exe.py
```

---

## Report

The complete 5-chapter capstone report is at:  
`outputs/ALNet_Capstone_Report.docx`

Includes: Introduction, Literature Review, Methodology, Results with real metrics, Conclusion, all figures (architecture diagram, training curves, confusion matrix, ROC curve, threshold analysis, app screenshots).

---

## Datasets

This project uses an extract from the **AML-Cytomorphology_LMU** dataset (Matek et al., 2019, The Cancer Imaging Archive).

**Dataset structure** (not included in this repo — download separately):
```
dataset/
  AML positive/
    MOB/   ← Monoblasts
    MYB/   ← Myeloblasts
  NEGATIVE/
    MON/   ← Monocytes
    MYO/   ← Myelocytes
```

Place the dataset at `dataset/dataset/` before running training.

---

## Results Summary

| Metric | Value |
|--------|-------|
| Architecture | ALNet (depthwise separable convs + sparse attention) |
| Parameters | 27,393 |
| Model Size | 385 KB |
| Input | 224×224×3 |
| AUC-ROC | 0.9675 |
| Sensitivity (Recall) | 0.5000 |
| Specificity | 0.9855 |
| F1-Score | 0.3846 |
| Inference Time (GPU) | < 50 ms |
| Inference Time (CPU) | < 100 ms |

**Limitation:** The model was trained on only 48 AML positive images (extreme 74:1 class imbalance). This limits sensitivity — the architecture works but needs more data for clinical-grade recall.

---

## Repository Structure

```
├── src/
│   ├── desktop_app.py          # CustomTkinter desktop application (★ main deliverable)
│   ├── alnet_model.py          # ALNet architecture + WeightedFocalLoss
│   ├── data_pipeline.py        # Data split, augmentation, dataloaders
│   ├── train.py                # Training script
│   ├── evaluate.py             # Test set evaluation
│   ├── plot_training.py        # Training curve plotter
│   ├── plot_architecture.py    # Architecture diagram generator
│   ├── threshold_analysis.py   # Classification threshold optimizer
│   ├── generate_report.py      # .docx report generator
│   ├── session_logger.py       # SQLite audit logging
│   ├── dataset_inventory.py    # Dataset statistics
│   └── build_exe.py            # PyInstaller build script
├── outputs/
│   ├── alnet_best.pt           # Trained model checkpoint
│   ├── ALNet_Capstone_Report.docx  # Complete 5-chapter report
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   ├── alnet_architecture.png
│   └── evaluation_results.json
├── screenshots/                # App screenshots for report
├── requirements.txt
├── WALKTHROUGH.md
├── PRESENTATION_OUTLINE.md
└── .gitignore
```

---

## Reference

Matek, C., Schwarz, S., Marr, C., & Spiekermann, K. (2019). *A Single-cell Morphological Dataset of Leukocytes from AML Patients and Non-malignant Controls (AML-Cytomorphology_LMU)*. The Cancer Imaging Archive. DOI: 10.7937/tcia.2019.36f5o9ld
