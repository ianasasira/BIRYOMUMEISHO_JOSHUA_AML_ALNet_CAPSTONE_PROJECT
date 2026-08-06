# ALNet AML Detection — Project Walkthrough

## Repository Structure

```
ian/
├── dataset/dataset/         # AML-Cytomorphology_LMU extract
│   ├── AML positive/
│   │   ├── MOB/             # Monoblasts (26 images)
│   │   └── MYB/             # Myeloblasts (42 images)
│   └── NEGATIVE/
│       ├── MON/             # Monocytes (1789 images)
│       └── MYO/             # Myelocytes (3268 images)
├── src/
│   ├── dataset_inventory.py # Count and report dataset statistics
│   ├── data_pipeline.py     # Data split, augmentation, dataloaders
│   ├── alnet_model.py       # ALNet architecture + WeightedFocalLoss
│   ├── train.py             # Training script (v1 — weighted sampler)
│   ├── train_v2.py          # Training script (v2 — balanced batches)
│   ├── evaluate.py          # Test set evaluation + figures
│   ├── threshold_analysis.py# Classification threshold optimization
│   ├── plot_training.py     # Training curve plotter
│   ├── plot_architecture.py # Architecture diagram generator
│   ├── desktop_app.py       # CustomTkinter desktop application
│   ├── session_logger.py    # SQLite logging for audit trail
│   ├── build_exe.py         # PyInstaller build script
│   └── generate_report.py   # 5-chapter .docx report generator
├── outputs/
│   ├── alnet_best.pt        # Best model checkpoint (epoch 34)
│   ├── alnet_final.pt       # Final model weights
│   ├── evaluation_results.json
│   ├── training_history.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_curves.png
│   ├── alnet_architecture.png
│   ├── threshold_analysis.png
│   └── ALNet_Capstone_Report.docx
├── requirements.txt
└── WALKTHROUGH.md           # This file
```

## How to Reproduce Training

### Prerequisites

- Python 3.11+
- NVIDIA GPU with CUDA (tested on RTX 3060 Ti, 8GB)
- Dataset at: `C:\Users\Kelvin\Desktop\ian\dataset\dataset`

### Step 1: Install Dependencies

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 2: Dataset Inventory

```powershell
python src/dataset_inventory.py
```

Output: `outputs/dataset_inventory.json` — class counts and imbalance ratio.

### Step 3: Data Pipeline (Split + Manifest)

```powershell
python src/data_pipeline.py
```

Creates `outputs/split_manifest.json` with 70/15/15 stratified split.

### Step 4: Train ALNet

```powershell
python src/train.py
```

- Best model saved to `outputs/alnet_best.pt`
- Training history saved to `outputs/training_history.json`
- ~25 seconds per epoch on RTX 3060 Ti
- Early stopping at ~49 epochs (patience=15)

### Step 5: Evaluate

```powershell
python src/evaluate.py
```

Generates:
- `outputs/evaluation_results.json` — all metrics
- `outputs/confusion_matrix.png`
- `outputs/roc_curve.png`

### Step 6: Generate Figures

```powershell
python src/plot_training.py
python src/plot_architecture.py
python src/threshold_analysis.py
```

### Step 7: Generate Report

```powershell
python src/generate_report.py
```

Output: `outputs/ALNet_Capstone_Report.docx`

## How to Run the Desktop App

### Direct Run

```powershell
cd C:\Users\Kelvin\Desktop\ian
python src/desktop_app.py
```

### Build Standalone Executable

```powershell
pip install pyinstaller
python src/build_exe.py
```

Output: `dist/ALNet_Screening_Tool.exe`

The executable bundles:
- Trained ALNet model
- All Python dependencies
- No separate Python/TensorFlow installation needed

### Using the App

1. Launch `ALNet_Screening_Tool.exe`
2. Click "Load Image" or drag-and-drop a blood smear image
3. Click "Run Analysis"
4. View prediction (AML/Non-AML) with confidence scores
5. All results logged to `outputs/prediction_log.db`

## How to Rebuild the Executable

```powershell
# 1. Ensure the model exists
ls outputs/alnet_best.pt

# 2. Install PyInstaller
pip install pyinstaller

# 3. Build
python src/build_exe.py

# 4. The .exe will be at:
#    dist/ALNet_Screening_Tool.exe
```

Note: The PyInstaller bundle will be large (~2-3 GB) because it includes PyTorch and all CUDA libraries. This is expected for PyTorch-based applications.

## Key Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Framework | PyTorch | GPU support on Windows; Keras 3 with PyTorch backend for .keras export |
| GUI | CustomTkinter | Lightweight, modern look, easy PyInstaller bundling |
| Loss | Weighted Focal Loss | Handles 74:1 class imbalance per proposal spec |
| Optimizer | AdamW + Cosine Annealing | As specified in proposal Section 3.4.1 |
| Augmentation | Rotation + Flip only | Proposal is explicit: no colour augmentation |
| Batch size | 32 | Fits 8GB VRAM comfortably with mixed precision |

## Model Summary

- Architecture: ALNet (dual-branch: depthwise separable convs + sparse attention)
- Parameters: 27,393
- Input: 224x224x3
- Output: Softmax (2 units): AML / Non-AML
- Key metrics on test set: AUC-ROC 0.9675, Recall 0.50, F1 0.38
