# ALNet AML Detection — Presentation Slides Outline (~25 slides)

## SLIDE 1: Title Slide
- Title: Automatic Detection of Acute Myeloblastic Leukemia Using ALNet Deep Learning Model
- Student: Biryomumeisho Joshua (JAN26/MAIDS/0819U)
- Supervisor: Dr. Kassim Kalinaki (PhD)
- Institution: UTAMU — MSc AI & Data Science

## SLIDE 2: Agenda
- Problem Context
- Research Objectives
- Related Work
- ALNet Architecture
- Methodology
- Results & Evaluation
- Desktop Application Demo
- Conclusion & Recommendations

## SLIDE 3: Background — AML
- AML: aggressive blood cancer, rapid proliferation of myeloblasts
- 20-25% of all leukemia cases globally
- In Uganda: 58.4% of adult acute leukemia admissions (UCI data)
- Early detection critical for survival

## SLIDE 4: Problem Statement
- LMICs lack advanced diagnostic tools
- Manual microscopy: time-consuming, subjective, specialist-dependent
- Symptom overlap with malaria complicates diagnosis
- No AI-based AML screening tool exists in Uganda

## SLIDE 5: Research Objectives
1. Develop & train ALNet detection model
2. Evaluate model performance
3. Build desktop UI for laboratory use

## SLIDE 6: Research Questions
- RQ1: Can ALNet be trained on blood smear images?
- RQ2: Can ALNet detect myeloblasts from unseen images?
- RQ3: Can a UI be integrated for direct utilisability?

## SLIDE 7: Scope
- Binary AML/Non-AML classification only (no subtypes)
- Publicly available dataset (AML-Cytomorphology_LMU)
- Decision-support tool — NOT a diagnostic system
- Flag for human review

## SLIDE 8: Literature Review — Current Methods
- Non-AI: microscopy, CBC, flow cytometry, PCR, FISH
- AI-based: CNNs, Vision Transformers (ReLViT), SVMs
- Key limitations: compute-heavy, poor stain generalization, high false negatives

## SLIDE 9: Literature Review — Key Gaps
- Most models require expensive compute clusters
- Stain/time variability causes performance drops
- Class imbalance not adequately addressed
- No LMIC-deployable solutions

## SLIDE 10: ALNet — Key Innovations
- Depthwise separable convolutions (lightweight)
- Localized sparse multi-head self-attention
- Weighted Focal Loss for extreme class imbalance
- 27,393 parameters — runs on standard hardware

## SLIDE 11: ALNet Architecture Diagram
- [Insert alnet_architecture.png]
- Conv Block 1 → Attention → Conv Block 2 → Attention → MaxPool → Dense → Output

## SLIDE 12: ALNet Components Explained
- Conv Blocks: depthwise separable, 32/64 filters
- Attention: channel + spatial attention (localized, sparse)
- Transition: max pooling (2x2, stride 2)
- Dense: 128→64 units with dropout (0.5, 0.3)
- Output: softmax, 2 units: AML / Non-AML

## SLIDE 13: Dataset
- Source: AML-Cytomorphology_LMU (extract)
- Total: 5,125 images
- AML Positive: 68 (1.3%) — MOB (26) + MYB (42)
- Non-AML: 5,057 (98.7%) — MON (1,789) + MYO (3,268)
- Imbalance ratio: 74:1

## SLIDE 14: Data Pipeline
- Split: 70% train / 15% val / 15% test (stratified)
- Resize: 224×224×3
- Augmentation (train only): rotation ±15°, horizontal flip
- No colour augmentation (preserves staining cues)
- Weighted random sampler for class balance

## SLIDE 15: Training Configuration
- GPU: NVIDIA RTX 3060 Ti (8 GB)
- Mixed precision (float16)
- Loss: Weighted Focal Loss (α=0.75, γ=2.0)
- Optimizer: AdamW (lr=0.001, wd=1e-4)
- Scheduler: Cosine Annealing (T_max=100)
- Early stopping (patience=15)
- Batch size: 32

## SLIDE 16: Training Results
- [Insert training_curves.png]
- Best epoch: 34 (val_loss=0.0076)
- Final train acc: 98.8%, val acc: 97.1%
- No overfitting — good generalization

## SLIDE 17: Evaluation Metrics
- [Insert confusion_matrix.png]
- Test set: 769 images (10 AML, 759 Non-AML)
- TN=748, FP=11, FN=5, TP=5

## SLIDE 18: Key Metrics Table
| Metric | Value |
|--------|-------|
| Accuracy | 97.9% |
| AUC-ROC | 0.9675 |
| F1-Score | 0.3846 |
| Sensitivity/Recall | 0.5000 |
| Specificity | 0.9855 |

## SLIDE 19: ROC Curve
- [Insert roc_curve.png]
- AUC = 0.9675 — strong discriminative ability despite limited training data

## SLIDE 20: Interpretation of Results
- AUC 0.97: model learns meaningful morphological features
- Recall 50%: 5/10 AML cases missed — clinically significant
- Root cause: only 48 positive training images
- Model does NOT default to majority class (would be 98.7% acc)
- Threshold analysis: cannot fix recall — need more data

## SLIDE 21: Desktop Application
- [Screenshots of the app]
- Built with CustomTkinter (modern, lightweight)
- Features: image load, auto-preprocess, inference, confidence display
- Explicit screening-flag labelling
- SQLite audit logging
- One-click .exe deployment (PyInstaller)

## SLIDE 22: Objective Achievement
| Objective | Status |
|-----------|--------|
| 1. Develop & train ALNet | ACHIEVED — 27K params, trained on GPU |
| 2. Evaluate performance | ACHIEVED — AUC 0.97, all metrics reported |
| 3. Build desktop UI | ACHIEVED — functional app, .exe ready |

All three research questions answered positively.

## SLIDE 23: Limitations
- 48 positive training images (extreme data scarcity)
- Single dataset source — limited generalizability
- Image-level split (not patient-level)
- Binary only (no subtype classification)
- Recall too low for clinical deployment

## SLIDE 24: Recommendations
1. Full AML-Cytomorphology_LMU dataset (18K images)
2. Multi-institutional data for robustness
3. Domain adversarial training for stain/microscope invariance
4. Multi-class subtype classification
5. Prospective clinical validation at UCI
6. Model quantization for edge deployment

## SLIDE 25: Thank You / Q&A
- Thank you for your attention
- Questions?
- Contact: Biryomumeisho Joshua
- Supervisor: Dr. Kassim Kalinaki
