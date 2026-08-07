import sys
sys.path.insert(0, "src")

import json, torch, numpy as np
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, average_precision_score
from data_pipeline import POSITIVE_TRANSFORM, TRAIN_TRANSFORM, EVAL_TRANSFORM, PerClassTransformDataset
from alnet_model import ALNet_DenseNet121, WeightedCrossEntropy

DEV = torch.device("cuda")
BS = 32
EPOCHS = 15
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

OUTPUT_DIR = Path("outputs")

with open(OUTPUT_DIR / "split_manifest.json") as f:
    m = json.load(f)

train_pos = m["counts"]["train_positive"]
train_neg = m["counts"]["train_negative"]

train_ds = PerClassTransformDataset(
    m["train"], pos_transform=POSITIVE_TRANSFORM, neg_transform=TRAIN_TRANSFORM,
    eval_transform=EVAL_TRANSFORM, is_train=True,
)
labels = [e["label"] for e in m["train"]]
w = [1.0 / train_pos if l == 1 else 1.0 / train_neg for l in labels]
sampler = WeightedRandomSampler(w, num_samples=train_pos * 50, replacement=True)
train_loader = DataLoader(train_ds, batch_size=BS, sampler=sampler)

val_ds = PerClassTransformDataset(
    m["val"], pos_transform=EVAL_TRANSFORM, neg_transform=EVAL_TRANSFORM,
    eval_transform=EVAL_TRANSFORM, is_train=False,
)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

test_ds = PerClassTransformDataset(
    m["test"], pos_transform=EVAL_TRANSFORM, neg_transform=EVAL_TRANSFORM,
    eval_transform=EVAL_TRANSFORM, is_train=False,
)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)

print(f"Train: {train_pos} pos + {train_neg} neg | samples/epoch: {train_pos * 50}")

model = ALNet_DenseNet121(num_classes=2).to(DEV)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Params: {total:,} total, {trainable:,} trainable")

crit = WeightedCrossEntropy(num_pos=train_pos, num_neg=train_neg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scaler = GradScaler("cuda")

best_recall = 0
best_val_loss = float("inf")
best_state = None

for epoch in range(1, EPOCHS + 1):
    model.train(); tl = 0; tc = 0; tt = 0
    for img, lab in train_loader:
        img, lab = img.to(DEV), lab.to(DEV); opt.zero_grad()
        with autocast("cuda"):
            loss = crit(model(img), lab)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tl += loss.item(); _, p = model(img).max(1)
        tt += lab.size(0); tc += p.eq(lab).sum().item()

    model.eval(); vl = 0; vpreds = []; vlabels = []
    with torch.no_grad():
        for img, lab in val_loader:
            img, lab = img.to(DEV), lab.to(DEV)
            with autocast("cuda"):
                out = model(img)
                vl += crit(out, lab).item()
            vpreds.append(out.argmax(1).cpu()); vlabels.append(lab.cpu())
    vpreds = torch.cat(vpreds).numpy(); vlabels = torch.cat(vlabels).numpy()
    vrec = recall_score(vlabels, vpreds, zero_division=0)
    vf1 = f1_score(vlabels, vpreds, zero_division=0)
    vacc = (vlabels == vpreds).mean()

    if vrec > best_recall or (vrec == best_recall and vl < best_val_loss):
        best_recall = vrec
        best_val_loss = vl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:2d} | Train Loss: {tl/len(train_loader):.4f} Acc: {100*tc/tt:.1f}% | "
          f"Val Loss: {vl/len(val_loader):.4f} Acc: {100*vacc:.1f}% | Rec: {vrec:.4f} F1: {vf1:.4f}")

# Save and evaluate
torch.save(best_state, OUTPUT_DIR / "alnet_densenet121_best.pt")
print(f"\nSaved best recall={best_recall:.4f}")

model.load_state_dict(best_state)
model.eval()
all_probs = []; all_labels = []
with torch.no_grad():
    for img, lab in test_loader:
        img = img.to(DEV)
        with autocast("cuda"):
            probs = torch.softmax(model(img), dim=1)
        all_probs.append(probs.cpu().numpy()); all_labels.append(lab.numpy())
all_probs = np.concatenate(all_probs); all_labels = np.concatenate(all_labels)
pos_probs = all_probs[:, 1]

print("_" * 50)
print("DENSENET121 TEST RESULTS")
print("_" * 50)
for t in [0.05, 0.10, 0.15, 0.20, 0.50]:
    p = (pos_probs >= t).astype(int)
    r = recall_score(all_labels, p, zero_division=0)
    pr = precision_score(all_labels, p, zero_division=0)
    fp = int(((all_labels == 0) & (p == 1)).sum())
    fn = int(((all_labels == 1) & (p == 0)).sum())
    print(f"  thr={t:.2f}: R={r:.0%} P={pr:.3f} FP={fp} FN={fn}")

auc_roc = roc_auc_score(all_labels, pos_probs)
auc_pr = average_precision_score(all_labels, pos_probs)
print(f"  AUC-ROC: {auc_roc:.4f}  AUC-PR: {auc_pr:.4f}")

aml_probs = sorted(pos_probs[all_labels == 1])
print(f"  AML probs: {[f'{p:.3f}' for p in aml_probs]}")

# Comparison table
print(f"\n{'='*70}")
print(f"COMPARISON: Original ALNet vs EfficientNet-B0 vs DenseNet121")
print(f"{'='*70}")
print(f"{'Model':<25} {'thr=0.10':>18} {'thr=0.15':>18} {'thr=0.50':>18}")
print(f"{'':25} {'Recall FP':>18} {'Recall FP':>18} {'Recall FP':>18}")
print(f"{'-'*70}")
print(f"{'Original ALNet':<25} {'80%  46':>18} {'70%  30':>18} {'50%  11':>18}")
print(f"{'EfficientNet-B0':<25} {'90% 217':>18} {'80% 146':>18} {'40%  38':>18}")

for t in [0.10, 0.15, 0.50]:
    p = (pos_probs >= t).astype(int)
    r = recall_score(all_labels, p, zero_division=0)
    fp = int(((all_labels == 0) & (p == 1)).sum())
    if t == 0.10:
        parts_10 = f"{r:.0%}  {fp}"
    elif t == 0.15:
        parts_15 = f"{r:.0%}  {fp}"
    else:
        parts_50 = f"{r:.0%}  {fp}"

print(f"{'DenseNet121':<25} {parts_10:>18} {parts_15:>18} {parts_50:>18}")
