import json
import os
from pathlib import Path

AUG_DIR = Path(r"E:\Mr Biryo\ian\dataset\dataset\AML positive_augmented")
MANIFEST_PATH = Path("outputs/split_manifest.json")

with open(MANIFEST_PATH) as f:
    m = json.load(f)

added = 0
for sub in ["MOB", "MYB"]:
    sub_path = AUG_DIR / sub
    for fname in os.listdir(str(sub_path)):
        if fname.lower().endswith(".tiff") and "_aug" in fname:
            m["train"].append({
                "path": str(sub_path / fname),
                "label": 1,
                "class_name": "AML positive",
                "subtype": sub,
                "filename": fname,
                "split": "train",
            })
            added += 1

m["counts"]["train"] = len(m["train"])
m["counts"]["train_positive"] = sum(1 for x in m["train"] if x["label"] == 1)
m["counts"]["train_negative"] = sum(1 for x in m["train"] if x["label"] == 0)

with open(MANIFEST_PATH, "w") as f:
    json.dump(m, f, indent=2)

print(f"Added {added} augmented positives to training set")
print(f"Train total: {m['counts']['train']} (pos={m['counts']['train_positive']}, neg={m['counts']['train_negative']})")
print(f"New pos/neg ratio in train: {m['counts']['train_positive']}/{m['counts']['train_negative']} = {m['counts']['train_positive']/m['counts']['train_negative']:.1%}")
print(f"New focal alpha: {round(m['counts']['train_negative']/(m['counts']['train_positive']+m['counts']['train_negative']), 4)}")
