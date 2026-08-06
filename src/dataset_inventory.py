import os
import json
from pathlib import Path

DATASET_DIR = Path(r"C:\Users\Kelvin\Desktop\ian\dataset\dataset")


def inventory():
    results = {}
    grand_total = 0
    pos_total = 0
    neg_total = 0

    for class_name in ["AML positive", "NEGATIVE"]:
        class_path = DATASET_DIR / class_name
        results[class_name] = {}
        class_total = 0

        for sub in sorted(os.listdir(class_path)):
            sub_path = class_path / sub
            if sub_path.is_dir():
                files = [f for f in os.listdir(sub_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
                count = len(files)
                results[class_name][sub] = count
                class_total += count

        results[class_name]["_total"] = class_total
        grand_total += class_total
        if class_name == "AML positive":
            pos_total = class_total
        else:
            neg_total = class_total

    results["_grand_total"] = grand_total
    results["_positive"] = pos_total
    results["_negative"] = neg_total
    results["_pos_pct"] = round(pos_total / grand_total * 100, 2) if grand_total else 0
    results["_neg_pct"] = round(neg_total / grand_total * 100, 2) if grand_total else 0
    results["_imbalance_ratio"] = round(max(pos_total, neg_total) / min(pos_total, neg_total), 2) if min(pos_total, neg_total) > 0 else float("inf")

    return results


def main():
    inv = inventory()

    print("=" * 50)
    print("AML Detection Dataset Inventory")
    print("=" * 50)
    for cls in ["AML positive", "NEGATIVE"]:
        print(f"\n{cls}:")
        for sub, count in inv[cls].items():
            if not sub.startswith("_"):
                print(f"  {sub}: {count} images")
        print(f"  TOTAL: {inv[cls]['_total']} images")

    print(f"\n{'=' * 50}")
    print(f"GRAND TOTAL: {inv['_grand_total']} images")
    print(f"Positive (AML):    {inv['_positive']} ({inv['_pos_pct']}%)")
    print(f"Negative (Non-AML): {inv['_negative']} ({inv['_neg_pct']}%)")
    print(f"Imbalance ratio:   {inv['_imbalance_ratio']}:1 (negative:positive)")

    out_path = Path("outputs/dataset_inventory.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(inv, f, indent=2)
    print(f"\nInventory saved to {out_path}")


if __name__ == "__main__":
    main()
