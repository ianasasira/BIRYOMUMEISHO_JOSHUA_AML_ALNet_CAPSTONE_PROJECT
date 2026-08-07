"""
augment_aml_positives.py

Purpose
-------
Your dataset is severely imbalanced:
    AML Positive (MOB + MYB) : 68    (1.3%)
    Negative     (MON + MYO) : 5,057 (98.7%)

This script augments ONLY the positive (minority) class images so that,
after augmentation, positives = TARGET_RATIO * negatives (default 1/2,
i.e. ~2,529 positive images to sit alongside your 5,057 negatives).

Important design choices (read before running):
1. Runs AFTER you have already split into train/val/test. Point SRC_DIR
   at the *training* positive folder only. Never augment val/test data -
   doing so lets near-duplicate cells leak across splits and inflates
   your metrics artificially.
2. Uses geometric + mild spatial transforms only (rotation, flips, crop/
   zoom, small translation, small elastic-like jitter). It deliberately
   avoids heavy colour/stain jitter, matching what your proposal already
   specifies ("avoiding color alterations to ensure the model preserves
   natural morphological variations").
3. Balances MOB and MYB proportionally so you don't accidentally end up
   with a positive class that's 95% one subtype.
4. Saves as .tiff, uncompressed, to match your source format.

Install deps first (if missing):
    pip install pillow numpy --break-system-packages
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# CONFIG - edit these paths/numbers for your machine
# ---------------------------------------------------------------------------
SRC_DIRS = {
    "MOB": r"E:\Mr Biryo\ian\dataset\dataset\AML positive\MOB",
    "MYB": r"E:\Mr Biryo\ian\dataset\dataset\AML positive\MYB",
}
OUT_DIR = r"E:\Mr Biryo\ian\dataset\dataset\AML positive_augmented"

NEGATIVE_COUNT_IN_TRAIN = 3540   # 70% of 5057 negatives (train split from data_pipeline.py).
                                  # After running data_pipeline.py, verify with the actual
                                  # 'train_negative' count from outputs/split_manifest.json
                                  # and update this value if it differs.
TARGET_RATIO = 1 / 2             # positives should reach >= 1:2 ratio (50%) of negatives
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def load_tiff(path: str) -> Image.Image:
    img = Image.open(path)
    return img.convert("RGB")


def random_rotate(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
    else:
        angle = random.uniform(-25, 25)
    return img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(255, 255, 255))


def random_flip(img: Image.Image) -> Image.Image:
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img


def random_crop_zoom(img: Image.Image, scale_range=(0.85, 1.0)) -> Image.Image:
    w, h = img.size
    scale = random.uniform(*scale_range)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BICUBIC)


def small_translate(img: Image.Image, max_shift=0.06) -> Image.Image:
    w, h = img.size
    dx = int(w * random.uniform(-max_shift, max_shift))
    dy = int(h * random.uniform(-max_shift, max_shift))
    return img.transform((w, h), Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=(255, 255, 255))


def mild_gaussian_noise(img: Image.Image, sigma=4.0) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


AUG_FUNCS = [random_rotate, random_flip, random_crop_zoom, small_translate, mild_gaussian_noise]


def augment_image(img: Image.Image) -> Image.Image:
    n_ops = random.randint(2, 3)          # stack 2-3 transforms per augmented copy
    for op in random.sample(AUG_FUNCS, n_ops):
        img = op(img)
    return img


def augment_class(src_dir: str, out_dir: str, n_needed: int, class_tag: str):
    src_files = sorted(glob.glob(os.path.join(src_dir, "*.tif*")))
    if not src_files:
        raise FileNotFoundError(f"No .tiff files found in {src_dir}")

    os.makedirs(out_dir, exist_ok=True)

    # keep the originals
    for f in src_files:
        img = load_tiff(f)
        img.save(os.path.join(out_dir, os.path.basename(f)), format="TIFF")

    generated = 0
    idx = 0
    while generated < n_needed:
        src_path = src_files[idx % len(src_files)]
        img = load_tiff(src_path)
        aug_img = augment_image(img)
        out_name = f"{Path(src_path).stem}_{class_tag}_aug{idx}.tiff"
        aug_img.save(os.path.join(out_dir, out_name), format="TIFF")
        generated += 1
        idx += 1

    print(f"[{class_tag}] originals={len(src_files)}  generated={generated}  total={len(src_files)+generated}")
    return len(src_files) + generated


def main():
    counts = {c: len(glob.glob(os.path.join(d, "*.tif*"))) for c, d in SRC_DIRS.items()}
    current_total = sum(counts.values())
    print(f"Current positive counts: {counts}  (total={current_total})")

    target_total_pos = int(np.ceil(NEGATIVE_COUNT_IN_TRAIN * TARGET_RATIO))
    print(f"Target positive total (>= {TARGET_RATIO*100:.0f}% of {NEGATIVE_COUNT_IN_TRAIN} negatives): {target_total_pos}")

    if target_total_pos <= current_total:
        print("Already at or above target ratio - nothing to do.")
        return

    # distribute the needed extra images proportionally across MOB/MYB
    # so the augmented set keeps roughly the same subtype balance
    extra_needed = target_total_pos - current_total
    result_totals = {}
    for i, (class_tag, src_dir) in enumerate(SRC_DIRS.items()):
        share = counts[class_tag] / current_total
        n_needed = int(round(extra_needed * share))
        out_dir = os.path.join(OUT_DIR, class_tag)
        result_totals[class_tag] = augment_class(src_dir, out_dir, n_needed, class_tag)

    grand_total = sum(result_totals.values())
    print(f"\nDone. New positive total: {grand_total} vs negatives: {NEGATIVE_COUNT_IN_TRAIN} "
          f"-> ratio = {grand_total/NEGATIVE_COUNT_IN_TRAIN:.2%}")
    print(f"Augmented images written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
