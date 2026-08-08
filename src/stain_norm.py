import numpy as np
from PIL import Image


def rgb_to_lab(rgb):
    """Convert RGB (0-255) to LAB color space."""
    rgb = np.array(rgb, dtype=np.float32) / 255.0
    mask = rgb > 0.04045
    rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb[~mask] = rgb[~mask] / 12.92

    xyz_matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb @ xyz_matrix.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])

    epsilon = 0.008856
    kappa = 903.3
    mask = xyz > epsilon
    f = np.zeros_like(xyz)
    f[mask] = xyz[mask] ** (1.0 / 3.0)
    f[~mask] = (kappa * xyz[~mask] + 16.0) / 116.0

    lab = np.zeros_like(xyz)
    lab[:, :, 0] = 116.0 * f[:, :, 1] - 16.0
    lab[:, :, 1] = 500.0 * (f[:, :, 0] - f[:, :, 1])
    lab[:, :, 2] = 200.0 * (f[:, :, 1] - f[:, :, 2])
    return lab


def lab_to_rgb(lab):
    """Convert LAB back to RGB (0-255)."""
    fy = (lab[:, :, 0] + 16.0) / 116.0
    fx = lab[:, :, 1] / 500.0 + fy
    fz = fy - lab[:, :, 2] / 200.0

    f = np.stack([fx, fy, fz], axis=2)
    epsilon = 0.008856
    mask = f ** 3 > epsilon
    xyz = np.zeros_like(f)
    xyz[mask] = f[mask] ** 3
    xyz[~mask] = (116.0 * f[~mask] - 16.0) / 903.3

    xyz = xyz * np.array([0.95047, 1.0, 1.08883])

    rgb_matrix = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252],
    ])
    rgb_lin = xyz @ rgb_matrix.T

    mask = rgb_lin > 0.0031308
    rgb = np.zeros_like(rgb_lin)
    rgb[mask] = 1.055 * (rgb_lin[mask] ** (1.0 / 2.4)) - 0.055
    rgb[~mask] = 12.92 * rgb_lin[~mask]

    rgb = np.clip(rgb * 255.0, 0, 255)
    return rgb.astype(np.uint8)


class ReinhardNormalizer:
    """Reinhard stain normalizer using LAB color space.

    Computes per-channel mean and std from reference images in LAB space,
    then maps each target image to those statistics. This works for
    Wright-Giemsa stained blood smears where Macenko (H&E-specific)
    fails.
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, img_arrays):
        """Fit from list of reference images (uint8 RGB arrays)."""
        all_means = []
        all_stds = []
        for img in img_arrays:
            if img is None or img.size == 0:
                continue
            try:
                lab = rgb_to_lab(img)
                all_means.append(np.array([lab[:, :, i].mean() for i in range(3)]))
                all_stds.append(np.array([lab[:, :, i].std() for i in range(3)]))
            except Exception:
                continue
        if not all_means:
            raise RuntimeError("No valid images to fit normalizer")
        self.target_means = np.mean(all_means, axis=0)
        self.target_stds = np.mean(all_stds, axis=0)

    def transform(self, img_array):
        """Apply Reinhard normalization to an RGB uint8 image."""
        if self.target_means is None:
            raise RuntimeError("Normalizer not fitted.")
        lab = rgb_to_lab(img_array)
        for i in range(3):
            mu = lab[:, :, i].mean()
            sigma = lab[:, :, i].std() + 1e-6
            lab[:, :, i] = (lab[:, :, i] - mu) * (self.target_stds[i] / sigma) + self.target_means[i]
        return lab_to_rgb(lab)


def load_and_pad_array(path, target_size=400):
    """Load image, pad to square, resize to target_size, return uint8 RGB array."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid dimensions: {w}x{h}")
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    new_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    if max_dim != target_size:
        new_img = new_img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(new_img, dtype=np.uint8)


def load_image_safe(path, target_size=400):
    """Load and pad with fallback for corrupt images."""
    try:
        return load_and_pad_array(path, target_size)
    except Exception:
        return None
