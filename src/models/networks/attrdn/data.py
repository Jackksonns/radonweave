# Radonweave (https://github.com/Jackksonns/radonweave): low-dose CT reconstruction/denoising toolkit.
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def default_theta(num_angles: int = 180) -> np.ndarray:
    return np.linspace(0.0, 180.0, num_angles, endpoint=False)


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    raise ValueError(f"Unsupported file type: {path}")


def radon_transform(image: np.ndarray, theta: np.ndarray) -> np.ndarray:
    try:
        from skimage.transform import radon
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for radon/iradon. Install scikit-image."
        ) from exc
    return radon(image, theta=theta, circle=False).astype(np.float32)


def simulate_low_dose(
    sinogram: np.ndarray, z0: float = 1e5, readout_noise: float = 0.0
) -> np.ndarray:
    intensity = z0 * np.exp(-sinogram) + readout_noise
    noisy = np.random.poisson(intensity).astype(np.float32)
    noisy = np.maximum(noisy, 1e-6)
    return -np.log(noisy / z0).astype(np.float32)


def create_sinogram_pair(
    ndct_image: np.ndarray,
    theta: np.ndarray,
    z0: float = 1e5,
    readout_noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    nd_sino = radon_transform(ndct_image, theta)
    ld_sino = simulate_low_dose(nd_sino, z0=z0, readout_noise=readout_noise)
    return ld_sino, nd_sino


def _pad_to_min_size(array: np.ndarray, size: int) -> np.ndarray:
    h, w = array.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h == 0 and pad_w == 0:
        return array
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    return np.pad(
        array,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode="constant",
        constant_values=0.0,
    )


def random_crop_pair(
    ld_sino: np.ndarray, nd_sino: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    ld_sino = _pad_to_min_size(ld_sino, patch_size)
    nd_sino = _pad_to_min_size(nd_sino, patch_size)
    h, w = ld_sino.shape
    top = np.random.randint(0, h - patch_size + 1)
    left = np.random.randint(0, w - patch_size + 1)
    ld_patch = ld_sino[top : top + patch_size, left : left + patch_size]
    nd_patch = nd_sino[top : top + patch_size, left : left + patch_size]
    return ld_patch, nd_patch


def normalize_pair(
    ld_sino: np.ndarray, nd_sino: np.ndarray, mode: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    if mode is None:
        return ld_sino, nd_sino
    if mode == "minmax":
        min_val = nd_sino.min()
        max_val = nd_sino.max()
        scale = max(max_val - min_val, 1e-6)
        ld_sino = (ld_sino - min_val) / scale
        nd_sino = (nd_sino - min_val) / scale
        return ld_sino, nd_sino
    raise ValueError(f"Unsupported normalization mode: {mode}")


class PrecomputedSinogramDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        split: str,
        patch_size: int = 55,
        training: bool = True,
        normalize: Optional[str] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.patch_size = patch_size
        self.training = training
        self.normalize = normalize

        nd_dir = self.root_dir / split / "ndct"
        ld_dir = self.root_dir / split / "ldct"
        if not nd_dir.exists() or not ld_dir.exists():
            raise FileNotFoundError(
                f"Expected ndct/ldct dirs under {self.root_dir}/{split}"
            )

        self.nd_paths = sorted(nd_dir.glob("*.npy"))
        if not self.nd_paths:
            raise FileNotFoundError(f"No .npy files found under {nd_dir}")
        self.ld_dir = ld_dir

    def __len__(self):
        return len(self.nd_paths)

    def __getitem__(self, idx):
        nd_path = self.nd_paths[idx]
        ld_path = self.ld_dir / nd_path.name
        if not ld_path.exists():
            raise FileNotFoundError(f"Missing LDCT sinogram for {nd_path.name}")

        nd_sino = _load_array(nd_path)
        ld_sino = _load_array(ld_path)
        ld_sino, nd_sino = normalize_pair(ld_sino, nd_sino, self.normalize)

        if self.training and self.patch_size:
            ld_sino, nd_sino = random_crop_pair(ld_sino, nd_sino, self.patch_size)

        ld_tensor = torch.from_numpy(ld_sino).unsqueeze(0)
        nd_tensor = torch.from_numpy(nd_sino).unsqueeze(0)
        return ld_tensor, nd_tensor


class NdctToSinogramDataset(Dataset):
    def __init__(
        self,
        ndct_dir: Path,
        patch_size: int = 55,
        training: bool = True,
        normalize: Optional[str] = None,
        theta: Optional[np.ndarray] = None,
        z0: float = 1e5,
        readout_noise: float = 0.0,
        cache: bool = False,
    ):
        self.ndct_dir = Path(ndct_dir)
        self.patch_size = patch_size
        self.training = training
        self.normalize = normalize
        self.theta = theta if theta is not None else default_theta()
        self.z0 = z0
        self.readout_noise = readout_noise
        self.cache = {} if cache else None

        self.nd_paths = sorted(self.ndct_dir.glob("*.npy"))
        if not self.nd_paths:
            raise FileNotFoundError(f"No .npy files found under {ndct_dir}")

    def __len__(self):
        return len(self.nd_paths)

    def __getitem__(self, idx):
        nd_path = self.nd_paths[idx]
        if self.cache is not None and nd_path in self.cache:
            ld_sino, nd_sino = self.cache[nd_path]
        else:
            ndct = _load_array(nd_path)
            ld_sino, nd_sino = create_sinogram_pair(
                ndct, self.theta, z0=self.z0, readout_noise=self.readout_noise
            )
            if self.cache is not None:
                self.cache[nd_path] = (ld_sino, nd_sino)

        ld_sino, nd_sino = normalize_pair(ld_sino, nd_sino, self.normalize)
        if self.training and self.patch_size:
            ld_sino, nd_sino = random_crop_pair(ld_sino, nd_sino, self.patch_size)

        ld_tensor = torch.from_numpy(ld_sino).unsqueeze(0)
        nd_tensor = torch.from_numpy(nd_sino).unsqueeze(0)
        return ld_tensor, nd_tensor
