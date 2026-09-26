"""TIFF data in class-named folders, returned as float32 CHW tensors."""

from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .config import DatasetConfig


def read_cube(path, config: DatasetConfig, image_size=None):
    cube = np.asarray(tifffile.imread(path), dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"{path}: expected a 3D TIFF, received {cube.shape}")
    if config.layout == "HWC":
        cube = cube.transpose(2, 0, 1)
    if config.band_indices is not None:
        if max(config.band_indices) >= cube.shape[0]:
            raise ValueError(f"{path}: band_indices exceed the {cube.shape[0]} source bands")
        cube = cube[config.band_indices]
    if cube.shape[0] != config.channels:
        raise ValueError(
            f"{path}: expected {config.channels} bands, received {cube.shape[0]}; "
            "check layout and explicitly configure band_indices"
        )
    if not np.isfinite(cube).all():
        raise ValueError(f"{path}: TIFF contains NaN or infinite values")
    if config.normalization == "scale":
        cube = cube / config.scale
    else:
        low = cube.min(axis=(1, 2), keepdims=True)
        high = cube.max(axis=(1, 2), keepdims=True)
        span = high - low
        normalized = np.divide(cube - low, span, out=np.zeros_like(cube), where=span > 0)
        # Preserve the original constant-positive-band behavior; zero stays zero.
        constant = np.divide(cube, high, out=np.zeros_like(cube), where=high != 0)
        cube = np.where(span > 0, normalized, constant)
    image = torch.from_numpy(np.ascontiguousarray(np.clip(cube, 0, 1)))
    if image_size is not None:
        if image_size < 8:
            raise ValueError("image_size must be >= 8")
        image = F.interpolate(image[None], (image_size, image_size), mode="bilinear", align_corners=False)[0]
    if min(image.shape[-2:]) < 8:
        raise ValueError(f"{path}: spatial dimensions must be >= 8 for three downsampling stages")
    return image


class HSIFolder(Dataset):
    def __init__(self, root, config, classes=None, image_size=None):
        self.root = Path(root)
        self.config = config
        self.image_size = image_size
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset folder does not exist: {self.root}")
        found = sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("."))
        self.classes = list(classes) if classes is not None else found
        if len(self.classes) != config.num_classes or len(set(self.classes)) != len(self.classes):
            raise ValueError(f"Expected {config.num_classes} distinct classes, found {self.classes}")
        unknown = set(found) - set(self.classes)
        if unknown:
            raise ValueError(f"Unknown class folders: {sorted(unknown)}")
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.samples = []
        for name, index in self.class_to_idx.items():
            paths = sorted(p for p in (self.root / name).rglob("*") if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
            if classes is None and not paths:
                raise ValueError(f"No TIFF images in class folder: {name}")
            self.samples.extend((path, index) for path in paths)
        if not self.samples:
            raise ValueError(f"No TIFF images in {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return read_cube(path, self.config, self.image_size), label
