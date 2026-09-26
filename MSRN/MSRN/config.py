"""Explicit dataset settings; all band indices are zero-based."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "datasets.json"


@dataclass(frozen=True)
class DatasetConfig:
    channels: int
    num_classes: int
    rgb_bands: list[int]
    layout: str
    normalization: str
    scale: float = 1.0
    band_indices: list[int] | None = None

    def __post_init__(self):
        if self.channels < 3 or self.num_classes < 2:
            raise ValueError("channels must be >= 3 and num_classes must be >= 2")
        if len(self.rgb_bands) != 3 or any(b < 0 or b >= self.channels for b in self.rgb_bands):
            raise ValueError("rgb_bands must contain three valid zero-based channel indices")
        if self.layout not in {"CHW", "HWC"}:
            raise ValueError("layout must be CHW or HWC")
        if self.normalization not in {"scale", "band_minmax"} or self.scale <= 0:
            raise ValueError("normalization must be scale or band_minmax; scale must be positive")
        if self.band_indices is not None:
            if len(self.band_indices) != self.channels or any(b < 0 for b in self.band_indices):
                raise ValueError("band_indices must contain exactly channels non-negative indices")
            if len(set(self.band_indices)) != len(self.band_indices):
                raise ValueError("band_indices must not contain duplicates")

    def to_dict(self):
        return asdict(self)


def load_config(name, path=DEFAULT_CONFIG):
    presets = json.loads(Path(path).read_text(encoding="utf-8"))
    if name not in presets:
        raise ValueError(f"Unknown dataset {name!r}; choose from {', '.join(presets)}")
    return DatasetConfig(**presets[name])
