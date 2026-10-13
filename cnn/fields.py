"""Read prepared RIS samples and construct total-field CNN inputs and targets."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import torch


def discover_groups(data_dir):
    groups = []
    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue
        match = re.fullmatch(r"(\d+)by(\d+)_(-?\d+)deg(?:_.*)?", folder.name)
        if match is None:
            continue
        size, other_size, angle = map(int, match.groups())
        if size != other_size:
            raise ValueError(f"Expected a square RIS array: {folder.name}")
        groups.append((size, angle, folder))
    if not groups:
        raise FileNotFoundError(f"No size/angle data folders found in {data_dir}")
    return sorted(groups, key=lambda group: (group[0], group[1], group[2].name))


def sample_files(folder):
    files = sorted(folder.glob("sample_*.fld"))
    if not files:
        raise FileNotFoundError(f"No prepared samples found in {folder}")
    return files


def build_sample_tensors(path, size, magnetic_scale):
    data = np.atleast_2d(np.loadtxt(path, skiprows=1, dtype=np.float64))
    if data.shape != (size * size, 17):
        raise ValueError(f"{path}: expected {(size * size, 17)}, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: non-finite field values")
    if not np.isin(data[:, 0], (0.0, 1.0)).all():
        raise ValueError(f"{path}: configuration states must be binary")
    indices = data[:, 1:3]
    if not np.equal(indices, np.rint(indices)).all() or np.any(indices < 1) or np.any(indices > size):
        raise ValueError(f"{path}: grid indices must be integers between 1 and {size}")
    if np.unique(indices, axis=0).shape[0] != size * size:
        raise ValueError(f"{path}: duplicate or missing grid locations")

    inputs = data[:, [0, 3, 4, 5, 6]].T.astype(np.float32).reshape(5, size, size)
    targets = data[:, 9:17].T.astype(np.float32)
    targets[4:] *= magnetic_scale
    return torch.from_numpy(inputs), torch.from_numpy(targets.reshape(8, size, size))
