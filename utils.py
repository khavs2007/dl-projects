"""Shared helpers. Run all scripts from the project root (textile_defect_project/)."""
import platform
import random
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_NAMES = {"train", "training", "val", "valid", "validation", "test", "testing"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def infer_class_and_split(path: Path, root: Path):
    """Class = nearest parent folder that is not a train/val/test folder."""
    parts = path.relative_to(root).parts[:-1]
    split = None
    for p in parts:
        if p.lower() in SPLIT_NAMES:
            split = p.lower()
    non_split = [p for p in parts if p.lower() not in SPLIT_NAMES]
    return (non_split[-1] if non_split else None), split


def env_info() -> dict:
    info = {"python": platform.python_version()}
    try:
        import torch, torchvision
        info.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else "none")
    except ImportError:
        info["torch"] = "not installed"
    return info
