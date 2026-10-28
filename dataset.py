"""PyTorch Dataset/DataLoader: loads ONE image at a time from disk (no big RAM use).
Optional extra TRAIN-only samples: GAN images (synthetic CSV) or oversampled real images."""
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from preprocessing import get_transforms
from utils import PROJECT_ROOT

DEFAULT_ROOT = PROJECT_ROOT / "data" / "reduced_dataset"
DEFAULT_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_class_names(processed_dir=DEFAULT_PROCESSED):
    return json.load(open(Path(processed_dir) / "class_names.json"))


def build_extra_train(split_csv, data_root, synthetic_csv=None, synthetic_classes=None,
                      oversample_to=0, oversample_classes=None, seed=42):
    """Returns a list of (image_path, label) that is appended to the TRAIN split only.
    - synthetic_csv: index written by generate_images.py (GAN images, always split='train')
    - oversample_to: control experiment - repeat REAL train images of the chosen classes
      (with the usual random augmentation) until each class has this many images."""
    extra = []
    if synthetic_csv:
        syn = pd.read_csv(synthetic_csv)
        assert (syn["split"] == "train").all() and (syn["source"] == "gan").all(), "synthetic must be train-only"
        if synthetic_classes:
            syn = syn[syn["class"].isin(synthetic_classes)]
        root = Path(synthetic_csv).parent / Path(synthetic_csv).stem.replace("synthetic_", "")
        extra += [(root / p.replace("\\", "/"), int(l)) for p, l in zip(syn["path"], syn["label"])]
    if oversample_to and oversample_classes:
        df = pd.read_csv(split_csv)
        df = df[df["split"] == "train"]
        rng = random.Random(seed)
        for cname in oversample_classes:
            d = df[df["class"] == cname]
            paths = [Path(data_root) / p.replace("\\", "/") for p in d["path"]]
            k = max(0, oversample_to - len(paths))
            extra += [(p, int(d["label"].iloc[0])) for p in rng.choices(paths, k=k)]
    return extra


class TextileDataset(Dataset):
    def __init__(self, split_csv, split, data_root=DEFAULT_ROOT, transform=None, extra=None):
        df = pd.read_csv(split_csv)
        df = df[df["split"] == split].reset_index(drop=True)
        # split CSVs made on Windows contain backslashes -> make them work everywhere
        self.paths = [Path(data_root) / p.replace("\\", "/") for p in df["path"]]
        self.labels = df["label"].astype(int).tolist()
        if extra:
            assert split == "train", "extra samples may only be added to the TRAIN split"
            self.paths += [p for p, _ in extra]
            self.labels += [l for _, l in extra]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)
        return im, self.labels[i]


def _worker_init(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)


def get_dataloaders(split_csv, data_root=DEFAULT_ROOT, batch_size=32, num_workers=4, seed=42, extra_train=None):
    g = torch.Generator(); g.manual_seed(seed)
    out = {}
    for split in ("train", "val", "test"):
        ds = TextileDataset(split_csv, split, data_root, get_transforms(train=(split == "train")),
                            extra=extra_train if split == "train" else None)
        out[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), generator=g if split == "train" else None,
            worker_init_fn=_worker_init, persistent_workers=num_workers > 0,
        )
    return out
