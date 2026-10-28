"""GAN quality check #2 ('GAN-test'): does a classifier trained on REAL training images only
recognise the GAN images as the class they were made for?  Uses synthetic images + the
VALIDATION split (never the test set).
Usage:  python eval_synthetic.py --seed 42 --data_root "<path to reduced_dataset>"
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from dataset import DEFAULT_PROCESSED, DEFAULT_ROOT, TextileDataset, load_class_names
from evaluate import predict
from preprocessing import get_transforms
from train_resnet import build_model
from utils import PROJECT_ROOT


class ImgList(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, l = self.items[i]
        with Image.open(p) as im:
            return self.transform(im.convert("RGB")), l


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--synthetic_csv", default=None)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = load_class_names()
    csv = Path(a.synthetic_csv) if a.synthetic_csv else PROJECT_ROOT / "data" / "augmented" / f"synthetic_seed{a.seed}.csv"
    syn = pd.read_csv(csv)
    root = csv.parent / csv.stem.replace("synthetic_", "")

    model = build_model(len(names), 0.5).to(dev)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "resnet50" / f"resnet50_finetune_seed{a.seed}_best.pt",
                                     map_location=dev))
    tf = get_transforms(train=False)
    ys, ps, _ = predict(model, DataLoader(ImgList([(root / p.replace("\\", "/"), int(l)) for p, l in
                                                    zip(syn["path"], syn["label"])], tf), batch_size=64), dev)
    split_csv = DEFAULT_PROCESSED / f"split_seed{a.seed}.csv"
    val = TextileDataset(split_csv, "val", a.data_root, tf)
    yv, pv, _ = predict(model, DataLoader(val, batch_size=64), dev)

    rows = []
    for l in sorted(set(ys)):
        m, mv = ys == l, yv == l
        wrong = pd.Series([names[i] for i in ps[m] if i != l]).value_counts().head(2).to_dict()
        rows.append({"class": names[l], "n_synthetic": int(m.sum()),
                     "synthetic_recognised_%": round(100 * float((ps[m] == l).mean()), 1),
                     "REAL_val_recall_%": round(100 * float((pv[mv] == l).mean()), 1),
                     "synthetic_confused_with": wrong})
    out = pd.DataFrame(rows)
    print("\nGAN-test (model trained on REAL train images only; no test set used)\n")
    print(out.to_string(index=False))
    f = PROJECT_ROOT / "results" / "metrics"; f.mkdir(parents=True, exist_ok=True)
    out.to_csv(f / f"gan_test_seed{a.seed}.csv", index=False)
    print("\nReading: synthetic recognised ~ real recall -> GAN images carry the class cues;"
          " much lower -> images are off-class/ambiguous.")


if __name__ == "__main__":
    main()
