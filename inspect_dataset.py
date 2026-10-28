"""STEP 1: inspect the dataset. Streams one image at a time (no big RAM use).
Usage (from project root):  python src/inspect_dataset.py --root data/raw
"""
import argparse, hashlib, json, os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import IMG_EXTS, PROJECT_ROOT, infer_class_and_split


def print_tree(root: Path, max_depth=3):
    print(f"\n=== DIRECTORY TREE (depth<={max_depth}) ===")
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        exts = Counter(Path(f).suffix.lower() or "<none>" for f in filenames)
        info = f"  [{sum(exts.values())} files: {dict(exts)}]" if filenames else ""
        print("  " * depth + (rel.name or str(root)) + "/" + info)


def analyse(path: Path) -> dict:
    rec = dict(path=str(path), bytes=path.stat().st_size, width=None, height=None,
               mode=None, format=None, corrupted=False, error="", md5="")
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        rec["md5"] = h.hexdigest()
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:   # re-open: verify() invalidates the handle
            im.load()                  # full decode catches truncated files
            rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
    except Exception as e:
        rec["corrupted"], rec["error"] = True, str(e)[:120]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.root).resolve()
    out_m, out_f = PROJECT_ROOT / "results" / "metrics", PROJECT_ROOT / "results" / "figures"
    out_m.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)

    print_tree(root)

    all_files = [Path(d) / f for d, _, fs in os.walk(root) for f in fs]
    imgs = sorted(p for p in all_files if p.suffix.lower() in IMG_EXTS)
    others = Counter(p.suffix.lower() or "<none>" for p in all_files if p.suffix.lower() not in IMG_EXTS)
    print(f"\nImage files: {len(imgs)} | Non-image files by extension: {dict(others)}")
    if not imgs:
        raise SystemExit("No images found - check --root.")
    if any(e in others for e in (".xml", ".json", ".txt", ".csv")):
        print("NOTE: annotation-like files exist -> dataset may be detection/segmentation "
              "format. Send me the tree above before continuing.")

    with ThreadPoolExecutor(a.workers) as ex:
        recs = list(tqdm(ex.map(analyse, imgs), total=len(imgs), desc="Analysing"))
    df = pd.DataFrame(recs)
    df[["class", "split_hint"]] = [infer_class_and_split(Path(p), root) for p in df["path"]]
    df["rel_path"] = [str(Path(p).relative_to(root)) for p in df["path"]]
    df = df.drop(columns="path")
    df.to_csv(out_m / "dataset_manifest.csv", index=False)

    ok = df[~df.corrupted]
    g = df.groupby("class")
    table = pd.DataFrame({
        "n_images": g.size(),
        "n_corrupted": g["corrupted"].sum(),
        "median_w": ok.groupby("class")["width"].median(),
        "median_h": ok.groupby("class")["height"].median(),
        "n_unique_sizes": ok.groupby("class").apply(lambda d: len(set(zip(d.width, d.height)))),
        "size_GB": (g["bytes"].sum() / 1e9).round(3),
    }).sort_values("n_images", ascending=False)
    table["percent"] = (100 * table.n_images / table.n_images.sum()).round(1)
    table["ratio_vs_largest"] = (table.n_images.max() / table.n_images).round(2)
    table.to_csv(out_m / "class_distribution.csv")
    print("\n=== CLASS DISTRIBUTION ===\n", table.to_string())
    print(f"\nClasses: {len(table)} -> {list(table.index)}")
    print(f"Imbalance ratio (largest/smallest): {table.n_images.max() / table.n_images.min():.2f}")

    print("\nFormats:", dict(Counter(ok.format)), "| Modes:", dict(Counter(ok["mode"])))
    print("Top image sizes (WxH):", Counter(zip(ok.width, ok.height)).most_common(5))
    print(f"Width  min/median/max: {ok.width.min()}/{ok.width.median()}/{ok.width.max()}")
    print(f"Height min/median/max: {ok.height.min()}/{ok.height.median()}/{ok.height.max()}")
    print(f"Total image size: {df.bytes.sum() / 1e9:.2f} GB")
    print(f"Corrupted/unreadable: {int(df.corrupted.sum())}")
    if df.corrupted.any():
        print(df[df.corrupted][["rel_path", "error"]].head(10).to_string())

    dup = ok[ok.duplicated("md5", keep=False)]
    n_groups = dup.md5.nunique()
    redundant = len(dup) - n_groups
    conflict = dup.groupby("md5")["class"].nunique()
    n_conflict = int((conflict > 1).sum())
    print(f"\nExact duplicate groups: {n_groups} | redundant files: {redundant} "
          f"| groups spanning >1 class (label conflict): {n_conflict}")

    print("\n=== PREDEFINED SPLIT? ===")
    if df.split_hint.notna().any():
        print(pd.crosstab(df["class"], df.split_hint.fillna("none")).to_string())
    else:
        print("No train/val/test folders detected -> we will create our own split.")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(table.index.astype(str), table.n_images, color="#3b6ea5")
    ax.bar_label(bars, padding=2)
    ax.set_ylabel("Number of images"); ax.set_xlabel("Class")
    ax.set_title("Original dataset: class distribution")
    plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    fig.savefig(out_f / "class_distribution_original.png", dpi=300); plt.close(fig)

    json.dump(dict(n_images=len(df), n_classes=len(table), classes=list(table.index),
                   imbalance_ratio=float(table.n_images.max() / table.n_images.min()),
                   corrupted=int(df.corrupted.sum()), duplicate_redundant_files=int(redundant),
                   label_conflict_groups=n_conflict, total_gb=round(df.bytes.sum() / 1e9, 3)),
              open(out_m / "inspection_summary.json", "w"), indent=2)
    print("\nSaved: results/metrics/{dataset_manifest.csv,class_distribution.csv,"
          "inspection_summary.json}, results/figures/class_distribution_original.png")


if __name__ == "__main__":
    main()
