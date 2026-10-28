"""STEP 2: controlled, reproducible dataset reduction.
Rules: drop corrupted + duplicate images, keep every class <= cap in full,
randomly (seeded) down-sample only classes larger than cap.
Usage:  python src/reduce_dataset.py --raw_root data/raw [--cap 600] [--max_side 0]
"""
import argparse, shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from utils import PROJECT_ROOT, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default=str(PROJECT_ROOT / "data" / "raw"))
    ap.add_argument("--manifest", default=str(PROJECT_ROOT / "results/metrics/dataset_manifest.csv"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "reduced_dataset"))
    ap.add_argument("--cap", type=int, default=None,
                    help="max images per class (default: 2 x median class size)")
    ap.add_argument("--max_side", type=int, default=0,
                    help="if >0, downscale so longest side <= this (0 = keep original)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    set_seed(a.seed)
    out = Path(a.out)

    df = pd.read_csv(a.manifest)
    n_raw = df.groupby("class").size().rename("raw")
    df = df[~df.corrupted].sort_values("rel_path").reset_index(drop=True)

    # duplicates: label-conflicting groups dropped entirely, others keep first copy
    cls_per_hash = df.groupby("md5")["class"].nunique()
    conflict = set(cls_per_hash[cls_per_hash > 1].index)
    df = df[~df.md5.isin(conflict)]
    df = df.drop_duplicates("md5", keep="first")
    n_clean = df.groupby("class").size().rename("after_cleaning")

    cap = a.cap or int(round(2 * n_clean.median()))
    print(f"Per-class cap = {cap}")
    parts, actions = [], {}
    for cls, d in df.groupby("class"):
        if len(d) > cap:
            d = d.sample(n=cap, random_state=a.seed); actions[cls] = f"down-sampled to {cap}"
        else:
            actions[cls] = "kept all"
        parts.append(d)
    red = pd.concat(parts).sort_values(["class", "rel_path"]).reset_index(drop=True)

    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            raise SystemExit(f"{out} not empty. Use --overwrite to rebuild.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    new_paths = []
    for _, r in tqdm(red.iterrows(), total=len(red), desc="Copying"):
        src = Path(a.raw_root) / r.rel_path
        dst_dir = out / str(r["class"]); dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / f"{src.stem}__{r.md5[:8]}{src.suffix.lower()}"
        if a.max_side > 0 and max(r.width, r.height) > a.max_side:
            with Image.open(src) as im:
                s = a.max_side / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                if dst.suffix in (".jpg", ".jpeg"):
                    im.save(dst, quality=95)
                else:
                    im.save(dst)
        else:
            shutil.copy2(src, dst)
        new_paths.append(str(dst.relative_to(out)))
    red["reduced_rel_path"] = new_paths
    red.to_csv(PROJECT_ROOT / "results/metrics/reduced_manifest.csv", index=False)

    summ = pd.concat([n_raw, n_clean, red.groupby("class").size().rename("after_reduction")],
                     axis=1).fillna(0).astype(int)
    summ["action"] = pd.Series(actions)
    summ = summ.sort_values("raw", ascending=False)
    summ.to_csv(PROJECT_ROOT / "results/metrics/reduction_summary.csv")
    print("\n=== BEFORE vs AFTER ===\n", summ.to_string())
    out_gb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e9
    print(f"\nTotal: {summ.raw.sum()} -> {summ.after_reduction.sum()} images | "
          f"disk: {out_gb:.2f} GB | imbalance ratio {summ.raw.max()/summ.raw.min():.2f} "
          f"-> {summ.after_reduction.max()/summ.after_reduction.min():.2f}")

    figs = PROJECT_ROOT / "results/figures"; figs.mkdir(parents=True, exist_ok=True)
    x, w = np.arange(len(summ)), 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, summ.raw, w, label="Original", color="#9aa5b1")
    b2 = ax.bar(x + w/2, summ.after_reduction, w, label="Reduced", color="#3b6ea5")
    ax.bar_label(b1, fontsize=8); ax.bar_label(b2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(summ.index, rotation=35, ha="right")
    ax.set_ylabel("Number of images"); ax.set_title("Class distribution before vs after reduction")
    ax.legend(); plt.tight_layout()
    fig.savefig(figs / "class_distribution_reduced.png", dpi=300); plt.close(fig)
    print("Saved results/metrics/{reduction_summary.csv,reduced_manifest.csv}, "
          "results/figures/class_distribution_reduced.png")


if __name__ == "__main__":
    main()
