"""STEP 3: stratified, leakage-aware train/val/test split (run BEFORE any GAN work).

What it does
 1. Reads images from data/reduced_dataset/<class>/...
 2. Finds near-duplicate images (perceptual dHash, same class) and puts each
    near-duplicate cluster into ONE split, so copies/consecutive frames can't leak.
 3. Splits each class ~70/15/15 (stratified), for several seeds.
 4. Writes data/processed/split_seed<seed>.csv + class_names.json.

Usage:  python split_dataset.py                (seeds 42 43 44, 70/15/15)
        python split_dataset.py --hash_thr 0   (disable near-duplicate grouping)
"""
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from utils import IMG_EXTS, PROJECT_ROOT


def dhash(path, size=8):
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming_matrix(h):
    n = len(h)
    D = np.zeros((n, n), dtype=np.uint8)
    for i in range(n):
        x = h[i] ^ h
        D[i] = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(1)
    return D


def cluster(idx, D, cls, thr, max_group):
    """Connected components of 'similar & same class' pairs. A component that grows
    bigger than max_group is a sign of chaining (A~B, B~C, C~D ...), so it is
    re-clustered with a STRICTER threshold until it is small (or thr reaches 0)."""
    idx = np.asarray(idx)
    sub = (D[np.ix_(idx, idx)] <= thr) & (cls[idx][:, None] == cls[idx][None, :])
    n_comp, lab = connected_components(csr_matrix(sub), directed=False)
    out = []
    for c in range(n_comp):
        members = idx[lab == c]
        if len(members) > max_group and thr > 0:
            out += cluster(members, D, cls, thr - 1, max_group)
        else:
            out.append(members)
    return out


def near_dup_groups(df, thr, max_group):
    n = len(df)
    if thr <= 0:
        return np.arange(n)
    h = np.array([dhash(p) for p in tqdm(df.abs_path, desc="Hashing")], dtype=np.uint64)
    D = hamming_matrix(h)
    gid = np.zeros(n, dtype=int)
    for g, members in enumerate(cluster(np.arange(n), D, np.asarray(df["class"].astype(str)), thr, max_group)):
        gid[members] = g
    return gid


def split_one_seed(df, ratios, seed):
    rng = random.Random(seed)
    names = ["train", "val", "test"]
    assign = {}
    for cls, d in df.groupby("class"):
        groups = d.groupby("group_id").size().to_dict()
        gids = list(groups)
        rng.shuffle(gids)
        gids.sort(key=lambda g: -groups[g])        # big groups first, ties stay shuffled
        total = sum(groups.values())
        target = {s: r * total for s, r in zip(names, ratios)}
        have = {s: 0 for s in names}
        for g in gids:
            # put the group in the split with the largest remaining deficit
            s = max(names, key=lambda k: (target[k] - have[k]) / max(target[k], 1e-9))
            have[s] += groups[g]
            assign[g] = s
    return df["group_id"].map(assign)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "reduced_dataset"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "processed"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.70, 0.15, 0.15])
    ap.add_argument("--hash_thr", type=int, default=4,
                    help="max Hamming distance (of 64 bits) to call two images near-duplicates")
    ap.add_argument("--max_group", type=int, default=6,
                    help="near-duplicate clusters larger than this are re-clustered more strictly "
                         "(prevents chaining into huge groups)")
    a = ap.parse_args()
    assert abs(sum(a.ratios) - 1) < 1e-6
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [(str(p.relative_to(data)), p.parent.name, str(p))
            for p in sorted(data.rglob("*")) if p.suffix.lower() in IMG_EXTS]
    if not rows:
        raise SystemExit(f"No images in {data}. Check --data.")
    df = pd.DataFrame(rows, columns=["path", "class", "abs_path"])
    classes = sorted(df["class"].unique())
    df["label"] = df["class"].map({c: i for i, c in enumerate(classes)})
    json.dump(classes, open(out / "class_names.json", "w"), indent=2)
    print(f"{len(df)} images, {len(classes)} classes: {classes}")

    df["group_id"] = near_dup_groups(df, a.hash_thr, a.max_group)
    gs = df.groupby("group_id").size()
    print(f"\nNear-duplicate groups (thr={a.hash_thr}): {len(gs)} groups for {len(df)} images; "
          f"{int((gs > 1).sum())} groups have >1 image; largest group = {gs.max()} images")
    if gs.max() > 15:
        print("WARNING: a large group remains (identical hashes). Send me this output.")
    multi = df[df.group_id.isin(gs[gs > 1].index)].groupby("class")["group_id"].nunique()
    print("Multi-image groups per class:", multi.to_dict())

    summaries = []
    for seed in a.seeds:
        df["split"] = split_one_seed(df, a.ratios, seed)
        # ---- leakage / sanity checks
        assert df.groupby("group_id")["split"].nunique().max() == 1, "group leaks across splits"
        ct = pd.crosstab(df["class"], df["split"])[["train", "val", "test"]]
        assert (ct.values >= 1).all(), "a class is missing from a split"
        ct["total"] = ct.sum(axis=1)
        print(f"\n=== SEED {seed} ===\n{ct.reindex(sorted(classes)).to_string()}")
        print("Split totals:", df.split.value_counts().to_dict())
        df.drop(columns="abs_path").to_csv(out / f"split_seed{seed}.csv", index=False)
        ct["seed"] = seed
        summaries.append(ct.reset_index())
    pd.concat(summaries).to_csv(out / "split_summary.csv", index=False)

    figs = PROJECT_ROOT / "results" / "figures"; figs.mkdir(parents=True, exist_ok=True)
    ct = summaries[0].drop(columns=["seed", "total"]).set_index("class")
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(ct))
    for s, c in zip(["train", "val", "test"], ["#3b6ea5", "#e0a030", "#b84a4a"]):
        ax.bar(ct.index, ct[s], bottom=bottom, label=s, color=c); bottom += ct[s].values
    ax.set_ylabel("Number of images"); ax.set_title(f"Train/val/test split per class (seed {a.seeds[0]})")
    ax.legend(); plt.xticks(rotation=35, ha="right"); plt.tight_layout()
    fig.savefig(figs / "split_distribution.png", dpi=300); plt.close(fig)
    print(f"\nSaved split CSVs + class_names.json in {out}")


if __name__ == "__main__":
    main()
