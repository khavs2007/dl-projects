"""STEP 7b: generate synthetic images for minority classes + real-vs-synthetic figure + copy check.
Synthetic images go to data/augmented/seed<seed><tag>/ and are meant for the TRAIN set only.
Options: --tag (keep variants apart), --classes (subset), --filter_conf (keep only images that a
classifier trained on REAL train images recognises as the intended class with >= this confidence).
Usage:  python generate_images.py --seed 42 --data_root "<reduced_dataset>" --target_total 300 --tag _n300
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import DEFAULT_PROCESSED, DEFAULT_ROOT, load_class_names
from preprocessing import IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from train_gan import Generator, load_train_images
from utils import PROJECT_ROOT


def small(x_uint8):                       # 32x32 vectors for fast nearest-neighbour checks
    return F.adaptive_avg_pool2d(x_uint8.float() / 255.0, 32).flatten(1)


def load_filter_model(seed, n_all, dev):
    from train_resnet import build_model   # fine-tuned ResNet50 trained on REAL train images only
    m = build_model(n_all, 0.5).to(dev)
    m.load_state_dict(torch.load(PROJECT_ROOT / "models" / "resnet50" / f"resnet50_finetune_seed{seed}_best.pt",
                                 map_location=dev))
    return m.eval()


@torch.no_grad()
def keep_mask(model, x_uint8, label, thr, dev):
    x = F.interpolate(x_uint8.float().div(255).to(dev), size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                      align_corners=False).repeat(1, 3, 1, 1)
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(1, 3, 1, 1)
    p = F.softmax(model((x - mean) / std), 1)
    return ((p.argmax(1) == label) & (p[:, label] >= thr)).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--data_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--target_total", type=int, default=200,
                    help="raise every generated class to this many TRAIN images (real + synthetic)")
    ap.add_argument("--classes", nargs="+", default=None, help="generate only for these classes")
    ap.add_argument("--filter_conf", type=float, default=0.0, help="0 = no filtering; e.g. 0.9")
    ap.add_argument("--tag", default="", help="e.g. _n300 -> data/augmented/seed42_n300")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed + 1000)
    split_csv = a.split_csv or str(DEFAULT_PROCESSED / f"split_seed{a.seed}.csv")
    ck = torch.load(PROJECT_ROOT / "models" / "gan" / f"cgan_seed{a.seed}.pt", map_location=dev)
    gan_classes, class_names = ck["gan_classes"], load_class_names()
    G = Generator(len(gan_classes), ck["z_dim"], ck["ngf"]).to(dev)
    G.load_state_dict(ck["G_ema"]); G.eval()
    todo = [c for c in gan_classes if (a.classes is None or c in a.classes)]
    filt = load_filter_model(a.seed, len(class_names), dev) if a.filter_conf > 0 else None
    print(f"GAN trained for {ck['iters_done']} iterations | generating for: {todo} | "
          f"filter_conf={a.filter_conf} | tag='{a.tag}'")

    real, rlab = load_train_images(split_csv, a.data_root, gan_classes, ck["img"])   # train split only
    out = PROJECT_ROOT / "data" / "augmented" / f"seed{a.seed}{a.tag}"
    rows, synth_by_class = [], {}
    print(f"\n{'class':16s} {'real':>5s} {'kept':>5s} {'tried':>6s} | NN-dist synth->real | NN-dist real->real (LOO)")
    for cname in todo:
        k = gan_classes.index(cname)
        R = real[rlab == k]
        n_gen, imgs, tried = max(0, a.target_total - len(R)), [], 0
        with torch.no_grad():
            while len(imgs) < n_gen and tried < 30 * max(n_gen, 1):
                z = torch.randn(64, ck["z_dim"], device=dev)
                x = G(z, torch.full((64,), k, device=dev, dtype=torch.long))
                x = ((x + 1) * 127.5).round().clamp(0, 255).byte().cpu()
                tried += 64
                if filt is not None:
                    x = x[keep_mask(filt, x, class_names.index(cname), a.filter_conf, dev)]
                imgs += list(x[:, 0])
        imgs = imgs[:n_gen]
        if len(imgs) < n_gen:
            print(f"WARNING: only {len(imgs)} of {n_gen} images passed the filter for {cname}")
        (out / cname).mkdir(parents=True, exist_ok=True)
        S = torch.stack(imgs)[:, None] if imgs else torch.empty(0, 1, ck["img"], ck["img"], dtype=torch.uint8)
        synth_by_class[cname] = S
        d_sr = d_rr = None
        if len(S) > 0:
            d_sr = torch.cdist(small(S), small(R)).min(1).values
            d_rr = torch.cdist(small(R), small(R)); d_rr.fill_diagonal_(float("inf")); d_rr = d_rr.min(1).values
        for i in range(len(S)):
            fn = f"synth_{i:04d}.png"
            Image.fromarray(S[i, 0].numpy()).save(out / cname / fn)
            rows.append({"path": f"{cname}/{fn}", "class": cname, "label": class_names.index(cname),
                         "split": "train", "source": "gan", "nn_dist_to_real": float(d_sr[i])})
        nn = f"{d_sr.mean():.3f} (min {d_sr.min():.3f})   | {d_rr.mean():.3f}" if d_sr is not None else "-"
        print(f"{cname:16s} {len(R):5d} {len(S):5d} {tried if filt is not None else len(S):6d} | {nn}")
    csv = out.parent / f"synthetic_seed{a.seed}{a.tag}.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"\nSaved {len(rows)} synthetic images in {out}\nIndex CSV: {csv}")
    print("Copy check: if synth->real distances are MUCH smaller than real->real, the GAN is copying.")

    ncol, K = 6, len(todo)
    fig, axs = plt.subplots(2 * K, ncol, figsize=(ncol * 1.6, 2 * K * 1.7), squeeze=False)
    rng = np.random.default_rng(a.seed)
    for j, cname in enumerate(todo):
        R, S = real[rlab == gan_classes.index(cname)], synth_by_class[cname]
        for r, (arr, lab) in enumerate(((R, "real"), (S, "GAN"))):
            sel = rng.choice(len(arr), min(ncol, len(arr)), replace=False) if len(arr) else []
            for c in range(ncol):
                ax = axs[2 * j + r, c]; ax.axis("off")
                if c < len(sel):
                    ax.imshow(arr[sel[c], 0].numpy(), cmap="gray", vmin=0, vmax=255)
            axs[2 * j + r, 0].text(-0.08, 0.5, f"{cname}\n({lab})", transform=axs[2 * j + r, 0].transAxes,
                                   ha="right", va="center", fontsize=8)
    plt.tight_layout()
    f = PROJECT_ROOT / "results" / "figures"; f.mkdir(parents=True, exist_ok=True)
    fig.savefig(f / f"real_vs_gan_seed{a.seed}{a.tag}.png", dpi=200, bbox_inches="tight")
    print(f"Figure: {f / f'real_vs_gan_seed{a.seed}{a.tag}.png'}")


if __name__ == "__main__":
    main()
