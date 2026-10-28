"""STEP 7: class-conditional GAN for minority defect classes (trained on TRAIN images only).
Architecture: conditional DCGAN-style generator (class-conditional BatchNorm) + projection
discriminator with spectral norm, hinge loss, DiffAugment (for tiny datasets) and an EMA generator.
Images: 128x128 grayscale. One GAN PER SEED (each seed has a different train split -> no leakage).
Usage:  python train_gan.py --seed 42 --data_root "<path to reduced_dataset>"
"""
import argparse, copy, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.utils.parametrizations import spectral_norm as SN
from torchvision.utils import save_image

from dataset import DEFAULT_PROCESSED, DEFAULT_ROOT, load_class_names
from utils import PROJECT_ROOT, env_info, set_seed


# ----------------------------------------------------------------- models
class CondBN(nn.Module):
    """BatchNorm whose scale/shift depend on the class label -> the class controls the image."""
    def __init__(self, ch, n_cls):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch, affine=False)
        self.gamma, self.beta = nn.Embedding(n_cls, ch), nn.Embedding(n_cls, ch)
        nn.init.ones_(self.gamma.weight); nn.init.zeros_(self.beta.weight)

    def forward(self, x, y):
        return self.bn(x) * self.gamma(y)[:, :, None, None] + self.beta(y)[:, :, None, None]


class Generator(nn.Module):
    def __init__(self, n_cls, z_dim=128, ngf=64):
        super().__init__()
        self.ch = [ngf * 8, ngf * 4, ngf * 2, ngf, ngf // 2, ngf // 2]      # 512,256,128,64,32,32
        self.fc = nn.Linear(z_dim, 4 * 4 * self.ch[0])
        self.convs = nn.ModuleList([nn.Conv2d(self.ch[i], self.ch[i + 1], 3, 1, 1) for i in range(5)])
        self.norms = nn.ModuleList([CondBN(self.ch[i + 1], n_cls) for i in range(5)])
        self.out = nn.Conv2d(self.ch[-1], 1, 3, 1, 1)                        # 4 -> 128 pixels (5 x2)

    def forward(self, z, y):
        x = self.fc(z).view(z.size(0), self.ch[0], 4, 4)
        for conv, norm in zip(self.convs, self.norms):
            x = F.relu(norm(conv(F.interpolate(x, scale_factor=2, mode="nearest")), y))
        return torch.tanh(self.out(x))


class Discriminator(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        ch, layers = [1, 64, 128, 256, 512, 512], []
        for i in range(5):                                                   # 128 -> 4
            layers += [SN(nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True)]
        self.features = nn.Sequential(*layers)
        self.linear = SN(nn.Linear(512, 1))
        self.embed = SN(nn.Embedding(n_cls, 512))                            # projection conditioning

    def forward(self, x, y):
        h = self.features(x).sum(dim=(2, 3))
        return self.linear(h).squeeze(1) + (self.embed(y) * h).sum(1)


# ----------------------------------------------------------------- helpers
def diff_augment(x):
    """Differentiable augmentation applied to BOTH real and fake before D (stops D memorising
    the tiny training set): random translation, brightness/contrast, cutout."""
    B, dev = x.size(0), x.device
    theta = torch.zeros(B, 2, 3, device=dev); theta[:, 0, 0] = 1; theta[:, 1, 1] = 1
    theta[:, :, 2] = (torch.rand(B, 2, device=dev) - 0.5) * 0.5
    x = F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False),
                      padding_mode="reflection", align_corners=False)
    x = x + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.2
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * (torch.rand(B, 1, 1, 1, device=dev) + 0.5) + m
    S = x.size(-1); ar = torch.arange(S, device=dev)
    cy, cx = torch.randint(0, S, (B, 1, 1), device=dev), torch.randint(0, S, (B, 1, 1), device=dev)
    mask = ((ar.view(1, S, 1) - cy).abs() < S // 4) & ((ar.view(1, 1, S) - cx).abs() < S // 4)
    return x * (~mask).unsqueeze(1).float()


def random_flips(x):
    """Real-data augmentation: flips keep horizontal defects horizontal / vertical ones vertical."""
    b = x.size(0)
    x = torch.where((torch.rand(b, device=x.device) < 0.5).view(-1, 1, 1, 1), x.flip(3), x)
    return torch.where((torch.rand(b, device=x.device) < 0.5).view(-1, 1, 1, 1), x.flip(2), x)


def load_train_images(split_csv, data_root, classes, img=128):
    """TRAIN split only. 128px grayscale uint8 -> a few MB, so it is kept in memory/GPU."""
    df = pd.read_csv(split_csv)
    df = df[(df["split"] == "train") & df["class"].isin(classes)].reset_index(drop=True)
    imgs, labels = [], []
    for _, r in df.iterrows():
        with Image.open(Path(data_root) / r["path"].replace("\\", "/")) as im:
            imgs.append(np.asarray(im.convert("L").resize((img, img), Image.LANCZOS)))
        labels.append(classes.index(r["class"]))
    return torch.from_numpy(np.stack(imgs))[:, None], torch.tensor(labels)


@torch.no_grad()
def ema_update(ema, G, decay):
    for pe, p in zip(ema.parameters(), G.parameters()):
        pe.lerp_(p, 1 - decay)


@torch.no_grad()
def standing_stats(G, n_cls, z_dim, dev, iters=50, bs=64):
    """Re-estimate BatchNorm statistics of the (EMA) generator so eval-mode sampling is reliable."""
    for m in G.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats(); m.momentum = None
    G.train()
    for _ in range(iters):
        G(torch.randn(bs, z_dim, device=dev), torch.randint(0, n_cls, (bs,), device=dev))
    G.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--data_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--classes", nargs="+", default=None, help="override target classes")
    ap.add_argument("--minority_below", type=int, default=150,
                    help="target = classes with fewer TRAIN images than this (rule uses train counts only)")
    ap.add_argument("--iters", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--z_dim", type=int, default=128)
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--lr_g", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=4e-4)
    ap.add_argument("--sample_every", type=int, default=1000)
    a = ap.parse_args()

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_csv = a.split_csv or str(DEFAULT_PROCESSED / f"split_seed{a.seed}.csv")
    class_names = load_class_names()
    tr = pd.read_csv(split_csv).query("split == 'train'")
    counts = tr.groupby("class").size()
    classes = sorted(a.classes) if a.classes else sorted(counts[counts < a.minority_below].index)
    print("TRAIN counts:", counts.to_dict()); print("GAN target classes:", classes)
    n_cls = len(classes)

    data, labels = load_train_images(split_csv, a.data_root, classes)
    data, labels = data.to(dev), labels.to(dev)
    print(f"GAN training images (train split only): {len(data)} | per class: "
          f"{ {c: int((labels == i).sum()) for i, c in enumerate(classes)} }")
    w = 1.0 / torch.bincount(labels, minlength=n_cls).float()[labels]        # class-balanced sampling

    G, D = Generator(n_cls, a.z_dim, a.ngf).to(dev), Discriminator(n_cls).to(dev)
    ema = copy.deepcopy(G).requires_grad_(False)
    oG = torch.optim.Adam(G.parameters(), lr=a.lr_g, betas=(0.0, 0.9))
    oD = torch.optim.Adam(D.parameters(), lr=a.lr_d, betas=(0.0, 0.9))

    out_img = PROJECT_ROOT / "results" / "generated_images" / f"gan_training_seed{a.seed}"
    out_img.mkdir(parents=True, exist_ok=True)
    ckpt = PROJECT_ROOT / "models" / "gan" / f"cgan_seed{a.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    z_fix = torch.randn(n_cls * 8, a.z_dim, device=dev)
    y_fix = torch.arange(n_cls, device=dev).repeat_interleave(8)
    log, t0 = [], time.time()

    def snapshot(it):
        snap = copy.deepcopy(ema); standing_stats(snap, n_cls, a.z_dim, dev)
        with torch.no_grad():
            save_image((snap(z_fix, y_fix) + 1) / 2, out_img / f"it_{it:06d}.png", nrow=8, padding=2)
        torch.save(dict(G_ema=snap.state_dict(), gan_classes=classes, class_names=class_names,
                        z_dim=a.z_dim, ngf=a.ngf, img=128, iters_done=it, args=vars(a)), ckpt)

    for it in range(1, a.iters + 1):
        idx = torch.multinomial(w, a.batch_size, replacement=True)
        real, y = random_flips(data[idx].float() / 127.5 - 1), labels[idx]
        # ---- discriminator step (hinge loss)
        with torch.no_grad():
            fake = G(torch.randn(a.batch_size, a.z_dim, device=dev), y)
        d_real, d_fake = D(diff_augment(real), y), D(diff_augment(fake), y)
        lossD = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
        oD.zero_grad(set_to_none=True); lossD.backward(); oD.step()
        # ---- generator step
        fake = G(torch.randn(a.batch_size, a.z_dim, device=dev), y)
        lossG = -D(diff_augment(fake), y).mean()
        oG.zero_grad(set_to_none=True); lossG.backward(); oG.step()
        ema_update(ema, G, min(0.999, (1 + it) / (10 + it)))

        if it % 200 == 0:
            rt = torch.sign(d_real.detach()).mean().item()   # -> 1.0 means D is overfitting/memorising
            log.append(dict(iter=it, loss_d=lossD.item(), loss_g=lossG.item(),
                            d_real=d_real.mean().item(), d_fake=d_fake.mean().item(), d_overfit_rt=rt))
            print(f"it {it:6d} | D {lossD.item():.3f} G {lossG.item():.3f} | D(real) {d_real.mean().item():+.2f} "
                  f"D(fake) {d_fake.mean().item():+.2f} | rt {rt:+.2f} | {it / (time.time() - t0):.1f} it/s")
        if it % a.sample_every == 0 or it == a.iters:
            snapshot(it)
    m = PROJECT_ROOT / "results" / "metrics"; m.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(log).to_csv(m / f"gan_seed{a.seed}_log.csv", index=False)
    json.dump(dict(vars(a), classes=classes, n_train_images=int(len(data)), env=env_info()),
              open(m / f"gan_seed{a.seed}_config.json", "w"), indent=2)
    print(f"\nDone. Checkpoint: {ckpt}\nSample grids: {out_img}  (rows = classes {classes})")


if __name__ == "__main__":
    main()
