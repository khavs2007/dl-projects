"""STEP 5 / Experiment 1: custom CNN baseline on the ORIGINAL training set.
Usage:  python train_cnn.py --seed 42
Model selection uses VALIDATION macro-F1 only. The test set is evaluated exactly once at the end.
"""
import argparse, json, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dataset import DEFAULT_PROCESSED, DEFAULT_ROOT, get_dataloaders, load_class_names
from evaluate import compute_metrics, predict, save_all
from utils import PROJECT_ROOT, env_info, set_seed


class CustomCNN(nn.Module):
    """4 x [Conv(3x3, padding=1) -> BatchNorm -> ReLU -> MaxPool(2)] -> GlobalAvgPool
       -> Dropout -> FC(256,128) -> ReLU -> Dropout -> FC(128, n_classes)"""

    def __init__(self, n_classes, widths=(32, 64, 128, 256), dropout=0.5):
        super().__init__()
        layers, c_in = [], 3
        for c_out in widths:
            layers += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split_csv", default=None, help="default: data/processed/split_seed<seed>.csv")
    ap.add_argument("--data_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=8, help="early-stop after N epochs without val macro-F1 gain")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", action="store_true", help="weighted loss (OFF for the baseline)")
    ap.add_argument("--tag", default="cnn_baseline")
    a = ap.parse_args()

    set_seed(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_csv = a.split_csv or str(DEFAULT_PROCESSED / f"split_seed{a.seed}.csv")
    class_names = load_class_names()
    loaders = get_dataloaders(split_csv, a.data_root, a.batch_size, a.num_workers, a.seed)
    sizes = {k: len(v.dataset) for k, v in loaders.items()}
    print(f"Device: {dev} | split sizes: {sizes}")

    model = CustomCNN(len(class_names)).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    weight = None
    if a.class_weights:
        cnt = np.bincount(loaders["train"].dataset.labels, minlength=len(class_names))
        weight = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp)

    tag = f"{a.tag}_seed{a.seed}"
    ckpt = PROJECT_ROOT / "models" / "cnn_baseline" / f"{tag}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rows, best_f1, bad = [], -1.0, 0

    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train()
        tl, tc, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                out = model(x); loss = criterion(out, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); tc += (out.argmax(1) == y).sum().item(); n += len(y)
        yv, pv, vl = predict(model, loaders["val"], dev, criterion)
        vs, _, _ = compute_metrics(yv, pv, class_names)
        sched.step(vs["macro_f1"])
        rows.append(dict(epoch=ep, train_loss=tl / n, train_acc=tc / n, val_loss=vl,
                         val_acc=vs["accuracy"], val_macro_f1=vs["macro_f1"], lr=opt.param_groups[0]["lr"]))
        print(f"Ep {ep:02d} | train loss {tl/n:.3f} acc {tc/n:.3f} | val loss {vl:.3f} "
              f"acc {vs['accuracy']:.3f} macroF1 {vs['macro_f1']:.3f} | {time.time()-t0:.0f}s")
        if vs["macro_f1"] > best_f1:
            best_f1, bad = vs["macro_f1"], 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stopping (no val macro-F1 gain for {a.patience} epochs)"); break

    hist = pd.DataFrame(rows)
    model.load_state_dict(torch.load(ckpt, map_location=dev))   # best-on-validation weights
    yv, pv, _ = predict(model, loaders["val"], dev)
    yt, pt, _ = predict(model, loaders["test"], dev)           # the ONLY time test is used
    vsum, vpc, vcm = compute_metrics(yv, pv, class_names)
    tsum, tpc, tcm = compute_metrics(yt, pt, class_names)
    save_all(tag, "val", vsum, vpc, vcm, class_names, hist)
    save_all(tag, "test", tsum, tpc, tcm, class_names)

    cfg = dict(vars(a), split_csv=split_csv, split_sizes=sizes, n_classes=len(class_names), n_params=n_params,
               optimizer="AdamW", loss="CrossEntropy" + (" (class-weighted)" if a.class_weights else ""),
               scheduler="ReduceLROnPlateau(max val macro-F1, factor 0.5, patience 3)",
               img_size=224, mixed_precision=amp, best_val_macro_f1=best_f1, env=env_info())
    json.dump(cfg, open(PROJECT_ROOT / "results" / "metrics" / f"{tag}_config.json", "w"), indent=2)
    print("\n=== TEST (best-on-validation model) ===")
    print(json.dumps(tsum, indent=2)); print(tpc.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
