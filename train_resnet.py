"""STEP 6 / Experiments 2 and 3: ResNet50 transfer learning (ImageNet weights).
  stage 'frozen'   : whole backbone frozen, only the new classifier head is trained  -> Experiment 2
  stage 'finetune' : starts from the frozen-stage model, unfreezes layer4 (+head)    -> Experiment 3
Usage:  python train_resnet.py --seed 42 --data_root "<path to reduced_dataset>"
Model selection uses VALIDATION macro-F1 only; the test set is evaluated once per stage at the end.
"""
import argparse, json, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

from dataset import DEFAULT_PROCESSED, DEFAULT_ROOT, build_extra_train, get_dataloaders, load_class_names
from evaluate import compute_metrics, predict, save_all
from utils import PROJECT_ROOT, env_info, set_seed


def build_model(n_classes, dropout):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)   # ImageNet pretrained
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, n_classes))  # new head
    return m


def set_trainable(model, layers):
    """Only parameters inside the named top-level blocks (e.g. 'layer4', 'fc') get gradients."""
    for name, p in model.named_parameters():
        p.requires_grad = name.split(".")[0] in layers


def set_train_mode(model):
    """model.train(), but keep fully-frozen blocks in eval mode so their BatchNorm
    running statistics (learned on ImageNet) are not disturbed."""
    model.train()
    for _, child in model.named_children():
        if not any(p.requires_grad for p in child.parameters()):
            child.eval()


def run_stage(stage, model, loaders, class_names, a, dev, trainable, epochs, lr, patience):
    set_trainable(model, trainable)
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    tag = f"resnet50{a.tag_suffix}_{stage}_seed{a.seed}"
    print(f"\n##### {tag}: trainable blocks {trainable} | trainable params {n_train:,} | lr {lr}")

    weight = None
    if a.class_weights:
        cnt = np.bincount(loaders["train"].dataset.labels, minlength=len(class_names))
        weight = torch.tensor(cnt.sum() / (len(cnt) * cnt), dtype=torch.float32, device=dev)
    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") \
        else torch.cuda.amp.GradScaler(enabled=amp)
    ckpt = PROJECT_ROOT / "models" / "resnet50" / f"{tag}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    rows, best_f1, bad = [], -1.0, 0

    for ep in range(1, epochs + 1):
        t0 = time.time(); set_train_mode(model)
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
            if bad >= patience:
                print(f"Early stopping (no val macro-F1 gain for {patience} epochs)"); break

    hist = pd.DataFrame(rows)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    yv, pv, _ = predict(model, loaders["val"], dev)
    yt, pt, _ = predict(model, loaders["test"], dev)             # test used once per stage
    vsum, vpc, vcm = compute_metrics(yv, pv, class_names)
    tsum, tpc, tcm = compute_metrics(yt, pt, class_names)
    save_all(tag, "val", vsum, vpc, vcm, class_names, hist)
    save_all(tag, "test", tsum, tpc, tcm, class_names)
    all_blocks = [n for n, _ in model.named_children() if sum(1 for _ in getattr(model, n).parameters())]
    cfg = dict(vars(a), tag=tag, stage=stage, epochs_max=epochs, lr=lr, patience=patience,
               trainable_blocks=trainable, frozen_blocks=[b for b in all_blocks if b not in trainable],
               trainable_params=n_train, optimizer="AdamW", dropout_in_head=a.dropout,
               loss="CrossEntropy" + (" (class-weighted)" if a.class_weights else ""),
               scheduler="ReduceLROnPlateau(max val macro-F1, factor 0.5, patience 2)", img_size=224,
               pretrained_weights="ResNet50_Weights.IMAGENET1K_V2", mixed_precision=amp,
               best_val_macro_f1=best_f1, env=env_info())
    json.dump(cfg, open(PROJECT_ROOT / "results" / "metrics" / f"{tag}_config.json", "w"), indent=2)
    print(f"\n=== TEST [{tag}] (best-on-validation model) ===")
    print(json.dumps(tsum, indent=2)); print(tpc.round(3).to_string(index=False))
    return ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage", choices=["frozen", "finetune", "both"], default="both")
    ap.add_argument("--split_csv", default=None)
    ap.add_argument("--data_root", default=str(DEFAULT_ROOT))
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs_frozen", type=int, default=20)
    ap.add_argument("--epochs_finetune", type=int, default=25)
    ap.add_argument("--lr_frozen", type=float, default=1e-3)
    ap.add_argument("--lr_finetune", type=float, default=1e-4)
    ap.add_argument("--unfreeze", nargs="+", default=["layer4"], help="blocks to unfreeze in finetune stage, e.g. layer3 layer4")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--class_weights", action="store_true", help="weighted loss (ablation only)")
    ap.add_argument("--tag_suffix", default="", help="e.g. _gan, _os, _cw (keeps result files separate)")
    ap.add_argument("--synthetic_csv", default=None, help="GAN index CSV from generate_images.py (Exp 4)")
    ap.add_argument("--synthetic_classes", nargs="+", default=None, help="use GAN images of these classes only")
    ap.add_argument("--oversample_to", type=int, default=0, help="control: repeat REAL images up to N per class")
    ap.add_argument("--oversample_classes", nargs="+", default=None)
    a = ap.parse_args()

    set_seed(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_csv = a.split_csv or str(DEFAULT_PROCESSED / f"split_seed{a.seed}.csv")
    class_names = load_class_names()
    extra = build_extra_train(split_csv, a.data_root, a.synthetic_csv, a.synthetic_classes,
                              a.oversample_to, a.oversample_classes, a.seed)
    a.n_extra_train = len(extra)
    loaders = get_dataloaders(split_csv, a.data_root, a.batch_size, a.num_workers, a.seed, extra_train=extra)
    print(f"Extra TRAIN-only images added (synthetic / oversampled): {len(extra)}")
    print(f"Device: {dev} | split sizes: { {k: len(v.dataset) for k, v in loaders.items()} }")
    model = build_model(len(class_names), a.dropout).to(dev)

    if a.stage in ("frozen", "both"):
        run_stage("frozen", model, loaders, class_names, a, dev, ["fc"], a.epochs_frozen, a.lr_frozen, patience=5)
    if a.stage in ("finetune", "both"):
        if a.stage == "finetune":      # start from the saved frozen-stage weights
            p = PROJECT_ROOT / "models" / "resnet50" / f"resnet50{a.tag_suffix}_frozen_seed{a.seed}_best.pt"
            model.load_state_dict(torch.load(p, map_location=dev))
        run_stage("finetune", model, loaders, class_names, a, dev, a.unfreeze + ["fc"],
                  a.epochs_finetune, a.lr_finetune, patience=6)


if __name__ == "__main__":
    main()
