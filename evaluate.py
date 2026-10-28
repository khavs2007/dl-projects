"""Metrics, confusion matrices and curves - reused by every experiment."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support)

from utils import PROJECT_ROOT


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval()
    ys, ps, loss_sum, n = [], [], 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy()); ps.append(out.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps), (loss_sum / n if criterion is not None else None)


def compute_metrics(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({"class": class_names, "precision": p, "recall": r, "f1": f, "support": s})
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()), "macro_f1": float(f.mean()),
    }
    return summary, per_class, confusion_matrix(y_true, y_pred, labels=labels)


def plot_confusion(cm, class_names, path, title):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right"); ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    fig.colorbar(im, fraction=0.046, label="Row-normalised"); plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300); plt.close(fig)


def plot_curves(hist, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train"); ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_acc"], label="train"); ax[1].plot(hist["epoch"], hist["val_acc"], label="val")
    ax[1].plot(hist["epoch"], hist["val_macro_f1"], "--", label="val macro-F1")
    ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Score"); ax[1].set_title("Accuracy / macro-F1"); ax[1].legend()
    fig.suptitle(title); plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300); plt.close(fig)


def save_all(tag, split_name, summary, per_class, cm, class_names, hist=None):
    """Writes metrics JSON, per-class CSV, confusion-matrix PNG (+ curves if hist)."""
    m, c, f = (PROJECT_ROOT / "results" / d for d in ("metrics", "confusion_matrices", "figures"))
    for d in (m, c, f):
        d.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(m / f"{tag}_{split_name}_metrics.json", "w"), indent=2)
    per_class.to_csv(m / f"{tag}_{split_name}_per_class.csv", index=False)
    np.savetxt(m / f"{tag}_{split_name}_confusion.csv", cm, fmt="%d", delimiter=",")
    plot_confusion(cm, class_names, c / f"{tag}_{split_name}_cm.png", f"{tag} ({split_name})")
    if hist is not None:
        hist.to_csv(m / f"{tag}_history.csv", index=False)
        plot_curves(hist, f / f"{tag}_curves.png", tag)
