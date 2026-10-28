"""Streamlit dashboard: upload a fabric image -> predicted defect class (9 classes).
Run from the folder that contains the .py files:   streamlit run app.py
Uses the SAME preprocessing as training/testing (grayscale -> 3 channels, 224x224, ImageNet normalisation).
"""
import re
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models

from dataset import DEFAULT_PROCESSED, load_class_names
from preprocessing import get_transforms
from utils import PROJECT_ROOT

HERE = Path(__file__).resolve().parent


def first_existing(*candidates):
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])


MODEL_DIR = first_existing(PROJECT_ROOT / "models" / "resnet50", HERE / "models" / "resnet50")
DATA_ROOT = first_existing(PROJECT_ROOT / "data" / "reduced_dataset", HERE / "data" / "reduced_dataset")
SPLIT_DIR = first_existing(DEFAULT_PROCESSED, HERE / "data" / "processed")

st.set_page_config(page_title="Textile Defect Classifier", layout="wide")


@st.cache_resource
def load_model(path: str, n_classes: int):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet50(weights=None)                       # same architecture as train_resnet.py
    m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, n_classes))
    m.load_state_dict(torch.load(path, map_location=dev))
    return m.to(dev).eval(), dev


@torch.no_grad()
def predict(model, dev, img):
    x = get_transforms(train=False)(img.convert("RGB")).unsqueeze(0).to(dev)
    return torch.softmax(model(x), 1)[0].cpu()


def show_result(model, dev, names, img, true_label=None):
    probs = predict(model, dev, img)
    top = torch.topk(probs, 3)
    pred, conf = names[int(top.indices[0])], float(top.values[0])
    c1, c2 = st.columns(2)
    shown = img.copy(); shown.thumbnail((460, 460))
    with c1:
        st.image(shown, caption="Your image")
    with c2:
        st.image(ImageOps.grayscale(shown).resize((224, 224)), caption="What the model sees (grayscale, 224x224)")
    st.success(f"Predicted class: **{pred}**  ({conf:.1%} confidence)")
    if true_label is not None:
        st.info(f"True label: **{true_label}**  ->  " + ("correct" if true_label == pred else "WRONG"))
    if conf < 0.60:
        st.warning("Low confidence: the image may be unlike the training data, or show an unclear defect.")
    df = pd.DataFrame({"class": names, "probability": probs.numpy()}).sort_values("probability", ascending=False)
    st.bar_chart(df.set_index("class"))
    top3 = df.head(3).copy()
    top3["probability"] = (top3["probability"] * 100).round(1).astype(str) + " %"
    st.dataframe(top3, hide_index=True)


st.title("Textile Surface Defect Classifier")
st.caption("ResNet50 (ImageNet pretrained, fine-tuned) - 9 classes: " + ", ".join(load_class_names(SPLIT_DIR)))
names = load_class_names(SPLIT_DIR)
ckpts = sorted(MODEL_DIR.glob("*finetune*_best.pt"))
if not ckpts:
    st.error(f"No fine-tuned checkpoints found in {MODEL_DIR}. Run train_resnet.py first.")
    st.stop()
default = next((i for i, p in enumerate(ckpts) if p.name == "resnet50_finetune_seed42_best.pt"), 0)
with st.sidebar:
    st.header("Model")
    ck = st.selectbox("Checkpoint", ckpts, index=default, format_func=lambda p: p.stem.replace("_best", ""))
    st.caption("'resnet50_finetune_seedN' = trained on original data. '..._gan_...' = with GAN images added. "
               "'_os' / '_cw' = oversampling / class-weight controls.")
model, dev = load_model(str(ck), len(names))

tab_up, tab_demo = st.tabs(["Upload an image", "Try a held-out test image"])
with tab_up:
    f = st.file_uploader("Upload a fabric image (jpg / png)", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    if f is not None:
        show_result(model, dev, names, ImageOps.exif_transpose(Image.open(f)))
    else:
        st.write("Upload an image to see the predicted defect class.")

with tab_demo:
    seed = re.search(r"seed(\d+)", ck.stem)
    split_csv = SPLIT_DIR / f"split_seed{seed.group(1) if seed else 42}.csv"
    if not split_csv.exists():
        st.warning(f"{split_csv} not found.")
    else:
        df = pd.read_csv(split_csv)
        test = df[df["split"] == "test"]
        st.write("These images were **never used for training** this model (its test split).")
        cls = st.selectbox("True class", sorted(test["class"].unique()))
        sub = test[test["class"] == cls].reset_index(drop=True)
        i = st.slider("Image number", 1, len(sub), 1) - 1 if len(sub) > 1 else 0
        path = DATA_ROOT / sub.loc[i, "path"].replace("\\", "/")
        if path.exists():
            show_result(model, dev, names, Image.open(path), true_label=cls)
        else:
            st.warning(f"Image not found: {path}")

with st.expander("About and limitations"):
    st.markdown(
        "- Trained on one public fabric-defect dataset; on held-out test images from that dataset the fine-tuned "
        "ResNet50 reached about 0.97 macro-F1 (3 random splits).\n"
        "- The model always chooses one of the 9 classes, even for an image that is not fabric (no 'unknown' option).\n"
        "- Images from other cameras, fabrics or resolutions may be classified much less reliably: in this dataset the "
        "image source is itself correlated with the class.\n"
        "- Research prototype - not validated for production quality control.")
