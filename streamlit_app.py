"""Alzheimer's Disease Stage Prediction: showcase app (Streamlit).
Six deep learning models + weighted ensemble trained in alzheimer_colab_v3.ipynb.
Research demo only, NOT for medical diagnosis."""
import os
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Alzheimer's Stage Prediction", page_icon="🧠", layout="wide")

# Keras flow_from_dataframe sorts class names alphabetically, so output index
# 0..3 = Mild, Moderate, No, Very Mild (matches the training labels).
CLASS_NAMES = ["Mild Impairment", "Moderate Impairment", "No Impairment", "Very Mild Impairment"]
DRIVE_FOLDER_ID = "1e4Nt-2UO2WUa0tJB92XCTzrDlAQC6VJ_"   # public folder with the 6 .keras files

# key: (display name, preprocessing type, ensemble weight, test accuracy %)
MODELS = {
    "vgg16":        ("VGG16",          "vgg",    0.2054, 93.66),
    "efficientnet": ("EfficientNetB0", "eff",    0.1687, 72.92),
    "cnn":          ("CNN",            "scale",  0.1596, 73.96),
    "resnet50":     ("ResNet50",       "resnet", 0.1563, 69.79),
    "rnn":          ("GRU (RNN)",      "scale",  0.1556, 69.10),
    "lstm":         ("LSTM",           "scale",  0.1544, 68.92),
}
ENSEMBLE_ACC = 88.54


@st.cache_resource(show_spinner="Downloading models (first start only)...")
def download_models():
    os.makedirs("models", exist_ok=True)
    if not all(os.path.exists(f"models/{k}.keras") for k in MODELS):
        import gdown
        gdown.download_folder(id=DRIVE_FOLDER_ID, output="models", quiet=True)
    return True


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    import tensorflow as tf
    return {k: tf.keras.models.load_model(f"models/{k}.keras", compile=False) for k in MODELS}


def preprocess(arr, kind):
    from tensorflow.keras.applications import vgg16, resnet50, efficientnet
    x = arr.copy()
    if kind == "vgg":
        return vgg16.preprocess_input(x)
    if kind == "resnet":
        return resnet50.preprocess_input(x)
    if kind == "eff":
        return efficientnet.preprocess_input(x)
    return x / 255.0


# ---------------- UI ----------------
st.title("🧠 Alzheimer's Disease Stage Prediction")
st.markdown(
    "Comparative analysis of six deep learning models (**CNN, GRU, LSTM, VGG16, ResNet50, "
    "EfficientNetB0**) and a **weighted ensemble** for classifying brain MRI slices into four stages: "
    "No, Very Mild, Mild and Moderate Impairment."
)
st.caption("Final-year project, Department of Computer Science, Adekunle Ajasin University, Akungba-Akoko. "
           "Research demo only, not for medical diagnosis.")

with st.expander("Model performance on the test set (1,152 images)"):
    perf = pd.DataFrame(
        [(v[0], v[3]) for v in MODELS.values()] + [("Weighted Ensemble", ENSEMBLE_ACC)],
        columns=["Model", "Test accuracy (%)"],
    ).sort_values("Test accuracy (%)", ascending=False)
    st.dataframe(perf, hide_index=True, use_container_width=True)

left, right = st.columns([1, 1.4])
with left:
    up = st.file_uploader("Upload a brain MRI slice (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")
        st.image(img, caption="Uploaded MRI", use_container_width=True)

with right:
    if not up:
        st.info("Upload an MRI image to see every model's prediction.")
    else:
        download_models()
        models = load_models()
        arr = np.asarray(img.resize((224, 224)), dtype="float32")[None]
        rows, ens = [], np.zeros(4)
        with st.spinner("Running all six models..."):
            for k, (name, kind, w, acc) in MODELS.items():
                p = models[k].predict(preprocess(arr, kind), verbose=0)[0]
                ens += w * p
                rows.append({"Model": name, "Prediction": CLASS_NAMES[int(p.argmax())],
                             "Confidence": f"{p.max()*100:.1f}%", "Test acc.": f"{acc}%"})
        ens /= sum(v[2] for v in MODELS.values())
        top = int(ens.argmax())
        st.subheader("Ensemble prediction")
        st.metric("Predicted stage", CLASS_NAMES[top], f"{ens[top]*100:.1f}% confidence", delta_color="off")
        st.bar_chart(pd.DataFrame({"Probability": ens}, index=CLASS_NAMES))
        st.subheader("Individual models")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("The ensemble averages the six models' probabilities, weighted by validation accuracy.")
