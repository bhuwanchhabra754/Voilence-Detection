import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import cv2

try:
    import tensorflow as tf
except Exception:
    tf = None


def load_model(path: str):
    if tf is None:
        return None, "tensorflow-not-installed"
    try:
        model = tf.keras.models.load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess_image(img: Image.Image, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


def predict_image(model, img_arr):
    preds = model.predict(img_arr)
    # Handle single-value output or two-class softmax
    if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[1] == 1):
        prob = float(preds.flatten()[-1])
        # assume prob is probability of 'Violence'
        return {
            "Violence": prob,
            "NonViolence": 1.0 - prob,
        }
    else:
        # multi-class output
        probs = preds.flatten()
        # if model has 2 outputs, assume [NonViolence, Violence]
        if probs.size == 2:
            return {"NonViolence": float(probs[0]), "Violence": float(probs[1])}
        else:
            # fallback: map argmax to 'label_X'
            idx = int(np.argmax(probs))
            return {f"label_{i}": float(p) for i, p in enumerate(probs)}


def analyze_video(model, video_path, target_size=(224, 224), sample_seconds=1.0, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "cannot-open-video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    sample_every = max(1, int(round(fps * sample_seconds)))
    frame_count = 0
    sampled = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_every == 0:
            # convert BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            arr = preprocess_image(pil, target_size)
            preds = predict_image(model, arr)
            results.append({"frame_index": frame_count, "preds": preds, "thumbnail": rgb})
            sampled += 1
            if sampled >= max_frames:
                break
        frame_count += 1

    cap.release()
    return results, None


def main():
    st.set_page_config(page_title="Violence Detection", layout="wide")
    st.title("Violence Detection — Streamlit UI")

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Model path", value="violence_detection_model.h5")
    threshold = st.sidebar.slider("Violence probability threshold", 0.0, 1.0, 0.5)
    sample_seconds = st.sidebar.slider("Video sample interval (s)", 0.1, 5.0, 1.0)
    max_frames = st.sidebar.number_input("Max sampled frames (video)", min_value=1, max_value=500, value=60)

    model, err = load_model(model_path)
    if model is None:
        if err == "tensorflow-not-installed":
            st.error("TensorFlow is not installed. Install dependencies listed in `requirements.txt`.")
        else:
            st.warning(f"Model could not be loaded from `{model_path}`: {err}")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Upload Image")
        uploaded_img = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="img_uploader")
        if uploaded_img is not None:
            img = Image.open(uploaded_img)
            st.image(img, caption="Uploaded image", use_column_width=True)
            if model is not None:
                target_size = tuple(model.input_shape[1:3]) if model.input_shape and len(model.input_shape) >= 3 else (224, 224)
                arr = preprocess_image(img, target_size)
                preds = predict_image(model, arr)
                # show results
                st.subheader("Prediction")
                for k, v in preds.items():
                    st.write(f"{k}: {v:.3f}")
                violence_prob = preds.get("Violence") or preds.get("label_1") or 0.0
                if float(violence_prob) >= threshold:
                    st.warning(f"Violence detected (p={float(violence_prob):.3f})")
                else:
                    st.success(f"No violence detected (p={float(violence_prob):.3f})")

    with col2:
        st.header("Upload Video")
        uploaded_vid = st.file_uploader("Choose a video", type=["mp4", "mov", "avi", "mkv"], key="vid_uploader")
        if uploaded_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                tfile.write(uploaded_vid.read())
                tfile.flush()
                st.video(tfile.name)
                if model is not None:
                    target_size = tuple(model.input_shape[1:3]) if model.input_shape and len(model.input_shape) >= 3 else (224, 224)
                    with st.spinner("Analyzing video — sampling frames..."):
                        results, verr = analyze_video(model, tfile.name, target_size=target_size, sample_seconds=sample_seconds, max_frames=int(max_frames))
                    if results is None:
                        st.error(f"Video analysis failed: {verr}")
                    else:
                        # summarize
                        violence_counts = 0
                        total = len(results)
                        for r in results:
                            p = r["preds"].get("Violence") or r["preds"].get("label_1") or 0.0
                            if float(p) >= threshold:
                                violence_counts += 1

                        st.subheader("Video analysis summary")
                        st.write(f"Sampled frames: {total}")
                        st.write(f"Frames above threshold: {violence_counts}")
                        if total > 0:
                            st.write(f"Violence ratio: {violence_counts/total:.2%}")

                        # show a few sampled frames and their predictions
                        st.markdown("---")
                        for i, r in enumerate(results[:12]):
                            st.write(f"Frame {r['frame_index']}:")
                            st.image(r["thumbnail"], width=240)
                            for k, v in r["preds"].items():
                                st.write(f"- {k}: {v:.3f}")

            finally:
                try:
                    os.unlink(tfile.name)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
