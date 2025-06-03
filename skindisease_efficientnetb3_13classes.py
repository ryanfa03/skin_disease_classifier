import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import gdown
from io import BytesIO
import zipfile

# === Konfigurasi halaman ===
st.set_page_config(page_title="🧬 Deteksi Penyakit Kulit", layout="wide")

# === Styling CSS ===
st.markdown("""
    <style>
        .main-title {
            font-size: 36px;
            font-weight: bold;
            color: #0d6efd;
        }
        .subtext {
            font-size: 16px;
            color: #6c757d;
        }
        .result-box {
            padding: 1.2rem;
            border-radius: 12px;
            background-color: #e9f5ff;
            border: 1px solid #b6e0fe;
            margin-top: 1rem;
        }
        .prediction-label {
            font-size: 22px;
            font-weight: 600;
            color: #0d6efd;
        }
    </style>
""", unsafe_allow_html=True)

# === Load Model ===
MODEL_PATH = "best_model_EfficientNetB3_fix.keras"
GDRIVE_FILE_ID = "1A2G5fpbw4Xmlvogqj1mYem9hacxeSJoX"

@st.cache_resource
def load_skin_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("📥 Mengunduh model dari Google Drive..."):
                gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

model = load_skin_model()

CLASS_NAMES = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma', 'Eczema', 
               'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 
               'Melanoma Skin Cancer Nevi', 'Nail Fungus and other Nail Disease', 
               'Psoriasis Lichen Planus', 'Seborrheic Keratoses Tumors', 'Urticaria Hives', 
               'Vascular Tumors', 'Vasculitis', 'Warts Molluscum Viral Infections']

# === Fungsi Prediksi ===
from tensorflow.keras.applications.efficientnet import preprocess_input

def predict(img: Image.Image):
    if model is None:
        st.error("❌ Model tidak tersedia. Pastikan file model berhasil dimuat.")
        return "Model Error", 0.0

    img = img.resize((300, 300))  # Pastikan sesuai model
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # Gunakan preprocessing bawaan model
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array, verbose=0)
    st.write("Softmax Output:", preds)  # Debug (opsional)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    return CLASS_NAMES[class_idx], confidence
# === Fungsi Log Prediksi ===
LOG_PATH = "predictions_log.csv"

def log_prediction(filename, label, confidence):
    new_entry = pd.DataFrame([{"filename": filename, "label": label, "confidence": confidence}])
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        df = pd.concat([old, new_entry], ignore_index=True)
    else:
        df = new_entry
    df.to_csv(LOG_PATH, index=False)

# === Layout Utama ===
st.markdown('<div class="main-title">🧬 Aplikasi Deteksi Gambar Penyakit Kulit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Model ini mendeteksi 13 jenis penyakit kulit berdasarkan gambar.</div>', unsafe_allow_html=True)
st.markdown("---")

# === Kolom Layout ===
col1, col2 = st.columns(2)

# === Upload Gambar Tunggal ===
with col1:
    st.subheader("📤 Unggah Gambar Kulit")
    uploaded_file = st.file_uploader("Pilih gambar JPG/JPEG/PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Pratinjau Gambar", use_container_width=True)

        if st.button("🔍 Prediksi Sekarang"):
            label, confidence = predict(img)
            filename = uploaded_file.name if hasattr(uploaded_file, "name") else "unknown.jpg"
            log_prediction(filename, label, confidence)

            # Tampilkan hasil di col2
            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-label">✅ Prediksi: {label}</div>', unsafe_allow_html=True)
                st.progress(confidence / 100)
                st.markdown(f"**Tingkat Keyakinan:** {confidence:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

# === Riwayat Prediksi ===
st.subheader("📋 Riwayat Prediksi")
if os.path.exists(LOG_PATH):
    df_log = pd.read_csv(LOG_PATH)
    st.dataframe(df_log)

# === Batch ZIP Upload ===
with st.expander("📦 Prediksi Batch (ZIP)", expanded=False):
    st.write("Unggah file ZIP berisi gambar untuk diproses secara massal.")
    batch_file = st.file_uploader("Unggah ZIP", type=["zip"], key="zip_uploader")

    if batch_file:
        with zipfile.ZipFile(BytesIO(batch_file.read())) as archive:
            image_files = [f for f in archive.namelist() if f.lower().endswith(('jpg', 'jpeg', 'png'))]

            if not image_files:
                st.warning("❗ ZIP tidak berisi file gambar yang valid.")
            else:
                st.success(f"📁 Ditemukan {len(image_files)} gambar dalam ZIP.")
                results = []

                for image_file in image_files:
                    with archive.open(image_file) as img_file:
                        try:
                            img = Image.open(img_file).convert("RGB")
                            label, confidence = predict(img)
                            results.append({"filename": image_file, "label": label, "confidence": confidence})
                            log_prediction(image_file, label, confidence)
                        except Exception as e:
                            st.error(f"❌ Gagal memproses {image_file}: {e}")

                if results:
                    st.write("📊 Hasil Prediksi Batch:")
                    st.dataframe(pd.DataFrame(results))
