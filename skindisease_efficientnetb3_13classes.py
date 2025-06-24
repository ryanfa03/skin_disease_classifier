import streamlit as st
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model# Bagian dalam with st.expander(...) sebelumnya

with st.expander("📦 Prediksi Batch (ZIP)", expanded=False):
    st.write("Unggah file ZIP berisi kumpulan gambar untuk diprediksi sekaligus.")
    batch_file = st.file_uploader("Unggah ZIP", type=["zip"])

    if batch_file is not None:
        with zipfile.ZipFile(BytesIO(batch_file.read())) as archive:
            image_files = [f for f in archive.namelist() if f.endswith(('jpg', 'jpeg', 'png'))]
            st.write(f"📁 Ditemukan {len(image_files)} gambar dalam ZIP.")
            
            # Simpan hasil prediksi batch dalam DataFrame
            batch_results = []

            for image_file in image_files:
                with archive.open(image_file) as img_file:
                    img = Image.open(img_file).convert("RGB")
                    label, confidence, _ = predict(img)
                    batch_results.append({
                        "Nama File": image_file,
                        "Kelas Prediksi": label,
                        "Confidence (%)": f"{confidence:.2f}"
                    })

            # Tampilkan sebagai tabel
            df_batch = pd.DataFrame(batch_results)
            st.subheader("📊 Tabel Hasil Batch Prediksi")
            st.dataframe(df_batch, use_container_width=True)

            # Simpan sebagai CSV opsional (jika dibutuhkan)
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh Hasil Batch (.csv)",
                data=csv,
                file_name='hasil_batch_prediksi.csv',
                mime='text/csv'
            )

from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import gdown
from io import BytesIO
import zipfile

# === Konfigurasi halaman ===
st.set_page_config(page_title="🧬 Deteksi Penyakit Kulit", layout="wide")

# === Styling CSS ===
st.markdown("""
    <style>
        body {
            background-color: #f8f9fa;
        }
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
        .confidence-bar .stProgress > div > div {
            background-color: #0d6efd;
        }
    </style>
""", unsafe_allow_html=True)

# === Load Model ===
MODEL_PATH = "best_model_EfficientNetB3_fix.keras" 
GDRIVE_FILE_ID = "1A2G5fpbw4Xmlvogqj1mYem9hacxeSJoX"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Mengunduh model dari Google Drive..."):
        gdown.download(id=GDRIVE_FILE_ID, output=MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)
CLASS_NAMES = ['Acne and Rosacea', 'Actinic Keratosis Basal Cell Carcinoma', 'Eczema', 'Exanthems and Drug Eruptions', 'Hair Loss Alopecia', 'Melanoma Skin Cancer Nevi', 'Nail Fungus and other Nail Disease', 'Psoriasis Lichen Planus', 'Seborrheic Keratoses Tumors', 'Urticaria Hives', 'Vascular Tumors', 'Vasculitis', 'Warts Molluscum Viral Infections']

# === Fungsi Prediksi ===
def predict(img):
    img = img.resize((300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    return CLASS_NAMES[class_idx], confidence, img_array

def log_prediction(filename, label, confidence):
    log_data.append({"filename": filename, "label": label, "confidence": confidence})
    df = pd.DataFrame(log_data)
    df.to_csv('predictions_log.csv', index=False)

log_data = []

# === Layout Utama ===
st.markdown('<div class="main-title">🧬 Aplikasi Deteksi Gambar Penyakit Kulit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext"> Model ini dirancang untuk melakukan klasifikasi otomatis terhadap 13 kategori penyakit kulit, yaitu: Acne and Rosacea, Actinic Keratosis Basal Cell Carcinoma, Eczema, Exanthems and Drug Eruptions, Hair Loss Alopecia, Melanoma Cancer Skin Nevi, Nail Fungus and Other Nail Diseases, Psoriasis Lichen Planus, Seborrheic Keratoses Tumors, Urticaria Hives, Vascular Tumors, Vasculitis, serta Warts Molluscum Viral Infections.</div>', unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Unggah Gambar Kulit")
    uploaded_file = st.file_uploader("Pilih gambar JPG/JPEG/PNG", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Pratinjau Gambar", use_container_width=True)

        if st.button("🔍 Prediksi Sekarang"):
            label, confidence, _ = predict(img)
            log_prediction(uploaded_file.name, label, confidence)

            with col2:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-label">✅ {label}</div>', unsafe_allow_html=True)
                st.progress(confidence / 100)
                st.markdown(f"*Tingkat Keyakinan:* {confidence:.2f}%")
                st.markdown('</div>', unsafe_allow_html=True)

                st.subheader("📋 Riwayat Prediksi")
                if os.path.exists('predictions_log.csv'):
                    df = pd.read_csv('predictions_log.csv')
                    st.dataframe(df)

with st.expander("📦 Prediksi Batch (ZIP)", expanded=False):
    st.write("Unggah file ZIP berisi kumpulan gambar untuk diprediksi sekaligus.")
    batch_file = st.file_uploader("Unggah ZIP", type=["zip"])

    if batch_file is not None:
        with zipfile.ZipFile(BytesIO(batch_file.read())) as archive:
            image_files = [f for f in archive.namelist() if f.endswith(('jpg', 'jpeg', 'png'))]
            st.write(f"📁 Ditemukan {len(image_files)} gambar dalam ZIP.")
            
            # Simpan hasil prediksi batch dalam DataFrame
            batch_results = []

            for image_file in image_files:
                with archive.open(image_file) as img_file:
                    img = Image.open(img_file).convert("RGB")
                    label, confidence, _ = predict(img)
                    batch_results.append({
                        "Nama File": image_file,
                        "Kelas Prediksi": label,
                        "Confidence (%)": f"{confidence:.2f}"
                    })

            # Tampilkan sebagai tabel
            df_batch = pd.DataFrame(batch_results)
            st.subheader("📊 Tabel Hasil Batch Prediksi")
            st.dataframe(df_batch, use_container_width=True)

            # Simpan sebagai CSV opsional (jika dibutuhkan)
            csv = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Unduh Hasil Batch (.csv)",
                data=csv,
                file_name='hasil_batch_prediksi.csv',
                mime='text/csv'
            )
