import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import time # Untuk animasi spinner

# --- Pindahkan st.set_page_config() ke sini, sebagai perintah Streamlit pertama ---
# Hapus parameter 'icon' karena menyebabkan masalah kompatibilitas di beberapa lingkungan.
# Streamlit akan menggunakan ikon default.
st.set_page_config(page_title="Klasifikasi Buah Segar/Busuk", layout="centered")

# --- Fungsi untuk memuat file CSS eksternal ---
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"Peringatan: File CSS '{file_name}' tidak ditemukan. Menggunakan gaya default.")
    except Exception as e:
        st.warning(f"Peringatan: Gagal memuat file CSS. Error: {e}")

# Panggil fungsi untuk memuat CSS
load_css('style.css')


# --- 1. Muat Model (di-cache agar tidak memuat ulang setiap kali) ---
@st.cache_resource
def load_fresh_rotten_model():
    model_path = 'fresh_rotten_classifier.h5'
    if not os.path.exists(model_path):
        st.error(f"Error: File model '{model_path}' tidak ditemukan. "
                 "Pastikan model 'fresh_rotten_classifier.h5' berada di direktori yang sama dengan skrip ini.")
        st.stop() # Hentikan eksekusi jika model tidak ditemukan
    try:
        # Muat model Keras yang sudah dilatih.
        # Pastikan input_shape model Anda cocok dengan IMG_TARGET_SIZE.
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        st.stop()

model_fresh_rotten = load_fresh_rotten_model()

# --- 2. Fungsi untuk Memuat dan Memproses Gambar untuk Prediksi ---
# Pastikan target_size sesuai dengan ukuran input model Anda saat pelatihan.
# Contoh: (100, 100) atau (64, 64)
IMG_TARGET_SIZE = (100, 100) # Pastikan ini sesuai dengan input model Anda!
def load_image_for_prediction(img_file):
    # image.load_img dari tf.keras.preprocessing secara otomatis menangani input dari file uploader Streamlit
    img = image.load_img(img_file, target_size=IMG_TARGET_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Tambahkan dimensi batch (misal: (1, 100, 100, 3))
    img_array = img_array / 255.0 # Normalisasi piksel ke rentang [0, 1]
    return img_array

# --- 3. Fungsi untuk Memprediksi Fresh atau Rotten ---
# Fungsi ini mengembalikan label dan tingkat keyakinan model.
def predict_fresh_or_rotten(model, img_file):
    img_array = load_image_for_prediction(img_file)
    prediction = model.predict(img_array)[0][0] # Ambil nilai skalar probabilitas untuk klasifikasi biner

    if prediction > 0.5:
        # Jika probabilitas > 0.5, model memprediksi 'Rotten'.
        # Keyakinan di sini adalah nilai probabilitas itu sendiri.
        label = 'Busuk' # Menggunakan "Busuk" untuk output bahasa Indonesia
        confidence = prediction
    else:
        # Jika probabilitas <= 0.5, model memprediksi 'Fresh'.
        # Keyakinan di sini adalah 1 minus probabilitas 'Rotten'.
        label = 'Segar' # Menggunakan "Segar" untuk output bahasa Indonesia
        confidence = 1 - prediction

    return label, confidence # Mengembalikan label dan tingkat keyakinan

# --- 4. Antarmuka Pengguna Streamlit ---

st.title("Klasifikasi Buah: Segar atau Busuk?")
st.markdown("---") # Garis pemisah untuk estetika

st.write("Unggah gambar buah (seperti apel, pisang, atau jeruk) untuk menentukan apakah buah tersebut segar atau busuk.")

uploaded_file = st.file_uploader("Pilih gambar buah...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah oleh pengguna
    st.image(uploaded_file, caption='Gambar yang Diunggah', use_container_width=True) # Menggunakan use_container_width
    st.write("") # Baris kosong untuk spasi

    # Menampilkan indikator loading saat model memproses
    with st.spinner('Menganalisis gambar buah...'):
        time.sleep(1.5) # Simulasi waktu pemrosesan model (sesuaikan jika model Anda butuh waktu lebih lama)
        # Panggil fungsi prediksi yang sudah diperbaiki
        label, confidence = predict_fresh_or_rotten(model_fresh_rotten, uploaded_file)

    # Menampilkan hasil prediksi beserta tingkat keyakinan dalam persentase
    confidence_percent = confidence * 100 # Konversi probabilitas ke persentase

    if label == 'Segar':
        st.success(f"🎉 **Hasil Prediksi:** Buah ini adalah **{label}** dengan keyakinan **{confidence_percent:.2f}%**!")
    else:
        st.error(f"⚠️ **Hasil Prediksi:** Buah ini adalah **{label}** dengan keyakinan **{confidence_percent:.2f}%**.")

st.markdown("---") # Garis pemisah di bagian bawah
st.markdown("""
<div class="footer">
    Aplikasi Klasifikasi Buah oleh [Nama Anda/Kelompok Anda] | Dibuat dengan Streamlit dan TensorFlow
</div>
""", unsafe_allow_html=True)
