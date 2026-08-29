import streamlit as st
import cv2
import numpy as np
import pandas as pd
import sqlite3
import urllib.request
import os
import pickle
from datetime import datetime

# ==========================================
# 1. SETUP DATABASE SQLITE
# ==========================================
def init_db():
    conn = sqlite3.connect('absensi.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS data_absen 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  nama TEXT, 
                  waktu DATETIME,
                  status TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS data_karyawan
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  nama TEXT UNIQUE)''')
    conn.commit()
    return conn

conn = init_db()

# ==========================================
# 2. LOAD MODEL DETEKSI & REKOGNISI
# ==========================================
MODEL_FILE = "face_recognizer.yml"
LABELS_FILE = "labels.pkl"
DATASET_DIR = "dataset_wajah"

@st.cache_resource
def load_cascade():
    xml_path = "haarcascade_frontalface_default.xml"
    # Jika file berada di folder Absensi atau root
    if not os.path.exists(xml_path):
        xml_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(xml_path)

face_cascade = load_cascade()

def train_from_dataset_folder():
    """Membaca folder dataset_wajah dan melatih model secara otomatis"""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR, exist_ok=True)
        return False, "Folder dataset_wajah baru saja dibuat. Silakan masukkan folder foto karyawan ke dalamnya."

    faces = []
    labels = []
    label_dict = {}
    current_id = 0
    karyawan_terproses = []

    # Iterasi setiap folder nama karyawan
    for user_name in os.listdir(DATASET_DIR):
        user_path = os.path.join(DATASET_DIR, user_name)
        if not os.path.isdir(user_path):
            continue
        
        if user_name not in label_dict:
            label_dict[user_name] = current_id
            current_id += 1
            
        target_label = label_dict[user_name]
        foto_count = 0

        # Baca semua gambar dalam folder karyawan
        for img_name in os.listdir(user_path):
            img_path = os.path.join(user_path, img_name)
            # Baca gambar dalam Grayscale
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is not None:
                # Deteksi wajah di dalam foto
                detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
                if len(detected_faces) > 0:
                    for (x, y, w, h) in detected_faces:
                        face_roi = gray[y:y+h, x:x+w]
                        face_resized = cv2.resize(face_roi, (200, 200))
                        faces.append(face_resized)
                        labels.append(target_label)
                        foto_count += 1
                else:
                    # Jika tidak terdeteksi via cascade, gunakan seluruh crop gambar
                    face_resized = cv2.resize(gray, (200, 200))
                    faces.append(face_resized)
                    labels.append(target_label)
                    foto_count += 1

        if foto_count > 0:
            karyawan_terproses.append(f"{user_name} ({foto_count} foto)")
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO data_karyawan (nama) VALUES (?)", (user_name,))
            conn.commit()

    if len(faces) == 0:
        return False, "Tidak ada foto yang ditemukan dalam folder dataset_wajah."

    # Latih model LBPH
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))
        recognizer.save(MODEL_FILE)
        with open(LABELS_FILE, "wb") as f:
            pickle.dump(label_dict, f)
        
        msg = f"Berhasil melatih AI dengan {len(karyawan_terproses)} karyawan: " + ", ".join(karyawan_terproses)
        return True, msg
    except AttributeError:
        return False, "Error: Module `opencv-contrib-python` belum diinstall. Jalankan `pip install opencv-contrib-python` di CMD."

def get_recognizer():
    if os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE):
        try:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(MODEL_FILE)
            with open(LABELS_FILE, "rb") as f:
                label_dict = pickle.load(f)
            id_to_name = {v: k for k, v in label_dict.items()}
            return recognizer, id_to_name
        except Exception:
            return None, {}
    return None, {}

# ==========================================
# 3. CONFIG STREAMLIT UI
# ==========================================
st.set_page_config(page_title="Absensi Realtime Face Recognition", layout="wide")

st.sidebar.title("📌 Menu Navigasi")
menu = st.sidebar.radio("Pilih Halaman", [
    "📷 Scan Absen Otomatis", 
    "📁 Kelola & Latih Database Foto", 
    "📊 Dashboard Admin"
])

# ------------------------------------------
# HALAMAN 1: SCAN ABSEN OTOMATIS
# ------------------------------------------
if menu == "📷 Scan Absen Otomatis":
    st.title("📱 Face Recognition Attendance")
    st.write("Arahkan wajah ke kamera. Sistem akan **otomatis mengenali nama Anda**.")
    
    recognizer, id_to_name = get_recognizer()
    
    if recognizer is None or not id_to_name:
        st.warning("⚠️ Belum ada model wajah terdaftar. Buka menu **📁 Kelola & Latih Database Foto** untuk memproses foto karyawan.")
    
    kamera = st.camera_input("Ambil Foto Wajah untuk Absen")
    
    if kamera is not None:
        bytes_data = kamera.getvalue()
        file_bytes = np.frombuffer(bytes_data, np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (200, 200))
            
            if recognizer is not None and id_to_name:
                label_id, confidence = recognizer.predict(face_resized)
                
                # Confidence threshold (makin kecil makin akurat)
                if confidence < 90 and label_id in id_to_name:
                    nama_terdeteksi = id_to_name[label_id]
                    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    c = conn.cursor()
                    c.execute("INSERT INTO data_absen (nama, waktu, status) VALUES (?, ?, ?)", 
                              (nama_terdeteksi, waktu_sekarang, "Hadir"))
                    conn.commit()
                    
                    st.balloons()
                    st.success(f"✅ **Absen Berhasil!**\n\nNama: **{nama_terdeteksi}**\nAkurasi Kemiripan: **{round(max(0, 100 - confidence), 1)}%**\nWaktu: **{waktu_sekarang}**")
                else:
                    st.error("❌ Wajah tidak terdaftar dalam database. Silakan daftarkan foto Anda terlebih dahulu.")
            else:
                st.warning("Model AI belum dilatih.")
        else:
            st.error("❌ Wajah tidak terdeteksi! Harap posisikan wajah menghadap ke kamera.")

# ------------------------------------------
# HALAMAN 2: KELOLA & LATIH DATABASE FOTO
# ------------------------------------------
elif menu == "📁 Kelola & Latih Database Foto":
    st.title("📁 Import Foto Karyawan & Pelatihan AI")
    st.info("💡 **Cara Penggunaan:** Masukkan foto-foto karyawan ke folder `dataset_wajah/[Nama_Karyawan]/` di laptop Anda, lalu klik tombol di bawah.")
    
    if st.button("⚡ Sinkronkan & Latih Data Foto Sekarang", type="primary"):
        with st.spinner("Sedang memproses dan melatih data foto karyawan..."):
            success, message = train_from_dataset_folder()
            st.cache_resource.clear()
            if success:
                st.success(f"🎉 **Sukses!** {message}")
            else:
                st.error(f"❌ {message}")
                
    st.divider()
    st.subheader("📋 Daftar Karyawan Terdaftar di System Database")
    df_karyawan = pd.read_sql_query("SELECT * FROM data_karyawan", conn)
    if not df_karyawan.empty:
        st.dataframe(df_karyawan[['id', 'nama']], use_container_width=True, hide_index=True)
    else:
        st.write("Belum ada data karyawan terdaftar.")

# ------------------------------------------
# HALAMAN 3: DASHBOARD ADMIN
# ------------------------------------------
elif menu == "📊 Dashboard Admin":
    st.title("📊 Real-time Attendance Dashboard")
    
    if st.button("🔄 Refresh Data"):
        st.rerun()
        
    df_absen = pd.read_sql_query("SELECT * FROM data_absen ORDER BY waktu DESC", conn)
    df_karyawan = pd.read_sql_query("SELECT * FROM data_karyawan", conn)
    
    if not df_absen.empty:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Absensi Recorded", len(df_absen))
        col2.metric("Total Karyawan Terdaftar", len(df_karyawan))
        col3.metric("Status Server", "Aktif 🟢")
        
        st.divider()
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("📋 Log Absensi Terbaru")
            st.dataframe(df_absen[['nama', 'waktu', 'status']], use_container_width=True, hide_index=True)
        with c2:
            st.subheader("📈 Grafis Kehadiran")
            df_absen['waktu_dt'] = pd.to_datetime(df_absen['waktu'])
            df_absen['jam'] = df_absen['waktu_dt'].dt.hour
            chart_data = df_absen.groupby('jam').size()
            st.bar_chart(chart_data)
    else:
        st.info("Belum ada log absensi yang tersimpan hari ini.")
