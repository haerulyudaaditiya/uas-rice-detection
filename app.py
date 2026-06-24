import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Coba import YOLO dari ultralytics, jika gagal gunakan demo mode
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Konfigurasi Halaman Streamlit - Clean & Simple layout
st.set_page_config(
    page_title="RiceDetect AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Judul Utama Aplikasi - Bersih & Minimalis
st.title("RiceDetect AI")
st.markdown("Sistem Deteksi Objek dan Klasifikasi Kualitas serta Varietas Beras")
st.markdown("---")

# Definisikan kelas beras
CLASSES = ['beras_gundukan', 'beras_patah', 'beras_utuh', 'ir42', 'ir64', 'ketan', 'pandan']
CLASS_COLORS = {
    'beras_gundukan': (255, 165, 0),    # Orange
    'beras_patah': (255, 0, 0),         # Merah
    'beras_utuh': (0, 255, 0),          # Hijau
    'ir42': (0, 191, 255),              # Biru Muda
    'ir64': (30, 144, 255),             # Biru Tua
    'ketan': (238, 130, 238),           # Violet
    'pandan': (144, 238, 144)           # Hijau Muda
}

# Sidebar untuk pengaturan model
st.sidebar.header("Pengaturan Model")

# Cek keberadaan model
weights_filenames = ["best.pt", "best_model.pt", "best_rice_model.pt"]
weights_path = None
for fname in weights_filenames:
    if os.path.exists(fname):
        weights_path = fname
        break

model_loaded = False

if weights_path:
    if YOLO_AVAILABLE:
        try:
            model = YOLO(weights_path)
            model_loaded = True
            st.sidebar.success(f"Model '{weights_path}' aktif.")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat model: {e}")
    else:
        st.sidebar.warning("Pustaka ultralytics belum terpasang. Mode Demo aktif.")
else:
    st.sidebar.info("Model weights belum terdeteksi.")

# Upload file weights via UI di sidebar
uploaded_weights = st.sidebar.file_uploader("Unggah file model (best.pt)", type=["pt"])
if uploaded_weights is not None:
    with open("best.pt", "wb") as f:
        f.write(uploaded_weights.getbuffer())
    st.sidebar.success("Model berhasil diunggah! Silakan refresh halaman.")
    st.rerun()

# Threshold Sliders
conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)
iou_threshold = st.sidebar.slider("IOU Threshold (NMS)", min_value=0.1, max_value=1.0, value=0.45, step=0.05)

# Main Page Layout: Dua Kolom Utama
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("Input Citra Beras")
    input_source = st.radio("Pilih Metode Input:", ["Unggah File", "Gunakan Kamera"], horizontal=True)
    
    img_rgb = None
    uploaded_file = None
    
    if input_source == "Unggah File":
        uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
    elif input_source == "Gunakan Kamera":
        uploaded_file = st.camera_input("Ambil foto...")
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img_rgb is not None:
        st.write("")
        st.subheader("Hasil Visualisasi Deteksi")
        
        if model_loaded:
            # Inference dengan model YOLOv8 riil
            results = model.predict(source=img_rgb, conf=conf_threshold, iou=iou_threshold)
            result = results[0]
            
            # Anotasi citra
            annotated_img = result.plot()
            annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
            
            # Hitung jumlah per kelas
            class_counts = {c: 0 for c in CLASSES}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = CLASSES[cls_id]
                class_counts[cls_name] += 1
                
            st.image(annotated_img_rgb, caption="Hasil Deteksi Model YOLOv8", width="stretch")
        else:
            # Mode simulasi (jika model tidak dimuat)
            st.warning("Menampilkan hasil simulasi (Mode Demo Aktif).")
            sim_img = img_rgb.copy()
            h, w, _ = sim_img.shape
            
            # Koordinat box simulasi relatif
            mock_boxes = [
                (int(w*0.25), int(h*0.3), int(w*0.35), int(h*0.4), 'beras_utuh', 0.92),
                (int(w*0.4), int(h*0.32), int(w*0.48), int(h*0.42), 'beras_utuh', 0.88),
                (int(w*0.3), int(h*0.5), int(w*0.38), int(h*0.56), 'beras_patah', 0.76),
                (int(w*0.55), int(h*0.45), int(w*0.63), int(h*0.53), 'beras_patah', 0.82),
                (int(w*0.62), int(h*0.25), int(w*0.72), int(h*0.35), 'ir64', 0.89),
                (int(w*0.15), int(h*0.6), int(w*0.28), int(h*0.72), 'pandan', 0.91),
                (int(w*0.45), int(h*0.65), int(w*0.75), int(h*0.88), 'beras_gundukan', 0.95),
            ]
            
            class_counts = {c: 0 for c in CLASSES}
            for x1, y1, x2, y2, cls_name, conf in mock_boxes:
                class_counts[cls_name] += 1
                color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                cv2.rectangle(sim_img, (x1, y1), (x2, y2), color, 3)
                label_text = f"{cls_name} {conf:.2f}"
                (txt_w, txt_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(sim_img, (x1, y1 - txt_h - 10), (x1 + txt_w, y1), color, -1)
                cv2.putText(sim_img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
            st.image(sim_img, caption="Hasil Deteksi Simulasi (Demo)", width="stretch")

with col2:
    st.subheader("Rincian Deteksi")
    
    if img_rgb is not None:
        df_data = []
        for cls, count in class_counts.items():
            if count > 0:
                cat = "Kualitas" if cls in ['beras_utuh', 'beras_patah', 'beras_gundukan'] else "Varietas"
                df_data.append({"Kategori": cat, "Kelas Objek": cls, "Jumlah": count})
        
        if df_data:
            st.dataframe(df_data, width="stretch", hide_index=True)
        else:
            st.info("Tidak ada objek yang terdeteksi.")
    else:
        st.info("Unggah gambar atau gunakan kamera di sebelah kiri untuk melihat rincian deteksi.")

st.markdown("---")
st.markdown("**Petunjuk Singkat:** Pilih metode input di sebelah kiri, unggah atau ambil gambar butiran beras, dan rincian deteksi akan ditampilkan secara otomatis.")
