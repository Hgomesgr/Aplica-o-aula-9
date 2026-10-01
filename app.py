import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detecção de Objetos com YOLO", layout="centered")

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Detecção de Objetos em Tempo Real")
st.write("Faça o upload de uma imagem para identificar objetos como pessoas, carros, animais e mais.")

uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)

    with st.spinner("Processando imagem..."):
        # imgsz=320 reduz significativamente o consumo de RAM no CPU
        results = model.predict(source=img_array, imgsz=320, conf=0.25)
        
        res_plotted = results[0].plot()
        res_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))

    with col2:
        st.subheader("Objetos Detectados")
        st.image(res_image, use_container_width=True)

    # Lista resumo dos objetos identificados
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.markdown("---")
        st.subheader("Resumo da Detecção")
        detected_names = [model.names[int(cls)] for cls in boxes.cls]
        
        counts = {}
        for name in detected_names:
            counts[name] = counts.get(name, 0) + 1
            
        for obj, count in counts.items():
            st.write(f"- **{obj.capitalize()}**: {count}")