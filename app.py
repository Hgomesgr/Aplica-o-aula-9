import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Detecção de Objetos", layout="centered")

@st.cache_resource
def load_model():
    # Carrega modelo leve otimizado para CPU
    return YOLO("yolov8n.pt")

model = load_model()

def process_and_display(img_input):
    """Função centralizada para processar e exibir os resultados"""
    image = Image.open(img_input).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Entrada")
        st.image(image, use_container_width=True)

    with st.spinner("Detectando objetos..."):
        # Processa em tamanho reduzido (320px) para economizar memória RAM do Render
        results = model.predict(source=img_array, imgsz=320, conf=0.25)
        
        res_plotted = results[0].plot()
        res_image = Image.fromarray(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))

    with col2:
        st.subheader("Resultado")
        st.image(res_image, use_container_width=True)

    # Exibe resumo numérico das detecções
    boxes = results[0].boxes
    if len(boxes) > 0:
        st.markdown("---")
        st.subheader("Objetos Identificados")
        detected_names = [model.names[int(cls)] for cls in boxes.cls]
        
        counts = {}
        for name in detected_names:
            counts[name] = counts.get(name, 0) + 1
            
        for obj, count in counts.items():
            st.write(f"- **{obj.capitalize()}**: {count}")
    else:
        st.info("Nenhum objeto identificado com o nível de confiança atual.")

st.title("Detector de Objetos (YOLOv8)")
st.write("Escolha o modo de captura abaixo:")

# Organização por Abas
tab1, tab2 = st.tabs(["📷 Câmera", "📁 Upload de Imagem"])

with tab1:
    st.caption("Tire uma foto com a sua câmera para analisar:")
    camera_file = st.camera_input("Capturar foto")
    if camera_file:
        process_and_display(camera_file)

with tab2:
    st.caption("Envie um arquivo de imagem do seu dispositivo:")
    uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        process_and_display(uploaded_file)
