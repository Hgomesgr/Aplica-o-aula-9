import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

st.set_page_config(page_title="YOLO Tempo Real", layout="centered")

@st.cache_resource
def load_model():
    # Modelo ultraleve para manter alta taxa de frames em CPU
    return YOLO("yolov8n.pt")

model = load_model()

st.title("Detecção de Objetos com Câmera")
st.write("Permita o acesso à sua câmera para detectar objetos em tempo real.")

# Configuração de servidores STUN para conexão WebRTC no Render
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class YOLOVideoProcessor(VideoTransformerBase):
    def transform(self, frame):
        # Converte o frame do WebRTC para array OpenCV BGR
        img = frame.to_ndarray(format="bgr24")

        # Processa a imagem com dimensão reduzida (imgsz=320) para economizar RAM e CPU
        results = model.predict(source=img, imgsz=320, conf=0.3, verbose=False)
        
        # Desenha os Bounding Boxes na imagem
        annotated_frame = results[0].plot()

        return annotated_frame

webrtc_streamer(
    key="yolo-realtime",
    video_processor_factory=YOLOVideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
