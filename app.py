# Python In-built packages
from pathlib import Path
import PIL
import time

# External packages
import streamlit as st


# Local Modules
import settings
import helper


start_time = time.time()

# Setting page layout
st.set_page_config(
    page_title="Tube Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Tube Detection using YOLOv8")

# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = 'Segmentation'
model_path = Path(settings.SEGMENTATION_MODEL)

#confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
confidence = st.sidebar.slider(
    'Confidence [%]', min_value=0.0, max_value=1.0, value=0.60)


# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

if source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)
    st.info('This will stream the current webcam', icon="⚠️")

elif source_radio == settings.MULTICAMS:
    helper.play_multi_webcam_test(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")

# Display processing time
end_time = time.time()
processing_time = end_time - start_time
st.sidebar.info(f"Processing time: {processing_time:.2f} seconds")
