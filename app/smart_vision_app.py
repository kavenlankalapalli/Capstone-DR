# smart_vision_app.py

import streamlit as st
from PIL import Image
import numpy as np
import time
import io
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from login import login_form
# import tensorflow as tf

# Loading model
@st.cache_resource
def load_model():
    model_path = '/Users/kalyanlankalapalli/documents/gcu/milestone-3/dr_model.keras'
    model = tf.keras.models.load_model(model_path)
    return model

# model = load_model()
# Set class labels
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']   

def predict_diabetic_retinopathy(processed):
    #stages = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[predicted_class] * 100  # Convert to percentage
    st.success(f"Prediction: **{class_labels[predicted_class]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
    st.success(f"Prediction: **{class_labels[predicted_class]}**")
    return np.random.choice(stages), confidence

def apply_preprocessing(image, grayscale=False, resize=False):
    if grayscale:
        image = image.convert("L")
    if resize:
        image = image.resize((224, 224))
    return image

# pre processing image 
def preprocess_image(image_array, target_size=(224, 224)):
    #img = cv2.imread(image_path)
    # Convert PIL image to NumPy array
    image_np = np.array(image_array)
    if image_array is None:
        return None  # Handle corrupted files

    # Convert RGB to BGR (OpenCV expects BGR)
    if image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    

    img = cv2.resize(image_np, target_size)

    # Convert to grayscale for CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Convert back to BGR for model (if needed)
    processed = cv2.merge([enhanced, enhanced, enhanced])
    processed = processed / 255.0  # Normalize
    return processed

# -------------------------------
# Set Page Config
# -------------------------------
st.set_page_config(
    page_title="Smart Retina - Diabetic Eye Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("Smart Vision")
app_mode = st.sidebar.radio("Navigation", ["Home", "Upload Image", "Preprocessing", "Analysis", "Report"])

# Global session state for uploaded image
if 'raw_image' not in st.session_state:
    st.session_state.raw_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# -------------------------------
# Home Screen
# -------------------------------
if app_mode == "Home":
    st.title("Smart Retina – Diabetic Eye Disease Detection System")
    st.markdown("""
    Welcome to **Smart Retina**, an AI-powered platform designed to assist in the early detection of diabetic retinopathy using retinal images.

    This tool helps clinicians and researchers:
    - Upload and view retinal images
    - Preprocess and enhance image data
    - Run deep learning analysis for diagnosis
    - Generate detailed diagnostic reports

    Use the sidebar to navigate through each step.
    """)
    col1, col2, col3 = st.columns([5, 2, 1])
    with col2:
     if st.button("Login"):
        st.session_state.show_login = True
        #st.write("Button was clicked!")  # This acts like an event handler
        login_form()

# -------------------------------
# Upload Image
# -------------------------------
elif app_mode == "Upload Image":
    st.header("Upload Retinal Images")
    uploaded_file = st.file_uploader("Choose a retinal image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.raw_image = image
        col1 = st.columns(1)[0]
        with col1:
            st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
        #st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
        st.success("Image successfully uploaded!")

# -------------------------------
# Preprocessing
# -------------------------------
elif app_mode == "Preprocessing":
    st.header("Preprocessing Options")
    if st.session_state.raw_image:
        grayscale = st.checkbox("Convert to Grayscale")
        resize = st.checkbox("Resize to 224x224")

        #processed = apply_preprocessing(st.session_state.raw_image.copy(), grayscale, resize)
        processed=preprocess_image(st.session_state.raw_image.copy())
        st.session_state.processed_image = processed

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.raw_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed, caption="Processed Image", use_container_width=True)
    else:
        st.warning("Please upload an image first in the 'Upload Image' section.")

# -------------------------------
# Analysis
# -------------------------------
elif app_mode == "Analysis":
    st.header("Run Model Analysis")
    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="Analyzing Processed Image", use_column_width=True)

        with st.spinner('Running prediction...'):
            time.sleep(2)  # simulate delay
            stage, confidence = predict_diabetic_retinopathy(np.array(st.session_state.processed_image))

        st.metric(label="Predicted Stage", value=stage)
        st.metric(label="Confidence", value=f"{confidence * 100}%")

        st.info("Heatmap visualization (Grad-CAM) feature is under development.")
    else:
        st.warning("Please preprocess an image before analysis.")

# -------------------------------
# Report
# -------------------------------
elif app_mode == "Report":
    st.header("Download Report")
    if st.session_state.raw_image:
        buffer = io.BytesIO()
        st.session_state.raw_image.save(buffer, format="PNG")
        byte_img = buffer.getvalue()

        st.download_button(
            label="Download Raw Image",
            data=byte_img,
            file_name="retinal_image.png",
            mime="image/png"
        )

        st.info("Additional report generation in PDF/CSV format will be added here.")
    else:
        st.warning("No image available. Please upload and analyze an image first.")

# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Smart Vision – Capstone Project")
