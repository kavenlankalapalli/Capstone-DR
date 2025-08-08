import streamlit as st
from PIL import Image
import numpy as np
import time
import io
import cv2
import os
import tensorflow as tf
from session_manager import initialize_session
from session_manager import update_patient_info
from session_manager import display_patient_summary
from session_manager import clear_patient_session
from report import generate_report
#import gdown
from tensorflow.keras.models import load_model
#from keras.models import load_model


initialize_session()
#import matplotlib.pyplot as plt
# from login import login_form
# import tensorflow as tf

# -------------------------------
# Session State Initialization
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Home"

if "patient_mode" not in st.session_state:
    st.session_state.patient_mode = "search"

if "raw_image" not in st.session_state:
    st.session_state.raw_image = None

if "processed_image" not in st.session_state:
    st.session_state.processed_image = None

# -------------------------------
# Model Loading
# -------------------------------

#if os.path.exists("report.py"):
   # model = load_model(model_path)
#    print("report file found")
#else:
   # raise FileNotFoundError(f"Model not found at: {model_path}")
model_path = "dr_model-4.keras"
#model_path = os.path.join("..", "model", "dr_model-4.keras")
# Optional: check the file if it exists
#if os.path.exists(model_path):
#    model = load_model(model_path)
#else:
#    raise FileNotFoundError(f"Model not found at: {model_path}")
@st.cache_resource

def load_model(model_path):
    
    #model_path = '/Users/kalyanlankalapalli/documents/gcu/milestone-3/dr_model.keras'
    file_id = "14b1NASH8S7JGaMc4z5gmo-7CXrkKqk5l"
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    # Download the model only if it doesn't already exist
    #if not os.path.exists(model_path):
        #url = f"https://drive.google.com/uc?id={file_id}"
        #gdown.download(url, model_path, quiet=False)
    model = tf.keras.models.load_model(model_path)
    #model = None  # Placeholder
    return model

#model = None  # 
#model = load_model(model_path)
class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']

def predict_diabetic_retinopathy(processed):
    prediction = model.predict(processed)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[predicted_class] * 100
    st.success(f"Prediction: **{class_labels[predicted_class]}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
    return class_labels[predicted_class], confidence

# -------------------------------
# Preprocessing Functions
# -------------------------------
def preprocess_image(image_array, target_size=(224, 224)):
    image_np = np.array(image_array)
    if image_array is None:
        return None
    if image_np.shape[-1] == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    img = cv2.resize(image_np, target_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    processed = cv2.merge([enhanced, enhanced, enhanced])
    processed = processed / 255.0
    return processed

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Smart Retina - Diabetic Eye Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Sidebar Navigation with Equal Buttons
# -------------------------------
st.sidebar.title("Smart Vision")

if "app_mode" not in st.session_state:
    st.session_state.app_mode = "Home1"

sidebar_buttons = {
    "🏠 Home": "Home",
    "🧍 Patient Details": "Patient Details",
    "📤 Upload Image": "Upload Image",
    "🧪 Preprocessing": "Preprocessing",
    "📊 Analysis": "Analysis",
    "📝 Report": "Report",
    "❓ Help": "Help"
}
if st.session_state.logged_in:
    st.sidebar.success(f"Logged in as {st.session_state.username}")
    if st.sidebar.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.session_state.app_mode = "Home"
        st.rerun()
else:
    st.sidebar.warning("🔒 Please log in")

for label, mode in sidebar_buttons.items():
    cols = st.sidebar.columns([1])
    if cols[0].button(label, use_container_width=True,disabled=not st.session_state.logged_in and mode != "Home"):
        st.session_state.app_mode = mode

app_mode = st.session_state.app_mode

# -------------------------------
# Session State Defaults
# -------------------------------
if 'raw_image' not in st.session_state:
    st.session_state.raw_image = None
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None

# -------------------------------
# Home Screen with Centered Login
# -------------------------------
if st.session_state.app_mode == "Home":
    st.title("Smart Retina – Diabetic Eye Disease Detection System")

    if st.session_state.logged_in:
        st.success("✅ You are logged in.")
        #st.sidebar.success("✅ Logged In")
        
        st.info("Please use the sidebar to access application features.")

        # Logout button
        if st.button("🔓 Logout"):
            # Reset session
            #st.session_state.logged_in = False
            #st.session_state.app_mode = "Home"
            clear_patient_session()
            st.session_state.clear()
            st.rerun()

            # Optional: clear login inputs if stored
            if "login_user" in st.session_state:
                del st.session_state["login_user"]
            if "login_pass" in st.session_state:
                del st.session_state["login_pass"]

            st.rerun()

    else:
        # Show login form
        st.markdown("### 🔐 Login to Access Features")

        with st.container():
            col1, col2, col3 = st.columns([2, 3, 2])
            with col2:
                st.subheader("User Login")
                username = st.text_input("Username", key="login_user")
                password = st.text_input("Password", type="password", key="login_pass")
                login_button = st.button("Login")

                if login_button:
                    if username and password:
                        st.session_state.logged_in = True
                        st.session_state.username = username

                        # Clear form fields
                        del st.session_state["login_user"]
                        del st.session_state["login_pass"]

                        st.success(f"Welcome, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Please enter both username and password.")
# -------------------------------
# Upload Image
# -------------------------------
elif app_mode == "Upload Image":
    st.header("Upload Retinal Image")
    if all(k in st.session_state for k in ["first_name", "last_name", "gender", "age"]):
        st.subheader("👤 Patient Information")
        st.markdown(f"""
        **Name**: {st.session_state.first_name} {st.session_state.last_name}  
        **Gender**: {st.session_state.gender}  
        **Age**: {st.session_state.age}
        """)
    else:
        st.warning("⚠️ No patient data found. Please go to 'Patient Details' and submit the form.")
         
    uploaded_file_left = st.file_uploader("Choose a retinal image Left", type=["jpg", "jpeg", "png"])
    uploaded_file_right = st.file_uploader("Choose a retinal image Right", type=["jpg", "jpeg", "png"])
  
    
    if uploaded_file_left:
        img_left = Image.open(uploaded_file_left)
        st.session_state.uploaded_patient_img_left = img_left
        st.image(img_left, caption="New Uploaded Image - Left Eye", use_container_width=True)
        st.success("Left eye image uploaded!")
    elif "uploaded_patient_img_left" in st.session_state:
        st.image(st.session_state.uploaded_patient_img_left, caption="Previously Uploaded Image - Left Eye", use_container_width=True)
    if uploaded_file_right:
        img_right = Image.open(uploaded_file_right)
        st.session_state.uploaded_patient_img_right = img_right
        st.image(img_right, caption="New Uploaded Image - Right Eye", use_container_width=True)
        st.success("Right eye image uploaded!")
    elif "uploaded_patient_img_right" in st.session_state:
        st.image(st.session_state.uploaded_patient_img_right, caption="Previously Uploaded Image - Right Eye", use_container_width=True)
    
    

# -------------------------------
# Preprocessing
# -------------------------------
elif app_mode == "Preprocessing":
    st.header("Preprocessing Options")
    if st.session_state.uploaded_patient_img_left:
        grayscale = st.checkbox("Convert to Grayscale")
        resize = st.checkbox("Resize to 224x224")
        try:
            img = Image.open(st.session_state.uploaded_patient_img_left)
        except AttributeError:
            img = st.session_state.uploaded_patient_img_left
        #img = Image.open(st.session_state.uploaded_patient_img_left)
        processed = preprocess_image(img)
        st.session_state.processed_image = processed

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.uploaded_patient_img_left, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed, caption="Processed Image", use_container_width=True)
    else:
        st.warning("Please upload an image first.")
    if st.session_state.uploaded_patient_img_right:
        #grayscale = st.checkbox("Convert to Grayscale")
        #resize = st.checkbox("Resize to 224x224")
        try:
            img = Image.open(st.session_state.uploaded_patient_img_right)
        except AttributeError:
            img = st.session_state.uploaded_patient_img_right
        #img = Image.open(st.session_state.uploaded_patient_img_right)
        processed = preprocess_image(img)
        st.session_state.processed_image = processed

        col1, col2 = st.columns(2)
        with col1:
            st.image(st.session_state.uploaded_patient_img_right, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed, caption="Processed Image", use_container_width=True)
    else:
        st.warning("Please upload an image first.")

# -------------------------------
# Analysis
# -------------------------------
elif app_mode == "Analysis":
    st.header("Run Model Analysis")
    if st.session_state.processed_image is not None and model:
        st.image(st.session_state.processed_image, caption="Analyzing Image", use_column_width=True)
        with st.spinner('Running prediction...'):
            time.sleep(2)
            stage, confidence = predict_diabetic_retinopathy(np.array(st.session_state.processed_image))
        st.metric(label="Predicted Stage", value=stage)
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
    elif not model:
        st.error("Model not loaded. Please load the model.")
    else:
        st.warning("Please preprocess an image before analysis.")

# -------------------------------
# Report
# -------------------------------
elif app_mode == "Report":
    st.header("Download Report")
    report_text = generate_report("123",st.session_state.get("first_name"),st.session_state.get("age"),st.session_state.get("gender"),"Mild","10%","demo")
    st.subheader("📄 DR AI Report")
    st.text(report_text)
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
        st.info("More report export options will be added.")
    else:
        st.warning("No image available.")

# -------------------------------
# Patient Details
# -------------------------------
# -------------------------------
# Patient Details (with Search + Add)
# -------------------------------
elif app_mode == "Patient Details":
    st.header("👤 Patient Information")

    # Maintain search/add state
    if "patient_mode" not in st.session_state:
        st.session_state.patient_mode = "search"  # options: 'search', 'add'

    # Top Row: Search + Add New Patient
    col1, col2 = st.columns([3, 1])
    with col1:
        patient_id = st.text_input(
            label="Patient ID", 
            label_visibility="collapsed",  # Hide the label but keep placeholder
            placeholder="🔍 Enter Patient ID"
        )
    with col2:
        if st.button("➕ Add New Patient"):
            st.session_state.patient_mode = "add"

    # Show or disable form based on mode
    form_disabled = st.session_state.patient_mode != "add"

    with st.form("patient_form"):
        col1, col2 = st.columns(2)
        first_name = col1.text_input("First Name", value=st.session_state.get("first_name", ""),disabled=form_disabled)
        last_name = col2.text_input("Last Name",value=st.session_state.get("last_name", ""), disabled=form_disabled)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"],index=["Male", "Female", "Other"].index(st.session_state.get("gender", "Male")),disabled=form_disabled)
        age = st.number_input("Age", min_value=0, max_value=120, step=1, value=st.session_state.get("age", 0),disabled=form_disabled)
        uploaded_patient_img_left = st.file_uploader("Upload Retinal Image Left", type=["jpg", "jpeg", "png"], disabled=form_disabled)
        uploaded_patient_img_right = st.file_uploader("Upload Retinal Image Right", type=["jpg", "jpeg", "png"], disabled=form_disabled)
        submitted = st.form_submit_button("Submit")
        display_patient_summary()

    if form_disabled:
        st.info("Enter a Patient ID to search or click 'Add New Patient' to enter new patient data.")

    if submitted and not form_disabled:
        if not all([first_name, last_name]) or age == 0:
            st.warning("Please complete all fields.")
        else:
            clear_patient_session()
            update_patient_info(first_name, last_name, gender, age, uploaded_patient_img_left, uploaded_patient_img_right)
            st.rerun()
            st.success("Patient details recorded successfully!")
            st.subheader("📋 Patient Summary")
            st.markdown(f"""
            **Name**: {first_name} {last_name}  
            **Gender**: {gender}  
            **Age**: {age}
            """)
            if uploaded_patient_img_left:
                st.markdown("**👁️ Left Eye Image**")
                left_img = Image.open(uploaded_patient_img_left)
                #img_right = Image.open(uploaded_patient_img_right)
                st.image(left_img, caption="Left Eye", use_container_width=True)
                #st.write(f"• **Filename**: `{left_img.name}`")
                #st.write(f"• **Type**: `{left_img.type}`")
                #st.write(f"• **Size**: `{round(left_img.size / 1024, 2)} KB`")
               
            else:
                st.info("No Left image uploaded.")
            if uploaded_patient_img_right:
                right_img = Image.open(uploaded_patient_img_right)
                st.markdown("**👁️ Right Eye Image**")
                st.image(right_img, caption="Right Eye", use_container_width=True)
                #st.write(f"• **Filename**: `{right_img.name}`")
                #st.write(f"• **Type**: `{right_img.type}`")
                #st.write(f"• **Size**: `{round(right_img.size / 1024, 2)} KB`")
                
                #st.image(img_left, caption="Uploaded Retinal Image", use_container_width=True)
                #st.image(img_right, caption="Uploaded Retinal Image", use_container_width=True)
            else:
                st.info("No Right image uploaded.")
        
        #update_patient_info(first_name, last_name, gender, age, uploaded_patient_img_left, uploaded_patient_img_right)
elif app_mode == "Help":
    render_help()
# -------------------------------
# Footer
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Smart Vision – Capstone Project")
