import streamlit as st

def render_help():
    with st.expander("❓ Help", expanded=False):
        st.markdown("""
        ### 🆘 Help Guide

        **👤 Patient Details**  
        Enter patient demographic information such as first name, last name, age, and gender.  
        You will also upload **retinal images for the left and right eyes** in this section.  
        - If images already exist for a patient, they will be displayed.  
        - You may continue with the existing images or upload new ones to replace them.  
        - Supported formats: JPG, JPEG, PNG.

        **📤 Upload Image**  
        You can re-upload left and/or right retinal images at any time.  
        - When a new image is selected, it will **replace** the previously uploaded image.  
        - Make sure the image is of sufficient quality for analysis (clear, focused, no artifacts).

        **🛠️ Pre-processing**  
        Uploaded images are resized, normalized, and optionally converted to grayscale (if selected).  
        The standard size for the AI model is **224x224 pixels**. Preprocessing ensures consistency before feeding images into the model.

        **🧠 Analysis**  
        Click the **Diagnose** button to trigger the AI model.  
        The model will predict the **Diabetic Retinopathy (DR) stage**, and display a **Grad-CAM heatmap** to highlight important regions that influenced the diagnosis.

        **📄 Reports**  
        After diagnosis, you can generate and download a comprehensive report.  
        The report includes:
        - Patient information
        - Predicted DR stage
        - Confidence score
        - Image explanation (Grad-CAM)
        - Download as `.txt` or `.pdf`

        ---
        For further assistance, please refer to the user documentation or contact the developer.
        """)
