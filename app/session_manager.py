# session_manager.py

import streamlit as st

def initialize_session():
    default_values = {
        "logged_in": False,
        "username": "",
        "patient_mode": "search",   # 'search' or 'add'
        "patient_data": {},         # patient_id: patient_info
        "gradcam_images": {},       # optional if used elsewhere
        "current_page": "Home",   # for navigation state
        "app_mode":"Home"
    }

    for key, default in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default
    
def reset_session():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
        
def update_patient_info(first_name, last_name, gender, age, img_left=None, img_right=None):
    st.session_state.first_name = first_name
    st.session_state.last_name = last_name
    st.session_state.gender = gender
    st.session_state.age = age
    st.session_state.uploaded_patient_img_left = img_left
    st.session_state.uploaded_patient_img_right = img_right
def display_patient_summary():
    if st.session_state.get("first_name"):
        st.subheader("📋 Patient Summary")
        st.markdown(f"""
        **Name**: {st.session_state.first_name} {st.session_state.last_name}  
        **Gender**: {st.session_state.gender}  
        **Age**: {st.session_state.age}
        """)

        # Left Eye Image
        left_img = st.session_state.get("uploaded_patient_img_left")
        if left_img:
            st.markdown("**👁️ Left Eye Image**")
            st.image(left_img, caption="Left Eye", use_container_width=True)
            #st.write(f"• **Filename**: `{left_img.name}`")
            #st.write(f"• **Type**: `{left_img.type}`")
            #st.write(f"• **Size**: `{round(left_img.size / 1024, 2)} KB`")

        # Right Eye Image
        right_img = st.session_state.get("uploaded_patient_img_right")
        if right_img:
            st.markdown("**👁️ Right Eye Image**")
            st.image(right_img, caption="Right Eye", use_container_width=True)
            #st.write(f"• **Filename**: `{right_img.name}`")
            #st.write(f"• **Type**: `{right_img.type}`")
            #st.write(f"• **Size**: `{round(right_img.size / 1024, 2)} KB`")

#def clear_patient_session():
#    keys = [
#        "first_name", 
#        "last_name", 
#        "gender", 
#        "age", 
#        "uploaded_patient_img_left", 
#        "uploaded_patient_img_right"
#    ]
#    for key in keys:
#        st.session_state.pop(key, None)
def clear_patient_session():
    keys_to_clear = ["first_name", "last_name", "gender", "age", "uploaded_patient_img_left", "uploaded_patient_img_right"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
