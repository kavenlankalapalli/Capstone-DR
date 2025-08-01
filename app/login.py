
# login.py

import streamlit as st

def login():
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["authenticated"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid credentials")
            st.session_state["authenticated"] = False

def login_form():
    st.subheader("Login")

    # Input fields for username and password
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    # Two columns: Submit and Cancel
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Submit", key="submit_login"):
            if username == "admin" and password == "1234":
                st.session_state.authenticated = True
                st.session_state.show_login = False
                st.success("Login successful!")
            else:
                st.error("Invalid credentials. Try again.")

    with col2:
        if st.button("Cancel", key="cancel_login"):
            st.session_state.show_login = False
