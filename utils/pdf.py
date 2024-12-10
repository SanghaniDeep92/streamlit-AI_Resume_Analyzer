import base64
import streamlit as st

def show_pdf_in_sidebar(uploaded_file):
    # Read the uploaded file and encode it as base64
    base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
    
    # Generate a data URL for the PDF
    pdf_display = f'<embed src="data:application/pdf;base64,{base64_pdf}" width="300" height="500" type="application/pdf" />'
    
    # Display the embed tag in the sidebar
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)