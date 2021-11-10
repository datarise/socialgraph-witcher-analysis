# Import dependencies
import streamlit as st

def app():
    # Read dataset (CSV)
    # Title of the main page
    st.title("Template")

    code = '''def hello():
    ...     print("Hello, Streamlit!")'''
    st.code(code, language='python')