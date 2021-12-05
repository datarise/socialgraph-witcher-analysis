# Import dependencies
import streamlit as st

def app():
    # Read dataset (CSV)
    # Title of the main page
    st.title("Template")

    st.markdown(
        """
    <style>
    .reportview-container .markdown-text-container {
        font-family: monospace;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e7bcf,#2e7bcf);
        color: white;
    }
    .Widget>label {
        color: white;
        font-family: monospace;
    }
    [class^="st-b"]  {
        color: white;
        font-family: monospace;
    }
    .st-bb {
        background-color: transparent;
    }
    .st-at {
        background-color: #0c0080;
    }
    footer {
        font-family: monospace;
    }
    .reportview-container .main footer, .reportview-container .main footer a {
        color: #0c0080;
    }
    header .decoration {
        background-image: none;
    }

    </style>
    """,
        unsafe_allow_html=True,
    )