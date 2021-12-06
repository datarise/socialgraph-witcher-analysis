# Import dependencies
import streamlit as st
from pathlib import Path


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def app():
    # Read dataset (CSV)
    # Title of the main page
    st.title("Summary")

    st.markdown(read_markdown_file("website/pages/text/summary.md"), unsafe_allow_html=True)
