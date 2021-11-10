# Import dependencies
import streamlit as st
from pathlib import Path


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def app():
    # Read dataset (CSV)
    # Title of the main page
    st.title("Graph Analysis of the Witcher")

    intro_markdown = read_markdown_file("pages/text/introduction.md")
    st.markdown(intro_markdown, unsafe_allow_html=True)