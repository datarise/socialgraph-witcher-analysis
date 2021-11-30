# Import dependencies
import streamlit as st
from pathlib import Path

from PIL import Image

def display_image(image_path):
    image = Image.open(image_path)
    return image

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def app():
    # Read dataset (CSV)
    # Title of the main page
    st.markdown(read_markdown_file("pages/text/introduction1.md"), unsafe_allow_html=True)

    st.image(display_image("pages/images/the_witcher.jpeg"))

    st.markdown(read_markdown_file("pages/text/introduction2.md"), unsafe_allow_html=True)
