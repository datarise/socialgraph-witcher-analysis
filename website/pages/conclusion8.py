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
    st.title("Summary")

    st.markdown(read_markdown_file("website/pages/text/summary.md"), unsafe_allow_html=True)

    st.text("Thanks for exploring the Witcher universe with us!")

    st.image(display_image("website/pages/images/ending.jpg"))

    st.balloons()
