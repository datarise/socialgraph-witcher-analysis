# Custom imports 
from multipage import MultiPage
from pages import intro1, explore2, statistics3, compare4, wordclouds5, communities6, conclusion7
import streamlit as st

st.set_page_config(page_title="The Witcher Network Analysis", layout="wide")
# Create an instance of the app 
app = MultiPage()

# Add all your applications (pages) here
app.add_page("1. Introduction", intro1.app)
app.add_page("2. Explore The Network", explore2.app)
app.add_page("3. Network Statistics", statistics3.app)
app.add_page("4. Comparison of Random Network", compare4.app)
app.add_page("5. Wordclouds and TF-IDF", wordclouds5.app)
app.add_page("6. Communities and Sentiment Analysis", communities6.app)
app.add_page("7. Conclusion", conclusion7.app)

# The main app
app.run()