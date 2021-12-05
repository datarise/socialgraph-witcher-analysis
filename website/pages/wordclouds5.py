# Import dependencies
from pandas.core.frame import DataFrame
import streamlit as st
import pandas as pd
from nltk.corpus import PlaintextCorpusReader
import nltk
import numpy as np
from nltk.probability import FreqDist
import math
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator 
from PIL import Image
from scipy.ndimage import gaussian_gradient_magnitude
import matplotlib
import os
matplotlib.rcParams["figure.dpi"] = 200
plt.style.use('dark_background')
import random

@st.cache()
def load_data():
    df = pd.read_csv("website/pages/data/cleaned_characters_attr.csv", index_col=0)

    return df

def generate_text_corpus(df_characters, overall_attribute, attribute_value):
    """
    Generates the text corpuses for the chosen attribute
    """
    data_root = r"website/pages/data/data"

    character_list = list(df_characters[df_characters[overall_attribute] == attribute_value]["title"].apply(lambda x: x + '.txt').values)
    wordlists = PlaintextCorpusReader(data_root, character_list)
    text = nltk.Text(wordlists.words())
    return text

    
def generate_dict_corpuses(df_characters, overall_attribute, number_of_clouds, specific_choices = None):
    """
    Generates dictionary of text_corpuses from the overall attribute.
    """

    attribute_values = []
    
    #Add specific choices first
    if specific_choices is not None:
        attribute_values.extend(specific_choices)

    #Get top represented from the overall attribute and drop unwanted entries
    top_attributes = df_characters[overall_attribute].value_counts()
    top_attributes = top_attributes.drop(index = f"No {overall_attribute}")
    if specific_choices is not None:
        top_attributes = top_attributes.drop(index = specific_choices)   
    attribute_values.extend(list(top_attributes.index))
    attribute_values = attribute_values[:number_of_clouds]
    
    #Generate dictionary of corpuses
    texts_dict = {}
    for attribute in attribute_values:
        attribute_text = generate_text_corpus(df_characters, overall_attribute, attribute)
        texts_dict[attribute] = attribute_text
    
    return texts_dict 

def generate_word_clouds(texts_dict, col1, col2):
    #Make TC of TF dictionary
    attribute_dict = {}
    for attr in texts_dict.keys():
        term_dict = {}
        for key, value in FreqDist(texts_dict[attr]).items():
            term_dict[key] = value
        attribute_dict[attr] = term_dict
    #Make IDF dictionary

    gathered_texts = list(list(texts_dict.values())[0])
    for text in list(texts_dict.values())[1:]:
        gathered_texts.extend(list(text))

    idf_dict = {}
    for word in np.unique(gathered_texts):
        number_of_documents = 0
        for attr in texts_dict.keys():
            if word in attribute_dict[attr].keys():
                number_of_documents += 1
        idf_dict[word] = math.log10(len(texts_dict.keys())/number_of_documents)


    icons_path = 'website/pages/images/icons'

    icons = os.listdir(icons_path)

    n_icons = len(icons) - 1
    #random.sample(n_icons,5)
    c = 0
    #Generate Wordclouds
    for i, attr in enumerate(attribute_dict.keys()):
        wordcloud_string = ''
        for word in attribute_dict[attr].keys():
            word_mod = word + ' ' 
            tf_idf_score = round(attribute_dict[attr][word] * idf_dict[word])
            wordcloud_string += word_mod * tf_idf_score

        if c % 2 == 0:
            with col1:
                make_cloud(wordcloud_string, attr, f"{icons_path}/{icons[random.randint(0, n_icons)]}")    
        else:
            with col2:
                make_cloud(wordcloud_string, attr, f"{icons_path}/{icons[random.randint(0, n_icons)]}")  
        if c == n_icons:
            c = 0
        else:
            c += 1
        

def make_cloud(wordcloud_string, attr, icon_path):

    wolf_mask = np.array(Image.open(icon_path))

    wc = WordCloud(background_color="black", max_words=2000, mask=wolf_mask, collocations=False, colormap='RdBu', height=1000, width=1000)

    wc.generate(wordcloud_string)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(f"{attr} Wordcloud")
    plt.gcf().set_facecolor('black')
    plt.axis("off")
    st.pyplot(fig)


def app():
    # Read dataset (CSV)
    # Title of the main page
    df = load_data()
    col1, col2 = st.columns(2)

    with col1:
        st.text("Some explainer")
    with col2:
        attribute = st.selectbox("Select the attribute to visualize", ("Race","Gender","Nationality","Family","Profession"))
        if attribute:
            choose = df[attribute].unique()
            attributes = st.multiselect("Select the attributes to visualize", choose)


    if attributes and len(attributes) >= 2:
        texts_dict = generate_dict_corpuses(df, attribute, len(attributes), attributes)
        generate_word_clouds(texts_dict, col1, col2)
    else:
        st.text("Please select at least two attributes to get started")