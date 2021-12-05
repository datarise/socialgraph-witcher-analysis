# Import dependencies
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import networkx as nx
from pyvis.network import Network
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import collections
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler



@st.cache(allow_output_mutation=True)
def load_graph():
    with open("website/pages/data/WG.gpickle", 'rb') as f:
        G = pickle.load(f)

    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    degrees = dict(G.degree)
    degrees.update((key, value*0.5) for key, value in degrees.items())
    nx.set_node_attributes(G, degrees, 'size')

    return G 

def set_attributes(G):
    df = pd.read_csv("website/pages/data/cleaned_characters_attr.csv")

    df = df.set_index("title")
    df = df[["Race", "Gender", "Nationality", "Family", "Profession"]]
    attributes = df.to_dict('index')
    nx.set_node_attributes(G, attributes)

    titles = {}
    for name, data in G.nodes(data = True):
        try:
            title = 'Name: ' + name + ' </br>Race: ' + data['Race'] + ' </br>Gender: ' + data['Gender'] + ' </br>Nationality: ' + data['Nationality'] + ' </br>Family: ' + data['Family'] + ' </br>Profession: ' + data['Profession']
            titles[name] = title
        except:
            pass
    nx.set_node_attributes(G, titles, 'title')

    scaler = MinMaxScaler((10,200))
    scaler.fit(np.array(list(dict(G.degree).values())).reshape(-1, 1))
    degrees = dict(G.degree)
    degrees.update((key, scaler.transform(np.array(value).reshape(1, -1))) for key, value in degrees.items())
    nx.set_node_attributes(G, degrees, 'size')


    return G



def set_color(G, key):

    attribute = st.selectbox(
     'Please select the attribute to color the network by',
     ('Race', 'Gender', 'Nationality', 'Family', 'Profession'), key=key)


    attributes = nx.get_node_attributes(G, attribute)
    unique_attributes = np.unique(list(attributes.values()))
    n_attributes = len(unique_attributes)

    cmap = plt.cm.get_cmap('ocean', n_attributes)
    hexmap = []
    for i in range(n_attributes):
        hexmap.append(matplotlib.colors.rgb2hex(cmap(i)))

    colors = {}
    for name, value in attributes.items():
        index = list(unique_attributes).index(value)
        color = hexmap[index]
        colors[name] = color
    nx.set_node_attributes(G, colors, 'color')
    return G



def pyviz_plot_network(G, layout, smooth=True):
    


    # Initiate PyVis network object
    char_net = Network(height='1000px', width='1500px', bgcolor='#222222', font_color='white')

    # Take Networkx graph and translate it to a PyVis graph format
    char_net.from_nx(G)

    if layout == "Barnes Hut":
        char_net.barnes_hut()

    elif layout ==  "Repulsion":
        char_net.repulsion()

    elif layout ==  "Hrepulsion":
        char_net.hrepulsion()
    else:
        char_net.force_atlas_2based()



    if smooth:
        char_net.set_edge_smooth('curvedCW')

    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = '/tmp'
        char_net.save_graph(f'{path}/pyvis_graph_by_characher.html')
        HtmlFile = open(f'{path}/pyvis_graph_by_characher.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = '/html_files'
        char_net.save_graph(f'{path}/pyvis_graph_by_characher.html')
        HtmlFile = open(f'{path}/pyvis_graph_by_characher.html', 'r', encoding='utf-8')

    # Load HTML file in HTML component for display on Streamlit page
    components.html( HtmlFile.read(), height=1000)

def pyviz_plot_by_character(G, selected_characters, layout, smooth=True):

    # Set info message on initial site load
    if len(selected_characters) == 0:
        st.text('Choose at least 1 character to get started')

    # Create network graph when user selects >= 1 item
    else:
        sub_list = []
        for character in selected_characters:

            sub_list = sub_list + list(G.neighbors(character))

            sub_list.append(character)

        H = G.subgraph(sub_list)

        for node in H.nodes():
            if node in selected_characters:
                H.nodes()[node]["color"] = "#099268"

            else:
                H.nodes()[node]["color"] = "##7048e8"

        # Initiate PyVis network object
        char_net = Network(height='1000px', width='1500x', bgcolor='#222222', font_color='white')

    

        # Take Networkx graph and translate it to a PyVis graph format
        char_net.from_nx(H)

        if layout == "Barnes Hut":
            char_net.barnes_hut()

        elif layout ==  "Repulsion":
            char_net.repulsion()

        elif layout ==  "Hrepulsion":
            char_net.hrepulsion()
        else:
            char_net.force_atlas_2based()

        if smooth:
            char_net.set_edge_smooth('curvedCW')

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = '/tmp'
            char_net.save_graph(f'{path}/pyvis_graph_by_characher.html')
            HtmlFile = open(f'{path}/pyvis_graph_by_characher.html', 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            path = '/html_files'
            char_net.save_graph(f'{path}/pyvis_graph_by_characher.html')
            HtmlFile = open(f'{path}/pyvis_graph_by_characher.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html( HtmlFile.read(), height=1000)

def select_layout(key):

    layout = st.selectbox("Select the layout algorithm", ("Barnes Hut", "Repulsion", "Hrepulsion", "Force Atlas"), key=key)

    return layout

def app():
    st.title("Explore The Witcher Network")

    G = load_graph()

    G = set_attributes(G)

    col1, col2 = st.columns(2)

    with col1:
        st.text("Some explainer text")


    with col2:
        st.text("Please select how you want to have the network shown")
        G = set_color(G, "sk1")
        layout = select_layout("s1")
        smooth = st.radio("Smooth Edges?", (True, False), key="rk3")
        display = st.radio("Display the network", (False, True), key="rk1")

    if display:
        pyviz_plot_network(G, layout, smooth)

    col1, col2 = st.columns(2)

    with col1:
        st.text("Some explainer text")


    with col2:

        G = set_color(G, "sk2")
        layout = select_layout("s2")
        # Implement multiselect dropdown menu for option selection (returns a list)
        st.text("Please select the character(s) you want to explore")
        character_list = list(G.nodes())
        selected_characters = st.multiselect('Select character', character_list, key="k2")
        smooth = st.radio("Smooth Edges?", (False, True), key="rk4")

        display = st.radio("Display the network", (False, True), key="rk2")

    if display:
        pyviz_plot_by_character(G, selected_characters, layout, smooth)


