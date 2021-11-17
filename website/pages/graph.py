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

def app():
    with open("pages/data/WG.pickle", 'rb') as f:
        G = pickle.load(f)

    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    character_list = list(G.nodes())


    # Implement multiselect dropdown menu for option selection (returns a list)
    selected_characters = st.multiselect('Select character', character_list)

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
                H.nodes()[node]["Profession"] = "Test1"

            else:
                H.nodes()[node]["color"] = "##7048e8"
                H.nodes()[node]["Profession"] = "Test2"


        # Initiate PyVis network object
        char_net = Network(height='1000px', width='1000px', bgcolor='#222222', font_color='white')

        # Take Networkx graph and translate it to a PyVis graph format
        char_net.from_nx(H)

        # Generate network with specific layout settings
        # char_net.repulsion(node_distance=420, central_gravity=0.33,
        #                 spring_length=110, spring_strength=0.10,
        #                 damping=0.95)

        char_net.force_atlas_2based()

        # Save and read graph as HTML file (on Streamlit Sharing)
        try:
            path = '/tmp'
            char_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

        # Save and read graph as HTML file (locally)
        except:
            path = '/html_files'
            char_net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

        # Load HTML file in HTML component for display on Streamlit page
        components.html( HtmlFile.read(), height=1000)