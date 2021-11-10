import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
#from fa2 import ForceAtlas2


def create_and_select_network():

    networks = ["ER-Graph", "Witcher-Graph"]

    select_network = st.selectbox("Chose network to draw: ", networks)


    if select_network == networks[0]:
        n = st.slider("Select number of nodes:", min_value=10, max_value=250, value=50, step=10)
        p = st.slider("Select probability of edges", min_value=0.001, max_value=1.0, value=0.1, step=0.001)

        G = nx.generators.random_graphs.erdos_renyi_graph(n, p, seed=42)
    elif select_network == networks[1]:
        print()
    else:
        print()

    return G 


def create_and_select_pos(G):

    positions = ["Circular", "Spring Layout", "Foce Atlas", "Kamada Kawai"]

    select_position = st.selectbox("Chose network to draw: ", positions)

    if select_position == positions[0]:
        pos = nx.circular_layout(G)
    elif select_position == positions[1]:
        pos = nx.spring_layout(G)
    # elif select_position == positions[2]: 
    #     forceatlas2 = ForceAtlas2()
    #     pos = forceatlas2.forceatlas2_networkx_layout(G, iterations=2000)
    elif select_position == positions[3]:
        pos = nx.kamada_kawai_layout(G) 

    return pos

def plot(G, pos):

    fig, ax = plt.subplots()

    nx.draw(G, pos, ax=ax)

    return fig 

def app():
    # Read dataset (CSV)
    # Title of the main page
    st.title("Comparizon of layout algorithms")

    G = create_and_select_network()

    pos = create_and_select_pos(G)

    fig = plot(G, pos)

    st.pyplot(fig)

