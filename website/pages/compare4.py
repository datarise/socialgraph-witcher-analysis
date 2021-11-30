import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from fa2 import ForceAtlas2


def create_and_select_network(col):
    with col:
        networks = ["Erdős-Rényi Graph", "Barabási–Albert Graph"]

        select_network = st.selectbox("Chose network to draw: ", networks)


        if select_network == networks[0]:
            n = st.slider("Select number of nodes:", min_value=10, max_value=250, value=50, step=10)
            p = st.slider("Select probability of edges", min_value=0.001, max_value=1.0, value=0.1, step=0.001)

            G = nx.generators.random_graphs.erdos_renyi_graph(n, p, seed=42)
        elif select_network == networks[1]:
            n = st.slider("Select number of nodes:", min_value=10, max_value=250, value=50, step=10)
            m = st.slider("Select number of edges", min_value=1, max_value=10, value=1, step=1)

            G = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=42)
        else:
            print()

    return G 


def create_and_select_pos(G, col):
    with col:
        positions = ["Circular", "Spring Layout", "Foce Atlas", "Kamada Kawai"]

        select_position = st.selectbox("Choose network layout to draw: ", positions)

        if select_position == positions[0]:
            pos = nx.circular_layout(G)
        elif select_position == positions[1]:
            pos = nx.spring_layout(G)
        elif select_position == positions[2]: 
             forceatlas2 = ForceAtlas2()
             pos = forceatlas2.forceatlas2_networkx_layout(G, iterations=2000)
        elif select_position == positions[3]:
            pos = nx.kamada_kawai_layout(G) 

    return pos

def plot(G, pos):

    fig, ax = plt.subplots()

    nx.draw(G, pos, node_color="#636EFA", edge_color="#FAFAFA")

    fig.set_facecolor('#0E1117')


    return fig 

def app():
    # Title of the main page
    st.title("Comparison with random networks")
    st.text("In this section the Witcher graph can be compared to multiple random networks.")

    col1, col2 = st.columns(2)

    G = create_and_select_network(col1)

    pos = create_and_select_pos(G, col1)

    fig = plot(G, pos)

    with col2:
        st.pyplot(fig)

