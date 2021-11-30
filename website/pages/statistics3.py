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
import collections
import plotly.figure_factory as ff

import plotly.express as px


@st.cache(allow_output_mutation=True)
def load_graph():
    with open("pages/data/WG.pickle", 'rb') as f:
        G = pickle.load(f)

    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    return G 




def create_and_select_pos(G):

    layouts = ["Circular", "Spring Layout", "Foce Atlas", "Kamada Kawai"]

    select_position = st.selectbox("Chose network to draw: ", layouts)

    if select_position == layouts[0]:
        pos = nx.circular_layout(G)
    elif select_position == layouts[1]:
        pos = nx.spring_layout(G)
    # elif select_position == positions[2]: 
    #     forceatlas2 = ForceAtlas2()
    #     pos = forceatlas2.forceatlas2_networkx_layout(G, iterations=2000)
    elif select_position == layouts[3]:
        pos = nx.kamada_kawai_layout(G) 

    return pos

def plot(G):

    pos = create_and_select_pos(G)

    fig, ax = plt.subplots()

    nx.draw(G, pos, ax=ax)

    return fig 

def plot_deg_dist(G, b):
    if b == True:
        degree_sequence = sorted([d for n, d in G.in_degree()], reverse=True)  
        title = "In Degree Histogram"
    else:
        degree_sequence = sorted([d for n, d in G.out_degree()], reverse=True)  
        title = "Out Degree Histogram"


    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig = px.bar(x=deg, y=cnt)


    fig.update_layout(
        title={
            'text': title,
            'y':0.91,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Degree",
        yaxis_title="Count",
        legend_title="Legend Title",
        template="plotly_dark",
        font=dict(
            family="Sans serif",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def graph_stats(G):
    in_degree = sorted([d for n, d in G.in_degree()], reverse=True)  

    out_degree = sorted([d for n, d in G.out_degree()], reverse=True)  


    df = pd.DataFrame()
    df["In Degree"] = in_degree
    df["Out Degree"] = out_degree

    st.write(df.describe().T.iloc[:,1:])





def plot_loglog_degree_histogram(G, b, normalized=True):
    if b:
        y = sorted([d for n, d in G.in_degree()], reverse=True)
        title = ('\nDistribution Of Node Linkages w.r.t. In-degree (log-log scale)')
    else:
        y = sorted([d for n, d in G.out_degree()], reverse=True)
        title = ('\nDistribution Of Node Linkages w.r.t. Out-degree (log-log scale)')
        
    x = np.arange(0,len(y)).tolist()
    
    n_nodes = G.number_of_nodes()
    
    if normalized:
        for i in range(len(y)):
            y[i] = y[i]/n_nodes

    fig = px.scatter(x=x, y=y, log_x=True, log_y=True)


    fig.update_layout(
        title={
            'text': title,
            'y':0.91,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title='Degree\n(log scale)',
        yaxis_title='Number of Nodes\n(log scale)',
        template="plotly_dark",
        font=dict(
            family="Sans serif",
        )
    )

    st.plotly_chart(fig, use_container_width=True)

def app():
    theme = "plotly_dark"

    st.title("Visualization And Statistics")

    st.text("In this section the network will be analysed with statistics and visualized. ")

    col1, col2 = st.columns(2)

    G = load_graph()

    with col1:
        plot_deg_dist(G, True)
        graph_stats(G)
        plot_loglog_degree_histogram(G, True, normalized=True)


    with col2:
        plot_deg_dist(G, False)
        st.text("Some explainer text")
        plot_loglog_degree_histogram(G, False, normalized=True)

    #fig = plot(G)

    #st.pyplot(fig)

  