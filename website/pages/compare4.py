import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from fa2 import ForceAtlas2
import streamlit as st
import pandas as pd
import numpy as np
import collections
import plotly.express as px
import powerlaw as pw
import pickle
from pathlib import Path



def create_and_select_network():
    networks = ["Barabási–Albert Graph", "Erdős-Rényi Graph"]

    select_network = st.selectbox("Chose network to draw: ", networks)


    if select_network == networks[1]:
        n = st.slider("Select number of nodes:", min_value=10, max_value=800, value=700, step=10)
        p = st.slider("Select probability of edges", min_value=0.001, max_value=1.0, value=0.1, step=0.001)

        G = nx.generators.random_graphs.erdos_renyi_graph(n, p, seed=42)
    elif select_network == networks[0]:
        n = st.slider("Select number of nodes:", min_value=10, max_value=800, value=700, step=10)
        m = st.slider("Select number of edges", min_value=1, max_value=10, value=2, step=1)

        G = nx.generators.random_graphs.barabasi_albert_graph(n, m, seed=42)
    else:
        print()

    return G 


def create_and_select_pos(G, select_position):
    positions = ["Spring Layout", "Force Atlas", "Kamada Kawai", "Circular"]

    if select_position == positions[0]:
        pos = nx.spring_layout(G)
    elif select_position == positions[1]:
        forceatlas2 = ForceAtlas2()
        pos = forceatlas2.forceatlas2_networkx_layout(G, iterations=2000)
    elif select_position == positions[2]: 
        pos = nx.kamada_kawai_layout(G) 
    elif select_position == positions[3]:
        pos = nx.circular_layout(G)

    return pos

def plot(G, pos):

    fig, ax = plt.subplots()

    d = nx.degree(G)
    d = [(d[node]+1) * 20 for node in G.nodes()]
    nx.draw(G, pos, node_color="#636EFA", edge_color="#FAFAFA", node_size=d)

    fig.set_facecolor('#0E1117')


    return fig 

def plot_deg_dist(G):

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  
    title = "Degree Histogram"


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


def plot_loglog_degree_histogram(G, normalized=True):
    y = sorted([d for n, d in G.degree()], reverse=True)
    title = ('\nDistribution Of Node Linkages w.r.t. In-degree (log-log scale)')

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

@st.cache(allow_output_mutation=True)
def load_graph():
    with open("website/pages/data/WG.gpickle", 'rb') as f:
        G = pickle.load(f)

    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])
    G = G.to_undirected()

    return G 

def whitespace(N):
    for i in range(N):
        st.text("")

def get_plt_data(fig):

    ax = plt.gca() 
    line = ax.lines[0] 

    return line.get_xydata()

def plot_powerlaw(G):

    degree = sorted([d for n, d in G.degree()], reverse=True)

    fit = pw.Fit(np.array(degree), xmin=0)

    fig = plt.figure()
    fit.power_law.plot_pdf( color= 'b',linestyle='--', label='fit pdf')
    pdf_fit = get_plt_data(fig)    

    fig = plt.figure()
    fit.plot_pdf(color= 'b', label='data')
    pdf_data = get_plt_data(fig)  

    df_fit = pd.DataFrame(pdf_fit)
    df_fit["Data"] = "PDF Powerlaw"
    df_data = pd.DataFrame(pdf_data)
    df_data["Data"] = "PDF Data"
    df = pd.concat([df_fit, df_data], axis=0)
    df.columns = ["Bin", "PDF", "Data"]

    fig = px.line(df, x="Bin", y="PDF", color='Data', log_x=True, log_y=True)

    fig.update_layout(
    title={
        'text': f"Degree - PDF fitted vs data - Exponent: {str(round(fit.power_law.alpha,2))}",
        'y':0.91,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    template="plotly_dark",
    font=dict(
        family="Sans serif",
    )
    )
    st.plotly_chart(fig, use_container_width=True)


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def whitespace(i):
    for i in range(i):
        st.text("")


def app():
    # Title of the main page
    st.title("Comparison with random networks")
    st.text("In this section the Witcher graph can be compared to multiple random networks.")

    col1, col2 = st.columns(2)

    with col1:
        G_com = create_and_select_network()
        G_witcher = load_graph()
        positions = ["Spring Layout", "Force Atlas", "Kamada Kawai", "Circular"]
        select_position = st.selectbox("Choose network layout to draw: ", positions)
        pos_com = create_and_select_pos(G_com, select_position) 
        fig_com = plot(G_com, pos_com)
        pos_witcher = create_and_select_pos(G_witcher, select_position)
        fig_witcher = plot(G_witcher, pos_witcher)

    with col2:
        st.markdown(read_markdown_file("website/pages/text/comparison1.md"), unsafe_allow_html=True)


    col3, col4 = st.columns(2)

    with col3:
        st.pyplot(fig_com)
        plot_deg_dist(G_com)
        plot_loglog_degree_histogram(G_com)
        plot_powerlaw(G_com)
    
    with col4:
        st.pyplot(fig_witcher)
        plot_deg_dist(G_witcher)
        plot_loglog_degree_histogram(G_witcher)
        plot_powerlaw(G_witcher)


    #with col2:

