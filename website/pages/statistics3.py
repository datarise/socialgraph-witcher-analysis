# Import dependencies
import streamlit as st
import pandas as pd
import networkx as nx
import pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
import plotly.express as px
import powerlaw as pw
from pathlib import Path



@st.cache(allow_output_mutation=True)
def load_graph():
    with open("website/pages/data/WG.gpickle", 'rb') as f:
        G = pickle.load(f)

    Gcc = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    return G 

@st.cache(allow_output_mutation=True)
def load_df():

    df = pd.read_csv("website/pages/data/characters_sentiment.csv")

    return df

def attribute_stats(df, attribute):
    df_count = df[attribute].value_counts().reset_index()
    df_count.columns = [attribute, "Count"]

    fig = px.bar(df_count, x=attribute, y='Count')

    fig.update_layout(
    title={
        'text': f"Barchart of {attribute}",
        'y':0.91,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
    xaxis={
            'tickangle': 45},
    template="plotly_dark",
    font=dict(
        family="Sans serif",
    )
    )
    st.plotly_chart(fig, use_container_width=True)


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


def top_n_degree(G, sort_by):
    df_in = pd.DataFrame.from_dict(dict(G.in_degree()), orient='index')
    df_out = pd.DataFrame.from_dict(dict(G.out_degree()), orient='index')
    df = pd.merge(df_in, df_out, left_on=df_in.index, right_on=df_out.index)
    df.columns = ["Character", "In-Degree", "Out-Degree"]
    df = df.sort_values(by=sort_by, ascending=False)

    st.dataframe(df)

def get_plt_data(fig):

    ax = plt.gca() 
    line = ax.lines[0] 

    return line.get_xydata()



def plot_powerlaw(G, degree_type):
    if degree_type == "In-Degree":
        degree = sorted([d for n, d in G.in_degree()], reverse=True)
    else:
        degree = sorted([d for n, d in G.out_degree()], reverse=True)

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
        'text': f"{degree_type} - PDF fitted vs data - Exponent: {str(round(fit.power_law.alpha,2))}",
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
    st.title("Visualization And Statistics")
    col1, col2 = st.columns(2)

    G = load_graph()

    df = load_df()

    with col1:
        st.markdown(read_markdown_file("website/pages/text/statistics1.md"), unsafe_allow_html=True)
        attribute = st.selectbox("Select the attribute to chart:", df.columns[1:-2])
        whitespace(10)
        plot_deg_dist(G, True)
        st.markdown(read_markdown_file("website/pages/text/statistics2.md"), unsafe_allow_html=True)
        plot_loglog_degree_histogram(G, True, normalized=True)
        st.markdown(read_markdown_file("website/pages/text/statistics2.md"), unsafe_allow_html=True)


    with col2:
        attribute_stats(df, attribute)
        plot_deg_dist(G, False)
        graph_stats(G)
        whitespace(3)
        plot_loglog_degree_histogram(G, False, normalized=True)
        analyse = st.radio("Select the degree to sort and analyse by:", ("In-Degree", "Out-Degree"))
        top_n_degree(G, analyse)
        plot_powerlaw(G, analyse)


  