# Import dependencies
import community as community_louvain
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from pyvis.network import Network
import networkx as nx
from collections import Counter
import plotly.express as px
import math
import pandas as pd
from nltk.corpus import PlaintextCorpusReader
import nltk
from nltk.probability import FreqDist
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import copy
import pickle
from sklearn.preprocessing import MinMaxScaler
import gensim.corpora as corpora
from gensim.models import TfidfModel, LdaModel
from nltk.corpus import stopwords
nltk.download('stopwords')
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis


@st.cache(allow_output_mutation=True)
def load_graph():
    with open("website/pages/data/WG.gpickle", 'rb') as f:
        G = pickle.load(f)

    G = G.to_undirected()

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G = G.subgraph(Gcc[0])

    return G 

def name_communities(G0, partition, community_size):
    # Find all node degress
    degrees = [val for (node, val) in G0.degree()]

    # Create node degree dataframe
    df_community = pd.DataFrame()
    df_community["Name"] = list(partition.keys())
    df_community["Community"] = list(partition.values())
    df_community["Degree"] = degrees

    df_community_group = df_community.groupby(['Community', 'Name', 'Degree']).count().reset_index()
    df_community = df_community_group.copy()

    df_community_group = df_community_group.sort_values('Degree', ascending = False).groupby('Community').head(3)

    df_community_group = df_community_group.groupby('Community')['Name'].agg(lambda col: '-'.join(col)).reset_index()
    df_community_group.columns = ['CommunityID', 'CommunityName']
    df_community_group["CommunitySize"] = community_size


    df_community = pd.merge(df_community, df_community_group, left_on="Community", right_on="CommunityID")

    return df_community_group, df_community


def community_stats(G, partition, df):
    

    st.text(f"Number of communities found: {max(partition.values())+1}")

    st.text(f'The value of modularity is: {round(community_louvain.modularity(partition, G),3)}')

    df = df.sort_values("CommunitySize", ascending=False)

    fig = px.bar(df, x="CommunityName", y="CommunitySize")

    

    fig.update_layout(
        title={
            'text': "Size of Communities",
            'y':0.91,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis_title="Community",
        yaxis_title="Count",
        legend_title="Legend Title",
        template="plotly_dark",
        font=dict(
            family="Sans serif",
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def set_attributes(G, df_community):
    df = pd.read_csv("website/pages/data/cleaned_characters_attr.csv")

    df = pd.merge(df, df_community, left_on="title", right_on="Name")

    df = df.set_index("title")
    df = df[["Race", "Gender", "Nationality", "Family", "Profession", "CommunityName"]]
    attributes = df.to_dict('index')
    nx.set_node_attributes(G, attributes)

    titles = {}
    for name, data in G.nodes(data = True):
        try:
            title = 'Name: ' + name + ' </br>Community: ' + data['CommunityName'] + ' </br>Race: ' + data['Race'] + ' </br>Gender: ' + data['Gender'] + ' </br>Nationality: ' + data['Nationality'] + ' </br>Family: ' + data['Family'] + ' </br>Profession: ' + data['Profession']
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

def set_color(G):

    attributes = nx.get_node_attributes(G, 'CommunityName')
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


def get_commutities(G):

    partition = community_louvain.best_partition(G, random_state=1)

    community = list()
    community_size = list()
    for i in range(max(partition.values())+1):
        community.append(i)
        community_size.append(Counter(partition.values())[i])

    df_group, df = name_communities(G, partition, community_size)

    community_stats(G, partition, df_group)

    return partition, df_group, df 


def community_top_words(chosen_comms, number_of_words, df_com_names, partition):

    if chosen_comms:

        chosen_comms_idx = []
        for i in chosen_comms:
            chosen_comms_idx.append(df_com_names.index[df_com_names['CommunityName'] == i].tolist()[0])

        
        #Generate dictionary of community texts
        texts_dict = {}
        data_root = "website/pages/data/data"


        for idx in chosen_comms_idx:
            char_list = []
            for char in partition.keys():
                if partition[char] == idx:
                    char_list.append(f"{char}.txt")
            
            wordlists_community = PlaintextCorpusReader(data_root, char_list)
            community_texts = nltk.Text(wordlists_community.words())
            texts_dict[idx] = community_texts


        #Make TF dictionary
        TF_dict = {}
        for comm in texts_dict.keys():
            term_dict = {}
            for key, value in FreqDist(texts_dict[comm]).items():
                term_dict[key] = value/len(texts_dict[comm])
            TF_dict[comm] = term_dict


        #Get list of all words from all communities
        gathered_texts = list(list(texts_dict.values())[0])
        for text in list(texts_dict.values())[1:]:
            gathered_texts.extend(list(text))

        #Make IDF dictionary
        idf_dict = {}
        for word in np.unique(gathered_texts):
            number_of_documents = 0
            for comm in texts_dict.keys():
                if word in TF_dict[comm].keys():
                    number_of_documents += 1
            idf_dict[word] = math.log10(len(texts_dict.keys())/number_of_documents)

        #Make tf_idf_dict
        tf_idf_dict = copy.deepcopy(TF_dict)
        for comm in tf_idf_dict.keys():
            for word in tf_idf_dict[comm].keys():
                tf_idf_dict[comm][word] = tf_idf_dict[comm][word] * idf_dict[word]

        #Make prints of top 5 words for each chosen community
        for key in tf_idf_dict.keys():
            st.text(f"Top {number_of_words} words from the {chosen_comms[chosen_comms_idx.index(key)]} community, based on TF-IDF is:")
            st.text(sorted(tf_idf_dict[key], key=tf_idf_dict[key].get, reverse=True)[:number_of_words])

def remove_stopwords(texts, stop_words):
    return [[word for word in doc 
             if word not in stop_words] for doc in texts]

def lda(chosen_comms, df_com_names, partition):

    chosen_comms_idx = []
    for i in chosen_comms:
        chosen_comms_idx.append(df_com_names.index[df_com_names['CommunityName'] == i].tolist()[0])

    stop_words = stopwords.words('english')
    stop_words.extend(['however', 'though', 'would', 'later', 'could'])

    #Generate dictionary of community texts
    comm_data = []
    data_root = "website/pages/data/data"

    for idx in chosen_comms_idx:
        char_list = []
        for char in partition.keys():
            if partition[char] == idx:
                char_list.append(f"{char}.txt")
        
        wordlists_community = PlaintextCorpusReader(data_root, char_list)
        words = wordlists_community.words()
        comm_data.append(words)

    comm_data = remove_stopwords(comm_data, stop_words)

    id2word = corpora.Dictionary(comm_data)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in comm_data]

    model = TfidfModel(corpus=corpus, normalize=True) #fit model


    vector = [model[corpus[i]] for i in range(len(comm_data))] #apply tf_idf

    lda_model = LdaModel(corpus=vector, id2word=id2word, num_topics=5)

    p = gensimvis.prepare(lda_model, corpus, id2word)

    try:
        path = '/tmp'
        pyLDAvis.save_html(p, f'{path}/lda.html')

        HtmlFile = open(f'{path}/lda.html', 'r', encoding='utf-8')

    # Save and read graph as HTML file (locally)
    except:
        path = '/html_files'
        pyLDAvis.save_html(p, f'{path}/lda.html')
        HtmlFile = open(f'{path}/lda.html', 'r', encoding='utf-8')

    # # Load HTML file in HTML component for display on Streamlit page
    background = """
    <head>
    <style>
      
    div {
      background-color: white;
    }
    
    </style>
  </head>
  
    """
    #HtmlFile =  open('website/pages/data/lda/lda5.html')
    #st.text(background + HtmlFile.read())
    components.html(background + HtmlFile.read(), height=1000)

def select_community_graph(G, community):

    selected_nodes = [n for n,v in G.nodes(data=True) if v['CommunityName'] in community]  

    st.text(selected_nodes)     

    H = G.subgraph(selected_nodes)
    return H

def app():

    st.title("Communities and sentiment analysis")

    G = load_graph()

    partition, df_com_names, df = get_commutities(G)

    G = set_attributes(G, df)

    G = set_color(G)

    display = st.radio("Display the network", (False, True), key="rk1")

    communities = np.unique(list(nx.get_node_attributes(G,'CommunityName').values()))

    if display:
        pyviz_plot_network(G, "Barnes Hut")

    chosen_comms = st.multiselect("Choose the communities to view:", communities)

    if chosen_comms:

        H = select_community_graph(G, chosen_comms)

        pyviz_plot_network(H, "Barnes Hut")

    number_of_words = st.slider("Select the number of words to view", 1, 25, 5, 1)

    community_top_words(chosen_comms, number_of_words, df_com_names, partition)

    if chosen_comms:
        with st.container():
            lda(chosen_comms, df_com_names, partition)





        
