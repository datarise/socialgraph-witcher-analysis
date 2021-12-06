# Import dependencies
import streamlit as st
import pandas as pd
import plotly.express as px


@st.cache()
def load_data():
    df = pd.read_csv("website/pages/data/characters_sentiment.csv")
    return df

def create_sentiment_analysis(df_sentiment, attribute):
    df_grouped = df_sentiment[[attribute, "Vader_Sentiment"]].groupby(attribute).mean().reset_index()
    df_grouped['Vader_std'] = df_sentiment[[attribute, "Vader_Sentiment"]].groupby(attribute).std().reset_index()['Vader_Sentiment']

    df_happy = df_grouped.sort_values("Vader_Sentiment", ascending=False).head(5)
    df_unhappy = df_grouped.sort_values("Vader_Sentiment").head(5)
    return df_happy, df_unhappy, df_grouped

def make_plot(df_grouped, attribute):
    if attribute == 'Name':
        x = [i for i in df_grouped["Name"]]
        y = [i for i in df_grouped["Vader_Sentiment"]]
        error = [i for i in df_grouped["Vader_std"]]



        for i in range(7):
            fig = px.bar(x=x[(i*105):((i+1)*105)], y=y[(i*105):((i+1)*105)], error_y=error[(i*105):((i+1)*105)])

            fig.update_layout(
            title={
                'text': f"Sentiment Grouped by {attribute}",
                'y':0.91,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis={
                'title': f"{attribute}"},
            yaxis={
                'title': f"Sentiment"},
            template="plotly_dark",
            font=dict(
                family="Sans serif",
            )
            )
            st.plotly_chart(fig, use_container_width=True)

    elif attribute == 'Profession':
        x = [i for i in df_grouped["Profession"]]
        y = [i for i in df_grouped["Vader_Sentiment"]]
        error = [i for i in df_grouped["Vader_std"]]


        for i in range(2):

            fig = px.bar(x=x[(i*80):((i+1)*80)], y=y[(i*80):((i+1)*80)], error_y=error[(i*80):((i+1)*80)])

            fig.update_layout(
            title={
                'text': f"Sentiment Grouped by {attribute}",
                'y':0.91,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            xaxis={
                'title': f"{attribute}"},
            yaxis={
                'title': f"Sentiment"},
            template="plotly_dark",
            font=dict(
                family="Sans serif",
            )
            )
            st.plotly_chart(fig, use_container_width=True)


    else:
        fig = px.bar(df_grouped, x=attribute, y='Vader_Sentiment', error_y='Vader_std')

        fig.update_layout(
        title={
            'text': f"Sentiment Grouped by {attribute}",
            'y':0.91,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'title': f"{attribute}"},
        yaxis={
            'title': f"Sentiment"},
        template="plotly_dark",
        font=dict(
            family="Sans serif",
        )
        )
        st.plotly_chart(fig, use_container_width=True)

from pathlib import Path
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def app():
    df = load_data()

    # Read dataset (CSV)
    # Title of the main page
    st.title("Sentiment Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(read_markdown_file("website/pages/text/sentiment1.md"), unsafe_allow_html=True)


    with col2:
        fig = px.histogram(df, x="Vader_Sentiment", nbins=25)
        fig.update_layout(
        title={
            'text': f"Histogram of Sentiment",
            'y':0.91,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        xaxis={
            'title': "Bins"},
        yaxis={
            'title': "Count"},
        template="plotly_dark",
        font=dict(
            family="Sans serif",
        )
        )
        st.plotly_chart(fig, use_container_width=True)

    attribute = st.selectbox("Select the attribute to group by: ", df.columns[:-1])
    df_happy, df_unhappy, df_grouped = create_sentiment_analysis(df, attribute=attribute)
    make_plot(df_grouped, attribute=attribute)