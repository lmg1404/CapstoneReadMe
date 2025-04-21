# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for tasks including how to modulize graphing functions, developing color-blind friendly palettes, and debugging.
#   https://chat.openai.com/
#
# Additional Resources Consulted:
# - Stack Overflow (stackoverflow.com) for troubleshooting and syntax patterns
# - Matplotlib documentation (https://matplotlib.org/stable/index.html)
# - Streamlit documentation (https://docs.streamlit.io/)
# - pandas documentation (https://pandas.pydata.org/docs/)
# - Streamlit dashboard example (https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/#2-perform-eda-analysis)
# - Ask Science Direct AI dashboard example (https://www.sciencedirect.com/ai)
#
# ============================================================


import streamlit as st
import os

#limit file watcher
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, get_single_color_func
from pathlib import Path
from collections import Counter
from textblob import TextBlob
import nltk #needed if stopwords need to be redownloaded
from nltk.corpus import stopwords
import re
import spacy
from spacy import displacy
from datetime import datetime, timedelta
import sys
import random
from utils import show_sidebar_logo, show_sidebar_arxiv
import plotly.express as px


#################
#Obtaining and processing papers
#################

sys.path.append(str(Path(__file__).resolve().parent.parent))

#import backend modueles
from utils import get_pg_connection

#color blind friendly palettes sourced and functioning code from ChatGPT (OpenAI, April 2025) as WordCloud does not accept palettes natively
COLORBLIND_PALETTES = {
    "Okabe-Ito": ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#999999"],
    "Color Universal Design (CUD)": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"],
    "Viridis": ["#440154", "#3B528B", "#21908C", "#5DC863", "#FDE725"],  
}

class PaletteColorFunc:
    def __init__(self, palette):
        self.palette = palette
        self.color_func = get_single_color_func(palette[0])

    def __call__(self, word, font_size, position, orientation, font_path, random_state):
        return random.choice(self.palette)



#download stop words, uncomment and run once
#nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2_000_000


def clean_text(text: str) -> str:
    """
    Clean text ahead of visualizing, setting case, keeping only letter, and removing stopwords and words smaller than 3 characters
    Author: Sam
    input:
        text: start date to filter fetched summaries by
        end_date: end date to filter fetched summaries by
        tag:user inputted tags to filter fetched summaries by
    returns:
        list of dictionaries, where each dictionary represents a single summary and its attributes
    """
    text2 = text.lower()  
    text3 = re.sub(r"[^a-z\s]", "", text2)
    words = [word for word in text3.split() if word not in stop_words and len(word) > 2]
    
    return " ".join(words)


@st.cache_data
def get_word_frequencies(text: str) -> Counter:
    """
    Clean text and return a Counter object of the word frequencies, using in multiple visuals to standardize results
    Author: Sam
    input:
        text: text needing word counts for
    returns:
        Counter object of the cleaned words and their frequencies
    """
    cleaned = clean_text(text)
    words = cleaned.lower().split()
    return Counter(words)



#################
#Visualization functions
#################

def show_wordcloud(text: Counter, palette: list[str]) -> None:
    """
    Display a word cloud visualization with a custom color palette.
    Author: Sam
    input:
        text: Counter object of word frequencies
        palette: color palette for the word cloud
    returns:
        None
    """    
    color_func = PaletteColorFunc(palette)
                
    wordcloud = WordCloud(width=800, height=500, background_color="white", color_func=color_func).generate_from_frequencies(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    st.subheader("Word Cloud")
    st.pyplot(fig)

#use this if a sentiment analysis is ever of interest, not used now because academic papers are mostly without sentiment
def show_sentiment_analysis(text: str, palette: list[str]) -> None:
    """
    Show the sentiment of the text with a pie chart
    Author: Sam
    input:
        text: text to analyze for sentiment
        palette: list of colors for the pie chart
    returns:
        None
    """
    sentiment = TextBlob(text).sentiment.polarity
    label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    fig, ax = plt.subplots()
    colors = palette[:2] if len(palette) >= 2 else ['#cccccc', '#999999']
    ax.pie([abs(sentiment), 1 - abs(sentiment)], labels=[label, "Other"], autopct="%1.1f%%", colors=colors)
    
    st.subheader("Sentiment Analysis")
    st.pyplot(fig)

#use this if named entities are ever of interest, not used now
def show_named_entities(text: str) -> None:
    """
    Display named entity recognition results for a snippet of the text.
    Author: Sam
    input:
        text: text for analysis
    returns:
        None
    """
    doc = nlp(text[:1000])
    html = displacy.render(doc, style="ent", page=True)
    
    st.subheader("Named Entity Recognition")
    st.components.v1.html(html, height=300, scrolling=True)

def show_frequent_words(text: Counter, palette: list[str]) -> None:
    """
    Plot the 20 most frequent words using a bar chart.
    Author: Sam
    input:
        text: Counter object of word frequencies, use same as word cloud for standardization
        palette: color palette for chart
    returns:
        None
    """
    
    df = pd.DataFrame(text.items(), columns=["Word", "Count"]).sort_values(by="Count", ascending=False).head(20)

    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(df["Word"], df["Count"], color=palette[:len(df)])
    ax.invert_yaxis()

    ax.set_xlabel("Frequency", fontsize=12)
    
    st.subheader("20 Most Frequent Words")
    st.pyplot(fig)

#use this if the top nouns are ever of interest, not used now
def show_top_nounss(text: str) -> None:
    """
    Display the top 15 most frequent nouns in the text in a dataframe
    Author: Sam
    input:
        text: text string to analyze
    returns:
        None
    """    
    st.subheader("Top Research Sub-Topics")

    doc = nlp(text)

    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3]

    word_counts = Counter(keywords)
    df_keywords = pd.DataFrame(word_counts.items(), columns=["Keyword", "Count"]).sort_values(by="Count", ascending=False).head(15)
    
    max_count = df_keywords["Count"].max()
    st.dataframe(
        df_keywords,
        column_order=("Keyword", "Count"),
        hide_index=True,
        width=None,
        column_config={
            "Keyword": st.column_config.TextColumn("Topic"),
            "Count": st.column_config.ProgressColumn(
                "Mentions", format="%d", min_value=0, max_value=int(max_count)
            )
        }
    )

def show_top_topics_from_tags(summaries: list[dict]) -> None:
    """
    Analyze and display the most common tags (topics) found in the paper summaries
    Author: Sam
    input:
        summaries: list of dictionaries of paper metadata
    returns:
        None
    """
    st.subheader("Top Research Sub-Topics (by Tags)")

    all_tags = []
    for paper in summaries:
        tags = paper.get("tags", [])
        all_tags.extend([tag for tag in tags if tag and tag.lower() != "none"])

    if not all_tags:
        st.info("No tags found in the summaries.")
        return

    tag_counts = Counter(all_tags)
    df_tags = pd.DataFrame(tag_counts.items(), columns=["Tag", "Count"]).sort_values(by="Count", ascending=False)

    max_count = df_tags["Count"].max()
    st.dataframe(
        df_tags,
        column_order=("Tag", "Count"),
        hide_index=True,
        width=None,
        column_config={
            "Tag": st.column_config.TextColumn("Topic"),
            "Count": st.column_config.ProgressColumn(
                "Mentions", format="%d", min_value=0, max_value=int(max_count)
            )
        }
    )

def get_top_term(text: str) -> str:
    """
    Identify the most frequent noun in the provided text for quick metrics
    Author: Sam
    input:
        text: text to analyze
    returns:
        the most common noun found or "N/A" if no nouns found
    """
    doc = nlp(text)
    keywords = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3]
    keyword_counts = Counter(keywords)
    df_keywords = pd.DataFrame(keyword_counts.items(), columns=["Keyword", "Count"]).sort_values(by="Count", ascending=False)
    
    top = df_keywords.iloc[0]["Keyword"] if not df_keywords.empty else "N/A"
    
    return top

def get_papers_over_time(start: datetime, end: datetime, query: list[str], conn) -> None:
    """
    Plot the number of papers published over time that we were able to summarize within a given date range
    Author: Sam
    input:
        start: start date 
        end: end date 
        query: list of tags/topics to filter by
        conn: active database connection
    returns:
        None
    """    
    cursor = conn.cursor()
    query_string = """
        SELECT p.publishdate::date, COUNT(*)
        FROM papers p
        JOIN summaries s ON p.paper_id = s.paper_id
        WHERE p.publishdate BETWEEN %s AND %s
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(p.tags) AS tag
            WHERE tag IN %s
        )
        GROUP BY p.publishdate
        ORDER BY p.publishdate
    """
    cursor.execute(query_string, (start, end, tuple(query)))
    rows = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(rows, columns=["Date", "Paper Count"])
    df["Date"] = pd.to_datetime(df["Date"])
    st.subheader("Summarized Papers by Publish Date")
    st.line_chart(df.set_index("Date"))

def get_papers_vs_summaries_over_time(start: datetime, end: datetime, tag: list[str], conn) -> pd.DataFrame:
    """
    Compare total papers published vs. papers we were able to summarize over time, very helpful for debugging
    Author: Sam
    input:
        start: start of date range
        end: end of date range
        tag: list of tags to filter by
        conn: active database connection
    returns:
        dataframe with columns for date, total papers, and summarized papers
    """
    cursor = conn.cursor()

    #total papers per day
    cursor.execute("""
        SELECT p.publishdate::date, COUNT(*)
        FROM papers p
        WHERE p.publishdate BETWEEN %s AND %s
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(p.tags) AS tag
            WHERE tag = ANY(%s)
        )
        GROUP BY p.publishdate
        ORDER BY p.publishdate
    """, (start, end, tag))
    papers = cursor.fetchall()

    #summarized papers per day
    cursor.execute("""
        SELECT p.publishdate::date, COUNT(*)
        FROM papers p
        JOIN summaries s ON p.paper_id = s.paper_id
        WHERE p.publishdate BETWEEN %s AND %s
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(p.tags) AS tag
            WHERE tag = ANY(%s)
        )
        GROUP BY p.publishdate
        ORDER BY p.publishdate
    """, (start, end, tag))
    summaries = cursor.fetchall()

    cursor.close()

    df_papers = pd.DataFrame(papers, columns=["date", "total papers"])
    df_summaries = pd.DataFrame(summaries, columns=["date", "summarized papers"])

    df_combined = pd.merge(df_papers, df_summaries, on="date", how="outer").fillna(0)
    df_combined["date"] = pd.to_datetime(df_combined["date"])
    df_combined = df_combined.sort_values("date")

    return df_combined


#show how many hits they got per tag
def show_topic_distribution(tags: list[str], start: datetime, end: datetime, conn, palette: list[str]) -> None:
    """
    Create a donut chart of paper counts grouped by selected tags/topics. Assistance from ChatGPT resizing chart
    Author: Sam
    input:
        tags: selected tags/topics
        start: start of date range
        end: end of date range
        conn: active database connection
        palette: color palette for the chart
    returns:
        None
    """
    cursor = conn.cursor()
    query = """
        SELECT tag, COUNT(*)
        FROM (
            SELECT jsonb_array_elements_text(p.tags) AS tag
            FROM papers p
            JOIN summaries s ON p.paper_id = s.paper_id
            WHERE p.publishdate BETWEEN %s AND %s
        ) AS tag_table
        WHERE tag = ANY(%s::text[])
        GROUP BY tag;
    """
    cursor.execute(query, (start, end, tags))
    rows = cursor.fetchall()
    cursor.close()

    df = pd.DataFrame(rows, columns=["Tag", "Count"])
    if df.empty:
        st.warning("No papers found for the selected tags in this time range.")
        return

    df["Label"] = df.apply(lambda row: f"{row['Tag']} ({row['Count']} papers)", axis=1)

    color_map = {label: color for label, color in zip(df["Label"], palette)}

    fig = px.pie(
        df,
        names="Label",
        values="Count",
        hole=0.4,
        color="Label",
        color_discrete_map=color_map
    )

    fig.update_traces(textinfo='none')

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            y=-0.15,
            x=0.5,
            xanchor="center",
            yanchor="bottom",
            font=dict(size=18),
        ),
        margin=dict(t=60, b=40)
    )

    st.subheader("Breakdown of Papers Found by Topic")
    
    st.plotly_chart(fig, use_container_width=True)



def avg_word_count(analytics: list[dict]) -> str:
    """
    Calculate the average word count across all papers for quick metrics
    Author: Sam
    input:
        analytics: list of analytics dictionaries from all papers
    returns:
        average word count as string for Streamlit, or "N/A" if no papers found
    """
    total_words = 0
    paper_count = 0

    for record in analytics:
        word_count_dict = record.get("word_count", {})
        if isinstance(word_count_dict, dict):
            total_words += sum(word_count_dict.values())
            paper_count += 1

    if paper_count > 0:
        avg = round(total_words / paper_count) 
    else: 
        avg = None
    
    #add comma if value goes into the thousands
    return f"{avg:,}" if avg else "N/A"



@st.cache_data
def get_combined_text_from_summaries(summaries: list[dict]) -> str:
    """
    Combine all summary texts into a single string
    Author: Sam
    input:
        summaries: list of summary dictionaries
    returns:
        one long string of all summary texts combined
    """
    return "\n".join([s["summary"] for s in summaries if "summary" in s])



#################
#streamlit UI
#################
#set page structure
st.set_page_config(
    page_title="Research Analytics Dashboard",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

#sidebar for user customization, accessiblity focused
with st.sidebar:
    st.title("User Customization")
    palette_choice = st.selectbox("Color-blind friendly palettes:", list(COLORBLIND_PALETTES.keys()))
    selected_palette = COLORBLIND_PALETTES[palette_choice]

#get data and variables from topic selection.py
start_date = st.session_state.get("start_date", datetime.today() - timedelta(days=100))
end_date = st.session_state.get("end_date", datetime.today())
summaries = st.session_state.get("summaries", [])
analytics = st.session_state.get("analytics", [])
query = st.session_state.get("query_tags", ["Artificial Intelligence"])

pdf_text = get_combined_text_from_summaries(summaries)

#changing to this for word cloud and word frequency for standardization
word_frequencies = get_word_frequencies(pdf_text)


#open connection once
conn = get_pg_connection()

#debugging, uncomment if visuals are not being displayed
# cursor = conn.cursor()
# cursor.execute("SELECT title, publishdate FROM papers ORDER BY publishdate DESC LIMIT 5")
# st.write("Recent papers:", cursor.fetchall())


#show logo in sidebar
show_sidebar_logo()
show_sidebar_arxiv()

st.title("Research Paper Analytics")

st.markdown("### Explore article content via the words, topics and trends that compose them.")

st.caption(f"Showing insights for papers on **{query}** from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ")



#quick metrics
m1, m2, m3 = st.columns(3)

articles_scraped = len(summaries)
avg_word_count = avg_word_count(analytics)
top_term = get_top_term(pdf_text)

m1.metric(label="Articles Summarized", value=articles_scraped)
m2.metric(label="Average Word Count", value=avg_word_count)
m3.metric(label="Most Frequent Word", value=top_term)


###################
#main page body


if pdf_text:

    tab1, tab2, tab3 = st.tabs(["Words", "Topics", "Trends"])

    #Term graphics
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            show_wordcloud(word_frequencies, selected_palette)
            
        with col2:
            show_frequent_words(word_frequencies, selected_palette)

        if st.button("View Paper Summaries", key="terms"):
            st.session_state["go_to_summaries"] = True

    #Topics
    with tab2:
        col1, col2 = st.columns([1.15, 1])

        with col1:
            show_topic_distribution(query, start_date, end_date, conn, selected_palette)
                
        with col2:
            show_top_topics_from_tags(summaries)
            
        if st.button("View Paper Summaries", key="topics"):
            st.session_state["go_to_summaries"] = True


    #Trend graphics
    with tab3:
        get_papers_over_time(start_date, end_date, query, conn)

        if st.button("View Paper Summaries", key="trends"):
            st.session_state["go_to_summaries"] = True
    
        #uncomment this line to visualize the number of papers actually able to be summarized
        #df_compare = get_papers_vs_summaries_over_time(start_date, end_date, query, conn)
        #st.subheader("Papers vs. Summarized Papers Over Time")
        #st.line_chart(df_compare.set_index("Date"))


    #ideas for future enhancements:
    
    #collaboration network using networkx
    #line chart comparing publish rate by topic

else:
    st.warning("No papers found. Please go to the Topic Selection page to get started!")



#close connection
conn.close()


#redirect if customer wants to go to summary page to cut down on the awkward loading time
if st.session_state.get("go_to_summaries"):
    st.session_state["go_to_summaries"] = False
    st.switch_page("pages/4_Summary_Library.py")



if __name__ == "__main__":

    #space for debugging
    pass
