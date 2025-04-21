# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for debugging.
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

from typing import List, Dict, Any
import datetime
import sys
from pathlib import Path
from utils import get_pg_connection


sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="README", page_icon="📖", layout="wide")


########################
#basic paper pulling queries
##########################
#updated to use postgres code to fetch papers, allow user to set date range and tag, with help from Luis
@st.cache_data
def fetch_summaries(start_date: datetime.date, end_date: datetime.date, tag: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch summaries from the database according to user inputted variables
    Author: Sam and Luis
    input:
        start_date: start date to filter fetched summaries by
        end_date: end date to filter fetched summaries by
        tag:user inputted tags to filter fetched summaries by
    returns:
        list of dictionaries, where each dictionary represents a single summary and its attributes
    """
    conn = get_pg_connection()
    cursor = conn.cursor()
    query = """
        SELECT papers.title, summaries.abstract, summaries.claude_summary, summaries.keywords,  papers.pdf_link, 
        papers.citations, summaries.sample_qa, papers.tags
        FROM papers
        JOIN summaries ON papers.paper_id = summaries.paper_id
        WHERE papers.publishdate BETWEEN %s AND %s
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(papers.tags) AS tag
            WHERE tag = ANY(%s)
        );
    """
    cursor.execute(query, (start_date, end_date, tag))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [
        {
            "title": row[0],
            "abstract": row[1],
            "summary": row[2],
            "technical_terms": row[3] if row[3] else [],
            "pdf_link": row[4],
            "citations": row[5],
            'sample_qa': row[6],
            "tags": row[7] if row[7] else []
        }
        for row in rows
    ]

@st.cache_data
def fetch_analytics(start_date: datetime.date, end_date: datetime.date, tag: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch analytics from the database according to user inputted variables
    Author: Sam and Luis
    input:
        start_date: start date to filter fetched papers by
        end_date: end date to filter fetched papers by
        tag:user inputted tags to filter fetched papers by
    returns:
        list of dictionaries, where each dictionary represents a single paper and its analytical attributes
    """
    conn = get_pg_connection()
    cursor = conn.cursor()
    query_string = """
        SELECT p.title, a.sentiment, a.named_entities, a.word_count
        FROM papers p
        JOIN analytics a ON p.paper_id = a.paper_id
        WHERE p.publishdate BETWEEN %s AND %s
        AND EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(p.tags) AS tag
            WHERE tag = ANY(%s)
        );
    """
    cursor.execute(query_string, (start_date, end_date, tag))
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    return [
        {
            "title": row[0],
            "sentiment": row[1],
            "named_entities": row[2] if row[2] else {},
            "word_count": row[3] if row[3] else {}
        }
        for row in rows
    ]

@st.cache_data
def get_unique_tags() -> List[str]:
    """
    Get a list of all the unique tags of the papers in the database for users to query by
    Author: Sam
    input:
        None
    returns:
        list of available tags, sorted alphabetically
    """
    conn = get_pg_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT jsonb_array_elements_text(tags) AS tag FROM papers;")
    tags = [row[0] for row in cursor.fetchall() if row[0] is not None]
    cursor.close()
    conn.close()
    return sorted(tags)

#show logo in sidebar
from utils import show_sidebar_logo, show_sidebar_arxiv
show_sidebar_logo()
show_sidebar_arxiv()


st.title("To begin, select one or more of the available Machine Learning sub-topics:")

#switch to search by tag, not category
available_tags = get_unique_tags()

default_tag = "Artificial Intelligence" if "Artificial Intelligence" in available_tags else available_tags[0]

selected_tags = st.multiselect("Choose sub-topics", available_tags, default=[default_tag])
st.session_state.query_tags = selected_tags


st.title("And, a date range to query by:")

min_date = datetime.date(2025, 1, 1)
max_date = datetime.date(2025, 3, 31)

start_date = st.date_input("Start date", value=max_date - datetime.timedelta(days=30), min_value=min_date, max_value=max_date)
end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

st.session_state["start_date"] = start_date
st.session_state["end_date"] = end_date


#uncomment to limit numbers of papers fetched, optional
# st.session_state.max_results = st.slider("How many papers would you like to fetch?", 1, 25, 5)

#go to next page
if st.button("Search"):
    if start_date and end_date and selected_tags:
        st.session_state["summaries"] = fetch_summaries(start_date, end_date, selected_tags)
        st.session_state["analytics"] = fetch_analytics(start_date, end_date, selected_tags)
    st.switch_page("pages/3_Overview_Analytics.py")

