# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for tasks including highlighting technical terms, defining a photo using html, and debugging.
#   https://chat.openai.com/
#
# Additional Resources Consulted:
# - Stack Overflow (stackoverflow.com) for troubleshooting and syntax patterns
# - Matplotlib documentation (https://matplotlib.org/stable/index.html)
# - Streamlit documentation (https://docs.streamlit.io/)
# - pandas documentation (https://pandas.pydata.org/docs/)
# - Streamlit dashboard example (https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/#2-perform-eda-analysis)
#
# ============================================================

import psycopg2
import shutil
import os
from dotenv import load_dotenv
load_dotenv()

def get_pg_connection():
  """
  Get PG connection to pass into other functions
  Author: Luis
  
  input:
    None
  returns:
    PGSQL connection
  """
  conn = psycopg2.connect(
    host=os.environ['PGHOST'],
    database=os.environ['PGDB'],
    user= os.environ['PGUSER'],
    password=os.environ['PGPASSWORD'],
    port='5432'
  )
  return conn

def teardown(start_date) -> None:
  """
  Delete pdf folder
  Author: Luis
  
  input:
    start_date: folder name which are start dates
  returns:
    None
  """ 
  shutil.rmtree(f'{start_date}')
  return None

"""
UI Functions
"""

import base64
from pathlib import Path
import streamlit as st

def show_sidebar_logo(path: str = "codebase/assets/readme_logo.png", width: int = 250) -> None:
    """
    Display the ReadMe logo in the Streamlit sidebar across all pages
    Author: Sam
    input:
        path: path to the logo image file
        width: pixel width to display the image
    returns:
        None
    """
    logo_path = Path(path)
    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode()
        st.sidebar.markdown(
            f"""
            <hr style="margin-top: 1em; margin-bottom: .25em;">
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" width="{width}" style="max-width: 100%;">
                <div style="margin-top: 0.5rem; font-weight: bold; font-size: 1rem; color: #444;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    #debugging
    else:
        st.sidebar.warning("logo file not found.")


def show_sidebar_arxiv(path: str = "codebase/assets/arxiv_logo_transparent.png", width: int = 150) -> None:
    """
    Display the arXiv logo in the Streamlit sidebar to make sure they are properly credited
    Author: Sam
    input:
        path: path to the logo image file
        width: pixel width to display the image
    returns:
        None
    """
    logo_path = Path(path)
    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode()
        st.sidebar.markdown(
            f"""
            <hr style="margin-top: .25em; margin-bottom: .25em;">
            <div style="text-align: center;">
                <img src="data:image/png;base64,{encoded}" width="{width}" style="max-width: 100%;">
                <div style="margin-top: 0.25rem; font-weight: bold; font-size: 1rem; color: #444;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    #debugging
    else:
        st.sidebar.warning("arxiv file not found.")



import html
import re

def highlight_keywords(text: str, keywords: dict[str, str]) -> str:
    """
    Allow user to highlight key terms when toggled on, furthering our accessibility mission
    Author: Sam, with guidance from ChatGPT, particularly tooltip styling and protecting tooltip style (OpenAI, April 2025)
    input:
        text: text to highlight keywords in
        keywords: dictionary of keywords and their definitions (as pairs)
    returns:
        Text with tooltip wrapping on each keyword
    """
    if not keywords:
        return text
    
    placeholder = "§§§PLACEHOLDER§§§"
    safe_text = text

    safe_text = re.sub(r'(<span class="tooltiptext">.*?</span>)', placeholder, safe_text, flags=re.DOTALL)

    for term, definition in keywords.items():
        safe_term = html.escape(term)
        safe_def = html.escape(definition)

        tooltip_html = f'''
            <span class="tooltip">{safe_term}
                <span class="tooltiptext">{safe_def}</span>
            </span>
        '''

        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        safe_text = pattern.sub(tooltip_html, safe_text)

    safe_text = safe_text.replace(placeholder, r'\1')

    return safe_text



#set tooltip style manually to override streamlit defaults
#tooltip styling created/debugged with guidance from ChatGPT (OpenAI, April 2025)

tooltip_style = """
<style>
.tooltip {
    position: relative;
    display: inline-block;
    background-color: yellow;
    cursor: help;
    padding: 1px 4px;
    border-radius: 3px;
}

.tooltip .tooltiptext {
    visibility: hidden;
    width: max-content;
    max-width: 300px;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 8px 10px;
    position: absolute;
    z-index: 9999;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.85em;
    line-height: 1.4;
    white-space: normal;
}

.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
</style>
"""

