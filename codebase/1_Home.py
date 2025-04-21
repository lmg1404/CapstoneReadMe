# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for tasks including centering html and debugging.
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


st.set_page_config(page_title="README", page_icon="📖", layout="wide")

#show logo in sidebar
from utils import show_sidebar_logo, show_sidebar_arxiv
show_sidebar_logo()
show_sidebar_arxiv()

#st.title("Welcome to README, your AI-Powered Dashboard for Understanding the Latest Machine Learning Research")


st.markdown("""
<div style='text-align: center;'>
            
# Welcome to README, your AI-Powered Dashboard for Understanding the Latest Machine Learning Research
            
## How to Use This Tool

### 1️⃣ Choose Sub-Topics  
Go to **Topic Selection** and choose your area of interest by sub-topics in Machine Learning (like AI, Robotics, etc.).

⬇️

### 2️⃣ Explore Analytics  
View publication trends, top sub-topics, word clouds, and more on the papers under that topic in the **Analytics Dashboard**.

⬇️

### 3️⃣ Read Summaries  
Go to **Summary Library** to browse and select simplified versions of academic papers, with key terms highlighted and defined.

⬇️

### 4️⃣ Ask the Chatbot  
Ask questions about selected papers using the **Q&A Chatbot** for deeper understanding.

</div>
""", unsafe_allow_html=True)



st.markdown(
    """
    ### Disclaimer

    This tool is designed to support researchers, students, and professionals in exploring and engaging with technical research papers. It leverages large language models (LLMs) to summarize and interact with paper content.

    Please note:

    - The responses generated are based on machine learning predictions and may not always reflect the original paper accurately.
    - This application should be used as a starting point for exploration, not a substitute for reading the source material.
    - We strongly encourage users to verify outputs against the original publications and, when necessary, consult domain experts.

    The development team has taken care to minimize errors and biases, but:
    - The underlying AI model may produce incomplete, imprecise, or misleading statements.
    - Generated content should not be cited or used as an authoritative source.

    This interface is intended to enhance comprehension, not replace critical thinking or scholarly review.
    """)

st.markdown("### Ready to get started?")

if st.button("Go to Topic Selection"):
    st.switch_page("pages/2_Topic_Selection.py") 