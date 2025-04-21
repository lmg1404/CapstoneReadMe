# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for tasks including highlighting technical terms, applying high contrast to tooltips and popups, 
#   using html to nest expanders, and debugging.
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
from utils import show_sidebar_logo, highlight_keywords, tooltip_style, show_sidebar_arxiv
#from rouge_score import rouge_scorer   #uncomment if you would like Rouge scores displayed as well
import bert_score


#limit file watcher
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"


def interpret_score(score: float, thresholds: tuple = (0.2, 0.5, 0.75)) -> tuple[str, str]:
    """
    Interpret the summary scoring emtrics and return a rating and associated color, enhancing user understanding
    Author: Sam
    input:
        score: the metric to evaluate
        thresholds: tuple defining the cutoff values for each score tier
    returns:
        A tuple of the rating and corresponding color
    """
    if score >= thresholds[2]:
        return "🟢 Good", "green"
    elif score >= thresholds[1]:
        return "🟡 Moderate", "orange"
    elif score >= thresholds[0]:
        return "🔴 Needs Improvement", "red"
    else:
        return "⚫️ Poor", "gray"


#########################
#start of Streamlit UI
#########################

st.set_page_config(page_title="README", page_icon="📖", layout="wide")
st.title("Summary Library")

st.markdown("#### Select and read simplified versions of academic papers. Activate ‘Highlight Technical Terms’ to identify key terms and hover over them for definitions. Click on the link to the original article to access full, authoritative content.")

#get tooltip style from utils
st.markdown(tooltip_style, unsafe_allow_html=True)

#add user customization toggles
with st.sidebar:
    st.title("User Customization")
    enable_keywords = st.checkbox("Highlight Technical Terms", value=True)
    enable_contrast = st.sidebar.checkbox("High Contrast Mode")
    
    font_size = st.selectbox(
        "Choose Font Size",
        options=["Small", "Medium", "Large"],
        index=1
    )

#applying high contract to additional text elements created/debugged with guidance from ChatGPT (OpenAI, April 2025)
if enable_contrast:
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #000 !important;
            color: #fff !important;
        }

        div[data-testid="stChatMessage"] {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #444;
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }

        textarea {
            background-color: #222 !important;
            color: #fff !important;
        }

        .stButton>button, .stSlider, .stTextInput>div>div>input {
            background-color: #111 !important;
            color: #fff !important;
            border: 1px solid #999 !important;
        }

        .streamlit-expanderHeader {
            color: #fff !important;
            background-color: #222 !important;
        }

        .streamlit-expanderContent {
            background-color: #111 !important;
        }

        .stMultiSelect > div {
            background-color: #111 !important;
            color: #fff !important;
        }

        .tooltip {
            background-color: #000 !important;
            color: #fff !important;
            border: 1px solid #888 !important;
        }

        strong {
            color: #f9c74f !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
tip_bg = "#111" if enable_contrast else "#eef6fb"
tip_border = "#f9c74f" if enable_contrast else "steelblue"
tip_text = "#fff" if enable_contrast else "#000"
citation_color = "#fff" if enable_contrast else "#000"


font_conversion = {"Small": 12, "Medium": 16, "Large": 20, "Extra-Large": 24}[font_size]
st.markdown(
        f"""
        <style>
        div[data-testid="stMarkdownContainer"] > * {{
            font-size: {font_conversion}px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
)



#show logo in sidebar
show_sidebar_logo()
show_sidebar_arxiv()


#guide user back to topic selection if they haven't searched yet
if "summaries" not in st.session_state:
    st.warning("No summaries found. Please visit the Topic Selection page first.")
else:
    summaries = st.session_state.get("summaries", [])
    left_col, right_col = st.columns([1,4])

    

    with left_col:
        st.markdown("### Select Papers")

        selected_titles = st.multiselect(
            label="Paper Titles",
            options=[paper["title"] for paper in summaries],
            label_visibility="collapsed",
            default=st.session_state.get("query_titles", []),
            max_selections=50,
        )
        with st.container():
            st.markdown(
                f"""
                <div style="
                    margin: 12px 0;
                    padding-left: 12px;
                    border-left: 4px solid {tip_border};
                    background: {tip_bg};
                    color: {tip_text};
                ">
                    <p style="margin-bottom: 4px;"><strong>Have a question about the paper?</strong></p>
                    Click below to interact with our chatbot.
                </div>
                """,
                unsafe_allow_html=True
            )
 
            st.markdown("")

            if st.button("Chatbot"):
                st.switch_page("pages/5_Q&A_Chatbot.py")

    st.session_state["query_titles"] = selected_titles


    with right_col:
        st.markdown("### Summaries (based on Articles from arXiv)")

        if not selected_titles:
            st.info("Select one or more papers to view their summaries.")
        else:
            for paper in summaries:
                if paper["title"] in selected_titles:
                    with st.expander(paper["title"]):
                        abstract = paper["abstract"]
                        summary = paper["summary"]
                        terms = paper.get("technical_terms", {})
                        qa = paper.get("sample_qa", {})



                        #optional keyword highlighting
                        if enable_keywords and terms:
                            abstract = highlight_keywords(abstract, terms)
                            summary = highlight_keywords(summary, terms)

                        #uncomment to display abstract alongside summary
                        #st.markdown(f"<p><strong>Abstract:</strong></p>{abstract}", unsafe_allow_html=True)
                        
                        st.markdown(f"<p><strong>Summary:</strong></p>{summary}", unsafe_allow_html=True)

                        abstract_link = paper.get("pdf_link").replace("/pdf/", "/abs/")
                        if abstract_link != None:
                            st.markdown(f"[View Abstract and Access Full PDF]({abstract_link})", unsafe_allow_html=True)
                        citations = paper.get("citations")
                        if citations is not None:
                            st.markdown(
                                f"<p style='font-size: 0.9em; color: {citation_color};'>Citation: {paper['citations']}</p>",
                                unsafe_allow_html=True )
                            
                        
                        
                        
                        #now, also offer stats about the summary quality

                        reference = paper["abstract"]
                        prediction = paper["summary"]

                        
                        #uncomment this out to add Rouge calculations back into displayed metrics
                        # rs = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                        # rouge = rs.score(reference, prediction)

                        # rouge1 = round(rouge["rouge1"].fmeasure, 4)
                        # rouge2 = round(rouge["rouge2"].fmeasure, 4)
                        # rougeL = round(rouge["rougeL"].fmeasure, 4)

                        P, R, F1 = bert_score.score([prediction], [reference], lang="en", model_type="distilbert-base-uncased")
                        bert = round(F1[0].item(), 4)

                        st.markdown("### **Summary Evaluation Metric**")

                        metrics = {
                            #uncomment this out to add Rouge calculations back into displayed metrics
                            #"ROUGE-1 F1": rouge1,
                            #"ROUGE-2 F1": rouge2,
                            #"ROUGE-L F1": rougeL,
                            "BERTScore F1": bert
                        }

                        cols = st.columns(len(metrics))

                        for col, (label, score) in zip(cols, metrics.items()):
                            rating, color = interpret_score(score)
                            col.metric(
                                label=f"{label} {rating}",
                                value=score
                            )
                        
                        #using HTML to nest expanders (not allowed with only Streamlit objects) created/debugged with guidance from ChatGPT (OpenAI, April 2025)
                        st.markdown(f"""
                        <details>
                        <summary style='cursor:pointer; font-weight:bold;'>What do these metrics mean?</summary>
                        <div style='margin-top: 10px; font-size: 0.95em;'>

                        <ul>

                        <li><strong>BERTScore</strong>: Uses a deep learning model (BERT) to compare the meaning of words and phrases in context. Even if the wording is different, BERTScore checks whether the summaries are semantically similar to the original article abstracts. Scores range from 0 to 1, and values above 0.85 generally indicate high-quality semantic matches.</li>
                        </ul>

                        </div>
                        </details>
                        """, unsafe_allow_html=True)