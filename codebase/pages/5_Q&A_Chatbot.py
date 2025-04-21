# ============================================================
# Project: README - Research Paper Analysis Tool
# Authors: Luis Gallego, Katherine Godwin, Samarth Somani, Samantha Russel
# Date: April 2025
#
# AI Assistance:
# - Portions of this code were developed with help from ChatGPT (OpenAI, April 2025)
#   for tasks including chatbot conversational logic, storing past conversation flows, 
#   and debugging.
#   https://chat.openai.com/
#
# Additional Resources Consulted:
# - Stack Overflow (stackoverflow.com) for troubleshooting and syntax patterns
# - Matplotlib documentation (https://matplotlib.org/stable/index.html)
# - Streamlit documentation (https://docs.streamlit.io/)
# - pandas documentation (https://pandas.pydata.org/docs/)
# - Streamlit dashboard example (https://blog.streamlit.io/crafting-a-dashboard-app-in-python-using-streamlit/#2-perform-eda-analysis)
# - Ask Science Direct AI dashboard example (https://www.sciencedirect.com/ai)
# ============================================================



import streamlit as st
import os
from utils import show_sidebar_logo, show_sidebar_arxiv
import random

#limit file watcher
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

#rag libraries
from qdrant_client import QdrantClient
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client.models import Filter, FieldCondition, MatchAny


#get api keys (moving these to .env)
from dotenv import load_dotenv

load_dotenv()

QDRANT_API_KEY = os.getenv("QKEY")
QDRANT_URL = os.getenv("QURL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY2")

def get_chain(lst: list[str]) -> ConversationalRetrievalChain:
    """
    Create a conversational retrieval chain using a Qdrant vector store and Anthropic language model
    Author: Samarth
    input:
        lst: list of paper titles to filter for
    returns:
        ConversationalRetrievalChain object
    """

    qdrant_client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
    embeddings = HuggingFaceEmbeddings(model_name="malteos/scincl")
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name="ReadMe",
        embeddings=embeddings
    )

    metadata_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.title",
                match=MatchAny(any=lst)
            )
        ]
    )

    retriever = qdrant_store.as_retriever(
        search_kwargs={
            "filter": metadata_filter
        }
    )

    llm = ChatAnthropic(
        model="claude-3-opus-20240229",
        temperature=0.5,
        anthropic_api_key=ANTHROPIC_API_KEY
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return chain



#########################
#start of Streamlit UI
#########################
st.set_page_config(page_title="README", page_icon="📖", layout="wide")

if "submitted_question" not in st.session_state:
    st.session_state.submitted_question = None

st.title("Questions on the papers? Consult with your Q&A Chatbot!")

st.markdown("Use the Q&A Chatbot to ask specific follow-up questions based on the research paper summaries. Tip: Avoid broad or general questions.")

st.markdown("As a reminder, the responses generated are based on machine learning predictions and may not always reflect the original paper accurately. This application should be used as a starting point for exploration, not a substitute for reading the source material. We strongly encourage users to verify outputs against the original publications and, when necessary, consult domain experts. The development team has taken care to minimize errors and biases, but the underlying AI model may produce incomplete, imprecise, or misleading statements. Generated content should not be cited or used as an authoritative source.")

#switching to the html style expander, modeled after the metrics expander in the Summary Library
st.markdown(f"""
<details>
<summary style='cursor:pointer; font-weight:bold;'>This Q&A Chatbot is built using RAG. What is RAG?</summary>
<div style='margin-top: 10px; font-size: 0.95em; line-height: 1.6;'>

<p><strong>Retrieval-Augmented Generation (RAG)</strong> is an approach that combines a <em>retriever</em> and a <em>generator</em> to produce more accurate and informed answers. The retriever first searches a knowledge base (like a document store or database) for relevant information. Those retrieved snippets or facts are then passed to the generator (like a large language model), which uses them to construct a response. This helps ensure that the final answer is grounded in real data, reducing errors and hallucinations.</p>

</div>
</details>
            
<div style='margin-bottom: 20px;'></div>
""", unsafe_allow_html=True)

#show logo in sidebar
show_sidebar_logo()
show_sidebar_arxiv()


selected_titles = st.session_state.get("query_titles")
if not selected_titles:
    st.warning("No paper titles selected. Please choose papers from the Summary Library or start a new search from Topic Selection.")
    st.stop()



#reuse the chain
if (
    "conversational_chain" not in st.session_state
    or st.session_state.get("last_titles") != selected_titles
):
    st.session_state.conversational_chain = get_chain(selected_titles)
    st.session_state.last_titles = selected_titles



def submit_prompt(question_text: str) -> None:
    """
    Submit a user question to the conversational retrieval chain and append the response to the chat history
    Author: Samarth
    input:
        question_text: the question to be processed
    returns:
        None
    """
    st.session_state.messages.append({"role": "user", "content": question_text})
    
    result = st.session_state.conversational_chain({"question": question_text})
    answer = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": answer})

    st.session_state.submitted_question = question_text
    
    #uncomment this if you would like the chunks displayed
    # for i, doc in enumerate(result.get("source_documents", []), start=1):
    #     snippet = doc.page_content[:200]
    #     st.session_state.messages.append({"role": f"Chunk {i}", "content": snippet})


#back-up hardcoded questions for the user if needed, inpsired by the Ask Science Direct AI dashboard (https://www.sciencedirect.com/ai)
categories = {
    "To gather insights on a topic...": [
        "What are the key trends in this field?",
        "How do recent papers approach this challenge?",
        "What is the most common method used?",
    ],
    "To identify new research opportunities...": [
        "What gaps are identified in these papers?",
        "Are there underexplored applications or techniques?",
        "Which future work is commonly suggested?",
    ],
    "To review methods or experimental design...": [
        "What datasets and tools are commonly used?",
        "How do authors validate their findings?",
        "What are the limitations discussed?",
    ]
}


st.markdown("### Suggested Questions")
summaries = st.session_state.get("summaries", [])
sample_questions = []

for paper in summaries:
    if paper["title"] in selected_titles:
        sample_qa = paper.get("sample_qa", [])
        for qa in sample_qa[:3]:  #only take first 3 questions, change if you would like more suggested questions displayed
            if "question" in qa:
                sample_questions.append(qa["question"])

#limit to 5 if user picks more than one paper,  change if you would like more suggested questions displayed
sample_questions = random.sample(sample_questions, min(5, len(sample_questions)))

for i, question in enumerate(sample_questions):
    if st.button(question, key=f"sample_{i}"):
        submit_prompt(question)
        st.rerun()

st.divider()



#conversation log
#preserving conversation history created/debugged with guidance from ChatGPT (OpenAI, April 2025)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**{msg['role']}**: {msg['content']} ...")


user_input = st.chat_input("Ask questions about the selected papers (or type 'exit'/'quit' to stop)...")

if user_input:
    if user_input.lower() in ["exit", "quit"]:
        st.stop()

    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = st.session_state.conversational_chain({"question": user_input})
    answer = result["answer"]

    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

    #option to display retrieved chunks
    # docs = result.get("source_documents", [])
    # for i, doc in enumerate(docs, start=1):
    #     snippet = doc.page_content[:200]
    #     st.session_state.messages.append(
    #         {"role": f"Chunk {i}", "content": snippet}
    #     )
    #     with st.chat_message("assistant"):
    #         st.markdown(f"**Chunk {i}**: {snippet} ...")