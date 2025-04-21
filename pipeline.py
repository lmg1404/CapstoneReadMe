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
print("Importing Modules, Torch slows this down for embedding model...")
import nltk
try:
  nltk.data.find('corpora/stopwords')
except LookupError:
  print('Downloading stopwords...')
  nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from codebase.extract import *
from codebase.processing import *
from codebase.upload import *
from codebase.utils import *
from datetime import date, timedelta

print("Getting Database connections (may take a while)...")
conn = get_pg_connection()
client = get_claude_conn()
vector_store = get_vector_store()

# change these
start_date = date(2025, 3, 1)
END = date(2025, 3, 31)

curr = start_date
while curr <= END:
  str_date = curr.strftime('%Y%m%d')
  start_date, end_date = f'{str_date}0001', f'{str_date}2359'

  print(f"Downloading for {start_date}...")
  download(start_date, end_date)
  print("Uploading papers to primary schema in PGSQL")
  upload_to_primary_table(str_date, conn)
  
  print("Beginning analytics processing...")
  pdf_to_text(str_date, stop_words)
  payload = analytics_processing(str_date)
  print(f'Begin upload for {str_date}')
  upload_to_analytics(payload, conn)
  
  print("Chunking portion of pipeline...")
  print(f"Converting PDF to Markdown for {str_date}")
  pdf_to_markdown(str_date)
  print("Working on Chunking using Langchain with associated meta-data")
  chunk(str_date, vector_store)
  
  print("Using Claude to get our summaries...")
  print(f'WORKING ON {str_date}')
  summaries = extract_summaries(str_date, client)
  upload_summaries(summaries, str_date, conn)
  
  teardown(str_date)
  curr += timedelta(days=1)
conn.close()

