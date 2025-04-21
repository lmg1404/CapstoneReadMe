"""
Contains all PDF processing including summaries, keyword extraction,
and analytic processing.
"""
from pathlib import Path
from codebase.constants import *
import pymupdf
import io
from PIL import Image
import pymupdf4llm
import fitz
import pandas as pd
import base64
import regex as re
import json
from pdf2image import convert_from_path
import time
from collections import Counter
from textblob import TextBlob
import spacy
import anthropic
from dotenv import load_dotenv
load_dotenv()

################################################################################
#
#                    FUNCTIONS FOR ANALYTICS
#
################################################################################

def clean_text(text: str, stop_words: list) -> str:
  """
  Cleans text for analytics usage
  Author: Samantha
  input:
    text: text to be cleaned
    stop_words: words to be removed
  returns:
    str
  """
  text2 = text.lower()  
  text3 = re.sub(r"[^a-z\s]", "", text2)
  words = [word for word in text3.split() if word not in stop_words and len(word) > 2]
  return " ".join(words)

def pdf_to_text(start_date: str, stop_words) -> None:
  """
  Converts PDF to text for analytics usage
  Author: Samantha
  input:
    start_date: start date which is the folder name
    end_date: end date which ends the folder name
  returns:
    str
  """
  pdfs = Path(f"./{start_date}/papers/").glob("*.pdf") # iterator
  
  for pdf in pdfs:
    print(f"Turning {str(pdf)} to text")
    try:
      doc = pymupdf.open(pdf)
      text = chr(12).join([page.get_text() for page in doc])
      text = clean_text(text, stop_words)
      with open(f'{start_date}/text_papers/{str(pdf).split("/")[-1].split(".")[0]}.txt', 'w') as file:
        file.write(text)
    except:
      print(f"Processing didn't work for {str(pdf)}")
  
  return None

def get_sentiment_analysis(text: str) -> float:
  """
  Get sentiment analysis of text
  Author: Samantha
  input:
    text: text for sentiment analysis
  returns:
    float
  """
  sentiment = TextBlob(text).sentiment.polarity
  return sentiment

def get_named_entities(text: str, nlp_) -> json:
  """
  Named entity extraction of text
  Author: Samantha
  input:
    text: text for extraction
    nlp_: obj
  returns:
    str
  """
  doc = nlp_(text[1000:])
  # needs to be converted so PGSQL can store it
  entities = [{
    "text": ent.text, 
    "label": ent.label_, 
    "start": ent.start_char, 
    "end": ent.end_char
    } for ent in doc.ents]
  return json.dumps(entities)

def get_frequent_words(text):
  """
  Getting 50 frequent words for PG storage
  Author: Samantha
  input:
    text: text for extraction
  returns:
    str
  """
  words = text.lower().split()
  word_counts = Counter(words)
  top_50 = word_counts.most_common(50)
  top_50 = {w:v for w, v in top_50}
  return json.dumps(top_50)

def analytics_processing(start_date: str) -> list:
  """
  Pipeline for getting analytics topics for PG upload
  Author: Luis
  input:
    start_date: date where folder will be located
  returns:
    list
  """
  print('Getting spacy model')
  
  nlp = spacy.load("en_core_web_sm")
  nlp.max_length = 2_000_000
  text_files = Path(f"./{start_date}/text_papers/").glob("*.txt")
  payload = []
  for filename in text_files:
    print(f'Processing {filename}')
    content = {}
    title = str(filename).split('/')[-1].split('.')[0]
    text = open(filename, 'r').read()
    content['title'] = title
    content['sentiment'] = get_sentiment_analysis(text)
    content['named_entities'] = get_named_entities(text, nlp)
    content['word_count'] = get_frequent_words(text)
    payload.append(content)
  return payload


################################################################################
#
#                    RAG OPERATIONS INCLUDING CHUNKING
#
################################################################################


def pdf_to_markdown(start_date: str) -> None:
  """
  Convert PDF to MD files
  Author: Luis 
  input:
    start_date: date where folder will be located
  returns:
    None
  """
  pdfs = Path(f"./{start_date}/papers/").glob("*.pdf") # iterator
  
  for i, pdf in enumerate(pdfs):
    print(f"Working on {pdf}")
    try:
      md_text = pymupdf4llm.to_markdown(pdf, show_progress=False)

      with open(f'{start_date}/markdown_papers/{str(pdf).split("/")[-1].split(".")[0]}.md', 'w') as file:
        file.write(md_text)
    except:
      print(f"Issues with {pdf}, ignoring")
  return None

################################################################################
#
#                          CLAUDE SUMMARIES
#
################################################################################


def image_to_base64(image, format="JPEG") -> base64:
  """
  Convert PIL images into base64 binary
  Author: Luis 
  input:
    image: image for turning to base64
    format: image format
  returns:
    base64
  """
  buffer = io.BytesIO()
  image.save(buffer, format=format)
  return base64.b64encode(buffer.getvalue()).decode("utf-8")

def get_summary_from_claude(content: dict, title, client) -> dict:
  """
  Get summaries form Claude using images
  Author: Samarth 
  input:
    start_date: date where folder will be located
  returns:
    dict
  """
  message = client.messages.create(
    model="claude-3-5-sonnet-20241022", 
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": content
        }
    ]
  )
  
  response_text = message.content[0].text

  summary_match = re.search(r'===SUMMARY===\s*(.*?)(?:===TECHNICAL_TERMS===|\Z)', response_text, re.DOTALL)
  terms_match = re.search(r'===TECHNICAL_TERMS===\s*(.*?)(?:===SAMPLE_QA===|\Z)', response_text, re.DOTALL)
  qa_match = re.search(r'===SAMPLE_QA===\s*(.*)', response_text, re.DOTALL)

  summary = summary_match.group(1).strip() if summary_match else ""
  json_terms_str = terms_match.group(1).strip() if terms_match else ""
  json_qa_str = qa_match.group(1).strip() if qa_match else ""

  json_terms_str = re.sub(r'^```json\s*', '', json_terms_str)
  json_terms_str = re.sub(r'\s*```$', '', json_terms_str)

  json_qa_str = re.sub(r'^```json\s*', '', json_qa_str)
  json_qa_str = re.sub(r'\s*```$', '', json_qa_str)

  try:
      technical_terms = json.loads(json_terms_str) if json_terms_str else {}
  except json.JSONDecodeError:
      technical_terms = {}

  try:
      sample_qa = json.loads(json_qa_str) if json_qa_str else []
  except json.JSONDecodeError:
      sample_qa = []

  final_output = {
      "title": title,
      "summary": summary,
      "technical_terms": technical_terms,
      "sample_qa": sample_qa
  }
  return final_output

def get_claude_conn() -> None:
  """
  Retrieve claude connection functionalized for easy calls in files
  Author: Samarth
  input:
    None
  returns:
    None
  """
  client = anthropic.Anthropic(
      api_key=ANTHROPIC_API_KEY2
  )
  return client

def extract_summaries(start_date: str, client) -> list:
  """
  Pipeline to extract summaries from Claude and use as a payload for 
  PG upload. Time sleep is there otherwise Anthropic throws a fit
  Author: Samart
  input:
    start_date: date where folder will be located
    client: connection to Anthropic
  returns:
    list
  """

  pdfs = Path(f"./{start_date}/papers/").glob("*.pdf") # iterator
  payload = []
  for pdf in pdfs:
    title = str(pdf).split('/')[-1].split('.pdf')[0]
    print(f"Claude is working on {title}")
    try:
      pdf_images = convert_from_path(pdf, dpi=200)
    except:
      print(f"\tCouldn't convert {title}, moving on")
      continue
    content = [{"type": "text", "text": SUMMARY_PROMPT}]
    max_pages = min(20, len(pdf_images))  
    content += [{
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_to_base64(pdf_image)
            }
        } for pdf_image in pdf_images[:max_pages]]
    try:
      payload.append(get_summary_from_claude(content, title, client))
    except:
      print(f"{title} didn't work with Claude")
    time.sleep(5)
  return payload




