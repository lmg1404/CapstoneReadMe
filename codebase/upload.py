"""
All functions that upload to cloud databases

Naturally follows all functions from processing.py
"""
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from typing import Dict
from datetime import datetime
from pathlib import Path
import json
import os
from dotenv import load_dotenv
load_dotenv()

def get_vector_store() -> QdrantVectorStore:
  """
  Get Qdrant vector store from langchain community
  Author: Luis
  input:
    None
  returns:
    QdrantVectorStore
  """
  COLLECTION_NAME = "ReadMe"
  qdrant_client = QdrantClient(
      url=os.environ['QURL'], 
      api_key=os.environ['QKEY'],
  )
  embeddings = HuggingFaceEmbeddings(model_name="malteos/scincl")
  if COLLECTION_NAME not in [c.name for c in qdrant_client.get_collections().collections]:
    print("Creating Collection ReadMe")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
  vector_store_ = QdrantVectorStore(
      client=qdrant_client,
      collection_name=COLLECTION_NAME,
      embedding=embeddings,
  )
  return vector_store_

def chunk(start_date: str, vector_store: QdrantVectorStore) -> None:
  """
  Chunks documents into pieces for LLM to understand and VDB to store
  Also embeds and uploads the chunks to the VDB
  Author: Luis
  
  input:
    start_date: date which is needed for metadata on VDB
    vector_store: Langchain Object which facilitates upload
  returns:
    None
  """
  mds = Path(f"./{start_date}/markdown_papers/").glob("*.md") # iterator
  tmp = datetime.strptime(start_date, "%Y%m%d")
  date = tmp.strftime("%m-%d-%Y")
  
  
  headers_to_split_on = [
      ("#", "Section A"),
      ("##", "Section B"),
      ("###", "Section C"),
  ]
  chunk_size = 1000
  chunk_overlap = 70

  markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
  )
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size, chunk_overlap=chunk_overlap
  )
  for md in mds:
    title = str(md).split('/')[-1].split('.')[0]
    print(f"Chunking up {title}")
    doc = open(md, 'r').read()
    # Split  
    md_header_splits = markdown_splitter.split_text(doc)
    splits = text_splitter.split_documents(md_header_splits)
    for split in splits:
      split.metadata['title'] = title
      split.metadata['date'] = date
    
    vector_store.add_documents(documents=splits)
  
  return None

################################################################################
#
#                          PGSQL Uploads
#
################################################################################

def upload_to_primary_table(start_date: str, conn) -> None:
  """
  Uploads papers to primary postgres table where id is serialized
  Author: Luis
  
  input:
    start_date: date which is needed for metadata on DB
    conn: postgres connection to use
  returns:
    None
  """
  query = """
    INSERT INTO papers (title, publishdate, tags, pdf_link, citations) 
    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (title, publishdate)
    DO UPDATE SET citations = EXCLUDED.citations;
  """
  cursor = conn.cursor()
  pdfs = Path(f"./{start_date}/papers/").glob("*.pdf")
  categories = Path(f"./{start_date}/categories/").glob("*.json")
  links = Path(f"./{start_date}/links/").glob("*.txt")
  citations = Path(f"./{start_date}/citations/").glob("*.txt")
  date = datetime.strptime(start_date, "%Y%m%d")
  payload = []
  
  for pdf, cats, link, citation in zip(pdfs, categories, links, citations):
    title = str(pdf).split('/')[-1].split('.')[0]
    cats_payload = json.loads(cats.read_bytes())
    link_ = open(link, 'r').read()
    citations_ = open(citation, 'r').read()
    payload.append((title, date, json.dumps(cats_payload), link_, citations_))
    
  cursor.executemany(query, payload)
  conn.commit()
  cursor.close()

def upload_summaries(summaries: dict, start_date, conn) -> None:
  """
  Upload summary payloads onto the summary PGSQL server
  Author: Luis
  
  input:
    summaries: contains all we need which are summaries, keywords, q&a pairs
    start_date: date which is needed for opening directories
    conn: PGSQL connection
  returns:
    None
  """
  cursor = conn.cursor()
  paperid_query = """
    SELECT paper_id, title
    FROM papers
    WHERE title in %s;
  """
  insert_query = """
    INSERT INTO summaries (paper_id, abstract, claude_summary, keywords, sample_qa)
    VALUES (%s, %s, %s, %s, %s) ON CONFLICT (paper_id) DO NOTHING;
  """
  titles = []
  for s in summaries:
    titles.append(s['title'])
    
  cursor.execute(paperid_query, (tuple(titles),))
  i2t_tup = cursor.fetchall()
  t2i = {title:id for id, title in i2t_tup}
  payload = []

  for s in summaries:
    if s['title'] in t2i.keys():
      abstract = open(f"./{start_date}/abstracts/{s['title']}.txt", "r").read()
      payload.append(
        (t2i[s['title']], 
        abstract, 
        s['summary'], 
        json.dumps(s['technical_terms']),
        json.dumps(s['sample_qa']))
      )
  cursor.executemany(insert_query, payload)
  conn.commit()
  cursor.close()

def upload_to_analytics(analytics_, conn) -> None:
  """
  Upload to analytics payload
  Author: Luis
  
  input:
    analytics_: payload containing metadata for analytical graphs
    conn: PGSQL connection
  returns:
    None
  """
  cursor = conn.cursor()
  paperid_query = """
    SELECT paper_id, title
    FROM papers
    WHERE title in %s;
  """
  insert_query = """
    INSERT INTO analytics (paper_id, sentiment, named_entities, word_count)
    VALUES (%s, %s, %s, %s) ON CONFLICT (paper_id) DO NOTHING;
  """
  titles = []
  for s in analytics_:
    titles.append(s['title'])
    
  cursor.execute(paperid_query, (tuple(titles),))
  i2t_tup = cursor.fetchall()
  t2i = {title:id for id, title in i2t_tup}
  payload = []

  for content in analytics_:
    payload.append(
      (t2i[content['title']], 
       content['sentiment'], 
       content['named_entities'], 
       content['word_count'])
    )
  cursor.executemany(insert_query, payload)
  conn.commit()
  cursor.close()

