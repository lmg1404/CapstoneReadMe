"""
Any API extractions and DB upload setups go here for ease go here
In a production setting many different categories would go here besides ML
"""
""" Note
The API provides one date filter, submittedDate, that allow you to select
data within a given date range of when the data was submitted to arXiv. 
The expected format is [YYYYMMDDTTTT+TO+YYYYMMDDTTTT] were the TTTT is 
provided in 24 hour time to the minute, in GMT. We could construct the 
following query using submittedDate
"""

import requests
from xml.etree import ElementTree as ET
import json
import re
import os
from codebase.constants import tag_mapping

def print_xml_tree(xml_content: str) -> None:
  """
  Debugging function to help understand what exactly is within the XML tree
  Prints out the tree, returns nothing
  Author: Luis
  input:
    xml_content: xml tree to show for debugging
  returns:
    None
  """
  root = ET.fromstring(xml_content)
  xml_string = ET.tostring(
    root, 
    encoding='utf-8', 
    xml_declaration=True
  ).decode('utf-8')
  print(xml_string)

def format_author(author) -> str:
  """
  Extracts authors the way arXiv recommends for citation, 
  which is last and first initial
  Author: Luis
  input:
    author: string of full author name
  returns:
    str
  """
  splits = author.split(" ")
  last = splits[-1]
  initials = ' '.join(f'{s[0]}.' for s in splits[:-1] if len(s) > 0)
  return f"{last}, {initials}"

def nature_format(title, authors, pdf_link, year) -> str:
  """
  Gets inputs and puts into nature citation for citing summaries and 
  during RAG
  Author: Luis
  input:
    title: Paper title
    authors: any authors that should be processed
    pdf_link: correct link for pdf
    year: year of publish
  returns:
    str
  """
  formatted_authors = ', '.join([format_author(a) for a in authors])
  doi = pdf_link.split('/')[-1]
  
  return f"{formatted_authors} *{title}* arXiv preprint arXiv:{doi} ({year})"

def download(start_date: str, end_date: str) -> None:
  """
  Downloads from API and uploads a couple things to certain folders for temp 
  storage
  Author: Luis
  input:
    start_date: start date inclusive which is needed, typically start of day
    end_date: end date inclusive which is needed by API, typically end of day
  returns:
    None
  """
  response = requests.get(
    f'https://export.arxiv.org/api/query?search_query=all:machine%20learning+AND+submittedDate:%5B{start_date}+TO+{end_date}%5D'
  )  
  root = ET.fromstring(response.text)
  titles = []
  
  os.makedirs(f"./{start_date[:-4]}/papers/", exist_ok=True)
  os.makedirs(f"./{start_date[:-4]}/markdown_papers/", exist_ok=True)
  os.makedirs(f"./{start_date[:-4]}/text_papers/", exist_ok=True)
  os.makedirs(f"./{start_date[:-4]}/abstracts/", exist_ok=True)
  os.makedirs(f"./{start_date[:-4]}/categories/", exist_ok=True)
  os.makedirs(f"./{start_date[:-4]}/links/", exist_ok=True)
  os.makedirs(f"./{start_date[:-4]}/citations/", exist_ok=True)

  for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
    # Get the title
    title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
    title = re.sub(r'[<>:"/\\|?*\n]', '', title)
    title = " ".join(title.split())
    abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()

    # get authors for citation
    # for author in entry.findall('{http://www.w3.org/2005/Atom}author'):
    names = [
      a.text 
      for author in entry.findall('{http://www.w3.org/2005/Atom}author') 
      for a in author.findall('{http://www.w3.org/2005/Atom}name')
    ]
    
    # find ALL categories
    categories = set()
    for category in entry.findall('{http://www.w3.org/2005/Atom}category'):
      cat = category.get('term')
      categories.add(tag_mapping.get(cat, None))

    # find the link tag with title="pdf"
    pdf_link = None
    for link in entry.findall('{http://www.w3.org/2005/Atom}link'):
      if link.get('title') == 'pdf':
        pdf_link = link.get('href')
        break
    
    nature_citation = nature_format(title, names, pdf_link, start_date[:4])

    # write to folder (important!)
    r = requests.get(pdf_link)
    with open(f"./{start_date[:-4]}/papers/{title}.pdf", 'wb') as f:
      f.write(r.content)
      
    with open(f"./{start_date[:-4]}/abstracts/{title}.txt", 'w') as f:
      f.write(abstract)
    
    with open(f"./{start_date[:-4]}/categories/{title}.json", 'w') as f:
      json.dump(list(categories), f)
      
    with open(f"./{start_date[:-4]}/links/{title}.txt", 'w') as f:
      f.write(pdf_link) 
      
    with open(f"./{start_date[:-4]}/citations/{title}.txt", 'w') as f:
      f.write(nature_citation) 
    
    titles.append(title)
