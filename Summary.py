"""
Author: Samarth
Summary testing for prompt used in the pipeline
Also extracts q&a pairs, and keywords with defintions
"""
from dotenv import load_dotenv
import os
load_dotenv()
from google.colab import files

uploaded = files.upload()

pdf_filename = list(uploaded.keys())[0]

import anthropic
import base64
from pdf2image import convert_from_path
import io
from PIL import Image
import re
import json

client = anthropic.Anthropic(
    api_key=os.environ['ANTHROPIC_API_KEY2']
)

pdf_images = convert_from_path(pdf_filename, dpi=200)

def image_to_base64(image, format="JPEG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

prompt = """
Please analyze this document and provide THREE separate outputs:

1. A comprehensive summary of this document.

2. A list of technical keywords from your summary that a general audience might not understand, along with their definitions.

3. Three sample Q&A pairs from the content of the document.

Structure your response exactly like this:
===SUMMARY===
[Your detailed summary here]

===TECHNICAL_TERMS===
{
  "term1": "definition1",
  "term2": "definition2",
  ...
}

===SAMPLE_QA===
[
  {"question": "sample question 1", "answer": "sample answer 1"},
  {"question": "sample question 2", "answer": "sample answer 2"},
  {"question": "sample question 3", "answer": "sample answer 3"}
]
"""

content = [{"type": "text", "text": prompt}]
max_pages = min(20, len(pdf_images))
for i in range(max_pages):
    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_to_base64(pdf_images[i])
        }
    })

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

summary_match = re.search(r'===SUMMARY===\s*(.*?)(?===TECHNICAL_TERMS===|\Z)', response_text, re.DOTALL)
terms_match = re.search(r'===TECHNICAL_TERMS===\s*(.*?)(?===SAMPLE_QA===|\Z)', response_text, re.DOTALL)
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
    "summary": summary,
    "technical_terms": technical_terms,
    "sample_qa": sample_qa
}

print(json.dumps(final_output, indent=2))
