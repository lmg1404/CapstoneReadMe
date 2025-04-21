# !pip install evaluate
# !pip install rouge-score
# !pip install bert-score

from codebase.utils import *
import langchain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatAnthropic
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os
load_dotenv()

query = """
SELECT
    p.title,
    p.publishdate,
    s.abstract,
    s.claude_summary,
    s.sample_qa
FROM summaries s
LEFT JOIN papers p ON p.paper_id = s.paper_id
ORDER BY p.publishdate ASC;
"""

conn = get_pg_connection()
df = pd.read_sql_query(query, conn)
conn.close()

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = df.apply(lambda row: scorer.score(row["abstract"], row["claude_summary"]), axis=1)
avg_rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(df)
avg_rouge2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(df)
avg_rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(df)

_, _, F1 = score(df["claude_summary"].tolist(), df["abstract"].tolist(), lang="en", verbose=False)
avg_bert_f1 = F1.mean().item()

print("avg ROUGE-1:", avg_rouge1)
print("avg ROUGE-2:", avg_rouge2)
print("avg ROUGE-L:", avg_rougeL)
print("avg BERTScore:", avg_bert_f1)

## RAG Evaluation

df['sample_qa'][0]
df["first_question"] = df["sample_qa"].apply(lambda x: x[0]["question"] if x else None)
df["first_answer"]   = df["sample_qa"].apply(lambda x: x[0]["answer"]   if x else None)


QDRANT_API_KEY = os.environ['QKEY']
QDRANT_URL = os.environ['QURL']
qdrant_client = QdrantClient(QDRANT_URL, api_key=QDRANT_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="malteos/scincl")
qdrant_store = Qdrant(client=qdrant_client, collection_name="ReadMe", embeddings=embeddings)
ANTHROPIC_API_KEY = os.environ['ANTHROPIC_API_KEY1']
llm = ChatAnthropic(model="claude-3-opus-20240229", temperature=0.5, anthropic_api_key=ANTHROPIC_API_KEY)
rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
def compute_rouge(reference,hypothesis):
    scores = rouge_scorer_instance.score(reference,hypothesis)
    return scores["rougeL"].fmeasure
mnli_model_name = "roberta-large-mnli"
mnli_tokenizer = AutoTokenizer.from_pretrained(mnli_model_name)
mnli_model = AutoModelForSequenceClassification.from_pretrained(mnli_model_name)
def compute_entailment_score(reference,hypothesis):
    inputs = mnli_tokenizer.encode_plus(reference,hypothesis,return_tensors="pt",truncation=True,max_length=512)
    with torch.no_grad():
        logits = mnli_model(**inputs).logits
    probs = torch.softmax(logits,dim=1).squeeze()
    return probs[2].item()
rag_answers = []
rouge_scores = []
bleu_scores = []
entail_scores = []
for idx,row in df.iterrows():
    doc_title = row["title"]
    print(doc_title)
    metadata_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.title",
                match=models.MatchValue(value=doc_title)
            )
        ]
    )
    retriever = qdrant_store.as_retriever(search_kwargs={"filter": metadata_filter})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        return_source_documents=True,
        output_key="answer"
    )
    question = row["first_question"]
    true_answer = row["first_answer"]
    result = chain({"question": question, "chat_history": []})
    rag_answer = result["answer"]
    source_docs = result.get("source_documents", [])
    if source_docs:
        print("Retrieved Chunks")
        for i, doc in enumerate(source_docs, 1):
            print(doc.page_content[:100] + "...")
        print("Metadata")
        for i, doc in enumerate(source_docs, 1):
            print(doc.metadata)
    r = compute_rouge(true_answer, rag_answer)
    e = compute_entailment_score(true_answer, rag_answer)
    rag_answers.append(rag_answer)
    rouge_scores.append(r)
    entail_scores.append(e)
df["rag_answer"] = rag_answers
df["rouge_score"] = rouge_scores
df["entailment_score"] = entail_scores
