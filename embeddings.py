import os
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json
import uuid

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found.")

client = OpenAI(api_key=api_key)

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="faq_embeddings",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
)

def load_faq_data(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def prepare_documents(faq_data):
    docs = []
    for item in faq_data:
        doc_id = str(uuid.uuid4())

        question = item.get("question", "")
        answer = item.get("text", "")
        section = item.get("section", "General")
        doc_type = item.get("type", "info")

        if question:
            text = f"Q: {question}\nA: {answer}"
        else:
            text = f"{section}: {answer}"

        metadata = {
            "section": section or "",
            "type": doc_type or "",
            "question": question or ""
        }

        docs.append({
            "id": doc_id,
            "text": text,
            "metadata": metadata
        })

    return docs

def add_to_chroma(docs):
    ids = [d["id"] for d in docs]
    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    collection.add(ids=ids, documents=texts, metadatas=metadatas)
    print(f"added {len(docs)} documents to chroma")

def query_faq(query_text, n_results=3):
    results = collection.query(query_texts=[query_text],n_results=n_results)
    answers = []
    print("\n🔹 Top Matches:")
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"][0] else {}
        answers.append({
            "match": i + 1,
            "answer": doc,
            "section": meta.get("section", "N/A"),
            "type": meta.get("type", "N/A"),
            "distance": results["distances"][0][i]
        })
        print(f"\nMatch {i+1}:")
        print(f"Q/A: {doc}")
        print(f"Section: {meta.get('section', 'N/A')}")
        print(f"Type: {meta.get('type', 'N/A')}")
        print(f"Distance: {results['distances'][0][i]:.4f}")
    return {"query": query_text, "results": answers}

def ingest_embeddings(filepath="faq_data.json"):
    faq_data = load_faq_data("faq_data.json")
    docs = prepare_documents(faq_data)
    add_to_chroma(docs)
    return len(docs)


    