from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Dict, List
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
import os
import sys
from dotenv import load_dotenv
from embeddings import query_faq, ingest_embeddings, collection

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🔹 Initializing Punch Support Bot...")

    try:
        existing = collection.get()
        count = len(existing["ids"]) if existing["ids"] else 0

        if count == 0:
            docs_added = ingest_embeddings("faq_data.json")
            print(f" Added {docs_added} FAQ entries to Chroma.")
        else:
            print(f" Chroma already has {count} FAQ entries.")
    except Exception as e:
        print(f" Failed to initialize embeddings: {e}")
        sys.exit(1)

    yield
    print(" Shutting down Punch Support Bot...")


app = FastAPI(title="Punch Support Bot", lifespan=lifespan)

# Allow all origins for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_history: Dict[str, List[Dict[str, str]]] = {}

class ChatRequest(BaseModel):
    session_id: str
    query: str


@app.get("/")
def root():
    return {"message": "Punch Support Bot is running!"}

@app.get("/welcome")
def welcome_message():
    """Send a default welcome message when the chat starts"""
    welcome_text = [
        "👋 Hi there! I’m Punch Support Assistant.",
        "How can I help you today?"
    ]
    return {"messages": welcome_text}


@app.post("/chat")
def chat_with_bot(request: ChatRequest):
    session_id = request.session_id
    user_query = request.query

    history = chat_history.get(session_id, [])
    recent_context = history[-5:]

    faq_results = query_faq(user_query)
    faq_context = "\n\n".join([r["answer"] for r in faq_results["results"]])

    # Make the model more concise and readable
    messages: list[
        ChatCompletionSystemMessageParam
        | ChatCompletionUserMessageParam
        | ChatCompletionAssistantMessageParam
    ] = [
        ChatCompletionSystemMessageParam(
            role="system",
            content=(
                "You are a friendly Punch customer support assistant. "
                "Use the FAQ context to answer the user's query briefly in clear, readable parts. "
                "Each part should be at most 2–3 lines long. "
                "If the context does not contain enough information, "
                "reply with: 'I'm sorry, but I don't have enough details to resolve this query right now. "
                "Would you like me to connect you with our support team? They'll be happy to help you further.'"
            ),
        )
    ]

    for msg in recent_context:
        if msg["role"] == "user":
            messages.append(ChatCompletionUserMessageParam(role="user", content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg["content"]))

    messages.append(
        ChatCompletionUserMessageParam(
            role="user",
            content=f"Context:\n{faq_context}\n\nUser Question: {user_query}",
        )
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.6,
    )

    answer = response.choices[0].message.content

    # Split response into small chunks for frontend
    message_chunks = [chunk.strip() for chunk in answer.split("\n\n") if chunk.strip()]

    chat_history.setdefault(session_id, []).append({"role": "user", "content": user_query})
    chat_history[session_id].append({"role": "assistant", "content": answer})

    return {"session_id": session_id, "messages": message_chunks}


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    chat_history.pop(session_id, None)
    return {"message": f"Session {session_id} cleared."}
