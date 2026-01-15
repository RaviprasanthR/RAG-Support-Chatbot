🤖 Punch Support Chatbot (POC)

An AI-powered customer support chatbot built as a proof of concept for Punch, designed to answer user queries using the FAQ content available on the Punch support page.

The chatbot uses:
- **FastAPI** for the backend API  
- **Chroma** as the vector database for storing and retrieving FAQ embeddings  
- **OpenAI GPT model** for generating natural language responses  
- A lightweight **HTML + JS frontend** for chat interaction  

---

🌟 Features

✅ Uses Punch’s public FAQ data for intelligent Q&A  
✅ Retains recent chat context for more natural conversations  
✅ Politely offers to connect to support if it cannot answer  
✅ Simple, dark-themed chat UI with typing animation  
✅ Built completely in Python (FastAPI) — easy to extend and deploy  

---

⚙️ Setup Instructions

1️⃣ Clone the Repository
```bash
git clone https://github.com/RaviprasanthR/RAG-Support-Chatbot.git
cd punch-chatbot-poc

2️⃣ Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure Environment Variables

Create a .env file in the root folder and add your OpenAI Key.
OPENAI_API_KEY=your_openai_api_key_here

5️⃣ Run the Application

uvicorn app:app --reload

---

💬 Access the Chatbot UI

- After running the server, open chatbot.html in your browser (use Live Server)

- You’ll see the Punch Support Assistant chat interface:

- The bot greets you on start.

- You can type questions or hit Enter to send.

- If the answer is unclear, it offers to connect with a support executive.

---

🧩 How It Works

Embeddings Creation

The FAQ data (faq_data.json) is embedded using text-embedding-3-small.

Stored locally in a Chroma vector DB.

Query Handling

User input is matched against top FAQ entries using semantic similarity.

The relevant context is sent to OpenAI’s GPT model for response generation.

Context Retention

The bot remembers the last 5 messages in each session for continuity.

---

🧱 Tech Stack

Python 3.10+

FastAPI

ChromaDB

OpenAI GPT API

HTML, CSS, JavaScript (Frontend)
