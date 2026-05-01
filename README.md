<div align="center">

# 📚 PDF RAG Chatbot

**Chat with any PDF document — 100% locally, zero API costs.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-black?style=for-the-badge)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>
<img width="1061" height="793" alt="image" src="https://github.com/user-attachments/assets/be684198-14c5-48fe-952d-f1a3a2adc6ab" />

---

## 🧠 What is RAG?

Large Language Models (LLMs) are powerful, but they don't know the contents of *your* documents. **Retrieval-Augmented Generation (RAG)** solves this by combining a vector search engine with an LLM:

```
Your PDF
   │
   ▼
[Chunking]  →  Split into smaller, overlapping pieces
   │
   ▼
[Embedding]  →  Convert each chunk into a vector (numerical meaning)
   │
   ▼
[FAISS Index]  →  Store vectors for fast similarity search
   │
   ▼
[Your Question]  →  Find the most relevant chunks
   │
   ▼
[LLM + Context]  →  Generate a grounded, accurate answer ✅
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Upload | Drag and drop any PDF directly in the browser |
| 🔒 Fully Local | No OpenAI key, no API costs — runs entirely on your machine |
| 🎯 Grounded Answers | Responses are based on your document, not hallucinated |
| 💬 Chat History | Conversation persists throughout your session |
| ⚡ Auto Re-embed | Upload a new PDF and it re-indexes automatically |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| 🤖 LLM | `llama3` via [Ollama](https://ollama.com) |
| 🔢 Embeddings | `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers) |
| 🗄️ Vector Store | FAISS (CPU) |
| 📑 PDF Parsing | LangChain + PyPDF |
| 🖥️ UI | Streamlit |

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- ~5 GB disk space for the llama3 model

### Step-by-step

**1. Clone the repository**
```bash
git clone https://github.com/Varn1t/PDF-RAG-Chatbot.git
cd PDF-RAG-Chatbot
```

**2. Install Python dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull the LLM model with Ollama**
```bash
ollama pull llama3
```

**4. Launch the app**
```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 🗂️ Project Structure

```
pdf-rag-model/
├── app.py              # Streamlit web app + RAG pipeline
├── main.py             # Command-line version of the pipeline
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ How It Works

1. **Upload** a PDF via the Streamlit file uploader
2. The app **chunks** the PDF into 1,000-character segments with 50-character overlap
3. Each chunk is **embedded** using `all-MiniLM-L6-v2` and stored in a FAISS index
4. When you ask a question, the **top 3 most relevant chunks** are retrieved
5. Those chunks are passed as context to **llama3**, which generates a precise answer

---

<div align="center">

Made with ❤️ using LangChain, FAISS, Ollama, and Streamlit

</div>
