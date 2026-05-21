<div align="center">

# 📚 Premium RAG Chatbot Suite

**Chat with any PDF or YouTube video — 100% locally, with a high-fidelity glassmorphic SaaS interface and zero API costs.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-black?style=for-the-badge)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 🧠 What is RAG?

Large Language Models (LLMs) are powerful, but they don't know the contents of *your* documents. **Retrieval-Augmented Generation (RAG)** solves this by combining a vector search engine with an LLM:

```
Your PDF / YouTube Video
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

## ✨ Features & Visual Upgrades

| Feature | Description |
|---|---|
| 🎨 **Premium SaaS UI/UX** | Dark-mode glassmorphic styling, custom gradient branding, and elegant `Plus Jakarta Sans` typography. |
| 🎛️ **Interactive Sidebar Control** | Complete control panel in the sidebar, separating data ingestion and configurations from the clean main chat view. |
| 📊 **Active Source Telemetry** | Glassmorphic telemetry panel detailing parsed chunk counts, character count metrics, active embeddings, and model status. |
| ⚙️ **Real-time Param Tuning** | Live sliders for adjusting **Chunk Size**, **Overlap**, and **Top K Retrieved Chunks** for advanced model optimization. |
| 🔄 **Smart Index Re-builder** | Notifies you when parameter settings differ from the active vector index, allowing on-demand rebuilds to avoid system lag. |
| 📄 **PDF Upload** | Drag and drop any PDF directly into the secure local ingestion area. |
| 🎥 **YouTube Support** | Paste any YouTube video URL to automatically parse video transcripts and query their contexts. |
| 🔒 **100% Fully Local** | Runs entirely on your local machine via Ollama and FAISS CPU. Zero API costs, zero tracking, total privacy. |
| 🖥️ **CLI Mode** | Toggle interactive terminal conversations using the standalone `main.py` client. |

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| 🤖 **LLM** | `llama3` (or customizable choices) via [Ollama](https://ollama.com) |
| 🔢 **Embeddings** | `all-MiniLM-L6-v2` (HuggingFace Sentence Transformers) |
| 🗄️ **Vector Store** | FAISS (CPU-based local vector store) |
| 📑 **PDF Parsing** | LangChain + PyPDF |
| 🎥 **YouTube Transcripts** | `youtube-transcript-api` |
| 🖥️ **Web UI Engine** | Streamlit (injected with custom glassmorphic CSS overrides) |

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- ~5 GB disk space for local models (e.g. `llama3` or `mistral`)

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

**3. Pull the local LLM model using Ollama**
```bash
ollama pull llama3
```

**4a. Launch the Premium Web App**
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your web browser.

**4b. Or run the CLI version in your terminal**
```bash
python main.py
```

---

## 🗂️ Project Structure

```
PDF-RAG-Chatbot/
├── app.py              # Redesigned Premium Web Interface (PDF + YouTube + Parameters + Telemetry)
├── main.py             # Standalone interactive CLI terminal script
├── requirements.txt    # Required project Python dependencies
├── .gitignore
└── README.md           # Project documentation
```

---

## ⚙️ How It Works (Web Application UI)

1. **Upload & Ingestion**: Select your source type (**PDF Upload** or **YouTube Video**) in the sidebar menu.
2. **Dynamic Splitting**: Customize chunk sizes (fully configurable from 200 to 2,000 characters) and overlap thresholds (from 0 to 300 characters) in the configuration panel rather than relying on hardcoded defaults.
3. **Local Vectorization**: Chunks are embedded locally via `all-MiniLM-L6-v2` and placed inside a local memory-resident FAISS vector database.
4. **Context Telemetry**: A dashboard updates dynamically to show you character lengths, chunk statistics, and active pipeline statuses.
5. **Grounded Question Answering**: When you chat, the engine retrieves the top $k$ most relevant context segments (configurable from 1 to 5 retrieved chunks) and feeds them into the local model, ensuring grounded answers without hallucinations.


---

<div align="center">

Made by Varnit using LangChain, FAISS, Ollama, Streamlit, and youtube-transcript-api.

</div>
