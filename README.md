<div align="center">

# 🌀 LoreLoop

**Chat with any PDF or YouTube video — 100% locally, with a self-correcting agentic loop, dual-graded answer verification, conversational memory, and zero API costs.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent-6366f1?style=for-the-badge)](https://github.com/langchain-ai/langgraph)
[![Ollama](https://img.shields.io/badge/Ollama-llama3-black?style=for-the-badge)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

</div>

---

## 🧠 What is this?

Most RAG demos stop at "retrieve chunks → generate answer." This project goes further — it uses a **LangGraph state-machine agent** that grades its own answers across two independent dimensions, detects failures, and automatically rewrites queries with targeted awareness of *what went wrong* before re-retrieving.

The agent catches two distinct failure modes that most RAG systems miss:
- **Hallucination** — the answer contains claims not supported by the retrieved chunks
- **Retrieval mismatch** — the chunks were retrieved but didn't match the intent of the query (e.g. asked for *examples*, got *limitations*)

On top of that, it has **conversational memory** — follow-up questions like *"give examples for it"* are automatically rephrased into standalone queries before retrieval, so context is never lost between turns.

Everything runs locally via Ollama and FAISS. No API keys. No data leaving your machine.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔁 **Self-Correcting Agent** | LangGraph loop retries with a rewritten query up to N iterations before giving up gracefully |
| 🎯 **Dual-Graded Verification** | Every answer is graded on both **groundedness** (hallucination check) AND **relevance** (did it actually answer the question?) — both must pass |
| 🔍 **Context-Aware Query Rewriter** | On failure, the rewriter receives the specific failure reason (hallucinated / off-topic / both) and rewrites accordingly |
| 💬 **Conversational Memory** | Resolves pronoun references and follow-up questions using the last N chat turns before retrieval |
| 🕵️ **Agentic Telemetry** | Expandable per-response panel showing every iteration's query, generated answer, retrieved chunks, and dual verdict |
| ⚙️ **Live Parameter Control** | Sidebar sliders for chunk size, overlap, top-K retrieval, max correction iterations, and memory window |
| 🔄 **Smart Re-indexer** | Detects when sidebar parameters differ from the active vector index and prompts a targeted rebuild |
| 📄 **PDF Support** | Drag-and-drop local PDF ingestion via PyPDF |
| 🎥 **YouTube Support** | Paste any YouTube URL to auto-fetch and query its transcript |
| 🔒 **Fully Local** | Ollama + FAISS CPU — zero API costs, zero tracking, total privacy |
| 🖥️ **CLI Mode** | Full terminal interface via `main.py` with diagnostic iteration logs |

---

## 🖥️ User Interface Gallery

### 🌐 Streamlit Web Application

LoreLoop features a custom dark-themed user interface with clean glassmorphic container cards and comprehensive system telemetry indicators.

#### 1. Active Document & Pipeline Status
Displays when a PDF or YouTube video is loaded, showing live chunking statistics and index parameters:
<img src="assets/active_pipeline.png" alt="Active Pipeline Telemetry" width="100%" />

#### 2. Agentic Self-Correction & Groundedness Grading
Shows the conversational assistant response alongside the expanded **Agentic Self-Corrective Telemetry** panel, displaying fact-checker status and retrieved context chunks:
<img src="assets/streamlit_question.png" alt="Streamlit Question and Telemetry" width="100%" />

*The conversational memory automatically reformulates follow-up queries like "Give examples for it" into clear standalone questions (e.g., "What are some common applications or examples of Deep Learning?") using prior chat history before running the agent node loop:*
<img src="assets/streamlit_memory.png" alt="Streamlit Conversational Memory Telemetry" width="100%" />

---

### 💻 CLI Diagnostic Interface

The CLI version provides detailed terminal output highlighting the query rephrasing and state transitions inside the agent loop:
<img src="assets/cli_telemetry.png" alt="CLI Diagnostic Telemetry" width="100%" />

---

## 🏗️ How the Agent Works

```mermaid
graph TD
    Start([User Question])
    Contextualize[contextualize_query<br>Rephrase using chat history]
    Retrieve[retrieve<br>Get top-K chunks from FAISS]
    Generate[generate<br>Answer using context]
    Check{check_hallucination<br>Grounded & Relevant?}
    Rewrite[rewrite_query<br>Targeted rewrite using failure reason]
    Success([Return Answer ✅])
    Failure([Information not found ❌])

    Start --> Contextualize
    Contextualize --> Retrieve
    Retrieve --> Generate
    Generate --> Check

    Check -- "Grounded & Relevant" --> Success
    Check -- "Either Fails & Iterations < N" --> Rewrite
    Rewrite --> Retrieve
    Check -- "Either Fails & Iterations == N" --> Failure

    %% Apply Styles
    classDef startEnd fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef nodeStyle fill:#0f172a,stroke:#38bdf8,stroke-width:1.5px,color:#e2e8f0;
    classDef checkStyle fill:#1e293b,stroke:#a855f7,stroke-width:2px,color:#fff;
    classDef successStyle fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef failureStyle fill:#7f1d1d,stroke:#f87171,stroke-width:2px,color:#fff;

    class Start startEnd;
    class Contextualize,Retrieve,Generate,Rewrite nodeStyle;
    class Check checkStyle;
    class Success successStyle;
    class Failure failureStyle;
```

**Two distinct query rewriting steps exist for two different reasons:**
- **`contextualize_query`** — runs once at the start of each turn to resolve ambiguous references using chat history. Never re-runs during the correction loop.
- **`rewrite_query`** — runs inside the correction loop when dual grading fails. Receives a specific failure reason so the rewrite is targeted, not generic.

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| 🧠 **Agent Framework** | LangGraph (state-machine with conditional edges and retry loop) |
| 🤖 **LLM** | `llama3` / `mistral` / `gemma` / `phi3` via Ollama |
| 🔢 **Embeddings** | `all-MiniLM-L6-v2` — HuggingFace Sentence Transformers |
| 🗄️ **Vector Store** | FAISS (CPU-based, no server required) |
| 📑 **PDF Parsing** | LangChain + PyPDFLoader |
| 🎥 **YouTube Transcripts** | `youtube-transcript-api` |
| 🖥️ **Web UI** | Streamlit with custom glassmorphic CSS |

---

## 🚀 Setup

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed and running locally
- ~5 GB disk space for the LLM model

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/Varn1t/LoreLoop.git
cd LoreLoop
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Pull a local model**
```bash
ollama pull llama3
```

**4a. Launch the web app**
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501)

**4b. Or run the CLI**
```bash
python main.py
```

---

## 🗂️ Project Structure

```
LoreLoop/
├── agent.py          # LangGraph agent — state, nodes, edges, dual grader, pipeline builder
├── app.py            # Streamlit web UI with telemetry and parameter controls
├── main.py           # CLI interface with iteration diagnostics
├── requirements.txt  # Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Key Dependencies

```
langchain
langchain-community
langchain-huggingface
langchain-ollama
langgraph
faiss-cpu
streamlit
youtube-transcript-api
pypdf
sentence-transformers
```

---

## ⚠️ Known Limitations

- **LLM-as-judge** — Implemented dual-graded self-correction; identified that local 8B models have reliability ceiling as LLM-as-judge — production deployment would require a stronger model or dedicated NLI scorer.
- **Single document per session** — the current implementation indexes one source at a time. Multi-document support is a planned upgrade.
- **Local model quality** — grounding and relevance accuracy are directly tied to the capability of the Ollama model you pull. `llama3` is recommended as the minimum. 

---

<div align="center">

Built by [Varnit](https://github.com/Varn1t) · LangGraph · FAISS · Ollama · Streamlit

</div>
