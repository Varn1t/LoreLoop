import hashlib
import json
import os
import re
from functools import lru_cache
from typing import Callable, List, Optional, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

EMBED_MODEL = "all-MiniLM-L6-v2"
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_index")
YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([\w-]{11})")

# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class RAGState(TypedDict):
    original_query: str          # Never changes — the user's actual question
    question: str                # Standalone (history-resolved) question; what we answer & grade against
    current_query: str           # Retrieval query; may be rewritten each iteration
    chunks: List[str]            # Raw text of retrieved chunks
    sources: List[str]           # Citation label for each chunk (same order)
    answer: str                  # Generated answer
    is_grounded: bool            # Hallucination check result
    is_relevant: bool            # Answer relevance check result
    iterations: int              # How many attempts so far
    max_iterations: int          # Cap (default 3)
    final_answer: str            # What gets returned to the caller
    citations: List[str]         # Deduped sources behind final_answer
    failed: bool                 # True if no grounded answer was found
    run_history: List[dict]      # Diagnostic history of the agentic loops
    chat_history: List[dict]     # Historical conversation messages for contextualization

# ─────────────────────────────────────────────
# Loaders & citations
# ─────────────────────────────────────────────

def load_pdf(path_or_url: str, name: Optional[str] = None) -> List[Document]:
    docs = PyPDFLoader(path_or_url).load()
    for d in docs:
        d.metadata["source"] = name or path_or_url
    return docs

def load_youtube(url: str) -> List[Document]:
    from youtube_transcript_api import YouTubeTranscriptApi
    m = YT_ID.search(url)
    if not m:
        raise ValueError("Invalid YouTube URL")
    vid = m.group(1)
    # One Document per ~60s window so every chunk keeps a timestamp to cite.
    # ponytail: chunks never span windows, so chunk_size above ~1000 chars has little effect on transcripts.
    docs, buf, start = [], [], None
    for t in YouTubeTranscriptApi().fetch(vid):
        start = t.start if start is None else start
        buf.append(t.text)
        if t.start - start >= 60:
            docs.append(Document(" ".join(buf), metadata={"source": url, "video_id": vid, "start": start}))
            buf, start = [], None
    if buf:
        docs.append(Document(" ".join(buf), metadata={"source": url, "video_id": vid, "start": start}))
    return docs

def load_source(s: str) -> List[Document]:
    """PDF path/URL or YouTube URL, auto-detected."""
    return load_youtube(s) if YT_ID.search(s) and "youtu" in s else load_pdf(s)

def cite(meta: dict) -> str:
    """Markdown citation label: 'file.pdf, p. 3' or a timestamped YouTube link."""
    if "start" in meta:
        s = int(meta["start"])
        return f"[▶ {s // 60}:{s % 60:02d} · YouTube {meta['video_id']}](https://www.youtube.com/watch?v={meta['video_id']}&t={s}s)"
    name = os.path.basename(str(meta.get("source", "document")))
    return f"{name}, p. {meta['page'] + 1}" if "page" in meta else name

def dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))

# ─────────────────────────────────────────────
# Hybrid retrieval: semantic (FAISS) + keyword (BM25), fused by rank
# ─────────────────────────────────────────────

def make_retriever(vectorstore: FAISS, k: int) -> Callable[[str], List[Document]]:
    bm25 = BM25Retriever.from_documents(list(vectorstore.docstore._dict.values()), k=k)

    def retrieve(query: str) -> List[Document]:
        # Reciprocal rank fusion: chunks ranked high by either search float to the top.
        scores, docs = {}, {}
        for hits in (vectorstore.similarity_search(query, k=k), bm25.invoke(query)):
            for rank, d in enumerate(hits):
                scores[d.page_content] = scores.get(d.page_content, 0) + 1 / (60 + rank)
                docs[d.page_content] = d
        return [docs[t] for t in sorted(scores, key=scores.get, reverse=True)[:k]]

    return retrieve

# ─────────────────────────────────────────────
# Build the graph (called once per session)
# ─────────────────────────────────────────────

def build_rag_graph(vectorstore: FAISS, llm: OllamaLLM, k: int = 3):
    """
    Returns a compiled LangGraph app.
    vectorstore and llm are injected so the graph is stateless/reusable.
    """

    search = make_retriever(vectorstore, k)

    # ── Node 0: Contextualize Query ──────────────────────────────────
    def contextualize_query(state: RAGState) -> dict:
        chat_history = state.get("chat_history", [])
        if not chat_history:
            return {"current_query": state["original_query"], "question": state["original_query"]}

        # Format chat history (limit to last 3 conversations, which is 6 messages)
        history_str = ""
        for msg in chat_history[-6:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = f"""Given the following chat history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the chat history.
Do NOT answer the question, just rephrase it if needed to refer to the correct subject, otherwise return it exactly as is.

Chat History:
{history_str}

Follow-up Question: {state['original_query']}

Standalone Rephrased Question (write ONLY the rephrased question):"""

        standalone_query = llm.invoke(prompt).strip().strip('"\'')
        return {"current_query": standalone_query, "question": standalone_query}

    # ── Node 1: Retrieve ──────────────────────────────────────────────
    def retrieve(state: RAGState) -> dict:
        docs = search(state["current_query"])
        return {"chunks": [d.page_content for d in docs], "sources": [cite(d.metadata) for d in docs]}

    # ── Node 2: Generate ─────────────────────────────────────────────
    def generate(state: RAGState) -> dict:
        context = "\n\n---\n\n".join(state["chunks"])
        prompt = f"""You are a helpful assistant. You must answer the specific question asked using ONLY the context provided below.

Strict Instructions:
- If the question asks for examples, provide examples from the context.
- If the question asks for limitations, provide limitations from the context.
- Do not cherry-pick or answer a different aspect of the topic than what was specifically asked.
- Do not make up facts or assumptions not present in the context.
- If the context does not contain enough information to answer the specific question, say so explicitly.

Context:
{context}

Question: {state['question']}

Answer:"""
        answer = llm.invoke(prompt)
        return {"answer": answer.strip()}

    # ── Node 3: Fact-Checker & Relevance Grader ──────────────────────
    def check_hallucination(state: RAGState) -> dict:
        context = "\n\n---\n\n".join(state["chunks"])
        prompt = f"""You are an expert fact-checker and QA relevance grader. Your job is to evaluate a generated answer based on the retrieved context and the user query.

Provide two binary grades (YES or NO) in the exact format shown below:
1. GROUNDED: Is the generated answer fully supported by and traceable to the context, without making assumptions or hallucinating outside facts? (YES or NO)
2. RELEVANT: Does the generated answer directly address and resolve what the question specifically asked for?
   - Output YES ONLY if the answer specifically addresses the core intent of the question.
   - Output NO if the answer describes a different aspect of the topic (e.g., if the question asks for EXAMPLES and the answer lists LIMITATIONS instead, that is NOT RELEVANT even if the general topic is correct).
   - (YES or NO)

Context:
{context}

User Query:
{state['question']}

Generated Answer:
{state['answer']}

Verdict format (MUST write exactly this and nothing else):
GROUNDED: <YES or NO>
RELEVANT: <YES or NO>"""

        verdict_raw = llm.invoke(prompt).strip().upper()

        # Tolerates "GROUNDED: **YES**", "GROUNDED - YES", etc.
        is_grounded = bool(re.search(r"GROUNDED\W*YES", verdict_raw))
        is_relevant = bool(re.search(r"RELEVANT\W*YES", verdict_raw))
        new_iterations = state.get("iterations", 0) + 1

        # Capture telemetry history
        history = list(state.get("run_history", []))
        history.append({
            "iteration": new_iterations,
            "query": state["current_query"],
            "answer": state["answer"],
            "verdict": f"Grounded: {'YES' if is_grounded else 'NO'} | Relevant: {'YES' if is_relevant else 'NO'}",
            "is_grounded": is_grounded,
            "is_relevant": is_relevant,
            "chunks": state["chunks"],
            "sources": state["sources"],
        })

        return {
            "is_grounded": is_grounded,
            "is_relevant": is_relevant,
            "iterations": new_iterations,
            "run_history": history
        }

    # ── Node 4: Query Rewriter ───────────────────────────────────────
    def rewrite_query(state: RAGState) -> dict:
        grounded = state.get("is_grounded", False)
        relevant = state.get("is_relevant", False)

        if not grounded and not relevant:
            failure_reason = "the generated answer was neither grounded in the retrieved context nor relevant to the query"
        elif not grounded:
            failure_reason = "the generated answer contained information not supported by the retrieved context chunks (hallucination)"
        else:
            failure_reason = "the generated answer was grounded but did not actually address or resolve the user's question (off-topic)"

        prompt = f"""The following question was asked and the generated answer failed verification because {failure_reason}.
Rewrite the question to be more specific, keyword-rich, and targeted so better context chunks can be retrieved from the vector index.

Original question: {state['question']}
Previous (failed) query: {state['current_query']}
Failed answer: {state['answer']}

Write ONLY the rewritten question, nothing else:"""

        return {"current_query": llm.invoke(prompt).strip().strip('"\'')}

    # ── Node 5: Finalize (success) ───────────────────────────────────
    def finalize_success(state: RAGState) -> dict:
        return {
            "final_answer": state["answer"],
            "citations": dedupe(state["sources"]),
            "failed": False
        }

    # ── Node 6: Finalize (failure) ───────────────────────────────────
    def finalize_failure(state: RAGState) -> dict:
        # Prefer the latest answer that was at least grounded over a flat "not found".
        best = next((h for h in reversed(state["run_history"]) if h["is_grounded"]), None)
        if best:
            return {
                "final_answer": "⚠️ I couldn't confirm this fully answers your question, "
                                "but it is supported by the document:\n\n" + best["answer"],
                "citations": dedupe(best["sources"]),
                "failed": False
            }
        fallback_msg = (
            "❌ The information you're looking for does not appear to be in the loaded document. "
            f"I tried {state['iterations']} time(s) with different query formulations but could not "
            "generate a grounded answer from the available chunks."
        )
        return {
            "final_answer": fallback_msg,
            "citations": [],
            "failed": True
        }

    # ── Routing logic ────────────────────────────────────────────────
    def route_after_check(state: RAGState) -> str:
        # Route to success ONLY if the response is fully grounded AND answers the question (is relevant)
        if state["is_grounded"] and state["is_relevant"]:
            return "finalize_success"
        elif state["iterations"] >= state["max_iterations"]:
            return "finalize_failure"
        else:
            return "rewrite_query"

    # ── Assemble Graph ───────────────────────────────────────────────
    graph = StateGraph(RAGState)

    graph.add_node("contextualize_query", contextualize_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("check_hallucination", check_hallucination)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("finalize_success", finalize_success)
    graph.add_node("finalize_failure", finalize_failure)

    graph.set_entry_point("contextualize_query")
    graph.add_edge("contextualize_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_check,
        {
            "finalize_success": "finalize_success",
            "finalize_failure": "finalize_failure",
            "rewrite_query": "rewrite_query",
        }
    )
    graph.add_edge("rewrite_query", "retrieve")   # Loop back
    graph.add_edge("finalize_success", END)
    graph.add_edge("finalize_failure", END)

    return graph.compile()

# ─────────────────────────────────────────────
# Helper: build vectorstore + LLM from docs
# (shared between app.py and main.py)
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings():
    # Load the sentence-transformer once per process, not on every (re)index.
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def index_key(docs: List[Document], *settings) -> str:
    h = hashlib.sha256("|".join(map(str, (EMBED_MODEL, *settings))).encode())
    for d in docs:
        h.update(repr(sorted(d.metadata.items())).encode())
        h.update(d.page_content.encode())
    return h.hexdigest()[:16]

def get_vectorstore(docs: List[Document], chunk_size: int, chunk_overlap: int) -> FAISS:
    """Loads the FAISS index from rag_index/ if these docs were indexed before, else builds and saves it."""
    path = os.path.join(INDEX_DIR, index_key(docs, chunk_size, chunk_overlap))
    if os.path.isdir(path):
        # Safe: this pickle was written by save_local below, never downloaded.
        return FAISS.load_local(path, get_embeddings(), allow_dangerous_deserialization=True)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectorstore = FAISS.from_documents(splitter.split_documents(docs), get_embeddings())
    # ponytail: rag_index/ grows forever; delete the folder to reclaim space.
    vectorstore.save_local(path)
    return vectorstore

def parse_overview(raw: str) -> dict:
    """Splits the LLM's SUMMARY / KEY POINTS / QUESTIONS reply into fields (tolerates markdown and numbering)."""
    sections, cur = {"summary": [], "key_points": [], "questions": []}, None
    for line in raw.splitlines():
        line = line.strip()
        head = re.match(r"^[#*\s]*(SUMMARY|KEY POINTS|QUESTIONS)\b[\s:*]*(.*)$", line, re.I)
        if head:
            cur = head.group(1).lower().replace(" ", "_")
            line = head.group(2)
        if cur and line:
            sections[cur].append(re.sub(r"^([-*•]|\d+[.)])\s*", "", line).strip("* "))
    return {"summary": " ".join(sections["summary"]),
            "key_points": sections["key_points"][:6],
            "questions": sections["questions"][:5]}

def get_overview(docs: List[Document], model_name: str, llm=None) -> dict:
    """Summary, key points and suggested questions for the sources; cached in rag_index/ per content + model."""
    path = os.path.join(INDEX_DIR, f"overview_{index_key(docs, model_name)}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    text = "\n".join(d.page_content for d in docs)
    if len(text) > 8000:
        # ponytail: 8 evenly spaced excerpts stand in for the whole text; switch to map-reduce if long sources summarize poorly.
        step = len(text) // 8
        text = "\n...\n".join(text[i * step: i * step + 1000] for i in range(8))

    prompt = f"""Below are excerpts from the user's sources. Using ONLY these excerpts, write exactly these three sections:

SUMMARY: a 2-3 sentence overview of what the sources are about.
KEY POINTS: 3-5 of the most important points, one per line starting with "- ".
QUESTIONS: 4 specific questions a reader could ask that these sources clearly answer, one per line starting with "- ".

Excerpts:
{text}"""
    overview = parse_overview((llm or OllamaLLM(model=model_name)).invoke(prompt))
    if overview["summary"]:
        os.makedirs(INDEX_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(overview, f, ensure_ascii=False, indent=2)
    return overview

def build_pipeline(docs: List[Document], chunk_size: int, chunk_overlap: int,
                   k: int, model_name: str):
    """
    Builds (or loads cached) FAISS vectorstore and wires up the LangGraph RAG.
    Returns (rag_app, num_chunks).
    """
    vectorstore = get_vectorstore(docs, chunk_size, chunk_overlap)
    rag_app = build_rag_graph(vectorstore, OllamaLLM(model=model_name), k=k)
    return rag_app, len(vectorstore.docstore._dict)

def run_query(rag_app, query: str, max_iterations: int = 3, chat_history: List[dict] = None,
              on_step: Optional[Callable[[str, dict], None]] = None) -> dict:
    """
    Runs a query through the self-correcting RAG graph.
    on_step(node_name, state) is called as each graph step finishes (for live progress).
    Returns a dict with: final_answer, citations, failed, iterations, run_history.
    """
    state: RAGState = {
        "original_query": query,
        "question": query,
        "current_query": query,
        "chunks": [],
        "sources": [],
        "answer": "",
        "is_grounded": False,
        "is_relevant": False,
        "iterations": 0,
        "max_iterations": max_iterations,
        "final_answer": "",
        "citations": [],
        "failed": False,
        "run_history": [],
        # Failed turns ("❌ not found") would only confuse follow-up rephrasing.
        "chat_history": [m for m in (chat_history or []) if not m.get("failed")],
    }
    for update in rag_app.stream(state, stream_mode="updates"):
        for node, changes in update.items():
            state.update(changes or {})
            if on_step:
                on_step(node, state)
    return {
        "final_answer": state["final_answer"],
        "citations": state["citations"],
        "failed": state["failed"],
        "iterations": state["iterations"],
        "run_history": state["run_history"],
    }
