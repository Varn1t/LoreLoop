import html
import os
import tempfile

import streamlit as st

import agent

# 1. Page Configuration (Wide Layout for premium SaaS appearance)
st.set_page_config(
    page_title="LoreLoop Intelligent Engine",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Premium Design System: Inject CSS for customized UI elements
st.markdown("""
<style>
    /* Import Plus Jakarta Sans for a clean, modern aesthetic */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Apply clean font styles globally */
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Glassmorphic card custom containers */
    .premium-card {
        background: rgba(17, 25, 40, 0.65);
        backdrop-filter: blur(12px) saturate(180%);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    
    .glow-card {
        background: linear-gradient(135deg, rgba(29, 38, 113, 0.2) 0%, rgba(195, 55, 100, 0.05) 100%);
        backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 40px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Sleek gradient brand titles */
    .gradient-text {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Advanced Telemetry metric style rules */
    .metric-label {
        font-size: 0.75rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 500;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f3f4f6;
    }
    
    /* Green status indicators */
    .status-dot {
        height: 10px;
        width: 10px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #10b981;
    }
    
    .sidebar-header {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 3. Helpers & Core RAG Pipeline functions
STEP_LABELS = {
    "contextualize_query": "🧭 Understanding the question",
    "retrieve": "🔎 Searching the sources",
    "generate": "✍️ Drafting an answer",
    "check_hallucination": "🕵️ Fact-checking the draft",
    "rewrite_query": "🔄 Rewriting the query and retrying",
    "finalize_success": "✅ Verified answer ready",
    "finalize_failure": "⚠️ Out of attempts",
}

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.getvalue())
        tmp_path = f.name
    try:
        return agent.load_pdf(tmp_path, name=uploaded_file.name)
    finally:
        os.unlink(tmp_path)

def load_sources(files, urls):
    docs = []
    for f in files:
        docs += process_pdf(f)
    for u in urls:
        docs += agent.load_youtube(u)
    return docs

def index_sources(docs):
    chain, num_chunks = agent.build_pipeline(docs, chunk_size, chunk_overlap, retrieval_k, ollama_model)
    st.session_state.loaded_docs = docs
    st.session_state.qa_chain = chain
    st.session_state.num_chunks = num_chunks
    st.session_state.total_chars = sum(len(d.page_content) for d in docs)
    st.session_state.indexed_chunk_size = chunk_size
    st.session_state.indexed_chunk_overlap = chunk_overlap
    st.session_state.indexed_k = retrieval_k
    st.session_state.indexed_model = ollama_model
    try:
        st.session_state.overview = agent.get_overview(docs, ollama_model)
    except Exception as e:  # e.g. Ollama not running; the index itself is still usable
        st.session_state.overview = None
        st.toast(f"Couldn't generate overview: {e}", icon="⚠️")

def render_citations(citations):
    if citations:
        st.caption("📎 Sources: " + " · ".join(citations))

def render_telemetry(run_history, msg_idx=0):
    if not run_history:
        return

    with st.expander("🕵️ Agentic Self-Corrective Telemetry", expanded=False):
        for entry in run_history:
            iter_num = entry["iteration"]
            q = entry["query"]
            ans = entry["answer"]
            is_grounded = entry["is_grounded"]
            is_relevant = entry.get("is_relevant", True)

            if is_grounded and is_relevant:
                status_emoji = "🟢 Grounded & Relevant"
                text_color = "#10b981"
            elif is_grounded:
                status_emoji = "⚠️ Grounded but Irrelevant (Retrying...)"
                text_color = "#f59e0b"
            else:
                status_emoji = "⚠️ Hallucinated / Not Grounded (Retrying...)"
                text_color = "#ef4444"

            st.markdown(f"### 🔄 Attempt {iter_num}")
            st.markdown(f"**Targeted Query:** `{q}`")
            st.markdown(f"**Verification Verdict:** <span style='color:{text_color}; font-weight:bold;'>{status_emoji}</span>", unsafe_allow_html=True)

            st.markdown(
                f'<div style="border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; background: rgba(255,255,255,0.02); margin-bottom: 15px;">'
                f'<strong>Draft Generated Answer:</strong><br>{html.escape(ans)}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown("**Retrieved Context Chunks Used:**")
            sources = entry.get("sources") or [""] * len(entry["chunks"])
            for i, (chunk, src) in enumerate(zip(entry["chunks"], sources)):
                st.text_area(f"Chunk {i+1} — {src}", value=chunk, height=100, disabled=True, key=f"chunk_{msg_idx}_{iter_num}_{i}_{len(chunk)}")

            st.markdown("---")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_source" not in st.session_state:
    st.session_state.current_source = None

# 4. Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-header"><span class="gradient-text">🌀 LoreLoop</span></div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader("📄 PDF documents", type="pdf", accept_multiple_files=True)
    yt_text = st.text_area(
        "🎥 YouTube URLs (one per line)",
        placeholder="https://youtube.com/watch?v=...",
        help="Mix any number of PDFs and videos — they are searched together.",
    )
    yt_urls = [u.strip() for u in yt_text.splitlines() if u.strip()]

    st.markdown("---")
    st.markdown("### ⚙️ Engine Parameters")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=200,
        max_value=2000,
        value=1000,
        step=100,
        help="Number of characters in each chunk split."
    )
    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=0,
        max_value=300,
        value=50,
        step=10,
        help="Overlap characters between chunk boundary splits."
    )
    retrieval_k = st.slider(
        "Top K Retrieved Chunks",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Number of matching context chunks fetched and sent to LLM."
    )
    ollama_model = st.text_input(
        "Ollama Model",
        value="llama3",
        help="The local model name deployed on your running Ollama server."
    )
    max_iterations = st.slider(
        "Max Iterations (N)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Maximum query-rewrite and re-generation attempts before giving up.",
        key="max_iterations"
    )

    st.markdown("---")

    # Sidebar quick-action button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared!", icon="🧹")

# Resolve active source key. file_id changes on every upload, so a new PDF
# with the same filename still re-indexes (the disk cache keeps that cheap).
source_key = tuple(f.file_id for f in uploaded_files) + tuple(yt_urls) or None
source_names = [f.name for f in uploaded_files] + yt_urls

# 5. Core logic mapping state index triggers
if source_key:
    # Evaluate changes to pipeline settings
    params_changed = (
        chunk_size != st.session_state.get("indexed_chunk_size") or
        chunk_overlap != st.session_state.get("indexed_chunk_overlap") or
        retrieval_k != st.session_state.get("indexed_k") or
        ollama_model != st.session_state.get("indexed_model")
    )

    if source_key != st.session_state.get("current_source"):
        st.session_state.current_source = source_key
        st.session_state.source_label = source_names[0] if len(source_names) == 1 else f"{len(source_names)} sources"
        st.session_state.messages = []
        st.session_state.pop("loaded_docs", None)

        with st.spinner("⚡ Parsing, indexing & summarizing sources..."):
            try:
                index_sources(load_sources(uploaded_files, yt_urls))
                st.toast("✅ Sources indexed successfully!", icon="🔥")
            except Exception as e:
                st.error(f"Error processing sources: {e}")
                st.session_state.current_source = None

    elif params_changed:
        # Prompt option to trigger re-indexing when parameters change
        with st.sidebar:
            st.warning("⚠️ Parameters differ from current vector index.")
            if st.button("🔄 Re-build Vector Index", use_container_width=True):
                with st.spinner("🔄 Updating vector index with new settings..."):
                    try:
                        # Reuse already parsed/fetched docs to avoid re-reading or re-fetching
                        docs = st.session_state.get("loaded_docs") or load_sources(uploaded_files, yt_urls)
                        index_sources(docs)
                        st.toast("✅ Vector index successfully updated!", icon="🔄")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during re-indexing: {e}")


# 6. Main Content Area Rendering
if st.session_state.get("current_source") and "qa_chain" in st.session_state:
    # Visual status card of loaded document telemetry
    source_name = st.session_state.source_label
    if len(source_name) > 65:
        source_name = source_name[:62] + "..."
    safe_source_name = html.escape(source_name)

    st.markdown(f"""
        <div class="premium-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #f3f4f6; display: flex; align-items: center;">
                    <span class="status-dot"></span> Active Source: <span class="gradient-text" style="margin-left: 8px;">{safe_source_name}</span>
                </span>
                <span style="background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.4); padding: 5px 12px; border-radius: 12px; font-size: 0.8rem; color: #a5b4fc; font-weight: 600; letter-spacing: 0.05em;">
                    ⚡ PIPELINE ACTIVE
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px;">
                <div style="background: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <div class="metric-label">Parsed Chunks</div>
                    <div class="metric-value">{st.session_state.num_chunks}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <div class="metric-label">Total Characters</div>
                    <div class="metric-value">{st.session_state.total_chars:,}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <div class="metric-label">Ollama Model</div>
                    <div class="metric-value" style="color: #ec4899; font-size: 1.6rem; line-height: 2.2rem;">{html.escape(st.session_state.indexed_model)}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <div class="metric-label">Retrieval</div>
                    <div class="metric-value" style="color: #6366f1; font-size: 1.4rem; line-height: 2.2rem;">MiniLM + BM25</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Auto-summary + suggested questions (generated once per source set at index time)
    overview = st.session_state.get("overview")
    suggested = None
    suggestions_box = st.empty()  # cleared as soon as a question is asked
    if overview:
        with st.expander("📝 Source Overview", expanded=not st.session_state.messages):
            st.markdown(overview["summary"])
            if overview["key_points"]:
                st.markdown("\n".join(f"- {p}" for p in overview["key_points"]))
        if overview["questions"] and not st.session_state.messages:
            with suggestions_box.container():
                st.markdown("**💡 Try asking:**")
                for i, q in enumerate(overview["questions"]):
                    if st.button(q, key=f"suggest_{i}"):
                        suggested = q

    # Render active dialog history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_citations(msg.get("citations"))
            if "history" in msg:
                render_telemetry(msg["history"], msg_idx)

    # Process user text message input query
    if query := st.chat_input(f"Ask a question about {source_name}...") or suggested:
        suggestions_box.empty()
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            result = None
            with st.status("💭 Running Self-Corrective Agent...", expanded=False) as status:
                def on_step(node, state):
                    label = STEP_LABELS.get(node, node)
                    if node in ("retrieve", "generate"):
                        label += f" (attempt {state['iterations'] + 1})"
                    status.update(label=label)
                    status.write(label)

                try:
                    result = agent.run_query(
                        st.session_state.qa_chain,
                        query,
                        max_iterations=st.session_state.get("max_iterations", 3),
                        chat_history=st.session_state.messages[:-1],
                        on_step=on_step,
                    )
                    status.update(
                        label=f"Done in {result['iterations']} attempt(s)",
                        state="error" if result["failed"] else "complete",
                    )
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Error querying model: {e}")

            if result:
                st.markdown(result["final_answer"])
                render_citations(result["citations"])
                render_telemetry(result["run_history"], len(st.session_state.messages))
                # Failed turns are flagged so they're left out of follow-up rephrasing
                st.session_state.messages[-1]["failed"] = result["failed"]
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["final_answer"],
                    "citations": result["citations"],
                    "history": result["run_history"],
                    "failed": result["failed"],
                })

else:
    # 7. Render Custom Hero Hub empty state
    st.markdown("""
        <div class="glow-card" style="text-align: center; margin-top: 10%;">
            <h1 style="margin: 0; font-size: 2.5rem;"><span class="gradient-text">🌀 LoreLoop Hub</span></h1>
            <p style="color: #9ca3af; font-size: 1.1rem; margin-top: 10px; margin-bottom: 30px;">
                Chat seamlessly with your PDFs or YouTube Video Transcripts — fully locally, privately, and with zero API costs.
            </p>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 20px; text-align: left; margin-bottom: 10px;">
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px;">
                    <h4 style="color: #a855f7; margin-top: 0; margin-bottom: 8px;">⚡ Hybrid Search</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.4;">Semantic FAISS + keyword BM25, cached on disk so re-opening a source is instant.</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px;">
                    <h4 style="color: #6366f1; margin-top: 0; margin-bottom: 8px;">🔒 100% Secure & Local</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.4;">Powered by local llama3 via Ollama. None of your data ever leaves your computer.</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px;">
                    <h4 style="color: #ec4899; margin-top: 0; margin-bottom: 8px;">📎 Cited Answers</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.4;">Mix PDFs and YouTube videos; every answer links back to its page or video timestamp.</p>
                </div>
            </div>
            <p style="color: #6b7280; font-size: 0.9rem; margin-top: 25px;">
                👈 <b>To get started, add PDFs or YouTube links in the Sidebar menu on the left.</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
