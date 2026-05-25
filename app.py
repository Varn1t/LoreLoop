import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import os
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
def build_qa_chain(docs, chunk_size, chunk_overlap, k, model_name):
    return agent.build_pipeline(docs, chunk_size, chunk_overlap, k, model_name)

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name
    try:
        loader = PyPDFLoader(tmp_path)
        return loader.load()
    finally:
        os.unlink(tmp_path)

def load_youtube(url):
    if "v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL")
    
    ytt = YouTubeTranscriptApi()
    transcript = ytt.fetch(video_id)
    full_text = " ".join([t.text for t in transcript])
    return [Document(page_content=full_text, metadata={"source": url})]

def render_telemetry(run_history, msg_idx=0):
    if not run_history:
        return
        
    with st.expander("🕵️ Agentic Self-Corrective Telemetry", expanded=False):
        for entry in run_history:
            iter_num = entry["iteration"]
            q = entry["query"]
            ans = entry["answer"]
            verdict = entry["verdict"]
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
            
            # Simple border container
            st.markdown(
                f'<div style="border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; background: rgba(255,255,255,0.02); margin-bottom: 15px;">'
                f'<strong>Draft Generated Answer:</strong><br>{ans}'
                f'</div>', 
                unsafe_allow_html=True
            )
            
            st.markdown("**Retrieved Context Chunks Used:**")
            for i, chunk in enumerate(entry["chunks"]):
                st.text_area(f"Chunk {i+1}", value=chunk, height=100, disabled=True, key=f"chunk_{msg_idx}_{iter_num}_{i}_{len(chunk)}")
            
            st.markdown("---")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_source" not in st.session_state:
    st.session_state.current_source = None

# 4. Sidebar configuration
with st.sidebar:
    st.markdown('<div class="sidebar-header"><span class="gradient-text">🌀 LoreLoop</span></div>', unsafe_allow_html=True)
    
    # Sleek document source selection tabs
    loader_type = st.radio(
        "Select Document Source Type", 
        ["📄 PDF Upload", "🎥 YouTube Link"], 
        help="Choose between uploading a PDF document or pasting a YouTube video URL."
    )
    
    uploaded_file = None
    yt_url = None
    
    if loader_type == "📄 PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf", label_visibility="collapsed")
    else:
        yt_url = st.text_input("Paste YouTube Video URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")
        
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

# Resolve active upload source key
source_key = None
if loader_type == "📄 PDF Upload" and uploaded_file:
    source_key = uploaded_file.name
elif loader_type == "🎥 YouTube Link" and yt_url and yt_url.strip():
    source_key = yt_url.strip()

# 5. Core logic mapping state index triggers
if source_key:
    # Evaluate changes to either active source or pipeline settings
    params_changed = (
        source_key != st.session_state.get("current_source") or
        chunk_size != st.session_state.get("indexed_chunk_size") or
        chunk_overlap != st.session_state.get("indexed_chunk_overlap") or
        retrieval_k != st.session_state.get("indexed_k") or
        ollama_model != st.session_state.get("indexed_model")
    )
    
    is_new_source = source_key != st.session_state.get("current_source")
    
    # Process if it is a brand new source loading in
    if is_new_source:
        st.session_state.current_source = source_key
        st.session_state.messages = []
        if "loaded_docs" in st.session_state:
            del st.session_state.loaded_docs
            
        with st.spinner("⚡ Parsing & Indexing document source..."):
            try:
                if uploaded_file:
                    docs = process_pdf(uploaded_file)
                else:
                    docs = load_youtube(yt_url)
                
                st.session_state.loaded_docs = docs
                
                total_chars = sum(len(d.page_content) for d in docs)
                chain, num_chunks = build_qa_chain(docs, chunk_size, chunk_overlap, retrieval_k, ollama_model)
                
                st.session_state.qa_chain = chain
                st.session_state.num_chunks = num_chunks
                st.session_state.total_chars = total_chars
                st.session_state.indexed_chunk_size = chunk_size
                st.session_state.indexed_chunk_overlap = chunk_overlap
                st.session_state.indexed_k = retrieval_k
                st.session_state.indexed_model = ollama_model
                st.toast("✅ Document indexed successfully!", icon="🔥")
            except Exception as e:
                st.error(f"Error processing document: {e}")
                st.session_state.current_source = None
                
    elif params_changed:
        # Prompt option to trigger re-indexing when parameters change
        with st.sidebar:
            st.warning("⚠️ Parameters differ from current vector index.")
            if st.button("🔄 Re-build Vector Index", use_container_width=True):
                with st.spinner("🔄 Updating vector index with new settings..."):
                    try:
                        # Retrieve already parsed/fetched docs from session state to avoid re-reading or re-fetching
                        docs = st.session_state.get("loaded_docs")
                        if not docs:
                            if uploaded_file:
                                docs = process_pdf(uploaded_file)
                            else:
                                docs = load_youtube(yt_url)
                            st.session_state.loaded_docs = docs
                        
                        total_chars = sum(len(d.page_content) for d in docs)
                        chain, num_chunks = build_qa_chain(docs, chunk_size, chunk_overlap, retrieval_k, ollama_model)
                        
                        st.session_state.qa_chain = chain
                        st.session_state.num_chunks = num_chunks
                        st.session_state.total_chars = total_chars
                        st.session_state.indexed_chunk_size = chunk_size
                        st.session_state.indexed_chunk_overlap = chunk_overlap
                        st.session_state.indexed_k = retrieval_k
                        st.session_state.indexed_model = ollama_model
                        st.toast("✅ Vector index successfully updated!", icon="🔄")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error during re-indexing: {e}")


# 6. Main Content Area Rendering
if st.session_state.get("current_source") and "qa_chain" in st.session_state:
    # Visual status card of loaded document telemetry
    source_name = st.session_state.current_source
    if len(source_name) > 65:
        source_name = source_name[:62] + "..."
        
    st.markdown(f"""
        <div class="premium-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 10px;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #f3f4f6; display: flex; align-items: center;">
                    <span class="status-dot"></span> Active Source: <span class="gradient-text" style="margin-left: 8px;">{source_name}</span>
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
                    <div class="metric-value" style="color: #ec4899; font-size: 1.6rem; line-height: 2.2rem;">{st.session_state.indexed_model}</div>
                </div>
                <div style="background: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.05);">
                    <div class="metric-label">Embeddings</div>
                    <div class="metric-value" style="color: #6366f1; font-size: 1.4rem; line-height: 2.2rem;">all-MiniLM-L6</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Render active dialog history
    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "history" in msg:
                render_telemetry(msg["history"], msg_idx)
            
    # Process user text message input query
    if query := st.chat_input(f"Ask a question about {source_name}..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            status_container = st.container()
            ans_placeholder = st.empty()
            with st.spinner("💭 Running Self-Corrective Agent..."):
                try:
                    # Get the last 3 turns of conversation history (last 6 messages)
                    # excluding the current user message which was just appended
                    history = st.session_state.messages[:-1][-6:] if len(st.session_state.messages) > 1 else []
                    
                    result = agent.run_query(
                        st.session_state.qa_chain, 
                        query, 
                        max_iterations=st.session_state.get("max_iterations", 3),
                        chat_history=history
                    )
                    answer = result["final_answer"]
                    ans_placeholder.markdown(answer)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "history": result["run_history"]
                    })
                    with status_container:
                        render_telemetry(result["run_history"], len(st.session_state.messages) - 1)
                except Exception as e:
                    st.error(f"Error querying model: {e}")
                    
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
                    <h4 style="color: #a855f7; margin-top: 0; margin-bottom: 8px;">⚡ High Performance</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.4;">Instant vector indexing using FAISS CPU combined with local sentence embeddings.</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px;">
                    <h4 style="color: #6366f1; margin-top: 0; margin-bottom: 8px;">🔒 100% Secure & Local</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.4;">Powered by local llama3 via Ollama. None of your data ever leaves your computer.</p>
                </div>
                <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 12px;">
                    <h4 style="color: #ec4899; margin-top: 0; margin-bottom: 8px;">🎥 YouTube Integration</h4>
                    <p style="color: #9ca3af; font-size: 0.85rem; margin: 0; line-height: 1.4;">Paste a video link to fetch its transcript automatically and run contextual queries.</p>
                </div>
            </div>
            <p style="color: #6b7280; font-size: 0.9rem; margin-top: 25px;">
                👈 <b>To get started, select your source and load it in the Sidebar menu on the left.</b>
            </p>
        </div>
    """, unsafe_allow_html=True)