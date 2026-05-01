import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="RAG Bot", page_icon = "📚")
st.title("📚 Book Chatbot (RAG)")
st.caption("Powered by RAG + llama3")

def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(uploaded_file.read())
        tmp_path = f.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource
def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("rag_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = OllamaLLM(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

qa_chain = load_qa_chain()

if "message" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    st.chat_message(msg['role']).markdown(msg['content'])

if query:=st.chat_input("Ask a question about the book..."):
    st.session_state.messages.append({'role':'user', 'content' : query})
    st.chat_message('user').write(query)

    with st.chat_message('assistant'):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": query})
            answer = response['result']
            st.write(answer)
    st.session_state.messages.append({'role':'assistant', 'content' : answer})