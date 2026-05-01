from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader("https://jcer.in/jcer-docs/E-Learning/Digital%20Library%20/E-Books/Data%20Science%20from%20Scratch%20by%20Joel%20Grus.pdf")
pages = loader.load()

print(f"Loaded {len(pages)} pages")

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap =50
)

chunks = splitter.split_documents(pages)
print(f"Created {len(chunks)} chunks")
print("\n Sample Chunk: ")
print(chunks[0].page_content)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#Load embedding model (download ~90 mb  on the first run)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#embed all chunks and store in FAISS
print("Building vector store... (may take a minute)")
vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("rag_index")
print("Done! Vector store saved")

from langchain_ollama import OllamaLLM
from langchain_classic.chains import RetrievalQA
# Load the saved vector store
vectorstore = FAISS.load_local("rag_index", embeddings, allow_dangerous_deserialization=True)
# Convert to retriever (fetches top 3 relevant chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

#Hook up llama3
llm = OllamaLLM(model="llama3")
# Build the QA chain

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
)

# Ask a question
print("\n RAG is ready! Ask anything about the book. Type 'exit' to quit \n")
while True: 
    query=(input("You: "))
    if query.lower() == 'exit':
        break
    else:
        response = qa_chain.invoke({"query":query})
        print(f"\nBot:   {response['result']}\n")