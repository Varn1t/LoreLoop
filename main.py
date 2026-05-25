import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import agent

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

def load_pdf(path_or_url):
    loader = PyPDFLoader(path_or_url)
    return loader.load()

def main():
    print("Welcome to the LoreLoop CLI!")
    print("1. Load a PDF (URL or local path)")
    print("2. Load a YouTube Video URL")
    
    choice = input("Select an option (1 or 2): ").strip()
    
    pages = []
    if choice == "1":
        path = input("Enter PDF URL or local path: ").strip()
        print(f"Loading PDF from {path}...")
        try:
            pages = load_pdf(path)
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return
    elif choice == "2":
        url = input("Enter YouTube Video URL: ").strip()
        print(f"Loading Transcript from {url}...")
        try:
            pages = load_youtube(url)
        except Exception as e:
            print(f"Error loading YouTube transcript: {e}")
            return
    else:
        print("Invalid choice. Exiting.")
        return

    if not pages:
        print("No content could be loaded. Exiting.")
        return

    print(f"Loaded {len(pages)} documents/pages")

    print("\nBuilding agent pipeline... (may take a minute)")
    rag_app, num_chunks = agent.build_pipeline(
        docs=pages,
        chunk_size=1000,
        chunk_overlap=50,
        k=3,
        model_name="llama3"
    )
    print(f"Created and indexed {num_chunks} chunks.")

    # Ask a question
    print("\nLoreLoop is ready! Ask anything about the document. Type 'exit' to quit \n")
    chat_history = []
    while True: 
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            print("Thanks for using LoreLoop!")
            break
        else:
            print("\nBot is thinking (Self-Correcting loop running)...")
            try:
                result = agent.run_query(
                    rag_app, 
                    query, 
                    max_iterations=3,
                    chat_history=chat_history[-6:]
                )
                
                # Print diagnostic logs of iterations to console
                print("\n================== AGENT DIAGNOSTICS ==================")
                for entry in result["run_history"]:
                    iter_num = entry["iteration"]
                    q = entry["query"]
                    verdict = entry["verdict"]
                    print(f"Attempt #{iter_num} | Query formulated: '{q}' | Verdict: {verdict}")
                print("=======================================================")
                
                print(f"\nBot:   {result['final_answer']}\n")
                
                # Append current conversation turn to history
                chat_history.append({"role": "user", "content": query})
                chat_history.append({"role": "assistant", "content": result["final_answer"]})
            except Exception as e:
                print(f"\nError querying agent: {e}\n")

if __name__ == "__main__":
    main()