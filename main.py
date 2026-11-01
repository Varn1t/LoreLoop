import sys

import agent

sys.stdout.reconfigure(encoding="utf-8")  # emoji crash Windows' cp1252 when output is piped


def main():
    print("Welcome to the LoreLoop CLI!")
    print("Add sources one per line: a PDF (local path or URL) or a YouTube URL.")
    print("Press Enter on an empty line when done.\n")

    pages = []
    while True:
        source = input("Source: ").strip()
        if not source:
            break
        try:
            docs = agent.load_source(source)
            pages += docs
            print(f"  ✓ loaded {len(docs)} section(s)")
        except Exception as e:
            print(f"  ✗ could not load: {e}")

    if not pages:
        print("No content could be loaded. Exiting.")
        return

    print(f"\nBuilding agent pipeline over {len(pages)} section(s)... (may take a minute)")
    rag_app, num_chunks = agent.build_pipeline(
        docs=pages,
        chunk_size=1000,
        chunk_overlap=50,
        k=3,
        model_name="llama3"
    )
    print(f"Indexed {num_chunks} chunks (reused from rag_index/ if seen before).")

    questions = []
    try:
        overview = agent.get_overview(pages, "llama3")
        questions = overview["questions"]
        print(f"\n📝 OVERVIEW\n{overview['summary']}")
        for p in overview["key_points"]:
            print(f"  • {p}")
        if questions:
            print("\n💡 Try asking (type the number):")
            for i, q in enumerate(questions, 1):
                print(f"  {i}. {q}")
    except Exception as e:
        print(f"(Couldn't generate overview: {e})")

    print("\nLoreLoop is ready! Ask anything about your sources. Type 'exit' to quit \n")
    chat_history = []
    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() == 'exit':
            print("Thanks for using LoreLoop!")
            break
        if query.isdigit() and 1 <= int(query) <= len(questions):
            query = questions[int(query) - 1]
            print(f"You: {query}")

        print()
        try:
            result = agent.run_query(
                rag_app,
                query,
                max_iterations=3,
                chat_history=chat_history,
                on_step=lambda node, state: print(f"  → {node}"),
            )

            # Print diagnostic logs of iterations to console
            print("\n================== AGENT DIAGNOSTICS ==================")
            for entry in result["run_history"]:
                print(f"Attempt #{entry['iteration']} | Query formulated: '{entry['query']}' | Verdict: {entry['verdict']}")
            print("=======================================================")

            print(f"\nBot:   {result['final_answer']}")
            if result["citations"]:
                print("Sources: " + " · ".join(result["citations"]))
            print()

            # Failed turns are flagged so they're left out of follow-up rephrasing
            chat_history.append({"role": "user", "content": query, "failed": result["failed"]})
            chat_history.append({"role": "assistant", "content": result["final_answer"], "failed": result["failed"]})
        except Exception as e:
            print(f"\nError querying agent: {e}\n")


if __name__ == "__main__":
    main()
