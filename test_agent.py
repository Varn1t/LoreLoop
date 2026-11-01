# Offline self-check: real embeddings/FAISS/BM25, scripted fake LLM (no Ollama needed).
# Run: python test_agent.py
import os, sys, time, tempfile, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM

agent.INDEX_DIR = tempfile.mkdtemp()  # keep the repo's rag_index/ clean

docs = [
    Document("The policy covers flood damage up to 50,000 dollars.", metadata={"source": "home.pdf", "page": 0}),
    Document("Claim form ZX-4471 must be filed within 30 days.", metadata={"source": "home.pdf", "page": 4}),
    Document("Photosynthesis converts light into chemical energy.",
             metadata={"source": "https://youtu.be/abcdefghijk", "video_id": "abcdefghijk", "start": 125.4}),
]

# citations
assert agent.cite(docs[1].metadata) == "home.pdf, p. 5"
assert agent.cite(docs[2].metadata).startswith("[▶ 2:05") and "&t=125s" in agent.cite(docs[2].metadata)

# disk cache: second build loads instead of re-embedding
vs = agent.get_vectorstore(docs, 1000, 50)
assert len(os.listdir(agent.INDEX_DIR)) == 1
t = time.time(); vs2 = agent.get_vectorstore(docs, 1000, 50)
assert len(vs2.docstore._dict) == 3 and len(os.listdir(agent.INDEX_DIR)) == 1
agent.get_vectorstore(docs, 500, 50); assert len(os.listdir(agent.INDEX_DIR)) == 2  # params change -> new index

# hybrid search: exact keyword (form ID) ranks first
hits = agent.make_retriever(vs, 2)("ZX-4471")
assert hits[0].metadata["page"] == 4, hits

def run(responses, max_it=3, history=None):
    app = agent.build_rag_graph(vs, FakeListLLM(responses=responses), k=2)
    steps = []
    r = agent.run_query(app, "how do I file form ZX-4471?", max_it, history, on_step=lambda n, s: steps.append(n))
    return r, steps

ok = "GROUNDED: **YES**\nRELEVANT: YES"
r, steps = run(["File within 30 days.", ok])
assert not r["failed"] and r["final_answer"] == "File within 30 days." and "home.pdf, p. 5" in r["citations"]
assert steps == ["contextualize_query", "retrieve", "generate", "check_hallucination", "finalize_success"], steps

# grounded-but-off-topic then hallucinated -> falls back to the grounded attempt
r, _ = run(["draft one", "GROUNDED: YES\nRELEVANT: NO", "better query", "draft two", "GROUNDED: NO\nRELEVANT: NO"], max_it=2)
assert not r["failed"] and r["final_answer"].startswith("⚠️") and r["final_answer"].endswith("draft one") and r["citations"]

# nothing grounded -> hard failure
r, _ = run(["made up", "GROUNDED: NO\nRELEVANT: NO"], max_it=1)
assert r["failed"] and r["final_answer"].startswith("❌") and r["citations"] == []

# failed turns are dropped from history -> no rephrase LLM call happens
hist = [{"role": "user", "content": "x", "failed": True}, {"role": "assistant", "content": "❌", "failed": True}]
r, _ = run(["answer", ok], history=hist)
assert r["final_answer"] == "answer"

# YouTube windowing into ~60s documents with timestamps
seg = lambda s, txt: types.SimpleNamespace(start=s, text=txt)
import youtube_transcript_api
class FakeApi:
    def fetch(self, vid): return [seg(0, "a"), seg(30, "b"), seg(61, "c"), seg(90, "d")]
youtube_transcript_api.YouTubeTranscriptApi = FakeApi
yt = agent.load_youtube("https://www.youtube.com/shorts/abcdefghijk")
assert [(d.page_content, d.metadata["start"]) for d in yt] == [("a b c", 0), ("d", 90)], yt
assert agent.load_source("https://youtu.be/abcdefghijk")[0].metadata["video_id"] == "abcdefghijk"
# overview parsing tolerates markdown headings, bold and numbering
ov = agent.parse_overview("**SUMMARY:** A home policy.\nIt covers floods.\n## Key Points\n1. Flood cover\n* Claims in 30 days\n"
                          "QUESTIONS:\n- What is covered?\n2) How do I claim?")
assert ov == {"summary": "A home policy. It covers floods.", "key_points": ["Flood cover", "Claims in 30 days"],
              "questions": ["What is covered?", "How do I claim?"]}, ov

# overview is cached on disk per content + model: the second call makes no LLM call
llm = FakeListLLM(responses=["SUMMARY: s\nKEY POINTS:\n- k\nQUESTIONS:\n- q?"])
assert agent.get_overview(docs, "m", llm)["questions"] == ["q?"]
assert agent.get_overview(docs, "m", FakeListLLM(responses=["SUMMARY: other"]))["summary"] == "s"

# eval scoring: whole-token matching and "a|b" alternatives
import evaluate
assert evaluate.mentions("trained in 2018 on 8 GPUs", "8") and not evaluate.mentions("in 2018", "8")
assert not evaluate.mentions("1706.03762, p. 50", "1706.03762, p. 5")
fake = {"final_answer": "Six layers.", "failed": False, "iterations": 1,
        "run_history": [{"sources": ["x.pdf, p. 3"]}]}
row = evaluate.score_case({"question": "q", "expect_source": ["x.pdf, p. 3"], "expect_in_answer": ["6|six"]}, fake, 1.0)
assert row["retrieval_hit"] and row["answer_match"] and row["verified"]
assert evaluate.summarize([row, {**row, "answer_match": False, "retrieval_hit": None}])["answer_match_%"] == 50
print("ALL CHECKS PASSED")
