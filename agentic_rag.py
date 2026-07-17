"""agentic_rag.py — agentic RAG with claim-level citations over the materials-science corpus.

Control loop (LangGraph StateGraph, not a one-shot chain):

    retrieve -> grade_docs -> identify_gaps -+-> [insufficient] -> retrieve (loop)
                                              +-> [sufficient]  -> generate_with_citations
                                                                       -> verify_faithfulness -> END

- retrieve: vector search on the current query, dedup by a stable content-hash chunk_id.
- grade_docs: LLM relevance filter (Corrective-RAG style) — only *new* chunks are graded,
  already-graded ones are skipped via `graded_ids`.
- identify_gaps: does the accumulated relevant context answer the question? If not, produce
  a rewritten query targeting the gap and loop back to retrieve (capped by max_iterations).
- generate_with_citations: answer as a list of claims, each tagged with the chunk_id(s) it
  relies on — the same "[chunk_id] pointer" pattern NotebookLM clones use.
- verify_faithfulness: the differentiator. Re-checks each claim against the literal text of
  its cited chunk(s) (entailment, not "does a source exist") and flags overreach — e.g. a
  paraphrase that upgrades "competitive with" into "state of the art".

Setup:
    uv run python agentic_rag.py --question "What formation-energy error does UMA report on Matbench?"

Requires the chroma_db built by materials_rag.py (same DB_DIR/COLLECTION by default).
"""
from __future__ import annotations

import argparse
import hashlib
import operator
import os
import sys
import uuid
from typing import Annotated, Literal, TypedDict

# chromadb needs this on systems with sqlite3 < 3.35 — must run before any chromadb import.
import pysqlite3 as _pysqlite3
sys.modules["sqlite3"] = _pysqlite3

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ── Config (parameterize freely — these are just defaults) ─────────────────
DB_DIR = "./chroma_db"
COLLECTION = "materials_papers"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-5.5"
RETRIEVAL_K = 4
MAX_ITERATIONS = 2  # 1 initial retrieve + up to 1 gap-driven re-retrieve


# ── Reducers for state fields that accumulate across loop iterations ───────
def _dedup_by_chunk_id(existing: list[dict], new: list[dict]) -> list[dict]:
    """Merge, keeping the first occurrence of each chunk_id."""
    seen = {d["chunk_id"] for d in existing}
    return existing + [d for d in new if d["chunk_id"] not in seen]


def _merge_dicts(existing: dict, new: dict) -> dict:
    return {**existing, **new}


class AgentState(TypedDict):
    question: str  # original research question, never mutated
    query: str  # current search query (starts == question, rewritten on gap loops)
    queries_tried: Annotated[list[str], operator.add]
    iteration: int
    max_iterations: int

    retrieved_docs: Annotated[list[dict], _dedup_by_chunk_id]  # every chunk ever retrieved
    graded_ids: Annotated[list[str], operator.add]  # chunk_ids already graded (skip re-grading)
    relevant_docs: Annotated[list[dict], _dedup_by_chunk_id]  # chunks graded relevant, accumulated
    chunk_lookup: Annotated[dict, _merge_dicts]  # chunk_id -> full chunk record

    gap: dict  # last identify_gaps verdict
    claims: list[dict]  # generate_with_citations output, pre-verification
    answer: str  # reconstructed prose with inline [chunk_id] tags
    citations: list[dict]  # verify_faithfulness output — the faithfulness report


# ── Shared model handles ────────────────────────────────────────────────────
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
vectorstore = Chroma(
    collection_name=COLLECTION,
    embedding_function=embeddings,
    persist_directory=DB_DIR,
)


def make_chunk_id(doc: Document) -> str:
    """Deterministic short id from content position, not Chroma's random per-insert UUID.

    Chroma.from_documents() assigns a fresh random UUID to every chunk on each ingest run
    (materials_rag.py doesn't pass explicit `ids=`), so those UUIDs are NOT stable across a
    re-index. Hashing (source, page, start_index) instead means the same physical passage
    gets the same citation id even if the collection is rebuilt.
    """
    src = doc.metadata.get("source", "?")
    page = doc.metadata.get("page", "?")
    start = doc.metadata.get("start_index", 0)
    raw = f"{src}|{page}|{start}"
    return hashlib.sha1(raw.encode()).hexdigest()[:8]


def format_context(docs: list[dict]) -> str:
    """Render chunks with their citation id so the model can only cite what it can see."""
    return "\n\n".join(
        f"[{d['chunk_id']}] ({d['paper']}, p.{d['page']})\n{d['text']}" for d in docs
    )


# ── Node: retrieve ───────────────────────────────────────────────────────────
def retrieve(state: AgentState) -> dict:
    query = state["query"]
    hits = vectorstore.similarity_search_with_score(query, k=RETRIEVAL_K)

    new_docs, lookup_update = [], {}
    for doc, distance in hits:  # Chroma score is L2 distance — lower is closer, not similarity
        cid = make_chunk_id(doc)
        record = {
            "chunk_id": cid,
            "paper": os.path.basename(doc.metadata.get("source", "?")).replace(".pdf", ""),
            "page": doc.metadata.get("page", "?"),
            "text": doc.page_content,
            "distance": round(float(distance), 4),
        }
        new_docs.append(record)
        lookup_update[cid] = record

    return {
        "retrieved_docs": new_docs,
        "chunk_lookup": lookup_update,
        "queries_tried": [query],
        "iteration": state["iteration"] + 1,
    }


# ── Node: grade_docs ─────────────────────────────────────────────────────────
class _DocGradeItem(BaseModel):
    chunk_id: str
    relevant: bool = Field(description="True if this chunk helps answer the question")


class _DocGrades(BaseModel):
    grades: list[_DocGradeItem]


_grade_llm = llm.with_structured_output(_DocGrades)


def grade_docs(state: AgentState) -> dict:
    graded_already = set(state.get("graded_ids", []))
    to_grade = [d for d in state["retrieved_docs"] if d["chunk_id"] not in graded_already]
    if not to_grade:
        return {}

    prompt = (
        f"Question: {state['question']}\n\n"
        "For each numbered chunk below, decide if it contains information relevant to "
        "answering the question. Be strict — tangential mentions don't count.\n\n"
        + "\n\n".join(f"chunk_id={d['chunk_id']}\n{d['text']}" for d in to_grade)
    )
    result: _DocGrades = _grade_llm.invoke(
        [SystemMessage("You are a strict relevance grader for a retrieval pipeline."),
         HumanMessage(prompt)]
    )
    relevant_ids = {g.chunk_id for g in result.grades if g.relevant}
    return {
        "graded_ids": [d["chunk_id"] for d in to_grade],
        "relevant_docs": [d for d in to_grade if d["chunk_id"] in relevant_ids],
    }


# ── Node: identify_gaps ──────────────────────────────────────────────────────
class GapAnalysis(BaseModel):
    sufficient: bool = Field(description="True if the context fully answers the question")
    missing_aspect: str = Field(default="", description="What info is still missing; '' if sufficient")
    rewritten_query: str = Field(
        default="", description="A refined search query targeting the missing aspect; '' if sufficient"
    )


_gap_llm = llm.with_structured_output(GapAnalysis)


def identify_gaps(state: AgentState) -> dict:
    # Always ask the LLM to propose a rewrite — even (especially) when relevant_docs is
    # empty. An earlier version special-cased "nothing relevant yet" by re-issuing the
    # SAME query, which made the retry a no-op: identical query -> identical top-k chunks
    # -> identical zero-relevance grade. Showing it the queries already tried lets it
    # actually vary phrasing/terminology instead of repeating a dead end.
    context = (
        format_context(state["relevant_docs"]) if state["relevant_docs"]
        else "(none of the chunks retrieved so far were judged relevant)"
    )
    tried = "; ".join(repr(q) for q in state["queries_tried"])
    prompt = (
        f"Original question: {state['question']}\n\n"
        f"Relevant context gathered so far:\n{context}\n\n"
        f"Queries already tried: {tried}\n\n"
        "Does this context fully answer the original question? If not, say what's "
        "missing and propose ONE rewritten search query — using different phrasing or "
        "more specific terminology than any query already tried — that would surface "
        "the missing information."
    )
    gap = _gap_llm.invoke(
        [SystemMessage("You are a retrieval gap analyst."), HumanMessage(prompt)]
    )
    next_query = gap.rewritten_query if not gap.sufficient and gap.rewritten_query else state["query"]
    return {"gap": gap.model_dump(), "query": next_query}


def route_after_gaps(state: AgentState) -> Literal["retrieve", "generate"]:
    if state["gap"]["sufficient"]:
        return "generate"
    if state["iteration"] >= state["max_iterations"]:
        return "generate"  # budget exhausted — answer with what we have, gaps show up in the trace
    return "retrieve"


# ── Node: generate_with_citations ────────────────────────────────────────────
class _Claim(BaseModel):
    text: str = Field(description="One factual claim, as a standalone sentence")
    chunk_ids: list[str] = Field(
        default_factory=list,
        description="chunk_id(s) from the context that directly support this claim. "
        "Empty ONLY for connective/transition sentences that assert no fact.",
    )


class _CitedAnswer(BaseModel):
    claims: list[_Claim]


_generate_llm = llm.with_structured_output(_CitedAnswer)


def generate_with_citations(state: AgentState) -> dict:
    # Structured claim+chunk_ids output (vs. asking for free prose and regexing out [id]
    # tags afterwards) trades a bit of prose fluency for citations that are never malformed
    # or dropped by the model, and it's what verify_faithfulness consumes directly.
    context_docs = state["relevant_docs"] or state["retrieved_docs"]
    if not context_docs:
        return {
            "claims": [],
            "answer": "No relevant passages were found in the corpus for this question.",
        }

    prompt = (
        f"Question: {state['question']}\n\n"
        f"Context:\n{format_context(context_docs)}\n\n"
        "Answer the question using ONLY this context. Break the answer into individual "
        "factual claims. Every claim must cite the chunk_id(s) it came from. Do not use "
        "any chunk_id that isn't listed above. If the context doesn't answer the question, "
        "say so as a single claim with an empty chunk_ids list."
    )
    result: _CitedAnswer = _generate_llm.invoke(
        [SystemMessage("You are a materials-science research assistant. Never state a "
                        "claim you cannot attribute to the given context."),
         HumanMessage(prompt)]
    )
    claims = [c.model_dump() for c in result.claims]
    prose = " ".join(
        f"{c['text']} [{', '.join(c['chunk_ids'])}]" if c["chunk_ids"] else c["text"]
        for c in claims
    )
    return {"claims": claims, "answer": prose}


# ── Node: verify_faithfulness ────────────────────────────────────────────────
class FaithfulnessCheck(BaseModel):
    verdict: Literal["supported", "partially_supported", "unsupported"]
    explanation: str = Field(description="One sentence: why the verdict")


_verify_llm = llm.with_structured_output(FaithfulnessCheck)


def verify_faithfulness(state: AgentState) -> dict:
    """The credibility feature: does the cited SPAN actually entail the CLAIM, not just
    'does a source with this chunk_id exist'. Catches the known NotebookLM failure mode —
    a correct pointer wrapped around an overreaching paraphrase (e.g. "competitive with"
    quietly becoming "state of the art").
    """
    citations = []
    for claim in state["claims"]:
        chunk_ids = claim["chunk_ids"]
        sources = [state["chunk_lookup"][cid] for cid in chunk_ids if cid in state["chunk_lookup"]]

        if not chunk_ids:
            citations.append({
                "claim": claim["text"], "chunk_ids": [], "sources": [],
                "verdict": "uncited", "explanation": "Claim has no supporting citation.",
            })
            continue

        span_text = "\n---\n".join(s["text"] for s in sources)
        prompt = (
            f"SOURCE SPAN(S):\n{span_text}\n\nCLAIM: {claim['text']}\n\n"
            "Does the source span literally support this claim? Numbers, comparatives "
            "('higher than', 'faster'), and superlatives ('state of the art', 'best') "
            "must be directly stated or computable from the span — do not credit a "
            "plausible-sounding paraphrase that isn't actually backed by the text."
        )
        check: FaithfulnessCheck = _verify_llm.invoke(
            [SystemMessage("You are a strict fact-checker verifying claim-to-source entailment."),
             HumanMessage(prompt)]
        )
        citations.append({
            "claim": claim["text"],
            "chunk_ids": chunk_ids,
            "sources": [{"chunk_id": s["chunk_id"], "paper": s["paper"], "page": s["page"],
                         "snippet": s["text"][:200]} for s in sources],
            "verdict": check.verdict,
            "explanation": check.explanation,
        })
    return {"citations": citations}


# ── Graph assembly ──────────────────────────────────────────────────────────
def build_graph():
    """Custom StateGraph, not create_agent's prebuilt ReAct loop — the node sequence here
    (grade -> gap-check -> conditional re-retrieve -> verify) is a fixed control flow, not
    a tool-calling loop the model steers, so a hand-wired graph gives exact control over it.
    """
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_docs", grade_docs)
    graph.add_node("identify_gaps", identify_gaps)
    graph.add_node("generate", generate_with_citations)
    graph.add_node("verify", verify_faithfulness)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade_docs")
    graph.add_edge("grade_docs", "identify_gaps")
    graph.add_conditional_edges(
        "identify_gaps", route_after_gaps, {"retrieve": "retrieve", "generate": "generate"}
    )
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    # Checkpointer gotcha: retrieved_docs / relevant_docs / graded_ids all persist under a
    # thread_id across separate .invoke() calls. Reuse a thread_id for a second, unrelated
    # question and grade_docs will skip re-grading chunks retrieved for the FIRST question,
    # and identify_gaps will judge sufficiency against a mixed-question context. Always mint
    # a fresh thread_id per question (see ask() below) — only reuse one for turns that are
    # genuinely follow-ups on the same question thread.
    return graph.compile(checkpointer=InMemorySaver())


# ── Driver ───────────────────────────────────────────────────────────────────
def ask(question: str, max_iterations: int = MAX_ITERATIONS, verbose: bool = True) -> AgentState:
    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial: AgentState = {
        "question": question, "query": question, "queries_tried": [],
        "iteration": 0, "max_iterations": max_iterations,
        "retrieved_docs": [], "graded_ids": [], "relevant_docs": [], "chunk_lookup": {},
        "gap": {}, "claims": [], "answer": "", "citations": [],
    }
    final_state = graph.invoke(initial, config=config)

    if verbose:
        _print_trace(final_state)
    return final_state


def _print_trace(state: AgentState) -> None:
    print(f"\n{'=' * 70}\nQUESTION: {state['question']}\n{'=' * 70}")
    print(f"\nQueries tried ({state['iteration']} retrieval pass(es)):")
    for i, q in enumerate(state["queries_tried"], 1):
        print(f"  {i}. {q!r}")
    print(f"\nRelevant chunks kept: {len(state['relevant_docs'])} "
          f"of {len(state['retrieved_docs'])} retrieved")
    if state["gap"]:
        print(f"Final gap verdict: sufficient={state['gap']['sufficient']}"
              + (f" (missing: {state['gap']['missing_aspect']})" if not state['gap']['sufficient'] else ""))

    print(f"\n--- ANSWER ---\n{state['answer']}\n")

    print("--- FAITHFULNESS REPORT ---")
    for c in state["citations"]:
        flag = {"supported": "✅", "partially_supported": "⚠️ ", "unsupported": "❌", "uncited": "❌"}[c["verdict"]]
        print(f"{flag} [{c['verdict']}] {c['claim']}")
        print(f"     {c['explanation']}")
        for s in c["sources"]:
            print(f"     source [{s['chunk_id']}] {s['paper']} p.{s['page']}: {s['snippet']!r}...")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--question", default="What formation-energy prediction error does UMA "
                                           "report, and how does it compare to MACE-MP-0?")
    ap.add_argument("--db-dir", default=DB_DIR)
    ap.add_argument("--collection", default=COLLECTION)
    ap.add_argument("--chat-model", default=CHAT_MODEL)
    ap.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    ap.add_argument("--k", type=int, default=RETRIEVAL_K)
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = ap.parse_args()

    # Re-point module-level handles if the caller overrode any connection params.
    if (args.db_dir, args.collection, args.chat_model, args.embedding_model) != (
        DB_DIR, COLLECTION, CHAT_MODEL, EMBEDDING_MODEL
    ):
        embeddings = OpenAIEmbeddings(model=args.embedding_model)
        llm = ChatOpenAI(model=args.chat_model, temperature=0)
        vectorstore = Chroma(
            collection_name=args.collection, embedding_function=embeddings,
            persist_directory=args.db_dir,
        )
    RETRIEVAL_K = args.k

    ask(args.question, max_iterations=args.max_iterations)
