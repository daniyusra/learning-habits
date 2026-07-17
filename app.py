# Materials-Science Paper Chat — agentic RAG with claim-level citations.
# Chat naturally, like NotebookLM. Every node in the retrieve -> grade -> gap-check
# -> generate -> verify loop (agentic_rag.py) is streamed to the UI as an inspectable
# Step, and every claim in the final answer shows its verdict from the faithfulness
# check, not just a source pointer.
# Run: uv run uvicorn app:app --reload

import pysqlite3 as _pysqlite3  # chromadb needs this on some systems
import sys
sys.modules["sqlite3"] = _pysqlite3

import uuid

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl

import agentic_rag

# ── Build the graph once; each chat turn gets its own thread_id (see the
# checkpointer note in agentic_rag.build_graph) so loop state from one question
# never bleeds into grading/gap-checking for the next. ─────────────────────────
_graph = agentic_rag.build_graph()

_STEP_TYPE = {
    "retrieve": "retrieval",
    "grade_docs": "llm",
    "identify_gaps": "llm",
    "generate": "llm",
    "verify": "llm",
}
_STEP_LABEL = {
    "retrieve": "retrieve",
    "grade_docs": "grade chunks",
    "identify_gaps": "check for gaps",
    "generate": "draft cited answer",
    "verify": "verify claims against sources",
}


# ── Per-node trace formatting — what shows up inside each Step in the UI ──────
def _describe(node_name: str, out: dict) -> tuple[str, str]:
    if node_name == "retrieve":
        query = out["queries_tried"][-1]
        docs = out["retrieved_docs"]
        body = "\n".join(
            f"  [{d['chunk_id']}] {d['paper']} p.{d['page']}  (distance={d['distance']})"
            for d in docs
        )
        return query, f"{len(docs)} chunk(s) retrieved:\n{body}"

    if node_name == "grade_docs":
        if not out:
            return "(no new chunks)", "skipped — every retrieved chunk was already graded on an earlier pass"
        kept = {d["chunk_id"] for d in out["relevant_docs"]}
        lines = [
            f"  [{cid}] {'kept — relevant' if cid in kept else 'dropped — not relevant'}"
            for cid in out["graded_ids"]
        ]
        return f"grading {len(out['graded_ids'])} chunk(s)", "\n".join(lines)

    if node_name == "identify_gaps":
        gap = out["gap"]
        if gap["sufficient"]:
            return "is the context enough to answer?", "sufficient — moving to generate"
        return (
            "is the context enough to answer?",
            f"insufficient — missing: {gap['missing_aspect']}\nrewritten query: {out['query']!r}",
        )

    if node_name == "generate":
        lines = [
            f"  - {c['text']}  [{', '.join(c['chunk_ids']) or 'uncited'}]"
            for c in out["claims"]
        ]
        return "drafting cited answer from relevant context", "\n".join(lines)

    if node_name == "verify":
        lines = [f"  {c['verdict']}: {c['claim']}\n      {c['explanation']}" for c in out["citations"]]
        return "checking each claim against its cited span", "\n".join(lines)

    return "", str(out)


# ── Final chat message formatting — inline citation markers + a sources panel,
# plus an explicit flag for any claim the verifier didn't fully back. ─────────
def _format_answer(state: dict) -> str:
    citations = state["citations"]
    if not citations:
        return state["answer"] or "I couldn't find anything relevant in the corpus for that."

    footnote_order: list[str] = []
    footnote_index: dict[str, int] = {}
    for c in citations:
        for cid in c["chunk_ids"]:
            if cid not in footnote_index:
                footnote_index[cid] = len(footnote_order) + 1
                footnote_order.append(cid)

    prose = " ".join(
        c["claim"] + "".join(f"[{footnote_index[cid]}]" for cid in c["chunk_ids"])
        for c in citations
    )

    lookup = state["chunk_lookup"]
    lines = [prose, "", "---", "**Sources**"]
    for cid in footnote_order:
        rec = lookup.get(cid, {})
        snippet = rec.get("text", "").replace("\n", " ").strip()[:180]
        lines.append(f"{footnote_index[cid]}. `{cid}` — *{rec.get('paper', '?')}*, p.{rec.get('page', '?')}: “{snippet}…”")

    flagged = [c for c in citations if c["verdict"] != "supported"]
    if flagged:
        lines += ["", "**⚠️ Flagged** — a citation exists but the wording overreaches the source:"]
        lines += [f"- *{c['verdict']}*: “{c['claim']}” — {c['explanation']}" for c in flagged]
    else:
        lines += ["", f"✅ All {len(citations)} claim(s) verified against their cited sources."]

    return "\n".join(lines)


def _contextualize(history: list[tuple[str, str]], new_message: str) -> str:
    """Fold recent turns into the question so follow-ups ('what about the other
    paper?') retrieve sensibly — the graph itself is stateless per question."""
    if not history:
        return new_message
    ctx = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    return f"Prior conversation:\n{ctx}\n\nNew question: {new_message}"


# ── Chainlit UI ────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    data = agentic_rag.vectorstore.get(include=["metadatas"])
    papers = sorted({md.get("source", "?").split("/")[-1].replace(".pdf", "") for md in data["metadatas"]})
    await cl.Message(
        content=(
            "**Materials-Science Paper Chat**\n\n"
            "Ask a question and I'll retrieve, grade, and — if needed — re-retrieve "
            "before answering. Every claim cites the exact chunk it came from, and "
            "each citation is checked against that chunk before you see it.\n\n"
            f"Loaded papers: {', '.join(papers)}"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    history = cl.user_session.get("history", [])
    question = _contextualize(history, message.content)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {
        "question": question, "query": question, "queries_tried": [],
        "iteration": 0, "max_iterations": agentic_rag.MAX_ITERATIONS,
        "retrieved_docs": [], "graded_ids": [], "relevant_docs": [], "chunk_lookup": {},
        "gap": {}, "claims": [], "answer": "", "citations": [],
    }

    async for update in _graph.astream(initial_state, stream_mode="updates", config=cfg):
        for node_name, node_output in update.items():
            step_input, step_output = _describe(node_name, node_output)
            step = cl.Step(
                name=_STEP_LABEL.get(node_name, node_name),
                type=_STEP_TYPE.get(node_name, "run"),
            )
            step.input = step_input
            step.output = step_output
            await step.send()

    final_state = (await _graph.aget_state(cfg)).values
    await cl.Message(content=_format_answer(final_state)).send()

    history.append((message.content, final_state["answer"]))
    cl.user_session.set("history", history[-3:])


# ── expose ASGI app for uvicorn ───────────────────────────────────────────────
from chainlit.server import app  # noqa: E402
