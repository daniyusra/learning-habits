# Materials-Science Paper Chat — agentic RAG with claim-level citations, PLUS
# the two-agent research pipeline (research_pipeline.py), served from the same
# Chainlit app as a second chat profile. Paper Chat streams every node in the
# retrieve -> grade -> gap-check -> generate -> verify loop (agentic_rag.py) as
# an inspectable Step, with every claim in the final answer showing its verdict
# from the faithfulness check. Research Pipeline streams the literature ->
# route -> synthesis handoff (research_pipeline.py) the same way.
# Run: uv run uvicorn app:app --reload

import pysqlite3 as _pysqlite3  # chromadb needs this on some systems
import sys
sys.modules["sqlite3"] = _pysqlite3

import uuid

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl

import agentic_rag
import research_pipeline

# ── Build each graph once; every chat turn gets its own thread_id (see the
# checkpointer notes in agentic_rag.build_graph / research_pipeline.run_pipeline)
# so loop state from one question never bleeds into the next. ─────────────────
_graph = agentic_rag.build_graph()
_pipeline_graph = research_pipeline.build_graph()

_PAPER_CHAT_PROFILE = "Paper Chat"
_RESEARCH_PIPELINE_PROFILE = "Research Pipeline"

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


# ── Research Pipeline: per-node trace formatting ───────────────────────────────
_PIPELINE_STEP_TYPE = {
    "literature": "tool",
    "synthesis": "llm",
    "end_no_papers": "run",
    "human_review_stub": "run",
}
_PIPELINE_STEP_LABEL = {
    "literature": "Agent 1 — literature search",
    "synthesis": "Agent 2 — synthesize hypotheses",
    "end_no_papers": "no relevant papers — stopping",
    "human_review_stub": "human review (stub — no-op today)",
}


def _describe_pipeline(node_name: str, out: dict) -> tuple[str, str]:
    if node_name == "literature":
        summaries = out["literature_findings"]["summaries"]
        if not summaries:
            return "search the corpus for grounded findings", "no relevant papers found"
        lines = [
            f"  [{s['paper']}] {s['key_claim']}\n"
            f"      method: {s['method']}\n"
            f"      finding: {s['relevant_finding']}\n"
            f"      source: {s['source_citation']}"
            for s in summaries
        ]
        return "search the corpus for grounded findings", f"{len(summaries)} paper(s) found:\n" + "\n".join(lines)

    if node_name == "synthesis":
        lines = [
            f"  {i}. {h['hypothesis']}  (confidence: {h['confidence']:.2f})"
            for i, h in enumerate(out["hypotheses"], 1)
        ]
        return "propose 3 hypotheses grounded ONLY in Agent 1's findings", "\n".join(lines)

    if node_name == "end_no_papers":
        return "route: is there anything to synthesize from?", out["status"]

    if node_name == "human_review_stub":
        return "", "stub — no-op; Wednesday's human-in-the-loop gate slots in here"

    return "", str(out)


def _format_pipeline_answer(state: dict) -> str:
    status = state.get("status", "")
    findings = state.get("literature_findings", {}).get("summaries", [])
    hypotheses = state.get("hypotheses", [])

    if not findings:
        return (
            "**No relevant papers found in the corpus for this question** — the "
            "literature agent came back empty, so the pipeline stopped before "
            "handing anything to the synthesis agent (no hypotheses invented from "
            "nothing).\n\n"
            f"_status: {status}_"
        )

    lines = ["**Literature findings** (Agent 1)"]
    for s in findings:
        lines.append(
            f"- **[{s['paper']}]** {s['key_claim']}\n"
            f"  method: {s['method']} · finding: {s['relevant_finding']} · source: {s['source_citation']}"
        )

    lines += ["", "**Proposed hypotheses** (Agent 2 — grounded only in the findings above)"]
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"{i}. **{h['hypothesis']}**")
        lines.append(f"   - variables: {', '.join(h['variables'])}")
        lines.append(f"   - expected outcome: {h['expected_outcome']}")
        lines.append(f"   - confidence (self-reported, not calibrated): {h['confidence']:.2f}")

    return "\n".join(lines)


def _contextualize(history: list[tuple[str, str]], new_message: str) -> str:
    """Fold recent turns into the question so follow-ups ('what about the other
    paper?') retrieve sensibly — the graph itself is stateless per question."""
    if not history:
        return new_message
    ctx = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
    return f"Prior conversation:\n{ctx}\n\nNew question: {new_message}"


# ── Chainlit UI ────────────────────────────────────────────────────────────────
# Two chat profiles share one app: pick at chat start, dispatched on every turn.
@cl.set_chat_profiles
async def chat_profiles():
    return [
        cl.ChatProfile(
            name=_PAPER_CHAT_PROFILE,
            markdown_description=(
                "Ask questions naturally, like NotebookLM. Retrieves, grades, and "
                "re-retrieves as needed, then answers with claim-level citations "
                "checked against their source chunks."
            ),
            default=True,
        ),
        cl.ChatProfile(
            name=_RESEARCH_PIPELINE_PROFILE,
            markdown_description=(
                "Give it a research question. Agent 1 searches the corpus for "
                "grounded literature findings, then Agent 2 proposes 3 testable "
                "experiment hypotheses from those findings alone."
            ),
        ),
    ]


def _loaded_papers() -> list[str]:
    data = agentic_rag.vectorstore.get(include=["metadatas"])
    return sorted({md.get("source", "?").split("/")[-1].replace(".pdf", "") for md in data["metadatas"]})


@cl.on_chat_start
async def on_chat_start():
    profile = cl.user_session.get("chat_profile")
    papers = _loaded_papers()

    if profile == _RESEARCH_PIPELINE_PROFILE:
        await cl.Message(
            content=(
                "**Two-Agent Research Pipeline**\n\n"
                "Ask a research question. Agent 1 searches the corpus and returns "
                "grounded literature findings; Agent 2 proposes 3 testable "
                "experiment hypotheses from those findings — nothing else. If "
                "Agent 1 finds nothing relevant, the pipeline stops there instead "
                "of letting Agent 2 hallucinate.\n\n"
                f"Loaded papers: {', '.join(papers)}"
            )
        ).send()
        return

    cl.user_session.set("history", [])
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
    profile = cl.user_session.get("chat_profile")
    if profile == _RESEARCH_PIPELINE_PROFILE:
        await _run_research_pipeline(message)
    else:
        await _run_paper_chat(message)


async def _run_paper_chat(message: cl.Message):
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


async def _run_research_pipeline(message: cl.Message):
    # Fresh thread_id per question by default (see research_pipeline.run_pipeline) —
    # this is a one-shot pipeline per question, not a running conversation.
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
    initial_state = {"question": message.content}

    async for update in _pipeline_graph.astream(initial_state, stream_mode="updates", config=cfg):
        for node_name, node_output in update.items():
            step_input, step_output = _describe_pipeline(node_name, node_output)
            step = cl.Step(
                name=_PIPELINE_STEP_LABEL.get(node_name, node_name),
                type=_PIPELINE_STEP_TYPE.get(node_name, "run"),
            )
            step.input = step_input
            step.output = step_output
            await step.send()

    final_state = (await _pipeline_graph.aget_state(cfg)).values
    await cl.Message(content=_format_pipeline_answer(final_state)).send()


# ── expose ASGI app for uvicorn ───────────────────────────────────────────────
from chainlit.server import app  # noqa: E402
