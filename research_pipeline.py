"""research_pipeline.py — Two-agent research pipeline (LangGraph, supervisor pattern).

Sits ON TOP of the existing knowledge layer (materials_rag.py's persisted Chroma
store + search_papers tool). Implements NO retrieval, embedding, or chunking here —
only imports it.

    question -> [Agent 1: literature] -> [route] -> [Agent 2: synthesis] -> [human review] -> END
                                            |
                                            +-> [no relevant papers] -> END

Why a plain StateGraph is "the supervisor" here, not a third LLM:
This is a fixed, dependent, two-step pipeline (find papers THEN hypothesize) — not
open-ended delegation between peers (that's the swarm pattern, and it's the wrong
shape for a dependent pipeline; see Cognition's "Don't Build Multi-Agents"). The
routing decision after Agent 1 (`route_after_literature`) is a deterministic check
on shared state, not a judgment call an LLM needs to make. Spending a third agent's
worth of latency/cost/failure-surface on a decision that's just "is this list empty"
would be the third-agent trap this task explicitly asks to avoid — so the graph's
conditional edges ARE the supervisor.

Handoff = shared state, not message passing:
Agent 2 never sees Agent 1's chat transcript (tool calls, intermediate reasoning,
retries). It receives the FULL validated LiteratureFindings — every field the
Pydantic schema demands — rendered into a fresh, purpose-built prompt. This is a
deliberate guard against Cemri et al. 2025 (arXiv 2503.13657) failure mode
"incomplete context propagation across the handoff": summarizing or truncating
Agent 1's output before Agent 2 sees it is exactly how that failure mode shows up
in practice, so the schema — not a prose summary — is the interface.

Termination:
`route_after_literature` is the explicit termination criterion guarding against
Cemri et al.'s "absent termination criteria" failure mode: if the literature agent
found nothing, the graph ends at `end_no_papers` and says why, instead of handing
Agent 2 an empty context and letting it hallucinate hypotheses from nothing. Every
path through the graph reaches END in a bounded number of steps — no open-ended
loop.

Human-in-the-loop gate (`human_review_node`):
Between synthesis and END sits a real `interrupt()` — not an automatic check, a
human one. It's the mitigation for the very gap flagged below (no automatic
grounding check on Agent 2's output): a person reads all 3 hypotheses, verbatim,
before anything is treated as final, and can approve/edit/reject each one. The
graph pauses at that point (checkpointer-backed — see build_graph) until a caller
resumes it with `Command(resume=decision)` on the SAME thread_id used for the
run that paused. Resuming a DIFFERENT thread_id, or one whose last step was a
half-finished tool call, is how checkpointer state gets corrupted — always
resume the exact thread_id the interrupt paused on. `decision` is
`{"approved_hypotheses": [...], "status": "<free text>"}`; see
`apply_review_decision` for the approve/edit/reject text convention both the
CLI (`__main__`, below) and app.py's Chainlit UI parse replies with.

What this design does NOT yet guard against (flagged, not solved):
- The human review gate is a manual check, not an automatic one — nothing stops
  a reviewer from rubber-stamping "all" without actually reading the hypotheses.
  There is still no automatic faithfulness/entailment pass (unlike
  agentic_rag.py's verify step) for whatever the human approves.
- No hard cap on Agent 1's tool-calling loop beyond LangGraph's default
  recursion_limit (25) on that agent's own internal graph — bounded by the model's
  own judgment via the system prompt, not enforced in code.
- No retry/escalation policy if `response_format` validation fails repeatedly.
"""
from __future__ import annotations

import argparse
import uuid
from typing import Literal, TypedDict

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

# The ONLY retrieval import in this file — three weeks of RAG work, called as a tool,
# never rebuilt. search_papers already tags each hit with the enriched metadata
# (paper/section_type/year/authors/method_type) added in materials_rag.py.
from materials_rag import search_papers

CHAT_MODEL = "gpt-5.5"
llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)


# ── Typed contracts — what each agent MUST return ───────────────────────────
class PaperSummary(BaseModel):
    paper: str = Field(description="Paper stem, e.g. 'mace-mp-0' — read from the retrieved chunk's metadata tag.")
    key_claim: str = Field(description="The paper's main claim relevant to the research question.")
    method: str = Field(description="The method/approach the paper uses.")
    relevant_finding: str = Field(description="The specific result, number, or finding relevant to the question.")
    source_citation: str = Field(description="Paper + page (from the metadata tag) this summary is grounded in.")


class LiteratureFindings(BaseModel):
    summaries: list[PaperSummary] = Field(
        default_factory=list,
        description="One entry per distinct paper with retrieved evidence. Empty if nothing relevant was found — "
                     "do not invent a summary for a paper you didn't actually retrieve.",
    )


class Hypothesis(BaseModel):
    hypothesis: str = Field(description="A concrete, testable experiment hypothesis.")
    variables: list[str] = Field(description="Independent/dependent variables the experiment would manipulate or measure.")
    expected_outcome: str = Field(description="What result would support the hypothesis.")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="SELF-REPORTED by the model — its own rough prior, not a calibrated probability. "
                     "Do not present this as measured or validated.",
    )


class SynthesisOutput(BaseModel):
    # min_length/max_length=3 makes "exactly 3" a structural constraint, not just a prompt
    # instruction — create_agent's structured-output strategy re-prompts on validation failure.
    hypotheses: list[Hypothesis] = Field(min_length=3, max_length=3)


# ── Shared graph state — the handoff mechanism ──────────────────────────────
class PipelineState(TypedDict, total=False):
    question: str
    literature_findings: dict     # LiteratureFindings.model_dump()
    hypotheses: list[dict]        # [Hypothesis.model_dump(), ...]
    status: str                   # how/why the run ended — always set on a terminal node


# ── Agent 1 — Literature agent ───────────────────────────────────────────────
_LITERATURE_SYSTEM_PROMPT = (
    "You are the literature-review agent in a two-agent materials-science research "
    "pipeline. Use the search_papers tool to find passages relevant to the research "
    "question — call it more than once with different phrasings if the first query "
    "doesn't surface enough, but stop once you've covered the relevant papers (don't "
    "search indefinitely). Each retrieved passage is tagged with its paper, section, "
    "year, authors, and method_type — use those tags, don't guess them.\n\n"
    "Produce exactly one structured summary per DISTINCT paper you found solid "
    "evidence for. Never summarize a paper you didn't actually retrieve a passage "
    "from. If nothing relevant turns up, return an empty summaries list rather than "
    "forcing an answer."
)

literature_agent = create_agent(
    model=llm,
    tools=[search_papers],
    system_prompt=_LITERATURE_SYSTEM_PROMPT,
    response_format=LiteratureFindings,
)


def literature_node(state: PipelineState) -> dict:
    result = literature_agent.invoke({
        "messages": [{"role": "user", "content": f"Research question: {state['question']}"}]
    })
    findings: LiteratureFindings = result["structured_response"]
    return {"literature_findings": findings.model_dump()}


def route_after_literature(state: PipelineState) -> Literal["synthesis", "end_no_papers"]:
    """The supervisor's one routing decision — deterministic, not an LLM's judgment call."""
    if not state["literature_findings"]["summaries"]:
        return "end_no_papers"
    return "synthesis"


def end_no_papers_node(state: PipelineState) -> dict:
    return {"hypotheses": [], "status": "terminated: literature agent found no relevant papers"}


# ── Agent 2 — Synthesis agent ────────────────────────────────────────────────
_SYNTHESIS_SYSTEM_PROMPT = (
    "You are the synthesis agent in a two-agent materials-science research pipeline. "
    "You do not have tools and cannot search — you only see the literature findings "
    "handed to you below. Ground every hypothesis in those findings; do not introduce "
    "papers, numbers, or claims that aren't in them.\n\n"
    "Propose exactly 3 concrete, testable experiment hypotheses. For each: the "
    "hypothesis itself, the variables an experiment would manipulate/measure, the "
    "expected outcome, and a confidence score. confidence is YOUR OWN self-reported "
    "estimate, not a calibrated probability — treat it as a rough prior."
)

synthesis_agent = create_agent(
    model=llm,
    tools=[],
    system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
    response_format=SynthesisOutput,
)


def _format_findings_for_handoff(findings: dict) -> str:
    """Render Agent 1's FULL structured output into Agent 2's prompt — every field
    the schema captured, not a shortened re-summary. See module docstring: this is
    the guard against lossy handoff, not decoration."""
    return "\n\n".join(
        f"- [{s['paper']}] {s['key_claim']}\n"
        f"    method: {s['method']}\n"
        f"    finding: {s['relevant_finding']}\n"
        f"    source: {s['source_citation']}"
        for s in findings["summaries"]
    )


def synthesis_node(state: PipelineState) -> dict:
    context = _format_findings_for_handoff(state["literature_findings"])
    result = synthesis_agent.invoke({
        "messages": [{"role": "user", "content": (
            f"Research question: {state['question']}\n\n"
            f"Literature findings gathered by the literature agent:\n{context}\n\n"
            "Propose exactly 3 experiment hypotheses grounded in these findings."
        )}]
    })
    out: SynthesisOutput = result["structured_response"]
    return {"hypotheses": [h.model_dump() for h in out.hypotheses], "status": "complete"}


# ── Human-in-the-loop review gate ────────────────────────────────────────────
def human_review_node(state: PipelineState) -> dict:
    """Pauses the graph for a human to approve, edit, or reject each of Agent 2's
    3 hypotheses. See the module docstring's "Human-in-the-loop gate" section for
    the interrupt/resume contract and the thread_id gotcha.

    The interrupt payload hands the reviewer state["hypotheses"] VERBATIM — same
    "full validated data, not a summary" rule as the literature->synthesis
    handoff. A caller resumes with Command(resume=decision) where decision is
    {"approved_hypotheses": [...], "status": "<free text>"}; approved_hypotheses
    becomes the pipeline's final hypotheses list, and status overwrites
    synthesis_node's "complete" with whatever the reviewer/caller reports (e.g.
    "reviewed", "review timed out — treated as reject").
    """
    decision = interrupt({
        "hypotheses": state["hypotheses"],
        "action": "approve, edit, or reject each hypothesis",
    })
    return {
        "hypotheses": decision.get("approved_hypotheses", state["hypotheses"]),
        "status": decision.get("status", "reviewed"),
    }


def apply_review_decision(hypotheses: list[dict], reply: str) -> list[dict]:
    """Parse a human reviewer's free-text reply into the approved/edited
    hypothesis list — the shared convention the CLI (__main__, below) and
    app.py's Chainlit UI both use to build the `approved_hypotheses` passed to
    Command(resume=...).

    Syntax: 'all' keeps every hypothesis unchanged; 'none' (or an empty reply)
    rejects everything; otherwise a comma-separated list of entries, each either
    a bare 1-based index to approve as-is ('2') or an index with replacement
    text to edit ('2: revised hypothesis text'). Any hypothesis whose index
    isn't mentioned is dropped (rejected).
    """
    reply = reply.strip()
    if not reply or reply.lower() == "none":
        return []
    if reply.lower() == "all":
        return list(hypotheses)

    approved = []
    for entry in reply.split(","):
        entry = entry.strip()
        if not entry:
            continue
        idx_part, _, edit_part = entry.partition(":")
        try:
            idx = int(idx_part.strip()) - 1
        except ValueError:
            continue
        if not (0 <= idx < len(hypotheses)):
            continue
        h = dict(hypotheses[idx])
        edit_text = edit_part.strip()
        if edit_text:
            h["hypothesis"] = edit_text
        approved.append(h)
    return approved


# ── Graph assembly — the supervisor ─────────────────────────────────────────
def build_graph(checkpointer=None):
    """Two agents, one deterministic router, one explicit early-exit, one real
    human-in-the-loop gate. Every path reaches END — human_review pauses via
    interrupt() but always resumes onto END, never loops."""
    graph = StateGraph(PipelineState)
    graph.add_node("literature", literature_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("end_no_papers", end_no_papers_node)
    graph.add_node("human_review", human_review_node)

    graph.add_edge(START, "literature")
    graph.add_conditional_edges(
        "literature", route_after_literature,
        {"synthesis": "synthesis", "end_no_papers": "end_no_papers"},
    )
    graph.add_edge("synthesis", "human_review")
    graph.add_edge("human_review", END)
    graph.add_edge("end_no_papers", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# ── Driver ───────────────────────────────────────────────────────────────────
def run_pipeline(question: str, graph=None, thread_id: str | None = None) -> PipelineState:
    """Mints a fresh thread_id per question by default — same gotcha as
    agentic_rag.py: reusing a thread_id across unrelated questions lets stale
    literature_findings/hypotheses from a PREVIOUS question leak into this run's
    state, since plain (non-Annotated) TypedDict fields persist across .invoke()
    calls on the same thread. Only pass thread_id explicitly for a genuine
    follow-up on the same question thread.

    NOTE: if the graph pauses at human_review, the returned dict is state-so-far
    with an extra "__interrupt__" key, not a finished run — this function does
    not resume it (it doesn't expose the thread_id needed to). For an
    interactive run that can actually resume, see __main__ below or app.py's
    Chainlit UI, both of which keep the thread_id around after the pause."""
    graph = graph or build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    return graph.invoke({"question": question}, config=config)


def _print_result(final_state: PipelineState) -> None:
    print(f"\nSTATUS: {final_state.get('status')}\n")
    print(f"Papers found: {[s['paper'] for s in final_state.get('literature_findings', {}).get('summaries', [])]}\n")
    for i, h in enumerate(final_state.get("hypotheses", []), 1):
        print(f"{i}. {h['hypothesis']}")
        print(f"   variables: {h['variables']}")
        print(f"   expected outcome: {h['expected_outcome']}")
        print(f"   confidence (self-reported): {h['confidence']}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?", default=(
        "What formation-energy prediction error do current foundation potentials "
        "(MACE-MP-0, Orb-v3, UMA) report, and where might a new benchmark or "
        "architecture change close the remaining gap?"
    ))
    args = ap.parse_args()

    # Built directly (not via run_pipeline) because resuming the interrupt below
    # needs the SAME thread_id the initial run paused on.
    _graph = build_graph()
    _config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    _state = _graph.invoke({"question": args.question}, config=_config)

    if "__interrupt__" in _state:
        payload = _state["__interrupt__"][0].value
        hyps = payload["hypotheses"]
        print(f"\nPapers found: {[s['paper'] for s in _state['literature_findings']['summaries']]}\n")
        print("Proposed hypotheses (pending your review):\n")
        for i, h in enumerate(hyps, 1):
            print(f"{i}. {h['hypothesis']}  (confidence: {h['confidence']:.2f})")
        reply = input(
            "\nApprove which? Comma-separated numbers, 'N: replacement text' to "
            "edit one, 'all', or 'none': "
        )
        approved = apply_review_decision(hyps, reply)
        _state = _graph.invoke(
            Command(resume={"approved_hypotheses": approved, "status": "reviewed via CLI"}),
            config=_config,
        )

    _print_result(_state)
