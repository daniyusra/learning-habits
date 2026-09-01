"""research_pipeline.py — Two-agent research pipeline (LangGraph, supervisor pattern).

Sits ON TOP of the existing knowledge layer (materials_rag.py's persisted Chroma
store + search_papers tool). Implements NO retrieval, embedding, or chunking here —
only imports it.

    question -> [Agent 1: literature] -> [route] -> [Agent 2: synthesis] -> [human review STUB] -> END
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

What this design does NOT yet guard against (flagged, not solved):
- No automatic check that Agent 2's hypotheses are actually grounded in Agent 1's
  findings (no faithfulness/entailment pass, unlike agentic_rag.py's verify step).
  A hallucinated hypothesis that cites real papers by name would pass silently.
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


# ── Wednesday seam — STUB ONLY, do not wire up today ─────────────────────────
def human_review_stub_node(state: PipelineState) -> dict:
    """Placeholder for Wednesday's human-in-the-loop gate. The edge (synthesis ->
    human_review_stub -> END) already exists so tomorrow's change is additive —
    replacing this no-op body, not restructuring the graph.

    Wednesday's real version, sketched (NOT executed today):

        from langgraph.types import interrupt

        def human_review_node(state: PipelineState) -> dict:
            decision = interrupt({
                "hypotheses": state["hypotheses"],
                "action": "approve, edit, or reject each hypothesis",
            })
            # graph pauses HERE; resumes via:
            #   graph.invoke(Command(resume=decision), config=thread_config)
            return {"hypotheses": decision.get("approved_hypotheses", state["hypotheses"])}

    interrupt() needs a checkpointer to survive the pause (already wired below via
    build_graph's checkpointer=) and the SAME thread_id on resume — that's the
    dangling-tool-call-corruption gotcha: resuming a DIFFERENT thread_id, or one
    whose last step was a half-finished tool call, is how checkpointer state gets
    corrupted. Always resume the exact thread_id the interrupt paused on.
    """
    return {}


# ── Graph assembly — the supervisor ─────────────────────────────────────────
def build_graph(checkpointer=None):
    """Two agents, one deterministic router, one explicit early-exit, one stubbed
    human gate. Every path reaches END."""
    graph = StateGraph(PipelineState)
    graph.add_node("literature", literature_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("end_no_papers", end_no_papers_node)
    graph.add_node("human_review_stub", human_review_stub_node)

    graph.add_edge(START, "literature")
    graph.add_conditional_edges(
        "literature", route_after_literature,
        {"synthesis": "synthesis", "end_no_papers": "end_no_papers"},
    )
    graph.add_edge("synthesis", "human_review_stub")
    graph.add_edge("human_review_stub", END)
    graph.add_edge("end_no_papers", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


# ── Driver ───────────────────────────────────────────────────────────────────
def run_pipeline(question: str, graph=None, thread_id: str | None = None) -> PipelineState:
    """Mints a fresh thread_id per question by default — same gotcha as
    agentic_rag.py: reusing a thread_id across unrelated questions lets stale
    literature_findings/hypotheses from a PREVIOUS question leak into this run's
    state, since plain (non-Annotated) TypedDict fields persist across .invoke()
    calls on the same thread. Only pass thread_id explicitly for a genuine
    follow-up on the same question thread."""
    graph = graph or build_graph()
    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    return graph.invoke({"question": question}, config=config)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="?", default=(
        "What formation-energy prediction error do current foundation potentials "
        "(MACE-MP-0, Orb-v3, UMA) report, and where might a new benchmark or "
        "architecture change close the remaining gap?"
    ))
    args = ap.parse_args()

    final_state = run_pipeline(args.question)
    print(f"\nSTATUS: {final_state.get('status')}\n")
    print(f"Papers found: {[s['paper'] for s in final_state.get('literature_findings', {}).get('summaries', [])]}\n")
    for i, h in enumerate(final_state.get("hypotheses", []), 1):
        print(f"{i}. {h['hypothesis']}")
        print(f"   variables: {h['variables']}")
        print(f"   expected outcome: {h['expected_outcome']}")
        print(f"   confidence (self-reported): {h['confidence']}\n")
