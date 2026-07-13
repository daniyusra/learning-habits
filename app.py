# Materials Research Agent — all tools from the notebooks in one place.
# Run: uv run uvicorn app:app --reload

import pysqlite3 as _pysqlite3  # chromadb needs this on some systems
import sys
sys.modules["sqlite3"] = _pysqlite3

import ast
import operator
import os
import uuid
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store

# ── LLM ───────────────────────────────────────────────────────────────────────

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0)

# ── Tool 1: safe calculator ───────────────────────────────────────────────────

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result.
    Supports + - * / ** % and parentheses, e.g. '23847 * 198 + 4471'."""
    try:
        return str(_safe_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Calculator error: {e}"

# ── Tool 2: web search ────────────────────────────────────────────────────────

web_search = TavilySearch(max_results=3)

# ── Tool 3: arXiv paper abstract ──────────────────────────────────────────────

@tool
def get_paper_abstract(title: str) -> str:
    """Look up an academic paper by its title on arXiv and return its abstract,
    authors, and year. Best for ML, physics, and materials-science papers."""
    try:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f'ti:"{title}"', "start": 0, "max_results": 1},
            timeout=20,
        )
        r.raise_for_status()
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = ET.fromstring(r.text).find("a:entry", ns)
        if entry is None:
            return f"No arXiv paper found matching '{title}'."
        found_title = entry.findtext("a:title", default="", namespaces=ns).strip()
        summary = entry.findtext("a:summary", default="", namespaces=ns).strip()
        year = entry.findtext("a:published", default="", namespaces=ns)[:4]
        authors = ", ".join(
            a.findtext("a:name", default="", namespaces=ns)
            for a in entry.findall("a:author", ns)[:3]
        )
        return f"{found_title} ({year}) — {authors}\n\n{summary}"
    except Exception as e:
        return f"arXiv error: {e}"

# ── Tool 4: current time ──────────────────────────────────────────────────────

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get today's date and current time. Call this FIRST whenever the user asks
    about 'recent', 'latest', or 'current' things. Pass an IANA timezone like
    'Asia/Tokyo' (defaults to UTC)."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"Unknown timezone '{timezone}'. Use an IANA name like 'Asia/Tokyo'."
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")

# ── Tools 5 & 6: long-term memory across sessions ────────────────────────────

_store = InMemoryStore(
    index={"embed": OpenAIEmbeddings(model="text-embedding-3-small"), "dims": 1536}
)

@tool
def remember_finding(note: str) -> str:
    """Save an important research finding to long-term memory so it can be
    recalled in future conversations."""
    get_store().put(("findings",), str(uuid.uuid4()), {"text": note})
    return "Saved to long-term memory."

@tool
def recall_findings(query: str) -> str:
    """Search long-term memory for previously saved findings relevant to the query.
    Call this at the start of a research task to reuse past knowledge."""
    hits = get_store().search(("findings",), query=query, limit=3)
    return "\n".join(f"- {h.value['text']}" for h in hits) or "No relevant findings."

# ── Tool 7: RAG over local PDF corpus ────────────────────────────────────────

_PDF_DIR = "papers"
_DB_DIR = "./chroma_db"
_COLLECTION = "materials_papers"

def _init_vectorstore() -> Chroma | None:
    if not os.path.isdir(_PDF_DIR):
        print(f"[RAG] No '{_PDF_DIR}/' directory — search_papers tool disabled.")
        return None
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = Chroma(collection_name=_COLLECTION, embedding_function=emb, persist_directory=_DB_DIR)
    if vs._collection.count() == 0:
        print("[RAG] Indexing PDFs (one-time cost)…")
        pages = PyPDFDirectoryLoader(_PDF_DIR).load()
        if not pages:
            print(f"[RAG] No PDFs found in {_PDF_DIR}/ — search_papers tool disabled.")
            return None
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True,
        ).split_documents(pages)
        vs.add_documents(chunks)
        print(f"[RAG] Indexed {len(chunks)} chunks.")
    else:
        print(f"[RAG] Loaded existing index: {vs._collection.count()} chunks.")
    return vs

_vectorstore = _init_vectorstore()

def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))} p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

@tool
def search_papers(query: str) -> str:
    """Search the local materials-science PDF collection for relevant passages.
    Use this BEFORE web_search for materials property prediction questions —
    these are peer-reviewed papers, not web pages."""
    if _vectorstore is None:
        return "Paper search unavailable: add PDFs to the 'papers/' directory first."
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 4})
    return _format_docs(retriever.invoke(query)) or "No relevant passages found."

# ── Agent ─────────────────────────────────────────────────────────────────────

_today = datetime.now(timezone.utc)
_tools = [calculator, get_paper_abstract, get_current_time,
          remember_finding, recall_findings, search_papers]
_system_prompt = (
    f"Today's date is {_today:%Y-%m-%d}. "
    "You are a materials-research assistant. "
    "Use calculator for arithmetic, web_search for current facts, "
    "get_paper_abstract for named arXiv papers, search_papers for the local PDF corpus, "
    "and remember_finding / recall_findings to persist insights across sessions."
)

_checkpointer = InMemorySaver()

agent = create_agent(
    model=llm,
    tools=_tools,
    system_prompt=_system_prompt,
    checkpointer=_checkpointer,
    store=_store,
)

# ── Chainlit UI ───────────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    await cl.Message(
        content=(
            "**Materials Research Assistant** ready.\n\n"
            "I can: calculate, search the web, look up arXiv papers, "
            "search local PDFs, and remember findings across turns."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    thread_id = cl.user_session.get("thread_id")
    cfg = {"configurable": {"thread_id": thread_id}}

    pending_steps: dict[str, cl.Step] = {}
    answer = cl.Message(content="")

    async for update in agent.astream(
        {"messages": [{"role": "user", "content": message.content}]},
        stream_mode="updates",
        config=cfg,
    ):
        for node_output in update.values():
            for msg in node_output.get("messages", []):
                if msg.type == "ai":
                    for tc in msg.tool_calls or []:
                        step = cl.Step(name=tc["name"], type="tool")
                        step.input = tc["args"]
                        await step.send()
                        pending_steps[tc["id"]] = step
                    if isinstance(msg.content, str) and msg.content.strip():
                        await answer.stream_token(msg.content)
                elif msg.type == "tool":
                    step = pending_steps.pop(msg.tool_call_id, None)
                    if step:
                        content = (
                            msg.content if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        step.output = content[:1000]
                        await step.update()

    await answer.send()


# ── expose ASGI app for uvicorn ───────────────────────────────────────────────
from chainlit.server import app  # noqa: E402
