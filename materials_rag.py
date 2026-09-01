"""materials_rag.py — RAG over a folder of materials-science PDFs.
 
Pipeline: load PDFs -> split into chunks -> embed -> ChromaDB -> retrieve -> grounded answer.
 
Setup:
    pip install -U langchain langchain-openai langchain-chroma langchain-community \
                   langchain-text-splitters pypdf
    export OPENAI_API_KEY=sk-...
 
Usage:
    mkdir papers && cp your_five_papers/*.pdf papers/
    python materials_rag.py                       # index (once) + demo queries
    python materials_rag.py --inspect "formation energy error"   # see raw chunks only
 
Note: langchain_community is being sunset but is still the home of
PyPDFDirectoryLoader. To drop it, read PDFs with pypdf directly and build
Document(page_content=..., metadata=...) yourself.
"""
import argparse
import os
import shutil
import sys

# chromadb needs this on systems with sqlite3 < 3.35 — must run before any chromadb
# import. Same fix already applied in agentic_rag.py / app.py; this file just lacked it.
import pysqlite3 as _pysqlite3
sys.modules["sqlite3"] = _pysqlite3

from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
 
PDF_DIR = "papers"
DB_DIR = "./chroma_db"
COLLECTION = "materials_papers"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-5.5")

# ── Catalog + section tagging ───────────────────────────────────────────────
# Proven out in rag-chunking-embedding-retrieval.ipynb as an in-memory experiment
# (throwaway chromadb.Client(), never written to the persisted store). Promoted
# here so the ACTUAL ./chroma_db carries it: paper stem, section_type, year,
# authors, method_type. Add a paper's stem here before reindexing, or its
# chunks fall back to "" for every catalog field.
PAPER_META = {
    "a-lab": {"year": 2023, "authors": "Szymanski et al.", "venue": "Nature",
              "method_type": "autonomous synthesis"},
    "chemcrow": {"year": 2023, "authors": "Bran et al.", "venue": "Nature Machine Intelligence",
                 "method_type": "LLM agent"},
    "mace-mp-0": {"year": 2024, "authors": "Batatia et al.", "venue": "arXiv",
                  "method_type": "foundation potential"},
    "omat-2024": {"year": 2026, "authors": "Barroso-Luque et al.", "venue": "arXiv",
                  "method_type": "dataset"},
    "orb3": {"year": 2025, "authors": "Rhodes, Vandenhaute et al.", "venue": "arXiv",
             "method_type": "foundation potential"},
    "universalmodelofatoms": {"year": 2025, "authors": "Wood et al.", "venue": "arXiv",
                               "method_type": "foundation potential"},
}
SECTIONS = ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]


def _section_spans(page_text: str) -> list[tuple[int, str]]:
    """(char_offset, section_type) for each header line in page_text, in reading order.

    Same keyword-header scan as the experiment notebook, but kept as OFFSETS instead
    of re-splitting the page — lets us tag chunks the splitter already produced rather
    than re-implementing chunking around section boundaries.
    """
    spans = [(0, "header")]
    offset = 0
    for line in page_text.split("\n"):
        if line.strip().lower() in SECTIONS:
            spans.append((offset, line.strip().lower()))
        offset += len(line) + 1  # +1 for the "\n" that .split("\n") consumed
    return spans


def _section_at(spans: list[tuple[int, str]], start_index: int) -> str:
    """Last section header at or before start_index — the section a chunk starts inside."""
    section = spans[0][1]
    for offset, name in spans:
        if offset > start_index:
            break
        section = name
    return section


def tag_metadata(chunks: list, pages: list) -> None:
    """Enrich chunk metadata IN PLACE with paper/section_type/year/authors/venue/method_type.

    Deliberately a post-hoc pass over chunks the splitter already made, not a rewrite of
    chunking: `start_index`/`page`/`source` (which agentic_rag.py hashes into chunk_id)
    are read, never overwritten, so this can't desync citations already in flight.
    """
    page_text_by_key = {(d.metadata["source"], d.metadata["page"]): d.page_content for d in pages}
    spans_cache: dict = {}
    for c in chunks:
        key = (c.metadata["source"], c.metadata["page"])
        if key not in spans_cache:
            spans_cache[key] = _section_spans(page_text_by_key[key])
        stem = os.path.basename(c.metadata["source"]).replace(".pdf", "")
        c.metadata["paper"] = stem
        c.metadata["section_type"] = _section_at(spans_cache[key], c.metadata.get("start_index", 0))
        c.metadata.update(PAPER_META.get(stem, {}))


# ── Steps 1+2: load PDFs, split into chunks ────────────────────────────────
def build_chunks():
    """One Document per PDF page -> overlapping character chunks -> tagged metadata.

    chunk_size ~1000 chars (~250 tokens) holds one idea; chunk_overlap=200
    keeps facts that straddle a boundary intact. add_start_index records the
    character offset so you can trace a chunk back into the page (and so
    tag_metadata can locate which section that offset falls in).
    """
    pages = PyPDFDirectoryLoader(PDF_DIR).load()
    if not pages:
        raise SystemExit(f"No PDFs found in ./{PDF_DIR}/ — add 5 papers first.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    ).split_documents(pages)
    tag_metadata(chunks, pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")
    return chunks
 
 
# ── Steps 3+4: embed and store (or load an existing index) ─────────────────
def get_vectorstore(reindex: bool = False) -> Chroma:
    """Embedding costs money, so index once and reload thereafter.

    Watch the kwarg asymmetry: the constructor takes `embedding_function=`,
    the from_documents classmethod takes `embedding=`.

    RECALCULATING METADATA: chunk metadata (paper/section_type/year/authors/
    method_type) is baked in at ingest time by tag_metadata(), inside
    build_chunks(). An existing persisted index does NOT pick up changes to
    PAPER_META, the section scanner, or chunk_size on its own — you have to
    rebuild it. Two ways to trigger that:
      - `python materials_rag.py --reindex`
      - `get_vectorstore(reindex=True)` from a notebook/REPL
    Both drop ./chroma_db and re-embed the whole corpus (~1000 chunks,
    text-embedding-3-small ~ a few cents) before writing it back.
    """
    if reindex and os.path.isdir(DB_DIR):
        print(f"--reindex: dropping {DB_DIR} to rebuild with current chunking + metadata")
        shutil.rmtree(DB_DIR)

    if os.path.isdir(DB_DIR):
        print(f"Loading existing index from {DB_DIR}")
        return Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=DB_DIR,
        )
    print("Indexing PDFs (one-time cost)...")
    vs = Chroma.from_documents(
        documents=build_chunks(),
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
    )
    print(f"Indexed {vs._collection.count()} chunks into {DB_DIR}")
    return vs


vectorstore = get_vectorstore(reindex="--reindex" in sys.argv)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
 
 
# ── Step 5a: inspect retrieval on its own, BEFORE trusting any answer ──────
def inspect(question: str, k: int = 4) -> None:
    """If the wrong chunks come back, no prompt engineering will fix the answer.

    Chroma's default score is L2 DISTANCE: lower = closer. Not similarity.
    """
    print(f"\n=== RETRIEVED for: {question}\n")
    for doc, score in vectorstore.similarity_search_with_score(question, k=k):
        m = doc.metadata
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  dist={score:.3f}  {m.get('paper', '?')} p.{m.get('page', '?')} "
              f"[{m.get('section_type', '?')}, {m.get('year', '?')}]\n    {snippet}…\n")
 
 
# ── Step 5b: format chunks into the prompt, generate a grounded answer ─────
prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Cite the source file and page for each claim.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
 
 
def format_docs(docs) -> str:
    """Inject citation + catalog metadata INTO the context — the model can only cite what it sees."""
    lines = []
    for d in docs:
        m = d.metadata
        tag = (f"[{m.get('paper', '?')} p.{m.get('page', '?')} "
               f"| section={m.get('section_type', '?')} | year={m.get('year', '?')} "
               f"| authors={m.get('authors', '?')} | method_type={m.get('method_type', '?')}]")
        lines.append(f"{tag}\n{d.page_content}")
    return "\n\n".join(lines)
 
 
def answer(question: str) -> str:
    docs = retriever.invoke(question)
    if not docs:
        return "No relevant passages found in the corpus."
    msg = prompt.format(context=format_docs(docs), question=question)
    return llm.invoke(msg).content
 
 
# ── Step 6: expose RAG as an agent tool (fold back into Part 1's agent) ────
@tool
def search_papers(query: str) -> str:
    """Search the local materials-science paper collection for passages relevant
    to the query. Use this BEFORE web search for questions about materials
    property prediction — these are peer-reviewed papers, not web pages."""
    docs = retriever.invoke(query)
    return format_docs(docs) or "No relevant passages found."
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", metavar="QUERY",
                    help="show retrieved chunks only, no LLM call")
    ap.add_argument("--reindex", action="store_true",
                    help="drop and rebuild ./chroma_db with current chunking + metadata "
                         "(re-embeds everything — see get_vectorstore docstring)")
    args = ap.parse_args()
 
    if args.inspect:
        inspect(args.inspect)
    else:
        for q in [
            "What is the formation energy prediction error?",   # should be in corpus
            "What is the melting point of tungsten?",           # should NOT be — expect "I don't know"
        ]:
            inspect(q)                       # always look at the chunks first
            print(f"✅ ANSWER: {answer(q)}\n" + "-" * 70)