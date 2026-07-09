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
 
 
# ── Steps 1+2: load PDFs, split into chunks ────────────────────────────────
def build_chunks():
    """One Document per PDF page -> overlapping character chunks.
 
    chunk_size ~1000 chars (~250 tokens) holds one idea; chunk_overlap=200
    keeps facts that straddle a boundary intact. add_start_index records the
    character offset so you can trace a chunk back into the page.
    """
    pages = PyPDFDirectoryLoader(PDF_DIR).load()
    if not pages:
        raise SystemExit(f"No PDFs found in ./{PDF_DIR}/ — add 5 papers first.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    ).split_documents(pages)
    print(f"Loaded {len(pages)} pages -> {len(chunks)} chunks")
    return chunks
 
 
# ── Steps 3+4: embed and store (or load an existing index) ─────────────────
def get_vectorstore() -> Chroma:
    """Embedding costs money, so index once and reload thereafter.
 
    Watch the kwarg asymmetry: the constructor takes `embedding_function=`,
    the from_documents classmethod takes `embedding=`.
    """
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
 
 
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
 
 
# ── Step 5a: inspect retrieval on its own, BEFORE trusting any answer ──────
def inspect(question: str, k: int = 4) -> None:
    """If the wrong chunks come back, no prompt engineering will fix the answer.
 
    Chroma's default score is L2 DISTANCE: lower = closer. Not similarity.
    """
    print(f"\n=== RETRIEVED for: {question}\n")
    for doc, score in vectorstore.similarity_search_with_score(question, k=k):
        src = os.path.basename(doc.metadata.get("source", "?"))
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content[:120].replace("\n", " ")
        print(f"  dist={score:.3f}  {src} p.{page}\n    {snippet}…\n")
 
 
# ── Step 5b: format chunks into the prompt, generate a grounded answer ─────
prompt = ChatPromptTemplate.from_template(
    "Answer the question using ONLY the context below. "
    "If the context does not contain the answer, say you don't know. "
    "Cite the source file and page for each claim.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)
 
 
def format_docs(docs) -> str:
    """Inject source+page INTO the context — the model can only cite what it sees."""
    return "\n\n".join(
        f"[{os.path.basename(d.metadata.get('source', '?'))} "
        f"p.{d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )
 
 
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