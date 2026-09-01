# AI Learning Environment

Python 3.11 environment managed by [uv](https://docs.astral.sh/uv/) for studying AI with Jupyter notebooks.

---

## Prerequisites

- `uv` must be installed. Check: `uv --version`
- If missing: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## Setup (first time)

```bash
# From this directory, create the venv and install all packages
uv sync

# Register the Jupyter kernel (only needed once)
uv run python -m ipykernel install --user --name learning-habits --display-name "Python 3.11 (learning-habits)"
```

---

## Daily usage

```bash
# Start JupyterLab
uv run jupyter lab

# Or classic Notebook
uv run jupyter notebook
```

When creating a notebook, select the kernel **"Python 3.11 (learning-habits)"** from the kernel picker.

---

## Adding packages

```bash
# Add a package (updates pyproject.toml and uv.lock automatically)
uv add <package-name>

# Examples
uv add torch
uv add transformers
uv add openai
uv add anthropic
```

No need to activate the venv manually — `uv run` handles it, and Jupyter uses the registered kernel.

---

## Installed packages

| Package | Purpose |
|---|---|
| `numpy` | Numerical computing, arrays |
| `pandas` | Data manipulation |
| `matplotlib` | Plotting and visualization |
| `scikit-learn` | Classical ML algorithms |
| `jupyterlab` | Notebook environment (primary) |
| `notebook` | Classic notebook UI |
| `ipykernel` | Jupyter kernel integration |

---

## Recalculating chunk metadata

`./chroma_db` (collection `materials_papers`) carries per-chunk metadata beyond
the raw PDF fields: `paper` (filename stem), `section_type`, `year`, `authors`,
`method_type` (see `PAPER_META` and `tag_metadata()` in `materials_rag.py`).
That metadata is baked in at **ingest time** — an already-persisted index does
NOT pick up catalog or chunking changes on its own.

Rebuild it whenever you:
- add/remove a PDF in `papers/`
- add or edit an entry in `PAPER_META` (new paper → add its stem here first,
  otherwise its chunks get `""` for every catalog field)
- change `chunk_size`/`chunk_overlap` in `build_chunks()`

```bash
uv run python materials_rag.py --reindex
```

This drops `./chroma_db` and re-embeds the whole corpus (~1,000 chunks,
`text-embedding-3-small`, a few cents) with the current chunking + tagging
logic. Sanity-check the result with:

```bash
uv run python materials_rag.py --inspect "formation energy prediction error"
```

and confirm the printed hits show real `paper`/`section_type`/`year` values,
not `?`. `section_type` is a per-page keyword scan (looks for a line that's
exactly "methods", "results", etc. on that page) — it defaults to `"header"`
on pages where no such line appears, which is expected for many pages, not a
bug to chase.

---

## Running the chat UI

`app.py` serves both agentic pipelines through one Chainlit app, as two chat
profiles you pick from the picker when a chat starts:

- **Paper Chat** (default) — `agentic_rag.py`'s retrieve -> grade -> gap-check
  -> generate -> verify loop. Every node streams as an inspectable Step, and
  the final answer shows a verdict for every claim, checked against its cited
  chunk.
- **Research Pipeline** — `research_pipeline.py`'s two-agent supervisor
  pipeline. Agent 1 searches the corpus for grounded literature findings;
  Agent 2 proposes exactly 3 testable hypotheses from those findings alone. If
  Agent 1 finds nothing relevant, the pipeline stops there instead of handing
  Agent 2 an empty context.

Both profiles share the same persisted `./chroma_db` index — see
[Recalculating chunk metadata](#recalculating-chunk-metadata) if it's stale.

```bash
uv run chainlit run app.py -w
```

`-w` enables auto-reload on file changes. Open the printed `localhost` URL,
choose a chat profile, and start asking questions. (`uv run uvicorn app:app
--reload` also works — `app.py` exposes the underlying ASGI app for that
entry point too.)

---

## Python version

This project pins **Python 3.11** (via `pyproject.toml`). To switch versions:

```bash
uv python pin 3.12   # pin to a different version
uv sync              # rebuild venv
```

Available on this machine: 3.8, 3.11

---

## Project files

| File | Purpose |
|---|---|
| `pyproject.toml` | Project metadata and dependencies |
| `uv.lock` | Locked dependency tree (do not edit manually) |
| `.venv/` | Virtual environment (auto-created by uv, do not commit) |
