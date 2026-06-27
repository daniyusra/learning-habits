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
