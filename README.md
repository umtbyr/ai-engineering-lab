# RAG Knowledge Starter — Parsing + normalized blocks

Small learning experiment; intentionally **no LLM, embeddings, database, or FastAPI** yet.

## Run (Python 3.10+)

Windows PowerShell (no environment activation required):

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe demo.py
.\.venv\Scripts\python.exe -m unittest -v
```

macOS / Linux:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python demo.py
./.venv/bin/python -m unittest -v
```

## Architecture

`sample.md -> markdown-it-py Token[] -> parser.py -> StructuredDocument -> chunking.py -> Chunk[]`

The parser is an **adapter**, while the models (`StructuredDocument`, `DocumentBlock`, `Chunk`) belong to our application.

## Decisions we are intentionally postponing

- Markdown tables, nested lists, quotes, raw HTML and frontmatter
- PDF and source page references
- Token-based budgets, overlap, splitting oversized code blocks
- Stable document source ID/version versus content fingerprint
- Embedding text normalization and metadata filtering
- Persistence, retrieval, evaluation and LLM integration

## Experiments

Try changing the size budget in `demo.py` from 160 to 80, 250 and 500.
Compare `chunk_by_size` with `chunk_by_section`. Note that sections can grow unlimited in this v0, while `chunk_by_size` never cuts an individual block (even a large one).

A key next question: how should the policy split an oversized section while preserving semantic boundaries and source offsets?
