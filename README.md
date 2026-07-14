# math-trainer

Material-extraction pipeline: ingest a **PDF or image**, extract clean, structured,
typed, embedded content with **Docling**, and persist it as a graph in **Neo4j**.

See [`docs/design/material_extraction_architecture.md`](docs/design/material_extraction_architecture.md)
for the full design.

## Pipeline

```
Docling → Picture Filter → Cleaner → Extractor → Seam Merger → Distributor → Refiner → Embedder
```

Docling (remote-VLM by default) is the sole text extractor and emits typed,
positioned items plus figure "blurbs". Content is stored as
`Source → Segment(page) → Element` (multi-label `:Element:<Type>`), with a single
cosine vector index on `:Element`.

## Quickstart

```bash
# 1. Start Neo4j
export NEO4J_PASSWORD=change-me
docker compose up -d

# 2. Install (Python 3.13)
uv venv --python 3.13 && source .venv/bin/activate
uv pip install -e ".[test]"

# 3. Configure — copy .env.example to .env and fill in model endpoints/keys
cp .env.example .env

# 4. Bootstrap schema (constraints + vector index)
math-trainer init-db

# 5. Ingest a document
math-trainer ingest path/to/textbook.pdf --title "Algebra Ch.1"
# resume a crashed run from a stage:
math-trainer ingest path/to/textbook.pdf --source-uuid <uuid> --from-stage 3
```

## Tests

Integration-first: storage and pipeline stages are verified against a real Neo4j
spun up via testcontainers (Docker required). LLM/Docling/embedder calls are faked.

```bash
source .venv/bin/activate
python -m pytest -q
```

## Layout

```
src/math_trainer/
  cli.py                  # CLI over the library core
  factory.py              # composition root (Config → wired IngestionService)
  core/                   # config, model types, DSPy wrapper
  providers/              # docling extraction, text embedding
  ingestion/
    service.py            # pipeline driver (resume-from-stage)
    pipeline/graph.py     # minimal LangGraph chain
    stages/               # picture_filter, cleaner, extractor, seam_merger,
                          #   distributor, refiner, embedder
  storage/neo4j/          # async driver, repository, schema bootstrap
```
