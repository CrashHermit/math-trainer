# math-trainer — Material Extraction Architecture

> Status: **Design / ready to execute**
> Scope: the **material-extraction pipeline** only — ingest a PDF or image, produce clean,
> structured, typed, embedded content in Neo4j. No knowledge-graph, mastery, or learner layer.
> Relationship to Paideia: **from-scratch reimplementation**. Paideia is a reference blueprint
> for stage design and prompts; none of its code is forked verbatim.

---

## 1. Goals & non-goals

### Goals
- Ingest **PDFs and standalone images** (photos/scans/screenshots of math content) as first-class inputs.
- Use **Docling as the sole text extractor** (via a remote VLM API), producing typed, position-aware content — including LaTeX for math and natural-language **blurbs** for figures.
- Persist a clean content graph in **Neo4j**: `Source → Segment(page) → Element`, with a single vector index for retrieval-readiness.
- Keep the system **simple**: minimal DSPy, minimal LangGraph, lightweight resumability.
- Be **resumable** — expensive Docling/VLM/LLM calls are never repeated unnecessarily.

### Non-goals (this phase)
- No Concept/Statement/Theorem/Procedure/Item knowledge layer.
- No mastery loop (Goals, Attempts, FSRS), no training/optimization of DSPy modules.
- No web API/frontend (library core + CLI only; API can wrap the core later).
- No DSPy judges/optimizers/training-data collection yet.

---

## 2. High-level architecture

```
                 ┌────────────────────────── library core ──────────────────────────┐
  CLI  ─────────▶│  Ingestion service (pipeline driver)                              │
  ingest <path>  │                                                                   │
                 │   Docling ─▶ Picture Filter ─▶ Cleaner ─▶ Extractor ─▶            │
                 │             Seam Merger ─▶ Refiner ─▶ Embedder                     │
                 │                                                                   │
                 │   DSPy (stage LLM I/O)   LangGraph (per-stage fan-out)            │
                 └───────────────┬───────────────────────────────────────────────────┘
                                 │  async Neo4j repository (Bolt)
                                 ▼
                         Neo4j (Docker, Community 5.x)
                         Source → Segment → Element  +  vector index on :Element
```

Two supporting subsystems:
- **Docling extraction provider** — wraps Docling, points its VLM/picture-description at remote APIs (config-switchable to local).
- **Text embedding provider** — one text embedder; images are embedded via their blurb text.

---

## 3. Pipeline stages

The pipeline is a **linear 7-stage** flow. Every stage is **idempotent** and skippable when its
work is already done (see §8). Intelligent stages are DSPy Signatures; each stage runs as a minimal
LangGraph fan-out (`dispatch → worker → condense`) over a batch of nodes.

| # | Stage | Kind | Reads | Writes |
|---|-------|------|-------|--------|
| 1 | **Docling** (provider) | Docling (remote VLM) | source file | `Source`, `Segment`(page), `Element` nodes (typed, Next-chained), Image files + blurbs |
| 2 | **Picture Filter** | DSPy (vision) | Image elements + page raster | deletes non-substantive Image elements |
| 3 | **Cleaner** | DSPy (text) | each Element + neighbors | normalized `content` on each Element |
| 4 | **Extractor** | DSPy (text) | Docling-typed Elements + neighbors | regrouped/retyped Elements (e.g. Instruction + Activities) |
| 5 | **Seam Merger** | DSPy (text) | Elements at page boundaries | merged cross-page continuations |
| 6 | **Refiner** | DSPy (text) | Code/Activity/Instruction/Admonition elements | refined `content` (+ structured fields) |
| 7 | **Embedder** | embedding provider | each embeddable Element | `embedding` vector on each Element |

### 3.1 Docling (provider)
- Accepts **PDF** (`InputFormat.PDF`) or **image** (`InputFormat.IMAGE`). A loose image is treated as a single-page document.
- Runs Docling's **VLM pipeline** against a **remote VLM API endpoint** (Docling API-VLM options). A config flag switches to a local model on GPU hosts. **CPU-only sandbox is the default target.**
- Enables **page image generation** (needed by Picture Filter as figure context) and **picture image generation** (cropped figures saved to disk).
- Enables **picture description** ("blurbs") via a remote vision API; the blurb is read from `PictureItem.annotations` and stored on the Image element.
- Math is transcribed to **LaTeX inline** by the VLM; no separate formula-enrichment model.
- **Materializes the graph immediately** by walking `DoclingDocument` in reading order:
  - `Source` (one per document/ingest).
  - `Segment` per page (`segment_index` = page number; `src` = page raster path).
  - One `Element` per Docling item, **typed from the Docling label** (see §4.3 mapping), Next-chained in reading order within the source, `Contains`-linked to its Segment.
  - `PictureItem` → `Image` element with `src` (file path), `blurb` (from annotations), and `bbox`/`page_no` from `prov`.
- Batches pages (config `batch_size`) with memory cleanup between batches.
- **Idempotent:** if a `Source` with the same uuid exists, reuse it and resume later stages instead of re-extracting.

### 3.2 Picture Filter (was Paideia `image_filter`)
- DSPy **vision** Signature: given the parent **page raster** for context and the individual **figure image**, decide `is_substantive` (keep) vs decorative/navigational noise (discard).
- Discarded Image elements are **hard-deleted** and unlinked from the Next chain (chain is re-stitched around the gap).
- Idempotency: skip Segments already filtered (stage marker).

### 3.3 Cleaner (replaces Paideia `ocr`)
- DSPy **text** Signature, **per-Element with prev/next neighbor context**.
- Docling's VLM already produced the text; the Cleaner **only ensures good format**: normalize markdown, enforce `$…$` / `$$…$$` LaTeX, repair VLM artifacts, strip leftover running-header/footer chrome.
- Writes normalized `content` back to the Element. Idempotent per element (skip if `cleaned_at` set).

### 3.4 Extractor
- DSPy **text** Signature operating over **Docling's structured, already-typed Element stream** (never flattened placeholder markdown).
- Its job is **pedagogical grouping/consolidation**, not raw parsing:
  - Group a lead **Instruction** with the **Activity** items it governs.
  - Split multi-exercise Lists into individual Activity items.
  - Reclassify where Docling's generic label is too coarse (e.g. a boxed Example → `Admonition`).
- Preserves reading order and re-links the Next chain / `Contains` edges for any nodes it creates or regroups.
- Because images arrive from Docling **already positioned with their files**, there are **no placeholders to reconcile** — the old Image Merger stage is gone.

### 3.5 Seam Merger
- DSPy **text** Signature that heals **cross-page continuations**: a paragraph, table, or math block split across a page boundary becomes one Element.
- **Single-pass** over adjacent page-boundary element pairs (simplified from Paideia's even/odd two-pass), followed by a deterministic **chain-anchor** pass that guarantees every reading-order adjacency has a live `Next` edge (so chain integrity never depends on a perfect LLM stitch).

### 3.6 Refiner
- DSPy **text** Signatures specialized per type: `Code`, `Activity`, `Instruction`, `Admonition`.
- Refines/normalizes those specific element types (e.g. clean code fences, structure an exercise stem + subparts). Other types pass through untouched.
- Idempotent per element.

### 3.7 Embedder
- One **text** embedding provider. For text-bearing Elements, embed the normalized (math-normalized) `content`; for **Image** elements, embed the **blurb** text.
- Stores the vector on `Element.embedding`. Uses a **content fingerprint** to skip re-embedding unchanged content on re-runs.
- After the source finishes, ensure the Neo4j vector index is **online/populated** (see §5.3).

---

## 4. Data model

### 4.1 Nodes
| Label | Purpose | Key props |
|-------|---------|-----------|
| `:Source` | one ingested document/image | `uuid`, `title`, `source_path`, `status`, `stage`, `created_at`, `updated_at` |
| `:Segment` | one page | `uuid`, `segment_index` (page no.), `src` (page raster), timestamps |
| `:Element:<Type>` | one content unit | `uuid`, `content`, `source_uuid`, `embedding`, stage markers, timestamps |
| `:Element:Image` | a kept figure | `uuid`, `src` (file), `blurb`, `bbox`, `page_no`, `embedding`, `source_uuid` |

`<Type>` ∈ `Paragraph, Heading, Math, Table, Caption, List, ListItem, Code, Image, Admonition, Instruction, Activity` (extend as needed). **Every content node carries the base `:Element` label** plus one concrete type label.

### 4.2 Edges
| Type | From → To | Meaning |
|------|-----------|---------|
| `Contains` | Source → Segment, Segment → Element | membership / structure |
| `Has` | Source → head Element | entry point into the element reading chain |
| `Next` | Element → Element | reading order |

Membership (`Contains`) is the **authoritative selection** for batching, so a broken `Next`
chain degrades ordering but never completeness.

### 4.3 Docling item → Element type mapping (starting point)
| Docling label | Element type |
|---------------|--------------|
| text / paragraph | `Paragraph` |
| section_header / title | `Heading` |
| formula | `Math` |
| table | `Table` |
| caption | `Caption` |
| list / list_item | `List` / `ListItem` |
| code | `Code` |
| picture | `Image` (with blurb) |

Refiner/Extractor may promote generic types to pedagogical ones (`Admonition`, `Instruction`, `Activity`).

### 4.4 Constraints & indexes (Neo4j)
- `CREATE CONSTRAINT element_uuid IF NOT EXISTS FOR (n:Element) REQUIRE n.uuid IS UNIQUE`
- Unique-uuid constraints likewise for `:Source`, `:Segment`.
- Range indexes: `Element.source_uuid`, `Segment.segment_index`.
- Vector index: see §5.3.

---

## 5. Storage layer

### 5.1 Deployment
- **Neo4j Community 5.x** in Docker (`docker-compose.yml`), Bolt on `:7687`, Browser on `:7474`.
- Community edition supports the native vector index — sufficient for this design.

### 5.2 Driver & repository
- Official **`neo4j` Python async driver** over Bolt.
- A thin **async repository** (`storage/neo4j/`) exposes: `create_node`, `update_node`, `delete_node` (DETACH), `link`, `link_chain`, `children_ordered`, `run` (read Cypher), `execute` (write Cypher), plus vector search.
- **No single-thread executor** — the driver is natively concurrent; use `AsyncSession` with **managed transactions** (`session.execute_read/execute_write`), a shared driver, and a bounded concurrency semaphore for stage fan-out.
- All access is **pure Cypher** (no ArcadeDB-SQL, no `expand()`/`vectorNeighbors` — those become Cypher + the vector-index procedure).

### 5.3 Vector search
- One index over the base label:
  ```cypher
  CREATE VECTOR INDEX element_embedding IF NOT EXISTS
  FOR (n:Element) ON n.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: $dims,          // from config (embedding model)
    `vector.similarity_function`: 'cosine'
  }};
  ```
- Query:
  ```cypher
  CALL db.index.vector.queryNodes('element_embedding', $k, $query_vector)
  YIELD node, score RETURN node, score;
  ```
- Multi-label means **all element subtypes share this one index** — collapsing Paideia's per-subtype indexes into a single one. Dimensions come from the configured embedding model.

---

## 6. Docling integration details
- Wrap `DocumentConverter` with a VLM pipeline option targeting a **remote API** (endpoint, key, model in config).
- Picture description enabled via a remote vision API option; blurb pulled from `PictureItem.annotations`.
- Page/picture image generation on; images written under `output/<source>/…`.
- Reading order from `DoclingDocument.body` / `iterate_items`; physical position from `prov[0].bbox`, `prov[0].page_no`.
- `PLAYWRIGHT`/GPU not involved; force CPU + remote models in the sandbox.

---

## 7. Frameworks

### 7.1 DSPy (kept, simple)
- Each intelligent stage = one DSPy `Signature` + a thin module wrapper with retries and a timeout.
- No judges, no optimizers, no training-data collection in this phase (add later without restructuring).
- A shared `normalize_math` utility is reused by Cleaner and Embedder.

### 7.2 LangGraph (kept, minimal)
- Each stage compiles a small `StateGraph`: `START → dispatch → worker (Send fan-out) → condense → END`.
- No checkpointing/streaming machinery beyond what a stage needs.
- Batch size and max-concurrency per stage come from config.

---

## 8. Resumability & idempotency (lightweight)
- `Source.stage` records the highest completed stage; the driver can **resume from a stage** (`--from-stage N`).
- Per-element stage markers (`cleaned_at`, `refined_at`, `embedded_at`, …) let a re-run **skip completed work**.
- Provider **reuses an existing `Source`** (by uuid) rather than re-extracting.
- Embedder uses a **content fingerprint** to avoid re-embedding unchanged content.
- **No audit-node tables** (`PipelineSourceJob`/`PipelineStageAttempt` are out of scope).
- Deletes are **hard** (`DETACH DELETE`); re-processing a source deletes and recreates its subgraph.

---

## 9. Configuration
- **One YAML file** (`config/math_trainer.yaml`) + **`.env`** for secrets, loaded into pydantic models.
- Structure:
  ```yaml
  database:
    uri: bolt://localhost:7687
    user: neo4j
    password: ${NEO4J_PASSWORD}
    database: math_trainer
  docling:
    mode: remote_vlm            # remote_vlm | local_vlm | standard
    vlm_api_base: ${VLM_API_BASE}
    vlm_api_key: ${VLM_API_KEY}
    vlm_model: ${VLM_MODEL}
    picture_description_api_base: ${PICDESC_API_BASE}
    batch_size: 4
  embedding:
    model: ${EMBED_MODEL}
    dimensions: 1024            # must match the vector index
  stages:
    picture_filter: { model: ${API_MODEL}, batch_size: 16, max_concurrent: 2, num_retries: 5, timeout_s: 300 }
    cleaner:        { model: ${API_MODEL}, batch_size: 32, max_concurrent: 2, ... }
    extractor:      { ... }
    seam_merger:    { ... }
    refiner:        { ... }
  ```
- Per-stage sections allow different models per stage (e.g. a vision model for Picture Filter).

---

## 10. CLI
- `math-trainer ingest <path> [--title T] [--from-stage N] [--source-uuid U]`
  runs the full pipeline (or resumes from stage N) on a PDF or image.
- `math-trainer stage <N> --source-uuid U` runs a single stage (debugging).
- `math-trainer init-db` applies constraints + the vector index.
- The CLI is a thin wrapper over the library core (`IngestionService`), so an API can wrap the same core later.

---

## 11. Testing
- **Integration-first** with a **real Neo4j via testcontainers**; `pytest` + `pytest-asyncio`.
- Storage/vector layer tested against real Neo4j (constraints, `Next`-chain integrity, vector query round-trip) — the riskiest part of the ArcadeDB→Neo4j move.
- Docling/VLM/embedder calls **mocked or run on tiny fixtures**; DSPy stages tested with recorded/mock LM responses.
- A small end-to-end test ingests a 1–2 page fixture PDF and a fixture image and asserts the resulting subgraph shape.
- (CI/sandbox must have Docker available for the integration tier.)

---

## 12. Project layout
```
math-trainer/
  pyproject.toml            # uv-managed, Python 3.13
  docker-compose.yml        # Neo4j Community 5.x
  config/math_trainer.yaml
  .env.example
  src/math_trainer/
    cli.py
    ingestion/
      service.py            # pipeline driver (resume-from-stage)
      pipeline/graph.py     # minimal LangGraph builders
      nodes/                # docling_provider, picture_filter, cleaner,
                            #   extractor, seam_merger, refiner, embedder
      normalize.py
    core/
      config.py             # pydantic config (YAML + .env)
      dspy/                 # module wrapper, image utils
      model/                # NodeType, EdgeType, element mapping
    providers/
      docling.py            # remote-VLM extraction provider
      embedding.py          # text embedder
    storage/neo4j/
      driver.py             # async driver lifecycle
      repository.py         # generic graph ops + vector search
      schema.py             # constraints + vector index bootstrap
  tests/
    conftest.py             # testcontainers Neo4j fixture
    ...
```

---

## 13. Build sequence
1. **Scaffold** — `pyproject.toml` (uv, py3.13), `docker-compose.yml`, config models, `.env.example`.
2. **Storage** — async driver + repository + schema bootstrap (`init-db`); testcontainers fixture; storage tests (CRUD, chain, vector round-trip).
3. **Docling provider** — remote-VLM conversion → `Source/Segment/Element` materialization + blurbs; fixture-based test.
4. **Picture Filter → Cleaner → Extractor → Seam Merger → Refiner** — one stage at a time, each with a DSPy Signature and an idempotency marker.
5. **Embedder** + vector-index population; retrieval round-trip test.
6. **Ingestion service** (resume-from-stage) + **CLI**; end-to-end fixture test (PDF + image).

---

## 14. Deferred / open
- Exact Seam Merger heuristics and Refiner per-type schemas (design at implementation time).
- Concrete VLM / embedding model choices + vector dimensions (config values).
- Hybrid keyword (full-text) search — deferred; vector-only for now.
- Knowledge-graph, mastery, API/frontend — future phases.
- Multimodal (pixel) image embedding — deferred in favor of blurb embedding.
