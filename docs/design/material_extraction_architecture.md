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
                 │             Seam Merger ─▶ Distributor ─▶ Assembler ─▶ Embedder    │
                 │                                                                   │
                 │   DSPy (stage LLM I/O)   LangGraph (per-stage fan-out)            │
                 └───────────────┬───────────────────────────────────────────────────┘
                                 │  async Neo4j repository (Bolt)
                                 ▼
                         Neo4j (Docker, Community 5.x)
                         Source → Segment → Element (backbone) → Block  +  vector index on :Block
```

Two supporting subsystems:
- **Docling extraction provider** — wraps Docling, points its VLM/picture-description at remote APIs (config-switchable to local).
- **Text embedding provider** — one text embedder; images are embedded via their blurb text.

---

## 3. Pipeline stages

The pipeline is a **linear 8-stage** flow. Every stage is **idempotent** and skippable when its
work is already done (see §8). Intelligent stages are DSPy Signatures; each stage runs as a minimal
LangGraph fan-out (`dispatch → worker → condense`) over a batch of nodes.

| # | Stage | Kind | Reads | Writes |
|---|-------|------|-------|--------|
| 1 | **Docling** (provider) | Docling (remote VLM) | source file | `Source`, `Segment`(page), `Element` nodes (typed, Next-chained), Image files + blurbs |
| 2 | **Picture Filter** | DSPy (vision) | Image elements + page raster | deletes non-substantive Image elements |
| 3 | **Cleaner** | DSPy (text) | each Element + neighbors | normalized `content` on each Element |
| 4 | **Extractor** | DSPy (text) | Docling-typed Elements + neighbors | regrouped/retyped Elements (e.g. Instruction + Activities) |
| 5 | **Seam Merger** | DSPy (text) | Elements at page boundaries | merged cross-page continuations |
| 6 | **Distributor** | DSPy (text) | Instruction + Activity elements | `Instructs` edges linking instructions → governed activities |
| 7 | **Assembler** | DSPy (text) + rules | Elements in reading order | `:Block` semantic units (the embed/retrieval unit) grouping the Elements |
| 8 | **Embedder** | embedding provider | each `:Block` | `embedding` vector on each Block |

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

### 3.6 Distributor (Activity/Instruction)
- Links each shared **Instruction** (a lead line, e.g. "1–20 Find the derivative…") to the **Activity** exercises it governs, via an `Instructs` edge, so every exercise carries its governing instruction.
- **Windowing:** an Instruction governs the Activities that follow it in reading order until the next Instruction or Heading (section boundary).
- An LLM confirms ambiguous pairs (`should_link`); with no decision it links the whole window. Idempotent via `distributed_at` on the Activity and a MERGE'd edge.
- (Ported from Paideia's `activity_instruction_distributor`, simplified: window + LLM confirmation instead of identifier-range matching.)

### 3.7 Assembler
Builds the **`:Block`** overlay — the semantic unit that becomes the embed/retrieval
unit. Elements stay untouched as the structural backbone; each Block `Groups` a
contiguous run of Elements. (The Refiner is retired: its format work folds into the
Cleaner; per-type structuring is deferred to the knowledge layer.)

- **Stateful, sequential walk** over the reading chain (the one stage that is not a
  fan-out — each window's start is the previous window's cut point). Resumable via a
  cursor; the carried open unit is transient. Idempotent: existing Blocks are dropped
  and rebuilt (Elements never touched).
- **Anchors** — `Activity` and `Instruction` only — are already at unit granularity:
  each is promoted 1:1 to a Block (no LLM), and an `Activity` **absorbs** the
  components that belong to it (its figure/table/equation). An exercise Block carries
  its governing `Instruction` text (via the `Instructs` edge). Anchors close any open
  ungrouped region.
- **Everything else** (Paragraph, Heading, Math, Table, Caption, List, Code, Image,
  Admonition) is **ungrouped content**: it accumulates into a token-bounded **main
  window**; when full, an LLM segments it into typed units (definition / theorem /
  example / prose …), keeping statement|proof and problem|solution as *separate* linked
  units. A read-only **context window** of upcoming blocks is peeked past the edge.
- **Commit-once handoff + greedy-to-unit-boundary:** only the main window commits; the
  LLM defers a trailing incomplete unit, which carries into the next window, so a unit
  never splits across windows. On `None`/failure the window degrades to one prose Block.
- Window sizes (`main_window_tokens`, `context_window_tokens`) are config.

### 3.8 Embedder
- Embeds each **`:Block`** from its composed (math-normalized) `content`. Image blurbs
  ride along inside their Block's content.
- Stores the vector on `Block.embedding`. Uses a **content fingerprint** to skip
  re-embedding unchanged Blocks on re-runs.
- The Neo4j vector index is on `:Block` (see §5.3).

---

## 4. Data model

### 4.1 Nodes
| Label | Purpose | Key props |
|-------|---------|-----------|
| `:Source` | one ingested document/image | `uuid`, `title`, `source_path`, `status`, `stage`, `created_at`, `updated_at` |
| `:Segment` | one page | `uuid`, `segment_index` (page no.), `src` (page raster), timestamps |
| `:Element:<Type>` | one atomic content element (backbone) | `uuid`, `content`, `source_uuid`, stage markers, timestamps |
| `:Element:Image` | a kept figure | `uuid`, `src` (file), `blurb`, `bbox`, `page_no`, `source_uuid` |
| `:Block` | a semantic unit (embed/retrieval unit) | `uuid`, `kind`, `label`, `content` (composed), `source_uuid`, `embedding`, timestamps |

`<Type>` ∈ `Paragraph, Heading, Math, Table, Caption, List, ListItem, Code, Image, Admonition, Instruction, Activity` (extend as needed). **Every content node carries the base `:Element` label** plus one concrete type label. `Block.kind` ∈ `prose, definition, theorem, example, remark, exercise, instruction, admonition, figure` (LLM-assigned kinds are free-form).

### 4.2 Edges
| Type | From → To | Meaning |
|------|-----------|---------|
| `Contains` | Source → Segment, Segment → Element, Source → Block | membership / structure |
| `Has` | Source → head Element | entry point into the element reading chain |
| `Next` | Element → Element, Block → Block | reading order |
| `Instructs` | Instruction → Activity | a lead instruction governs an exercise |
| `Groups` | Block → Element | a semantic unit groups these elements |

Membership (`Contains`) is the **authoritative selection** for batching, so a broken `Next`
chain degrades ordering but never completeness. **Elements are the structural backbone;
Blocks are a non-destructive overlay** (drop-and-rebuild safe).

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

The Extractor may promote generic types to pedagogical ones (`Admonition`, `Instruction`, `Activity`).

### 4.4 Constraints & indexes (Neo4j)
- `CREATE CONSTRAINT element_uuid IF NOT EXISTS FOR (n:Element) REQUIRE n.uuid IS UNIQUE`
- Unique-uuid constraints likewise for `:Source`, `:Segment`, `:Block`.
- Range indexes: `Element.source_uuid`, `Block.source_uuid`, `Segment.segment_index`.
- Vector index on `:Block` (see §5.3).

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
- One index over the semantic unit (`:Block`):
  ```cypher
  CREATE VECTOR INDEX block_embedding IF NOT EXISTS
  FOR (n:Block) ON n.embedding
  OPTIONS { indexConfig: {
    `vector.dimensions`: $dims,          // from config (embedding model)
    `vector.similarity_function`: 'cosine'
  }};
  ```
- Query:
  ```cypher
  CALL db.index.vector.queryNodes('block_embedding', $k, $query_vector)
  YIELD node, score RETURN node, score;
  ```
- The `:Block` overlay is the retrieval unit, so there is a **single** index (no per-subtype indexes, unlike Paideia). Dimensions come from the configured embedding model.

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
    assembler:      { model, main_window_tokens: 1500, context_window_tokens: 400, ... }
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
      stages/               # picture_filter, cleaner, extractor, seam_merger,
                            #   distributor, assembler, embedder
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
4. **Picture Filter → Cleaner → Extractor → Seam Merger → Distributor → Assembler** — one stage at a time, each with a DSPy Signature and an idempotency marker (Assembler builds the `:Block` overlay).
5. **Embedder** + vector-index population; retrieval round-trip test.
6. **Ingestion service** (resume-from-stage) + **CLI**; end-to-end fixture test (PDF + image).

---

## 14. Deferred / open
- Exact Seam Merger heuristics and Assembler LLM prompt/kinds (refine against real Docling output).
- Concrete VLM / embedding model choices + vector dimensions (config values).
- Hybrid keyword (full-text) search — deferred; vector-only for now.
- Knowledge-graph, mastery, API/frontend — future phases.
- Multimodal (pixel) image embedding — deferred in favor of blurb embedding.
