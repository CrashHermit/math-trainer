"""Real full-chain smoke test.

Runs the entire IngestionService on a real born-digital PDF with live models:
Docling (standard, local layout model) → Picture Filter (no-op, no images) →
Cleaner → Extractor → Seam Merger → Distributor → Assembler → Embedder, all against
a throwaway Neo4j. Then prints the Elements, the Blocks, and a vector-search result.

Text stages use DeepSeek; embeddings use OpenAI text-embedding-3-small. Creds come
from ../Paideia/.env (or the environment). No secrets are printed.

Run:  python scripts/smoke_full_chain.py
"""
import asyncio
import os
import sys

from dotenv import dotenv_values
from testcontainers.neo4j import Neo4jContainer

from math_trainer.core.config import (
    Config, DatabaseConfig, DoclingConfig, EmbeddingConfig, StageConfig,
)
from math_trainer.core.model.types import NodeType
from math_trainer.factory import build_service
from math_trainer.providers.embedding import LiteLLMEmbedder
from math_trainer.storage.neo4j.schema import VECTOR_INDEX_NAME

C = {**dotenv_values("/home/user/Paideia/.env"), **os.environ}
DEEPSEEK = C.get("DEEPSEEK_API_KEY", "")
OPENAI = C.get("OPEN_AI_API_KEY", "")
# A tiny born-digital sample ships in tests/fixtures; override with argv[1].
PDF = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/sample_textbook.pdf")


def _stage(**kw) -> StageConfig:
    return StageConfig(model="deepseek/deepseek-chat", api_key=DEEPSEEK,
                       max_concurrent=4, num_retries=2, timeout_s=120.0, **kw)


def _vision_stage() -> StageConfig:  # Picture Filter needs a vision model
    return StageConfig(model="openai/gpt-4o-mini", api_key=OPENAI,
                       max_concurrent=2, num_retries=2, timeout_s=120.0)


def build_config(container: Neo4jContainer) -> Config:
    return Config(
        database=DatabaseConfig(uri=container.get_connection_url(), user="neo4j",
                                password=container.password, database="neo4j"),
        docling=DoclingConfig(
            mode="standard", do_ocr=False, do_formula_enrichment=False,
            output_dir="scratch_out",
            # Docling generates a blurb per figure via a remote vision API (OpenAI).
            picture_description_api_base="https://api.openai.com/v1/chat/completions",
            picture_description_api_key=OPENAI,
            picture_description_model="gpt-4o-mini",
        ),
        embedding=EmbeddingConfig(model="openai/text-embedding-3-small", api_key=OPENAI,
                                  dimensions=1536, similarity="cosine"),
        stages={
            "picture_filter": _vision_stage(), "cleaner": _stage(), "extractor": _stage(),
            "seam_merger": _stage(), "distributor": _stage(),
            "assembler": _stage(main_window_tokens=1500, context_window_tokens=400),
        },
    )


async def main() -> None:
    with Neo4jContainer("neo4j:5.26-community") as container:
        config = build_config(container)
        service, driver = build_service(config)
        repo = service._repo
        try:
            print(f"Ingesting {PDF} through the full chain (live models)…\n")
            uid = await service.ingest(PDF, title="Calculus §3.2")

            source = await repo.get_node(uid, NodeType.SOURCE)
            print(f"Source: stage={source['stage']} status={source['status']}\n")

            els = await repo.source_elements_ordered(uid)
            print(f"--- {len(els)} Elements (backbone) ---")
            for e in els:
                t = next((l for l in e["_labels"] if l != "Element"), "?")
                c = (e.get("content") or e.get("blurb") or "")[:70].replace("\n", " ")
                print(f"  {t:<11} {c}")

            blocks = await repo.source_blocks_ordered(uid)
            print(f"\n--- {len(blocks)} Blocks (semantic units) ---")
            for b in blocks:
                dim = len(b["embedding"]) if b.get("embedding") else 0
                label = f" [{b['label']}]" if b.get("label") else ""
                c = (b.get("content") or "")[:80].replace("\n", " ")
                print(f"  {b['kind']:<14}{label} (emb={dim}d)  {c}")

            # vector search sanity
            embedder = LiteLLMEmbedder(config.embedding)
            qvec = await embedder.embed("the power rule for differentiating x^n")
            hits = await repo.vector_search(VECTOR_INDEX_NAME, qvec, k=3)
            print("\n--- vector search: 'power rule for differentiating x^n' ---")
            for node, score in hits:
                c = (node.get("content") or "")[:70].replace("\n", " ")
                print(f"  {score:.3f}  {node.get('kind')}: {c}")
        finally:
            await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
