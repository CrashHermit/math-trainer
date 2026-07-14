"""Stage 8 — Embedder.

Embed each ``:Block`` (the semantic unit) from its composed content into
``embedding`` on the block. A content fingerprint skips re-embedding unchanged
blocks on re-runs.
"""

import hashlib

from math_trainer.core.config import StageConfig
from math_trainer.ingestion.normalize import normalize_math
from math_trainer.ingestion.stages.base import concurrent_map, now_iso
from math_trainer.providers.embedding import Embedder
from math_trainer.storage.neo4j.repository import GraphRepository


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbedderStage:
    name = "embedder"

    def __init__(self, repo: GraphRepository, embedder: Embedder, config: StageConfig) -> None:
        self._repo = repo
        self._embedder = embedder
        self._config = config

    async def run(self, source_uuid: str) -> None:
        blocks = await self._repo.source_blocks_ordered(source_uuid)

        targets: list[tuple[dict, str, str]] = []
        for block in blocks:
            text = normalize_math(block.get("content"))
            if not text:
                continue
            fp = _fingerprint(text)
            if block.get("embed_fingerprint") == fp:
                continue
            targets.append((block, text, fp))

        async def embed(item: tuple[dict, str, str]) -> None:
            block, text, fp = item
            vector = await self._embedder.embed(text)
            await self._repo.update_node(
                block["uuid"],
                embedding=vector,
                embed_fingerprint=fp,
                embedded_at=now_iso(),
            )

        await concurrent_map(targets, embed, self._config.max_concurrent)
