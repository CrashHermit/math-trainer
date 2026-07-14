"""Stage 7 — Embedder.

Embed text-bearing Elements from their normalized content and Image Elements from
their blurb text, into ``embedding`` on the node. A content fingerprint skips
re-embedding unchanged content on re-runs.
"""
from __future__ import annotations

import hashlib

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import (
    TEXT_EMBEDDABLE_TYPES,
    NodeType,
    element_subtype,
    has_type,
)
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

    def _embed_text(self, element: dict) -> str | None:
        if has_type(element, NodeType.IMAGE):
            return (element.get("blurb") or "").strip() or None
        if element_subtype(element) in TEXT_EMBEDDABLE_TYPES:
            return normalize_math(element.get("content")) or None
        return None

    async def run(self, source_uuid: str) -> None:
        ordered = await self._repo.source_elements_ordered(source_uuid)

        targets: list[tuple[dict, str, str]] = []
        for element in ordered:
            text = self._embed_text(element)
            if not text:
                continue
            fp = _fingerprint(text)
            if element.get("embed_fingerprint") == fp:
                continue
            targets.append((element, text, fp))

        async def embed(item: tuple[dict, str, str]) -> None:
            element, text, fp = item
            vector = await self._embedder.embed(text)
            await self._repo.update_node(
                element["uuid"],
                embedding=vector,
                embed_fingerprint=fp,
                embedded_at=now_iso(),
            )

        await concurrent_map(targets, embed, self._config.max_concurrent)
