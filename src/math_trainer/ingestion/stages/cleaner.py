"""Stage 3 — Cleaner (replaces OCR).

Docling already produced the text; the Cleaner only ensures good format: normalize
markdown, enforce $…$/$$…$$ LaTeX, repair VLM artifacts. Per-Element, with prev/next
context. Idempotent via ``cleaned_at``.
"""

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import NodeType, has_type
from math_trainer.ingestion.stages.base import LMModule, concurrent_map, neighbors, now_iso
from math_trainer.storage.neo4j.repository import GraphRepository


class CleanerStage:
    name = "cleaner"

    def __init__(self, repo: GraphRepository, module: LMModule, config: StageConfig) -> None:
        self._repo = repo
        self._module = module
        self._config = config

    async def run(self, source_uuid: str) -> None:
        ordered = await self._repo.source_elements_ordered(source_uuid)
        targets = [
            (i, e) for i, e in enumerate(ordered)
            if not has_type(e, NodeType.IMAGE)
            and e.get("content")
            and not e.get("cleaned_at")
        ]

        async def clean(pair: tuple[int, dict]) -> None:
            index, element = pair
            prev, nxt = neighbors(ordered, index)
            prediction = await self._module.aforward(
                previous_context=(prev or {}).get("content") if prev else None,
                current_content=element.get("content"),
                next_context=(nxt or {}).get("content") if nxt else None,
            )
            content = None if prediction is None else getattr(prediction, "content", None)
            update = {"cleaned_at": now_iso()}
            if content:
                update["content"] = content
            await self._repo.update_node(element["uuid"], **update)

        await concurrent_map(targets, clean, self._config.max_concurrent)
