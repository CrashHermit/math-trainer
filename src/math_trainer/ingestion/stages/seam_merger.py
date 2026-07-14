"""Stage 5 — Seam Merger.

Heal cross-page continuations: an element whose content continues onto the next
page is merged with its successor when they straddle a page boundary. Single pass
over adjacent pairs, then the reading chain is rebuilt.
"""
from __future__ import annotations

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import NodeType, has_type
from math_trainer.ingestion.stages.base import LMModule, now_iso
from math_trainer.storage.neo4j.repository import GraphRepository


class SeamMergerStage:
    name = "seam_merger"

    def __init__(self, repo: GraphRepository, module: LMModule, config: StageConfig) -> None:
        self._repo = repo
        self._module = module
        self._config = config

    async def run(self, source_uuid: str) -> None:
        ordered = await self._repo.source_elements_ordered(source_uuid)
        merged_any = False
        skip_next = False

        for i in range(len(ordered) - 1):
            if skip_next:
                skip_next = False
                continue
            left, right = ordered[i], ordered[i + 1]
            if has_type(left, NodeType.IMAGE) or has_type(right, NodeType.IMAGE):
                continue
            # Only consider a merge across a page boundary.
            if left.get("page_no") == right.get("page_no"):
                continue
            if not (left.get("content") and right.get("content")):
                continue

            prediction = await self._module.aforward(
                left_content=left.get("content"),
                right_content=right.get("content"),
            )
            if prediction is None or not getattr(prediction, "merged", False):
                continue

            merged_content = getattr(prediction, "merged_content", None) or (
                f"{left['content']}\n{right['content']}"
            )
            await self._repo.update_node(
                left["uuid"], content=merged_content, seam_merged_at=now_iso()
            )
            await self._repo.delete_node(right["uuid"])
            merged_any = True
            skip_next = True  # don't merge the just-deleted right into i+2

        if merged_any:
            await self._repo.rebuild_reading_chain(source_uuid)
