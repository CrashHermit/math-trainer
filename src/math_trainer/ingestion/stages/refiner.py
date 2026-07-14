"""Stage 6 — Refiner.

Refine specific element types (Code / Activity / Instruction / Admonition); other
types pass through untouched. Idempotent via ``refined_at``.
"""

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import REFINABLE_TYPES, element_subtype
from math_trainer.ingestion.stages.base import LMModule, concurrent_map, neighbors, now_iso
from math_trainer.storage.neo4j.repository import GraphRepository


class RefinerStage:
    name = "refiner"

    def __init__(self, repo: GraphRepository, module: LMModule, config: StageConfig) -> None:
        self._repo = repo
        self._module = module
        self._config = config

    async def run(self, source_uuid: str) -> None:
        ordered = await self._repo.source_elements_ordered(source_uuid)
        targets = [
            (i, e) for i, e in enumerate(ordered)
            if element_subtype(e) in REFINABLE_TYPES and not e.get("refined_at")
        ]

        async def refine(pair: tuple[int, dict]) -> None:
            index, element = pair
            prev, nxt = neighbors(ordered, index)
            prediction = await self._module.aforward(
                element_type=element_subtype(element).value,
                current_content=element.get("content"),
                previous_context=(prev or {}).get("content") if prev else None,
                next_context=(nxt or {}).get("content") if nxt else None,
            )
            content = None if prediction is None else getattr(prediction, "content", None)
            update = {"refined_at": now_iso()}
            if content:
                update["content"] = content
            await self._repo.update_node(element["uuid"], **update)

        await concurrent_map(targets, refine, self._config.max_concurrent)
