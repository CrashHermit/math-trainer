"""Stage 6 — Activity/Instruction Distributor.

Links each shared Instruction (a lead line like "1–20 Find the derivative…") to the
Activity exercises it governs, via an ``Instructs`` edge, so every exercise carries
its governing instruction. Windowing: an Instruction governs the Activities that
follow it in reading order until the next Instruction or Heading (section boundary).
An LLM confirms ambiguous pairs (``should_link``); with no module it links the whole
window. Idempotent via ``distributed_at`` on the Activity and a MERGE'd edge.
"""

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import EdgeType, NodeType, element_subtype
from math_trainer.ingestion.stages.base import LMModule, concurrent_map, now_iso
from math_trainer.storage.neo4j.repository import GraphRepository

# Element types that close an instruction's governing window.
_WINDOW_BREAKERS = frozenset({NodeType.HEADING, NodeType.INSTRUCTION})


class DistributorStage:
    name = "distributor"

    def __init__(self, repo: GraphRepository, module: LMModule, config: StageConfig) -> None:
        self._repo = repo
        self._module = module
        self._config = config

    async def run(self, source_uuid: str) -> None:
        ordered = await self._repo.source_elements_ordered(source_uuid)

        # Walk reading order, pairing each Activity with the instruction in scope.
        pairs: list[tuple[dict, dict]] = []
        current_instruction: dict | None = None
        for element in ordered:
            subtype = element_subtype(element)
            if subtype in _WINDOW_BREAKERS:
                current_instruction = element if subtype is NodeType.INSTRUCTION else None
                continue
            if (
                subtype is NodeType.ACTIVITY
                and current_instruction is not None
                and not element.get("distributed_at")
            ):
                pairs.append((current_instruction, element))

        async def distribute(pair: tuple[dict, dict]) -> None:
            instruction, activity = pair
            prediction = await self._module.aforward(
                instruction_text=instruction.get("content") or "",
                activity_content=activity.get("content") or "",
            )
            should_link = True if prediction is None else bool(
                getattr(prediction, "should_link", True)
            )
            if should_link:
                await self._repo.link(
                    EdgeType.INSTRUCTS, instruction["uuid"], activity["uuid"]
                )
            await self._repo.update_node(activity["uuid"], distributed_at=now_iso())

        await concurrent_map(pairs, distribute, self._config.max_concurrent)
