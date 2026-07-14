"""Stage 4 — Extractor.

Consumes Docling's already-typed Element stream and consolidates it into the
pedagogical/content types: retype a coarse item, or split one item into several
(e.g. an exercise list → one Activity per item). Structural changes mutate nodes
+ order_index, then the reading chain is rebuilt deterministically.

The LLM module returns, per element, a prediction with ``items``: a list of
``{type, content}`` (dicts or attr-objects). One item with the same type is a
no-op; a different type retypes; multiple items split in place.
"""

from typing import Any

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import (
    NodeType,
    element_subtype,
    has_type,
)
from math_trainer.ingestion.stages.base import LMModule, neighbors, now_iso
from math_trainer.storage.neo4j.repository import GraphRepository

_SPLIT_EPS = 1e-3


def _read(item: Any, key: str) -> Any:
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def _to_type(value: Any, default: NodeType) -> NodeType:
    if not value:
        return default
    try:
        return NodeType(str(value))
    except ValueError:
        return default


class ExtractorStage:
    name = "extractor"

    def __init__(self, repo: GraphRepository, module: LMModule, config: StageConfig) -> None:
        self._repo = repo
        self._module = module
        self._config = config

    async def run(self, source_uuid: str) -> None:
        ordered = await self._repo.source_elements_ordered(source_uuid)
        structural_change = False

        for index, element in enumerate(ordered):
            if has_type(element, NodeType.IMAGE) or element.get("extracted_at"):
                continue
            if not element.get("content"):
                continue

            prev, nxt = neighbors(ordered, index)
            prediction = await self._module.aforward(
                previous_context=(prev or {}).get("content") if prev else None,
                current_content=element.get("content"),
                next_context=(nxt or {}).get("content") if nxt else None,
            )
            items = None if prediction is None else getattr(prediction, "items", None)
            current_type = element_subtype(element) or NodeType.PARAGRAPH

            if not items:
                await self._repo.update_node(element["uuid"], extracted_at=now_iso())
                continue

            if len(items) == 1:
                new_type = _to_type(_read(items[0], "type"), current_type)
                new_content = _read(items[0], "content") or element.get("content")
                if new_type is not current_type:
                    await self._repo.set_element_type(element["uuid"], new_type, current_type)
                await self._repo.update_node(
                    element["uuid"], content=new_content, extracted_at=now_iso()
                )
                continue

            # Split: replace the element with several, ordered between neighbors.
            base = float(element.get("order_index") or (index + 1))
            segment = await self._repo.segment_for_element(element["uuid"])
            new_uuids: list[str] = []
            for j, sub in enumerate(items):
                sub_type = _to_type(_read(sub, "type"), current_type)
                node = await self._repo.create_node(
                    [NodeType.ELEMENT, sub_type],
                    source_uuid=source_uuid,
                    order_index=base + j * _SPLIT_EPS,
                    content=_read(sub, "content") or "",
                    page_no=element.get("page_no"),
                    extracted_at=now_iso(),
                )
                new_uuids.append(node["uuid"])
            if segment is not None:
                await self._repo.link_children(segment["uuid"], new_uuids)
            await self._repo.delete_node(element["uuid"])
            structural_change = True

        if structural_change:
            await self._repo.rebuild_reading_chain(source_uuid)
