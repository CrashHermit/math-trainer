"""Stage 2 — Picture Filter.

Judge each Image element (with its page raster as context) as substantive vs
decorative; delete the decorative ones and re-stitch the reading chain.
"""

from typing import Any, Callable

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import NodeType, has_type
from math_trainer.ingestion.stages.base import LMModule, concurrent_map, now_iso
from math_trainer.storage.neo4j.repository import GraphRepository


def _default_image_loader(path: str | None) -> Any | None:
    if not path:
        return None
    try:
        from PIL import Image
        return Image.open(path)
    except Exception:
        return None


class PictureFilterStage:
    name = "picture_filter"

    def __init__(
        self,
        repo: GraphRepository,
        module: LMModule,
        config: StageConfig,
        image_loader: Callable[[str | None], Any | None] | None = None,
    ) -> None:
        self._repo = repo
        self._module = module
        self._config = config
        self._load = image_loader or _default_image_loader

    async def run(self, source_uuid: str) -> None:
        elements = await self._repo.source_elements_ordered(source_uuid)
        images = [
            e for e in elements
            if has_type(e, NodeType.IMAGE) and not e.get("filtered_at")
        ]

        async def judge(img: dict) -> tuple[dict, bool]:
            seg = await self._repo.segment_for_element(img["uuid"])
            page_image = self._load(seg.get("src")) if seg else None
            picture_image = self._load(img.get("src"))
            prediction = await self._module.aforward(
                page_image=page_image,
                picture_image=picture_image,
                blurb=img.get("blurb"),
            )
            keep = True if prediction is None else bool(prediction.is_substantive)
            return img, keep

        results = await concurrent_map(images, judge, self._config.max_concurrent)

        deleted = False
        for img, keep in results:
            if keep:
                await self._repo.update_node(img["uuid"], filtered_at=now_iso())
            else:
                await self._repo.delete_node(img["uuid"])
                deleted = True

        if deleted:
            await self._repo.rebuild_reading_chain(source_uuid)
