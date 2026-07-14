"""IngestionService — the pipeline driver (resume-from-stage) over a LangGraph chain."""

from pathlib import Path

from math_trainer.ingestion.pipeline.graph import build_pipeline
from math_trainer.ingestion.stages.cleaner import CleanerStage
from math_trainer.ingestion.stages.distributor import DistributorStage
from math_trainer.ingestion.stages.embedder import EmbedderStage
from math_trainer.ingestion.stages.extractor import ExtractorStage
from math_trainer.ingestion.stages.picture_filter import PictureFilterStage
from math_trainer.ingestion.stages.refiner import RefinerStage
from math_trainer.ingestion.stages.seam_merger import SeamMergerStage
from math_trainer.providers.docling import DoclingProvider
from math_trainer.storage.neo4j.repository import GraphRepository
from math_trainer.storage.neo4j.schema import Schema

STAGE_NAMES = {
    1: "docling",
    2: "picture_filter",
    3: "cleaner",
    4: "extractor",
    5: "seam_merger",
    6: "distributor",
    7: "refiner",
    8: "embedder",
}
STAGE_MAX = max(STAGE_NAMES)


class IngestionService:
    def __init__(
        self,
        repo: GraphRepository,
        schema: Schema,
        provider: DoclingProvider,
        picture_filter: PictureFilterStage,
        cleaner: CleanerStage,
        extractor: ExtractorStage,
        seam_merger: SeamMergerStage,
        distributor: DistributorStage,
        refiner: RefinerStage,
        embedder: EmbedderStage,
    ) -> None:
        self._repo = repo
        self._schema = schema
        self._provider = provider
        self._stages = {
            2: picture_filter,
            3: cleaner,
            4: extractor,
            5: seam_merger,
            6: distributor,
            7: refiner,
            8: embedder,
        }
        self._graph = build_pipeline(self)

    async def init_db(self) -> None:
        await self._schema.bootstrap()

    async def _set_stage(self, source_uuid: str, stage_no: int) -> None:
        status = "extracted" if stage_no < STAGE_MAX else "complete"
        await self._repo.update_node(source_uuid, stage=stage_no, status=status)

    async def ingest(
        self,
        source_path: str | Path,
        *,
        title: str | None = None,
        source_uuid: str | None = None,
        from_stage: int = 1,
    ) -> str:
        if from_stage < 1 or from_stage > STAGE_MAX:
            raise ValueError(f"from_stage must be 1–{STAGE_MAX}; got {from_stage}")
        if from_stage > 1 and source_uuid is None:
            raise ValueError("Resuming (from_stage > 1) requires source_uuid.")
        await self._schema.bootstrap()
        result = await self._graph.ainvoke(
            {
                "source_path": str(source_path),
                "title": title,
                "source_uuid": source_uuid,
                "from_stage": from_stage,
            }
        )
        return result["source_uuid"]

    async def run_stage(self, stage_no: int, source_uuid: str) -> None:
        """Run a single stage in isolation (debugging)."""
        if stage_no == 1:
            raise ValueError("Stage 1 (docling) runs via ingest(), not run_stage().")
        await self._stages[stage_no].run(source_uuid)
        await self._set_stage(source_uuid, stage_no)
