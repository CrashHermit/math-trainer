"""Composition root: build a fully-wired IngestionService from Config.

DSPy/Docling/litellm imports happen lazily inside ``build_service`` so importing
this module (and the package) never requires the heavy ML stack.
"""

from math_trainer.core.config import Config
from math_trainer.ingestion.service import IngestionService
from math_trainer.ingestion.stages.cleaner import CleanerStage
from math_trainer.ingestion.stages.distributor import DistributorStage
from math_trainer.ingestion.stages.embedder import EmbedderStage
from math_trainer.ingestion.stages.extractor import ExtractorStage
from math_trainer.ingestion.stages.picture_filter import PictureFilterStage
from math_trainer.ingestion.stages.refiner import RefinerStage
from math_trainer.ingestion.stages.seam_merger import SeamMergerStage
from math_trainer.providers.docling import DoclingProvider
from math_trainer.providers.embedding import LiteLLMEmbedder
from math_trainer.storage.neo4j.driver import Neo4jDriver
from math_trainer.storage.neo4j.repository import GraphRepository
from math_trainer.storage.neo4j.schema import Schema


def build_service(config: Config) -> tuple[IngestionService, Neo4jDriver]:
    from math_trainer.core.dspy.module import DSPyModule, load_dspy_image
    from math_trainer.ingestion.stages.signatures import (
        CleanerSignature,
        ExtractorSignature,
        LinkDecisionSignature,
        PictureFilterSignature,
        RefinerSignature,
        SeamMergerSignature,
    )

    driver = Neo4jDriver(config.database)
    repo = GraphRepository(driver)
    schema = Schema(repo, config.embedding)
    provider = DoclingProvider(config.docling, repo)

    def stage_module(name: str, signature: object) -> DSPyModule:
        return DSPyModule(config.stage(name), signature)

    picture_filter = PictureFilterStage(
        repo, stage_module("picture_filter", PictureFilterSignature),
        config.stage("picture_filter"), image_loader=load_dspy_image,
    )
    cleaner = CleanerStage(repo, stage_module("cleaner", CleanerSignature), config.stage("cleaner"))
    extractor = ExtractorStage(repo, stage_module("extractor", ExtractorSignature), config.stage("extractor"))
    seam_merger = SeamMergerStage(repo, stage_module("seam_merger", SeamMergerSignature), config.stage("seam_merger"))
    distributor = DistributorStage(repo, stage_module("distributor", LinkDecisionSignature), config.stage("distributor"))
    refiner = RefinerStage(repo, stage_module("refiner", RefinerSignature), config.stage("refiner"))
    embedder = EmbedderStage(repo, LiteLLMEmbedder(config.embedding), config.stage("embedder"))

    service = IngestionService(
        repo, schema, provider,
        picture_filter, cleaner, extractor, seam_merger, distributor, refiner, embedder,
    )
    return service, driver
