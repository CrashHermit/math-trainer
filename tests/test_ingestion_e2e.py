"""End-to-end pipeline test: full ingest through the IngestionService with fakes."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from math_trainer.core.config import DoclingConfig, EmbeddingConfig, StageConfig
from math_trainer.core.model.types import NodeType, has_type
from math_trainer.ingestion.service import IngestionService
from math_trainer.ingestion.stages.assembler import AssemblerStage
from math_trainer.ingestion.stages.cleaner import CleanerStage
from math_trainer.ingestion.stages.distributor import DistributorStage
from math_trainer.ingestion.stages.embedder import EmbedderStage
from math_trainer.ingestion.stages.extractor import ExtractorStage
from math_trainer.ingestion.stages.picture_filter import PictureFilterStage
from math_trainer.ingestion.stages.seam_merger import SeamMergerStage
from math_trainer.providers.docling import DoclingProvider, NormalizedDocument, NormalizedItem
from math_trainer.storage.neo4j.schema import Schema

CFG = StageConfig(max_concurrent=2)


class FakeImage:
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"\x89PNG fake")

    def close(self) -> None:
        pass


class FakeProvider(DoclingProvider):
    """DoclingProvider with the Docling coupling stubbed out."""

    def convert(self, source: Path) -> object:
        return object()

    def normalize(self, document: object) -> NormalizedDocument:
        doc = NormalizedDocument()
        doc.page_images = {1: FakeImage(), 2: FakeImage()}
        doc.items = [
            NormalizedItem(1, NodeType.HEADING, content="# Ch 1"),
            NormalizedItem(1, NodeType.PARAGRAPH, content="Intro."),
            NormalizedItem(1, NodeType.IMAGE, image=FakeImage(), blurb="A unit circle."),
            NormalizedItem(1, NodeType.IMAGE, image=FakeImage(), blurb="decorative flourish"),
            NormalizedItem(1, NodeType.CODE, content="print( 1 )"),
            NormalizedItem(2, NodeType.INSTRUCTION, content="Solve for x:"),
            NormalizedItem(2, NodeType.ACTIVITY, content="x + 1 = 2"),
            NormalizedItem(2, NodeType.ACTIVITY, content="x - 3 = 0"),
            NormalizedItem(2, NodeType.PARAGRAPH, content="Body text."),
        ]
        return doc


class FakeModule:
    def __init__(self, fn):
        self._fn = fn

    async def aforward(self, **kwargs):
        return self._fn(**kwargs)


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [float(len(text)), 0, 0, 0, 0, 0, 0, 0]


@pytest_asyncio.fixture
async def service(repo, tmp_path):
    schema = Schema(repo, EmbeddingConfig(dimensions=8, similarity="cosine"))
    provider = FakeProvider(DoclingConfig(output_dir=str(tmp_path / "out")), repo)
    embedder = FakeEmbedder()
    svc = IngestionService(
        repo, schema, provider,
        PictureFilterStage(
            repo,
            FakeModule(lambda **kw: SimpleNamespace(is_substantive="decorative" not in (kw.get("blurb") or ""))),
            CFG,
            image_loader=lambda p: None,
        ),
        CleanerStage(repo, FakeModule(lambda **kw: SimpleNamespace(content=(kw["current_content"] or "").strip())), CFG),
        ExtractorStage(repo, FakeModule(lambda **kw: SimpleNamespace(parts=None)), CFG),
        SeamMergerStage(repo, FakeModule(lambda **kw: SimpleNamespace(merged=False)), CFG),
        DistributorStage(repo, FakeModule(lambda **kw: SimpleNamespace(should_link=True)), CFG),
        AssemblerStage(
            repo,
            FakeModule(lambda **kw: SimpleNamespace(
                units=[{"kind": "prose", "members": [b["uuid"] for b in kw["window"]]}],
                deferred=[],
            )),
            CFG,
        ),
        EmbedderStage(repo, embedder, CFG),
    )
    return SimpleNamespace(svc=svc, embedder=embedder)


async def test_full_ingest(service, repo):
    uid = await service.svc.ingest("/tmp/fake.pdf", title="Doc")

    source = await repo.get_node(uid, NodeType.SOURCE)
    assert source["stage"] == 8
    assert source["status"] == "complete"

    # Element backbone intact: decorative image dropped → 8 elements remain.
    els = await repo.source_elements_ordered(uid)
    assert len(els) == 8
    images = [e for e in els if has_type(e, NodeType.IMAGE)]
    assert len(images) == 1 and images[0]["blurb"] == "A unit circle."

    # distributor linked the instruction to both activities
    links = await repo.run(
        "MATCH (:`Instruction`)-[:`Instructs`]->(a:`Activity`) "
        "WHERE a.source_uuid = $s RETURN count(*) AS n", s=uid,
    )
    assert links[0]["n"] == 2

    # Assembler: prose grouped; each Activity promoted to its own exercise block
    # carrying its instruction; a lone prose block for the trailing paragraph.
    blocks = await repo.source_blocks_ordered(uid)
    assert [b["kind"] for b in blocks] == ["prose", "instruction", "exercise", "exercise", "prose"]
    exercises = [b for b in blocks if b["kind"] == "exercise"]
    assert len(exercises) == 2
    assert all("Solve for x:" in b["content"] for b in exercises)   # instruction carried in
    # the first prose block absorbed the kept figure's blurb
    assert "[figure] A unit circle." in blocks[0]["content"]

    # every block embedded (blocks are the embed unit now)
    assert all(b.get("embedding") for b in blocks)
    assert service.embedder.calls == 5


async def test_resume_reembeds_only(service, repo):
    uid = await service.svc.ingest("/tmp/fake.pdf", title="Doc")
    calls_after_first = service.embedder.calls
    assert calls_after_first == 5  # one call per block

    # Resume from the embedder stage: fingerprints match → no re-embedding.
    await service.svc.ingest("/tmp/fake.pdf", source_uuid=uid, from_stage=8)
    assert service.embedder.calls == calls_after_first


async def test_resume_requires_uuid(service):
    with pytest.raises(ValueError):
        await service.svc.ingest("/tmp/fake.pdf", from_stage=3)
