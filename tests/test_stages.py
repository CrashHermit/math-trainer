"""Stage graph-mutation tests with fake LLM modules, against real Neo4j."""

from types import SimpleNamespace

import pytest
import pytest_asyncio

from math_trainer.core.config import StageConfig
from math_trainer.core.model.types import EdgeType, NodeType, element_subtype, has_type
from math_trainer.ingestion.stages.cleaner import CleanerStage
from math_trainer.ingestion.stages.distributor import DistributorStage
from math_trainer.ingestion.stages.embedder import EmbedderStage
from math_trainer.ingestion.stages.extractor import ExtractorStage
from math_trainer.ingestion.stages.picture_filter import PictureFilterStage
from math_trainer.ingestion.stages.refiner import RefinerStage
from math_trainer.ingestion.stages.seam_merger import SeamMergerStage

CFG = StageConfig(max_concurrent=2)


class FakeModule:
    def __init__(self, fn):
        self._fn = fn

    async def aforward(self, **kwargs):
        return self._fn(**kwargs)


async def _mk_element(repo, source_uuid, seg_uuid, ntype, oi, **props):
    node = await repo.create_node(
        [NodeType.ELEMENT, ntype], source_uuid=source_uuid, order_index=oi, **props
    )
    await repo.link_children(seg_uuid, [node["uuid"]])
    return node


@pytest_asyncio.fixture
async def source(repo):
    src = await repo.create_node([NodeType.SOURCE], title="t")
    seg1 = await repo.create_node([NodeType.SEGMENT], segment_index=1, src="/p1.png")
    seg2 = await repo.create_node([NodeType.SEGMENT], segment_index=2, src="/p2.png")
    await repo.link(EdgeType.CONTAINS, src["uuid"], seg1["uuid"])
    await repo.link(EdgeType.CONTAINS, src["uuid"], seg2["uuid"])
    return SimpleNamespace(uuid=src["uuid"], seg1=seg1["uuid"], seg2=seg2["uuid"])


async def test_picture_filter_deletes_and_rebuilds(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="a", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.IMAGE, 2, blurb="decorative logo", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 3, content="b", page_no=1)
    await repo.rebuild_reading_chain(source.uuid)

    module = FakeModule(lambda **kw: SimpleNamespace(
        is_substantive="decorative" not in (kw.get("blurb") or "")
    ))
    await PictureFilterStage(repo, module, CFG).run(source.uuid)

    els = await repo.source_elements_ordered(source.uuid)
    assert not any(has_type(e, NodeType.IMAGE) for e in els)
    chain = await repo.run(
        "MATCH p=(:`Source` {uuid:$s})-[:`Has`]->()-[:`Next`*]->() "
        "RETURN max(length(p)) AS len", s=source.uuid,
    )
    assert chain[0]["len"] == 2  # Has + one Next across the two surviving paragraphs


async def test_picture_filter_keeps_substantive(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.IMAGE, 1, blurb="a labeled graph", page_no=1)
    module = FakeModule(lambda **kw: SimpleNamespace(is_substantive=True))
    await PictureFilterStage(repo, module, CFG).run(source.uuid)
    els = await repo.source_elements_ordered(source.uuid)
    assert len(els) == 1 and els[0].get("filtered_at")


async def test_cleaner_updates_content_skips_images(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="  messy  ", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.IMAGE, 2, blurb="x", page_no=1)
    module = FakeModule(lambda **kw: SimpleNamespace(content=(kw["current_content"] or "").strip() + " [c]"))
    await CleanerStage(repo, module, CFG).run(source.uuid)

    els = await repo.source_elements_ordered(source.uuid)
    para = next(e for e in els if element_subtype(e) is NodeType.PARAGRAPH)
    img = next(e for e in els if has_type(e, NodeType.IMAGE))
    assert para["content"] == "messy [c]" and para["cleaned_at"]
    assert not img.get("cleaned_at")


async def test_extractor_split(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="intro", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.LIST, 2, content="1. do x\n2. do y", page_no=1)

    def split(**kw):
        if kw.get("current_content", "").startswith("1."):
            return SimpleNamespace(items=[
                {"type": "Instruction", "content": "Solve:"},
                {"type": "Activity", "content": "do x"},
                {"type": "Activity", "content": "do y"},
            ])
        return SimpleNamespace(items=[{"type": "Paragraph", "content": kw["current_content"]}])

    await ExtractorStage(repo, FakeModule(split), CFG).run(source.uuid)
    els = await repo.source_elements_ordered(source.uuid)
    types = [element_subtype(e).value for e in els]
    contents = [e.get("content") for e in els]
    assert types == ["Paragraph", "Instruction", "Activity", "Activity"]
    assert contents == ["intro", "Solve:", "do x", "do y"]


async def test_extractor_retype(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="Note: careful here", page_no=1)
    module = FakeModule(lambda **kw: SimpleNamespace(
        items=[{"type": "Admonition", "content": kw["current_content"]}]
    ))
    await ExtractorStage(repo, module, CFG).run(source.uuid)
    els = await repo.source_elements_ordered(source.uuid)
    assert element_subtype(els[0]) is NodeType.ADMONITION


async def test_seam_merger_merges_across_page_boundary(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="The quick brown", page_no=1)
    await _mk_element(repo, source.uuid, source.seg2, NodeType.PARAGRAPH, 2, content="fox jumps.", page_no=2)
    await repo.rebuild_reading_chain(source.uuid)

    module = FakeModule(lambda **kw: SimpleNamespace(
        merged=True, merged_content=kw["left_content"] + " " + kw["right_content"]
    ))
    await SeamMergerStage(repo, module, CFG).run(source.uuid)
    els = await repo.source_elements_ordered(source.uuid)
    assert len(els) == 1
    assert els[0]["content"] == "The quick brown fox jumps."


async def test_seam_merger_ignores_same_page(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="a", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 2, content="b", page_no=1)
    module = FakeModule(lambda **kw: SimpleNamespace(merged=True, merged_content="x"))
    await SeamMergerStage(repo, module, CFG).run(source.uuid)
    els = await repo.source_elements_ordered(source.uuid)
    assert len(els) == 2  # same page → never considered


async def test_refiner_only_refinable_types(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="prose", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.CODE, 2, content="print( 'x' )", page_no=1)
    module = FakeModule(lambda **kw: SimpleNamespace(content=kw["current_content"].replace(" ", "")))
    await RefinerStage(repo, module, CFG).run(source.uuid)

    els = await repo.source_elements_ordered(source.uuid)
    para = next(e for e in els if element_subtype(e) is NodeType.PARAGRAPH)
    code = next(e for e in els if element_subtype(e) is NodeType.CODE)
    assert para["content"] == "prose" and not para.get("refined_at")
    assert code["content"] == "print('x')" and code["refined_at"]


class FakeEmbedder:
    def __init__(self):
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        # deterministic 8-dim vector from text length
        n = float(len(text))
        return [n, 0, 0, 0, 0, 0, 0, 0]


async def test_embedder_text_and_blurb_and_skip(repo, source):
    await _mk_element(repo, source.uuid, source.seg1, NodeType.PARAGRAPH, 1, content="hello", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.IMAGE, 2, blurb="a graph", page_no=1)
    embedder = FakeEmbedder()

    await EmbedderStage(repo, embedder, CFG).run(source.uuid)
    els = await repo.source_elements_ordered(source.uuid)
    assert all(e.get("embedding") for e in els)
    assert embedder.calls == 2

    # Re-run: fingerprints match, nothing re-embedded.
    await EmbedderStage(repo, embedder, CFG).run(source.uuid)
    assert embedder.calls == 2


async def _instructs_count(repo, instruction_uuid) -> int:
    rows = await repo.run(
        "MATCH (:`Instruction` {uuid:$u})-[:`Instructs`]->(:`Activity`) RETURN count(*) AS n",
        u=instruction_uuid,
    )
    return rows[0]["n"]


async def test_distributor_links_window_and_respects_boundary(repo, source):
    ins = await _mk_element(repo, source.uuid, source.seg1, NodeType.INSTRUCTION, 1, content="Solve:", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.ACTIVITY, 2, content="a1", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.ACTIVITY, 3, content="a2", page_no=1)
    # A heading closes the window; the activity after it is NOT governed.
    await _mk_element(repo, source.uuid, source.seg1, NodeType.HEADING, 4, content="Next section", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.ACTIVITY, 5, content="a3", page_no=1)

    module = FakeModule(lambda **kw: SimpleNamespace(should_link=True))
    await DistributorStage(repo, module, CFG).run(source.uuid)

    assert await _instructs_count(repo, ins["uuid"]) == 2
    els = await repo.source_elements_ordered(source.uuid)
    a3 = next(e for e in els if e.get("content") == "a3")
    incoming = await repo.run(
        "MATCH (:`Instruction`)-[:`Instructs`]->(a:`Activity` {uuid:$u}) RETURN count(*) AS n",
        u=a3["uuid"],
    )
    assert incoming[0]["n"] == 0


async def test_distributor_skips_when_should_link_false(repo, source):
    ins = await _mk_element(repo, source.uuid, source.seg1, NodeType.INSTRUCTION, 1, content="Solve:", page_no=1)
    await _mk_element(repo, source.uuid, source.seg1, NodeType.ACTIVITY, 2, content="unrelated", page_no=1)
    module = FakeModule(lambda **kw: SimpleNamespace(should_link=False))
    await DistributorStage(repo, module, CFG).run(source.uuid)
    assert await _instructs_count(repo, ins["uuid"]) == 0
    # still marked processed so a re-run won't re-ask
    els = await repo.source_elements_ordered(source.uuid)
    act = next(e for e in els if element_subtype(e) is NodeType.ACTIVITY)
    assert act.get("distributed_at")
