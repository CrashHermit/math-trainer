"""Tests for DoclingProvider.materialize — pure graph logic, no Docling needed."""
from __future__ import annotations

from pathlib import Path

import pytest

from math_trainer.core.config import DoclingConfig
from math_trainer.core.model.types import EdgeType, NodeType
from math_trainer.providers.docling import (
    DoclingProvider,
    NormalizedDocument,
    NormalizedItem,
)


class FakeImage:
    """Stand-in for a PIL image: records where it was saved."""

    def __init__(self) -> None:
        self.saved_to: str | None = None

    def save(self, path: str) -> None:
        self.saved_to = path
        Path(path).write_bytes(b"\x89PNG fake")

    def close(self) -> None:  # provider calls close() after save
        pass


@pytest.fixture
def provider(repo, tmp_path) -> DoclingProvider:
    cfg = DoclingConfig(output_dir=str(tmp_path / "output"))
    return DoclingProvider(cfg, repo)


def _doc() -> NormalizedDocument:
    doc = NormalizedDocument()
    doc.page_images = {1: FakeImage(), 2: FakeImage()}
    doc.items = [
        NormalizedItem(page_no=1, node_type=NodeType.HEADING, content="# Chapter 1"),
        NormalizedItem(page_no=1, node_type=NodeType.PARAGRAPH, content="Intro text."),
        NormalizedItem(page_no=1, node_type=NodeType.IMAGE, image=FakeImage(), blurb="A unit circle."),
        NormalizedItem(page_no=2, node_type=NodeType.MATH, content="$$x^2 + y^2 = 1$$"),
        NormalizedItem(page_no=2, node_type=NodeType.PARAGRAPH, content="More text."),
    ]
    return doc


async def test_materialize_builds_source_segment_element(provider, repo):
    source_uuid = await provider.materialize(
        _doc(), source_path="/tmp/doc.pdf", document_title="Doc"
    )

    source = await repo.get_node(source_uuid, NodeType.SOURCE)
    assert source["status"] == "extracted"
    assert source["title"] == "Doc"

    # Two segments, ordered by page.
    segments = await repo.children_ordered(source_uuid, EdgeType.CONTAINS)
    assert [s["segment_index"] for s in segments] == [1, 2]
    assert all(s["src"] and s["src"].endswith("page.png") for s in segments)

    # Elements in global reading order.
    elements = await repo.source_elements_ordered(source_uuid)
    assert [e["order_index"] for e in elements] == [1, 2, 3, 4, 5]
    assert elements[0]["content"] == "# Chapter 1"


async def test_materialize_image_element_has_src_and_blurb(provider, repo):
    source_uuid = await provider.materialize(
        _doc(), source_path="/tmp/doc.pdf", document_title="Doc"
    )
    rows = await repo.run(
        "MATCH (n:`Element`:`Image` {source_uuid: $s}) RETURN n", s=source_uuid
    )
    assert len(rows) == 1
    img = rows[0]["n"]
    assert img["blurb"] == "A unit circle."
    assert img["src"] and img["src"].endswith(".png")
    assert Path(img["src"]).exists()


async def test_materialize_next_chain_and_head(provider, repo):
    source_uuid = await provider.materialize(
        _doc(), source_path="/tmp/doc.pdf", document_title="Doc"
    )
    # Head edge Source-[:Has]->first element
    head = await repo.run(
        "MATCH (:`Source` {uuid: $s})-[:`Has`]->(h) RETURN h.order_index AS oi", s=source_uuid
    )
    assert head[0]["oi"] == 1

    # Full 5-node Next chain reachable from the head.
    chain = await repo.run(
        "MATCH p=(:`Source` {uuid: $s})-[:`Has`]->()-[:`Next`*]->(t) "
        "RETURN length(p) AS len ORDER BY len DESC LIMIT 1", s=source_uuid
    )
    assert chain[0]["len"] == 5  # Has + 4 Next hops across 5 elements


async def test_materialize_idempotent_reuse(provider, repo):
    # create_document reuses an existing Source with the same uuid (no extraction).
    await repo.create_node([NodeType.SOURCE], uuid="fixed-1", title="pre")
    result = await provider.create_document(
        source=Path("/nonexistent.pdf"), source_uuid="fixed-1"
    )
    assert result == "fixed-1"
    # No extraction happened (convert never called), Source unchanged.
    again = await repo.get_node("fixed-1", NodeType.SOURCE)
    assert again["title"] == "pre"
