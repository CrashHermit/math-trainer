
import asyncio

import pytest

from math_trainer.core.model.types import EdgeType, NodeType
from math_trainer.storage.neo4j.schema import VECTOR_INDEX_NAME


async def test_create_and_get_node(repo):
    node = await repo.create_node(
        [NodeType.ELEMENT, NodeType.PARAGRAPH], content="hello", source_uuid="s1"
    )
    assert node["uuid"]
    assert node["content"] == "hello"
    assert node["created_at"] is not None

    fetched = await repo.get_node(node["uuid"], NodeType.ELEMENT)
    assert fetched is not None
    assert fetched["content"] == "hello"


async def test_multi_label(repo):
    node = await repo.create_node([NodeType.ELEMENT, NodeType.MATH], content="$x^2$")
    rows = await repo.run(
        "MATCH (n:`Element`:`Math` {uuid: $u}) RETURN labels(n) AS labels",
        u=node["uuid"],
    )
    assert set(rows[0]["labels"]) == {"Element", "Math"}


async def test_update_and_delete(repo):
    node = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="a")
    await repo.update_node(node["uuid"], content="b", cleaned_at="2026-01-01")
    fetched = await repo.get_node(node["uuid"])
    assert fetched["content"] == "b"
    assert fetched["cleaned_at"] == "2026-01-01"

    await repo.delete_node(node["uuid"])
    assert await repo.get_node(node["uuid"]) is None


async def test_children_ordered_by_order_index(repo):
    source = await repo.create_node([NodeType.SOURCE], title="doc")
    seg = await repo.create_node([NodeType.SEGMENT], segment_index=1)
    await repo.link(EdgeType.CONTAINS, source["uuid"], seg["uuid"])

    # Insert out of order; order_index should drive ordering.
    e2 = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="two", order_index=2)
    e1 = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="one", order_index=1)
    e3 = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="three", order_index=3)
    await repo.link_children(seg["uuid"], [e2["uuid"], e1["uuid"], e3["uuid"]])

    ordered = await repo.children_ordered(seg["uuid"])
    assert [n["content"] for n in ordered] == ["one", "two", "three"]


async def test_link_chain(repo):
    a = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="a", order_index=1)
    b = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="b", order_index=2)
    c = await repo.create_node([NodeType.ELEMENT, NodeType.PARAGRAPH], content="c", order_index=3)
    await repo.link_chain([a["uuid"], b["uuid"], c["uuid"]], EdgeType.NEXT)

    rows = await repo.run(
        "MATCH (a {uuid: $a})-[:`Next`]->(b)-[:`Next`]->(c) "
        "RETURN a.content AS a, b.content AS b, c.content AS c",
        a=a["uuid"],
    )
    assert rows[0] == {"a": "a", "b": "b", "c": "c"}


async def test_source_elements_ordered(repo):
    for i, txt in enumerate(["x", "y", "z"], start=1):
        await repo.create_node(
            [NodeType.ELEMENT, NodeType.PARAGRAPH],
            content=txt, source_uuid="src-1", order_index=i,
        )
    # noise from another source
    await repo.create_node(
        [NodeType.ELEMENT, NodeType.PARAGRAPH], content="other", source_uuid="src-2", order_index=1
    )
    ordered = await repo.source_elements_ordered("src-1")
    assert [n["content"] for n in ordered] == ["x", "y", "z"]


async def test_vector_roundtrip(repo):
    # 8-dim vectors (matches conftest TEST_EMBED_DIMS)
    near = [1.0, 0, 0, 0, 0, 0, 0, 0]
    far = [0, 0, 0, 0, 0, 0, 0, 1.0]
    query = [0.9, 0.1, 0, 0, 0, 0, 0, 0]

    n1 = await repo.create_node([NodeType.BLOCK], kind="prose", content="near", embedding=near)
    await repo.create_node([NodeType.BLOCK], kind="prose", content="far", embedding=far)

    # Vector index population can lag a commit briefly; retry a few times.
    results: list = []
    for _ in range(10):
        results = await repo.vector_search(VECTOR_INDEX_NAME, query, k=2)
        if results:
            break
        await asyncio.sleep(0.5)

    assert results, "vector index returned no results"
    top_node, _score = results[0]
    assert top_node["uuid"] == n1["uuid"]
    assert top_node["content"] == "near"
