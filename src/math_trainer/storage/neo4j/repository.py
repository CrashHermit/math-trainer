"""Generic async graph repository over Neo4j (pure Cypher).

Node identity is the external ``uuid`` property (unique on :Element/:Source/:Segment).
Nodes are returned as plain property dicts; callers read ``node["uuid"]`` etc. Labels
cannot be parametrized in Cypher, so every label is validated against NodeType before
being interpolated.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from math_trainer.core.model.types import EdgeType, NodeType
from math_trainer.storage.neo4j.driver import Neo4jDriver

_VALID_LABELS = {t.value for t in NodeType}
_VALID_EDGES = {t.value for t in EdgeType}


def _labels_clause(labels: list[NodeType | str]) -> str:
    parts: list[str] = []
    for label in labels:
        value = label.value if isinstance(label, NodeType) else str(label)
        if value not in _VALID_LABELS:
            raise ValueError(f"Unknown node label: {value!r}")
        parts.append(f"`{value}`")
    return ":".join(parts)


def _edge_label(edge: EdgeType | str) -> str:
    value = edge.value if isinstance(edge, EdgeType) else str(edge)
    if value not in _VALID_EDGES:
        raise ValueError(f"Unknown edge type: {value!r}")
    return f"`{value}`"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GraphRepository:
    def __init__(self, driver: Neo4jDriver) -> None:
        self._driver = driver

    # ── raw query helpers ──────────────────────────────────────────────────
    async def run(self, cypher: str, **params: object) -> list[dict]:
        """Read query → list of row dicts."""
        async with self._driver.session() as session:
            result = await session.run(cypher, **params)
            return [record.data() async for record in result]

    async def execute(self, cypher: str, **params: object) -> None:
        """Write query in a managed transaction."""
        async with self._driver.session() as session:
            await session.execute_write(lambda tx: tx.run(cypher, **params))

    # ── node CRUD ──────────────────────────────────────────────────────────
    async def create_node(
        self, labels: list[NodeType | str], **props: object
    ) -> dict:
        node_uuid = str(props.pop("uuid", None) or uuid4())
        now = _now()
        props = {k: v for k, v in props.items() if v is not None}
        rows = await self._create_node(labels, node_uuid, now, props)
        return rows[0]["n"]

    async def _create_node(
        self, labels: list[NodeType | str], node_uuid: str, now: datetime, props: dict
    ) -> list[dict]:
        clause = _labels_clause(labels)
        async with self._driver.session() as session:
            async def _tx(tx):
                result = await tx.run(
                    f"CREATE (n:{clause}) "
                    "SET n += $props, n.uuid = $uuid, "
                    "n.created_at = $now, n.updated_at = $now "
                    "RETURN n",
                    props=props, uuid=node_uuid, now=now,
                )
                return [r.data() async for r in result]
            return await session.execute_write(_tx)

    async def get_node(self, uuid: str, label: NodeType | str | None = None) -> dict | None:
        lbl = f":{_labels_clause([label])}" if label else ""
        rows = await self.run(
            f"MATCH (n{lbl} {{uuid: $uuid}}) RETURN n LIMIT 1", uuid=uuid
        )
        return rows[0]["n"] if rows else None

    async def update_node(self, uuid: str, **props: object) -> None:
        props = {k: v for k, v in props.items() if v is not None}
        await self.execute(
            "MATCH (n {uuid: $uuid}) SET n += $props, n.updated_at = $now",
            uuid=uuid, props=props, now=_now(),
        )

    async def delete_node(self, uuid: str) -> None:
        await self.execute("MATCH (n {uuid: $uuid}) DETACH DELETE n", uuid=uuid)

    # ── edges ──────────────────────────────────────────────────────────────
    async def link(
        self, edge: EdgeType, from_uuid: str, to_uuid: str, **props: object
    ) -> None:
        rel = _edge_label(edge)
        await self.execute(
            "MATCH (a {uuid: $from_uuid}), (b {uuid: $to_uuid}) "
            f"MERGE (a)-[r:{rel}]->(b) SET r += $props",
            from_uuid=from_uuid, to_uuid=to_uuid, props=props,
        )

    async def unlink(self, edge: EdgeType, from_uuid: str, to_uuid: str) -> None:
        rel = _edge_label(edge)
        await self.execute(
            f"MATCH (a {{uuid: $from_uuid}})-[r:{rel}]->(b {{uuid: $to_uuid}}) DELETE r",
            from_uuid=from_uuid, to_uuid=to_uuid,
        )

    async def link_children(
        self, parent_uuid: str, child_uuids: list[str], edge: EdgeType = EdgeType.CONTAINS
    ) -> None:
        if not child_uuids:
            return
        rel = _edge_label(edge)
        await self.execute(
            "MATCH (p {uuid: $parent_uuid}) "
            "UNWIND $child_uuids AS cu MATCH (c {uuid: cu}) "
            f"MERGE (p)-[:{rel}]->(c)",
            parent_uuid=parent_uuid, child_uuids=child_uuids,
        )

    async def link_chain(
        self, uuids: list[str], edge: EdgeType = EdgeType.NEXT
    ) -> None:
        """Link consecutive uuids: uuids[i] -[edge]-> uuids[i+1]."""
        if len(uuids) < 2:
            return
        rel = _edge_label(edge)
        pairs = list(zip(uuids, uuids[1:]))
        await self.execute(
            "UNWIND $pairs AS pair "
            "MATCH (a {uuid: pair[0]}), (b {uuid: pair[1]}) "
            f"MERGE (a)-[:{rel}]->(b)",
            pairs=[list(p) for p in pairs],
        )

    # ── traversal ──────────────────────────────────────────────────────────
    async def children_ordered(
        self, parent_uuid: str, edge: EdgeType = EdgeType.CONTAINS
    ) -> list[dict]:
        """Members of parent via `edge`, ordered by reading order.

        Selection is by membership (authoritative); ordering uses order_index
        with segment_index as a fallback so a broken Next chain degrades order,
        never completeness.
        """
        rel = _edge_label(edge)
        rows = await self.run(
            f"MATCH (p {{uuid: $parent_uuid}})-[:{rel}]->(c) "
            "RETURN c ORDER BY coalesce(c.order_index, c.segment_index, 0)",
            parent_uuid=parent_uuid,
        )
        return [r["c"] for r in rows]

    async def source_elements_ordered(self, source_uuid: str) -> list[dict]:
        """All Element nodes for a source in reading order (membership-based)."""
        rows = await self.run(
            "MATCH (e:`Element` {source_uuid: $source_uuid}) "
            "RETURN e ORDER BY coalesce(e.order_index, 0)",
            source_uuid=source_uuid,
        )
        return [r["e"] for r in rows]

    # ── vector search ──────────────────────────────────────────────────────
    async def vector_search(
        self, index_name: str, embedding: list[float], k: int
    ) -> list[tuple[dict, float]]:
        rows = await self.run(
            "CALL db.index.vector.queryNodes($index_name, $k, $embedding) "
            "YIELD node, score RETURN node, score",
            index_name=index_name, k=k, embedding=embedding,
        )
        return [(r["node"], r["score"]) for r in rows]
