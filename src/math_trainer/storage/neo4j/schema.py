"""Schema bootstrap: uniqueness constraints, range indexes, and the vector index.

Idempotent — safe to run on every startup / via `math-trainer init-db`.
"""

from math_trainer.core.config import EmbeddingConfig
from math_trainer.core.model.types import NodeType
from math_trainer.storage.neo4j.repository import GraphRepository

# The vector index lives on :Block — the semantic unit is the embed/retrieval unit.
VECTOR_INDEX_NAME = "block_embedding"


class Schema:
    def __init__(self, repo: GraphRepository, embedding: EmbeddingConfig) -> None:
        self._repo = repo
        self._embedding = embedding

    async def bootstrap(self) -> None:
        await self._constraints()
        await self._indexes()
        await self._vector_index()

    async def _constraints(self) -> None:
        for label in (NodeType.SOURCE, NodeType.SEGMENT, NodeType.ELEMENT, NodeType.BLOCK):
            await self._repo.execute(
                f"CREATE CONSTRAINT {label.value.lower()}_uuid IF NOT EXISTS "
                f"FOR (n:`{label.value}`) REQUIRE n.uuid IS UNIQUE"
            )

    async def _indexes(self) -> None:
        await self._repo.execute(
            "CREATE INDEX element_source_uuid IF NOT EXISTS "
            "FOR (n:`Element`) ON (n.source_uuid)"
        )
        await self._repo.execute(
            "CREATE INDEX block_source_uuid IF NOT EXISTS "
            "FOR (n:`Block`) ON (n.source_uuid)"
        )
        await self._repo.execute(
            "CREATE INDEX segment_index IF NOT EXISTS "
            "FOR (n:`Segment`) ON (n.segment_index)"
        )

    async def _vector_index(self) -> None:
        await self._repo.execute(
            f"CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS "
            "FOR (n:`Block`) ON (n.embedding) "
            "OPTIONS { indexConfig: { "
            "`vector.dimensions`: $dims, "
            "`vector.similarity_function`: $sim } }",
            dims=self._embedding.dimensions,
            sim=self._embedding.similarity,
        )
