from __future__ import annotations

import pytest
import pytest_asyncio
from testcontainers.neo4j import Neo4jContainer

from math_trainer.core.config import DatabaseConfig, EmbeddingConfig
from math_trainer.storage.neo4j.driver import Neo4jDriver
from math_trainer.storage.neo4j.repository import GraphRepository
from math_trainer.storage.neo4j.schema import Schema

# Small embedding dimension keeps vector-index tests fast.
TEST_EMBED_DIMS = 8


@pytest.fixture(scope="session")
def neo4j_container():
    with Neo4jContainer("neo4j:5.26-community") as container:
        yield container


@pytest.fixture(scope="session")
def db_config(neo4j_container) -> DatabaseConfig:
    return DatabaseConfig(
        uri=neo4j_container.get_connection_url(),
        user="neo4j",
        password=neo4j_container.password,
        database="neo4j",
    )


@pytest_asyncio.fixture
async def repo(db_config) -> GraphRepository:
    driver = Neo4jDriver(db_config)
    repo = GraphRepository(driver)
    await repo.execute("MATCH (n) DETACH DELETE n")
    schema = Schema(repo, EmbeddingConfig(dimensions=TEST_EMBED_DIMS, similarity="cosine"))
    await schema.bootstrap()
    yield repo
    await repo.execute("MATCH (n) DETACH DELETE n")
    await driver.close()
