"""Neo4j async driver lifecycle.

The official driver is natively concurrent, so there is no single-thread executor
here (unlike the ArcadeDB-embedded design this project is modeled on). A single
shared ``AsyncDriver`` is used for the process; sessions are cheap and created
per unit of work.
"""

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession

from math_trainer.core.config import DatabaseConfig


class Neo4jDriver:
    """Owns a shared AsyncDriver and hands out sessions bound to the database."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._config = config
        self._driver: AsyncDriver | None = None

    @property
    def driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
            )
        return self._driver

    def session(self, **kwargs: object) -> AsyncSession:
        return self.driver.session(database=self._config.database, **kwargs)

    async def verify(self) -> None:
        await self.driver.verify_connectivity()

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
