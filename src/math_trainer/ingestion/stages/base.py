"""Shared helpers for pipeline stages.

Each stage exposes ``async run(source_uuid)`` and is constructed with a repo, an
LLM ``module`` (duck-typed ``aforward(**kwargs) -> prediction``), and a StageConfig.
Keeping the LLM behind a small interface makes every stage testable with a fake
module against a real Neo4j, with no DSPy/model dependency in the test path.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Protocol, TypeVar

T = TypeVar("T")
R = TypeVar("R")


class LMModule(Protocol):
    async def aforward(self, **kwargs: Any) -> Any: ...


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def concurrent_map(
    items: list[T], fn: Callable[[T], Awaitable[R]], max_concurrent: int
) -> list[R]:
    """Run ``fn`` over ``items`` with a bounded concurrency semaphore."""
    if not items:
        return []
    sem = asyncio.Semaphore(max(1, max_concurrent))

    async def _run(item: T) -> R:
        async with sem:
            return await fn(item)

    return await asyncio.gather(*[_run(i) for i in items])


def neighbors(
    ordered: list[dict], index: int
) -> tuple[dict | None, dict | None]:
    prev = ordered[index - 1] if index > 0 else None
    nxt = ordered[index + 1] if index < len(ordered) - 1 else None
    return prev, nxt
