"""Thin DSPy wrapper: a stage LLM module with retries + timeout.

Kept deliberately simple (no judges/optimizers/training). Returns ``None`` on
failure or timeout — every stage treats ``None`` as "leave unchanged", so a flaky
call degrades to a no-op rather than corrupting the graph. DSPy is imported lazily
so importing this module does not require dspy to be installed.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from math_trainer.core.config import StageConfig

logger = logging.getLogger(__name__)


class DSPyModule:
    def __init__(self, config: StageConfig, signature: Any) -> None:
        import dspy

        self._lm = dspy.LM(
            model=config.model,
            api_base=config.api_base or None,
            api_key=config.api_key or None,
            num_retries=config.num_retries,
        )
        self._predictor = dspy.Predict(signature)
        self._timeout = config.timeout_s

    async def aforward(self, **kwargs: Any) -> Any | None:
        import dspy

        try:
            with dspy.context(lm=self._lm):
                return await asyncio.wait_for(
                    self._predictor.acall(**kwargs), timeout=self._timeout
                )
        except asyncio.TimeoutError:
            logger.warning("DSPy module timed out after %.0fs", self._timeout)
            return None
        except Exception as exc:  # noqa: BLE001 — degrade to no-op, never crash a stage
            logger.warning("DSPy module failed: %s", exc)
            return None


def load_dspy_image(path: str | None) -> Any | None:
    """Load an image path as a dspy.Image (for vision stages). None on failure."""
    if not path:
        return None
    try:
        import dspy

        return dspy.Image.from_file(path)
    except Exception:
        return None
