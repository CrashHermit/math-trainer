"""Text embedding provider (integration seam).

A thin async wrapper over litellm's embedding API. Kept behind a small interface
so the Embedder stage can be tested with a fake embedder and no model calls.
"""

from typing import Protocol

from math_trainer.core.config import EmbeddingConfig


class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


class LiteLLMEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    async def embed(self, text: str) -> list[float]:
        import litellm

        response = await litellm.aembedding(
            model=self._config.model,
            input=[text],
            api_base=self._config.api_base or None,
            api_key=self._config.api_key or None,
        )
        return list(response["data"][0]["embedding"])
