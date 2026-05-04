from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class QwenEmbeddingProvider:
    """Qwen dense embeddings for production retrieval."""

    model_name: str = "Qwen/Qwen3-Embedding-8B"
    dimensions: int = 4096

    @cached_property
    def model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "QwenEmbeddingProvider requires the embeddings extra: "
                "`uv sync --extra embeddings` or `uv run --extra embeddings ...`"
            ) from exc
        return SentenceTransformer(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(f"query: {text}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def _encode(self, text: str) -> list[float]:
        vector = self.model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector.tolist()]

def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimensionality")
    return sum(a * b for a, b in zip(left, right, strict=True))
