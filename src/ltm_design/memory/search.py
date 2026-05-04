from __future__ import annotations

from dataclasses import dataclass, field

from ltm_design.memory.embeddings import EmbeddingProvider, cosine_similarity
from ltm_design.memory.models import Facets, MemoryItem
from ltm_design.memory.store import MemoryStore


@dataclass(frozen=True)
class CandidateMemory:
    mem_id: str
    channels: tuple[str, ...]
    channel_evidence: dict
    reference_count: int


@dataclass(frozen=True)
class MemorySearchResult:
    candidates: tuple[CandidateMemory, ...]


class MemoryIndexer:
    def __init__(self, store: MemoryStore, embeddings: EmbeddingProvider) -> None:
        self.store = store
        self.embeddings = embeddings

    def index_item(self, item: MemoryItem) -> None:
        vector = self.embeddings.embed_documents([item.semantic_text])[0]
        self.store.store_embedding(
            item.mem_id,
            model=self.embeddings.model_name,
            dimensions=self.embeddings.dimensions,
            vector=vector,
        )

    def index_all(self) -> None:
        for item in self.store.list_items():
            self.index_item(item)


class MemorySearch:
    def __init__(self, store: MemoryStore, embeddings: EmbeddingProvider) -> None:
        self.store = store
        self.embeddings = embeddings

    def search(
        self,
        *,
        text: str,
        facets: Facets | None = None,
        vector_top_k: int = 20,
        fts_top_k: int = 20,
        facet_top_k: int = 20,
    ) -> MemorySearchResult:
        merged: dict[str, CandidateMemory] = {}

        query_vector = self.embeddings.embed_query(text)
        vector_hits = sorted(
            (
                (mem_id, cosine_similarity(query_vector, vector))
                for mem_id, vector in self.store.list_embeddings(model=self.embeddings.model_name)
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:vector_top_k]
        for mem_id, score in vector_hits:
            self._merge(merged, mem_id, "vector", {"similarity": score})

        if text.strip():
            try:
                for item in self.store.search_fts(self._fts_query(text), limit=fts_top_k):
                    self._merge(merged, item.mem_id, "full_text", {"matched": text})
            except Exception:
                # FTS query syntax is intentionally best-effort for the harness.
                pass

        if facets is not None:
            for item in self.store.match_facets(facets, limit=facet_top_k):
                self._merge(merged, item.mem_id, "facet", {"matched_facets": facets.to_dict()})

        ordered = sorted(
            merged.values(),
            key=lambda candidate: (
                len(candidate.channels),
                candidate.channel_evidence.get("vector", {}).get("similarity", 0.0),
                candidate.reference_count,
            ),
            reverse=True,
        )
        return MemorySearchResult(candidates=tuple(ordered))

    def _merge(self, merged: dict[str, CandidateMemory], mem_id: str, channel: str, evidence: dict) -> None:
        item = self.store.get(mem_id)
        if item is None:
            return
        existing = merged.get(mem_id)
        if existing is None:
            merged[mem_id] = CandidateMemory(
                mem_id=mem_id,
                channels=(channel,),
                channel_evidence={channel: evidence},
                reference_count=item.reference_count,
            )
            return
        if channel in existing.channels:
            return
        merged[mem_id] = CandidateMemory(
            mem_id=mem_id,
            channels=(*existing.channels, channel),
            channel_evidence={**existing.channel_evidence, channel: evidence},
            reference_count=existing.reference_count,
        )

    def _fts_query(self, text: str) -> str:
        terms = [term for term in text.replace('"', " ").split() if term]
        return " OR ".join(terms[:16])
