from __future__ import annotations

from dataclasses import dataclass

from ltm_design.memory.memorization import ConnectionProposal
from ltm_design.memory.models import Facets, MemoryCandidate, MemoryItem, MemoryRelation
from ltm_design.memory.store import MemoryStore


@dataclass(frozen=True)
class ConnectionSearchResult:
    candidate_id: str
    nearby_memories: tuple[MemoryItem, ...]
    proposed_links: tuple[ConnectionProposal, ...]


class ConnectionDiscovery:
    """Retrieval-like candidate connection discovery before main-model approval."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def prefilter_existing_memories(self, candidate: MemoryCandidate, *, limit: int = 12) -> tuple[MemoryItem, ...]:
        by_facets = self.store.match_facets(candidate.facets, limit=limit)
        if len(by_facets) >= limit:
            return tuple(by_facets[:limit])

        terms = self._query_from_candidate(candidate)
        by_fts = self.store.search_fts(terms, limit=limit) if terms else []
        seen = {item.mem_id for item in by_facets}
        merged = list(by_facets)
        for item in by_fts:
            if item.mem_id not in seen:
                merged.append(item)
                seen.add(item.mem_id)
            if len(merged) >= limit:
                break
        return tuple(merged)

    def heuristic_proposals(
        self,
        *,
        candidate_id: str,
        candidate: MemoryCandidate,
        existing: tuple[MemoryItem, ...],
    ) -> tuple[ConnectionProposal, ...]:
        proposals: list[ConnectionProposal] = []
        candidate_text = self._norm(candidate.text)
        for item in existing:
            if self._norm(item.text) == candidate_text:
                proposals.append(
                    ConnectionProposal(
                        candidate_id=candidate_id,
                        existing_mem_id=item.mem_id,
                        relation=MemoryRelation.SAME_AS.value,
                        reason="Candidate text exactly matches an existing memory.",
                    )
                )
            elif self._shares_facets(candidate.facets, item.facets):
                proposals.append(
                    ConnectionProposal(
                        candidate_id=candidate_id,
                        existing_mem_id=item.mem_id,
                        relation=MemoryRelation.ELABORATES.value,
                        reason="Candidate and existing memory share typed facets; model should verify whether it adds useful detail.",
                    )
                )
        return tuple(proposals)

    def _query_from_candidate(self, candidate: MemoryCandidate) -> str:
        values: list[str] = [candidate.text]
        for facet_values in candidate.facets.to_dict().values():
            values.extend(facet_values)
        return " OR ".join(value for value in values if value.strip())[:512]

    def _shares_facets(self, left: Facets, right: Facets) -> bool:
        for key, left_values in left.to_dict().items():
            right_values = right.to_dict().get(key, [])
            if {self._norm(value) for value in left_values}.intersection(
                self._norm(value) for value in right_values
            ):
                return True
        return False

    def _norm(self, value: str) -> str:
        return " ".join(value.casefold().split())
