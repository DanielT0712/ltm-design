from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from ltm_design.memory.models import EvidenceRef, Facets, MemoryCandidate, MemoryLink, MemorizationTrigger
from ltm_design.memory.store import MemoryStore


@dataclass(frozen=True)
class MemorizationRequest:
    text: str
    facets: Facets = Facets()
    evidence: tuple[EvidenceRef, ...] = ()
    trigger: MemorizationTrigger = "active_model"


@dataclass(frozen=True)
class MemorizationResult:
    mem_ids: tuple[str, ...]
    trigger: MemorizationTrigger


@dataclass(frozen=True)
class ProposedMemoryResult:
    candidate_ids: tuple[str, ...]
    trigger: MemorizationTrigger


@dataclass(frozen=True)
class ConnectionProposal:
    candidate_id: str
    existing_mem_id: str
    relation: str
    reason: str


@dataclass(frozen=True)
class ExtractedMemory:
    text: str
    facets: Facets = Facets()
    evidence: tuple[EvidenceRef, ...] = ()
    connection_hints: tuple[str, ...] = ()


@dataclass(frozen=True)
class MainModelMemoryDecision:
    approve_candidate_ids: tuple[str, ...] = ()
    reject_candidate_ids: tuple[str, ...] = ()
    approve_links: tuple[tuple[str, MemoryLink], ...] = ()


class MemoryWriter:
    """Shared writer used by active and automatic memorization."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def memorize(self, candidates: Iterable[MemoryCandidate]) -> tuple[str, ...]:
        mem_ids: list[str] = []
        for candidate in candidates:
            if not candidate.text.strip():
                continue
            item = self.store.remember(candidate)
            mem_ids.append(item.mem_id)
        return tuple(mem_ids)


class ActiveMemorizationSkill:
    """Callable skill surface for an agent that decides something should be remembered."""

    name = "active_memorization"
    description = (
        "Store concise user/context-specific memories as text items with facets and evidence. "
        "Do not store generic world knowledge or filler."
    )

    def __init__(self, writer: MemoryWriter) -> None:
        self.writer = writer

    def call(self, request: MemorizationRequest) -> MemorizationResult:
        candidate = MemoryCandidate(
            text=request.text,
            facets=request.facets,
            evidence=request.evidence,
        )
        return self.call_candidate(candidate, trigger=request.trigger)

    def call_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        trigger: MemorizationTrigger = "active_model",
    ) -> MemorizationResult:
        candidate = MemoryCandidate(
            text=candidate.text,
            facets=candidate.facets,
            evidence=candidate.evidence,
            links=candidate.links,
            title=candidate.title,
        )
        return MemorizationResult(
            mem_ids=self.writer.memorize((candidate,)),
            trigger=trigger,
        )

    def propose_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        trigger: MemorizationTrigger = "active_model",
    ) -> ProposedMemoryResult:
        candidate_id = self.writer.store.propose_candidate(candidate, trigger=trigger)
        return ProposedMemoryResult(candidate_ids=(candidate_id,), trigger=trigger)

    def apply_main_model_decision(self, decision: MainModelMemoryDecision) -> tuple[str, ...]:
        mem_ids: list[str] = []
        for candidate_id in decision.approve_candidate_ids:
            item = self.writer.store.approve_candidate(candidate_id)
            mem_ids.append(item.mem_id)
        for candidate_id in decision.reject_candidate_ids:
            self.writer.store.reject_candidate(candidate_id)
        for from_mem_id, link in decision.approve_links:
            if link.relation.value == "same_as":
                continue
            self.writer.store.add_links(from_mem_id, (link,))
        for mem_id in mem_ids:
            self.writer.store.assign_to_recall_fragment(mem_id)
        return tuple(mem_ids)


class IdleAutoMemorizationTrigger:
    """Triggers extraction after a quiet period and surfaces candidates for approval."""

    def __init__(
        self,
        skill: ActiveMemorizationSkill,
        *,
        idle_after: timedelta = timedelta(minutes=20),
    ) -> None:
        self.skill = skill
        self.idle_after = idle_after

    def should_run(self, *, last_interaction_at: datetime, now: datetime) -> bool:
        return now - last_interaction_at >= self.idle_after

    def run(
        self,
        *,
        last_interaction_at: datetime,
        now: datetime,
        candidates: Iterable[MemoryCandidate],
    ) -> ProposedMemoryResult | None:
        if not self.should_run(last_interaction_at=last_interaction_at, now=now):
            return None
        candidate_ids: list[str] = []
        for candidate in candidates:
            result = self.skill.propose_candidate(candidate, trigger="auto_idle")
            candidate_ids.extend(result.candidate_ids)
        return ProposedMemoryResult(
            candidate_ids=tuple(candidate_ids),
            trigger="auto_idle",
        )
