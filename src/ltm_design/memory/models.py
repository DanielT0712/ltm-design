from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def datetime_to_str(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def datetime_from_str(value: str) -> datetime:
    return datetime.fromisoformat(value)


class MemoryRelation(StrEnum):
    UPDATES = "updates"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    ELABORATES = "elaborates"
    CAUSED_BY = "caused_by"
    PART_OF = "part_of"
    SAME_AS = "same_as"


class CandidateStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


class FragmentKind(StrEnum):
    RECALL = "recall"
    CONNECTION = "connection"


@dataclass(frozen=True)
class Facets:
    people: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    emotions: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    places: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    times: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "Facets":
        if not value:
            return cls()
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: tuple(value.get(key, ()) or ()) for key in allowed})

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "people": list(self.people),
            "topics": list(self.topics),
            "emotions": list(self.emotions),
            "events": list(self.events),
            "places": list(self.places),
            "objects": list(self.objects),
            "times": list(self.times),
        }

    def lexical_text(self) -> str:
        parts: list[str] = []
        for key, values in self.to_dict().items():
            if values:
                parts.append(f"facets.{key}: {' '.join(values)}")
        return "\n".join(parts)


@dataclass(frozen=True)
class EvidenceRef:
    source_id: str
    path: str | None = None
    locator: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EvidenceRef":
        return cls(
            source_id=value["source_id"],
            path=value.get("path"),
            locator=dict(value.get("locator") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "path": self.path,
            "locator": self.locator,
        }


@dataclass(frozen=True)
class SourceSpan:
    source_id: str
    message_id: str | None = None
    event_id: str | None = None
    speaker: str = "unknown"
    char_start: int | None = None
    char_end: int | None = None
    matched_text: str = ""

    def to_evidence_ref(self) -> EvidenceRef:
        return EvidenceRef(
            source_id=self.source_id,
            locator={
                "message_id": self.message_id,
                "event_id": self.event_id,
                "speaker": self.speaker,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "matched_text": self.matched_text,
            },
        )


@dataclass(frozen=True)
class SourceAnchorRef:
    source_id: str
    start_anchor: str | None = None
    end_anchor: str | None = None
    message_id: str | None = None
    event_id: str | None = None
    speaker: str = "unknown"
    whole_source: bool = False


@dataclass(frozen=True)
class EvidenceRecord:
    source_id: str
    text: str
    conversation_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryLink:
    to_mem_id: str
    relation: MemoryRelation

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "MemoryLink":
        return cls(to_mem_id=value["to_mem_id"], relation=MemoryRelation(value["relation"]))

    def to_dict(self) -> dict[str, str]:
        return {"to_mem_id": self.to_mem_id, "relation": self.relation.value}


@dataclass(frozen=True)
class MemoryCandidate:
    text: str
    facets: Facets = field(default_factory=Facets)
    evidence: tuple[EvidenceRef, ...] = ()
    links: tuple[MemoryLink, ...] = ()
    title: str | None = None


@dataclass(frozen=True)
class PendingMemoryCandidate:
    candidate_id: str
    text: str
    facets: Facets
    evidence: tuple[EvidenceRef, ...]
    title: str | None
    status: CandidateStatus
    trigger: str
    created_at: datetime


@dataclass(frozen=True)
class MemoryItem:
    mem_id: str
    text: str
    facets: Facets
    evidence: tuple[EvidenceRef, ...]
    title: str | None
    semantic_text: str
    lexical_text: str
    created_at: datetime
    updated_at: datetime
    reference_count: int = 0


@dataclass(frozen=True)
class MemoryFragment:
    fragment_id: str
    kind: FragmentKind
    mem_ids: tuple[str, ...]
    title: str
    text: str
    token_count_estimate: int
    created_at: datetime
    updated_at: datetime


MemorizationTrigger = Literal["active_model", "auto_idle"]
