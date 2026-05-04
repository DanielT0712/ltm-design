from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from ltm_design.memory.models import (
    CandidateStatus,
    EvidenceRecord,
    EvidenceRef,
    Facets,
    FragmentKind,
    MemoryCandidate,
    MemoryFragment,
    MemoryItem,
    MemoryLink,
    MemoryRelation,
    datetime_from_str,
    datetime_to_str,
    utc_now,
)


class MemoryStore:
    """SQLite-backed canonical store for text memory items."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.migrate()

    def close(self) -> None:
        self._conn.close()

    def migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memory_items (
              mem_id TEXT PRIMARY KEY,
              title TEXT,
              text TEXT NOT NULL,
              facets_json TEXT NOT NULL,
              evidence_json TEXT NOT NULL,
              semantic_text TEXT NOT NULL,
              lexical_text TEXT NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              reference_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS evidence_records (
              source_id TEXT PRIMARY KEY,
              conversation_id TEXT,
              text TEXT NOT NULL,
              metadata_json TEXT NOT NULL,
              created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS pending_memory_candidates (
              candidate_id TEXT PRIMARY KEY,
              title TEXT,
              text TEXT NOT NULL,
              facets_json TEXT NOT NULL,
              evidence_json TEXT NOT NULL,
              status TEXT NOT NULL CHECK (status IN ('proposed', 'approved', 'rejected')),
              trigger TEXT NOT NULL,
              created_at TEXT NOT NULL,
              decided_at TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_links (
              from_mem_id TEXT NOT NULL,
              to_mem_id TEXT NOT NULL,
              relation TEXT NOT NULL CHECK (
                relation IN ('updates', 'contradicts', 'supports', 'elaborates', 'caused_by', 'part_of', 'same_as')
              ),
              created_at TEXT NOT NULL,
              PRIMARY KEY (from_mem_id, to_mem_id, relation),
              FOREIGN KEY (from_mem_id) REFERENCES memory_items(mem_id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_items_fts USING fts5(
              mem_id UNINDEXED,
              title,
              lexical_text,
              facets,
              tokenize = 'unicode61 remove_diacritics 2'
            );

            CREATE TABLE IF NOT EXISTS memory_fragments (
              fragment_id TEXT PRIMARY KEY,
              kind TEXT NOT NULL CHECK (kind IN ('recall', 'connection')),
              title TEXT NOT NULL,
              mem_ids_json TEXT NOT NULL,
              text TEXT NOT NULL,
              token_count_estimate INTEGER NOT NULL,
              created_at TEXT NOT NULL,
              updated_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    def add_evidence(self, record: EvidenceRecord) -> EvidenceRecord:
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO evidence_records (
                  source_id, conversation_id, text, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record.source_id,
                    record.conversation_id,
                    record.text,
                    json.dumps(record.metadata, sort_keys=True),
                    datetime_to_str(record.created_at),
                ),
            )
        return record

    def get_evidence(self, source_id: str) -> EvidenceRecord | None:
        row = self._conn.execute(
            "SELECT * FROM evidence_records WHERE source_id = ?",
            (source_id,),
        ).fetchone()
        if not row:
            return None
        return EvidenceRecord(
            source_id=row["source_id"],
            conversation_id=row["conversation_id"],
            text=row["text"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime_from_str(row["created_at"]),
        )

    def propose_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        trigger: str,
        now: datetime | None = None,
    ) -> str:
        now = now or utc_now()
        candidate_id = "cand_" + self._candidate_digest(candidate)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO pending_memory_candidates (
                  candidate_id, title, text, facets_json, evidence_json, status, trigger, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate_id,
                    candidate.title or self._title_for(candidate.text),
                    candidate.text,
                    json.dumps(candidate.facets.to_dict(), sort_keys=True),
                    json.dumps([ref.to_dict() for ref in candidate.evidence], sort_keys=True),
                    CandidateStatus.PROPOSED.value,
                    trigger,
                    datetime_to_str(now),
                ),
            )
        return candidate_id

    def approve_candidate(self, candidate_id: str) -> MemoryItem:
        row = self._conn.execute(
            "SELECT * FROM pending_memory_candidates WHERE candidate_id = ?",
            (candidate_id,),
        ).fetchone()
        if not row:
            raise KeyError(candidate_id)
        candidate = MemoryCandidate(
            text=row["text"],
            facets=Facets.from_mapping(json.loads(row["facets_json"])),
            evidence=tuple(EvidenceRef.from_mapping(value) for value in json.loads(row["evidence_json"])),
            title=row["title"],
        )
        item = self.remember(candidate)
        with self._conn:
            self._conn.execute(
                "UPDATE pending_memory_candidates SET status = ?, decided_at = ? WHERE candidate_id = ?",
                (CandidateStatus.APPROVED.value, datetime_to_str(utc_now()), candidate_id),
            )
        return item

    def reject_candidate(self, candidate_id: str) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "UPDATE pending_memory_candidates SET status = ?, decided_at = ? WHERE candidate_id = ?",
                (CandidateStatus.REJECTED.value, datetime_to_str(utc_now()), candidate_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(candidate_id)

    def remember(self, candidate: MemoryCandidate, *, now: datetime | None = None) -> MemoryItem:
        now = now or utc_now()
        mem_id = self._mem_id(candidate)
        existing = self.get(mem_id)
        if existing:
            return existing

        title = candidate.title or self._title_for(candidate.text)
        semantic_text = self._semantic_text(candidate)
        lexical_text = self._lexical_text(title, candidate)
        evidence_json = json.dumps([ref.to_dict() for ref in candidate.evidence], sort_keys=True)
        facets_json = json.dumps(candidate.facets.to_dict(), sort_keys=True)
        created_at = datetime_to_str(now)

        with self._conn:
            self._conn.execute(
                """
                INSERT INTO memory_items (
                  mem_id, title, text, facets_json, evidence_json, semantic_text,
                  lexical_text, created_at, updated_at, reference_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    mem_id,
                    title,
                    candidate.text,
                    facets_json,
                    evidence_json,
                    semantic_text,
                    lexical_text,
                    created_at,
                    created_at,
                ),
            )
            self._upsert_fts(mem_id, title, lexical_text, candidate.facets)
            self.add_links(mem_id, candidate.links, now=now)

        item = self.get(mem_id)
        if item is None:
            raise RuntimeError(f"memory insert did not persist: {mem_id}")
        return item

    def get(self, mem_id: str) -> MemoryItem | None:
        row = self._conn.execute(
            "SELECT * FROM memory_items WHERE mem_id = ?",
            (mem_id,),
        ).fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    def list_items(self) -> list[MemoryItem]:
        rows = self._conn.execute(
            "SELECT * FROM memory_items ORDER BY created_at, mem_id"
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def search_fts(self, query: str, *, limit: int = 20) -> list[MemoryItem]:
        rows = self._conn.execute(
            """
            SELECT mi.*
            FROM memory_items_fts fts
            JOIN memory_items mi ON mi.mem_id = fts.mem_id
            WHERE memory_items_fts MATCH ?
            ORDER BY bm25(memory_items_fts)
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        return [self._row_to_item(row) for row in rows]

    def match_facets(self, facets: Facets, *, limit: int = 50) -> list[MemoryItem]:
        wanted = {
            key: {self._norm(value) for value in values}
            for key, values in facets.to_dict().items()
            if values
        }
        if not wanted:
            return []

        matches: list[MemoryItem] = []
        for item in self.list_items():
            item_facets = item.facets.to_dict()
            for key, values in wanted.items():
                existing = {self._norm(value) for value in item_facets.get(key, [])}
                if existing.intersection(values):
                    matches.append(item)
                    break
            if len(matches) >= limit:
                break
        return matches

    def add_links(
        self,
        from_mem_id: str,
        links: Iterable[MemoryLink],
        *,
        now: datetime | None = None,
    ) -> None:
        now_str = datetime_to_str(now or utc_now())
        with self._conn:
            for link in links:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_links (from_mem_id, to_mem_id, relation, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (from_mem_id, link.to_mem_id, link.relation.value, now_str),
                )

    def links_for(self, mem_id: str) -> list[MemoryLink]:
        rows = self._conn.execute(
            "SELECT to_mem_id, relation FROM memory_links WHERE from_mem_id = ? ORDER BY relation, to_mem_id",
            (mem_id,),
        ).fetchall()
        return [
            MemoryLink(to_mem_id=row["to_mem_id"], relation=MemoryRelation(row["relation"]))
            for row in rows
        ]

    def connected_mem_ids(self, mem_id: str, *, relation_depth: int = 2) -> tuple[str, ...]:
        if relation_depth < 1:
            return ()
        seen: set[str] = {mem_id}
        frontier: set[str] = {mem_id}
        connected: list[str] = []
        for _ in range(relation_depth):
            next_frontier: set[str] = set()
            for current_mem_id in frontier:
                for link in self.links_for(current_mem_id):
                    if link.to_mem_id in seen:
                        continue
                    seen.add(link.to_mem_id)
                    next_frontier.add(link.to_mem_id)
                    connected.append(link.to_mem_id)
            frontier = next_frontier
            if not frontier:
                break
        return tuple(connected)

    def record_reference(self, mem_id: str, *, amount: int = 1) -> None:
        if amount < 1:
            raise ValueError("amount must be positive")
        with self._conn:
            cursor = self._conn.execute(
                """
                UPDATE memory_items
                SET reference_count = reference_count + ?, updated_at = ?
                WHERE mem_id = ?
                """,
                (amount, datetime_to_str(utc_now()), mem_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(mem_id)

    def record_final_acceptance(self, mem_ids: Iterable[str]) -> None:
        """Track memories accepted by the final main conversation model."""
        self.record_references(mem_ids)

    def record_references(self, mem_ids: Iterable[str]) -> None:
        for mem_id in mem_ids:
            self.record_reference(mem_id)

    def build_fragment(
        self,
        *,
        kind: FragmentKind,
        mem_ids: Iterable[str],
        title: str,
        now: datetime | None = None,
    ) -> MemoryFragment:
        now = now or utc_now()
        mem_id_tuple = tuple(mem_ids)
        items = [self.get(mem_id) for mem_id in mem_id_tuple]
        if any(item is None for item in items):
            missing = [mem_id for mem_id, item in zip(mem_id_tuple, items, strict=True) if item is None]
            raise KeyError(f"unknown mem ids: {missing}")
        text = self._fragment_text([item for item in items if item is not None])
        fragment_id = "frag_" + hashlib.sha256(
            json.dumps({"kind": kind.value, "mem_ids": mem_id_tuple}, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]
        created_at = datetime_to_str(now)
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            kind=kind,
            mem_ids=mem_id_tuple,
            title=title,
            text=text,
            token_count_estimate=max(1, len(text.split())),
            created_at=now,
            updated_at=now,
        )
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_fragments (
                  fragment_id, kind, title, mem_ids_json, text, token_count_estimate, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fragment.fragment_id,
                    fragment.kind.value,
                    fragment.title,
                    json.dumps(list(fragment.mem_ids)),
                    fragment.text,
                    fragment.token_count_estimate,
                    created_at,
                    created_at,
                ),
            )
        return fragment

    def _upsert_fts(self, mem_id: str, title: str, lexical_text: str, facets: Facets) -> None:
        self._conn.execute("DELETE FROM memory_items_fts WHERE mem_id = ?", (mem_id,))
        self._conn.execute(
            """
            INSERT INTO memory_items_fts (mem_id, title, lexical_text, facets)
            VALUES (?, ?, ?, ?)
            """,
            (mem_id, title, lexical_text, facets.lexical_text()),
        )

    def _row_to_item(self, row: sqlite3.Row) -> MemoryItem:
        evidence = tuple(
            EvidenceRef.from_mapping(value)
            for value in json.loads(row["evidence_json"])
        )
        return MemoryItem(
            mem_id=row["mem_id"],
            text=row["text"],
            facets=Facets.from_mapping(json.loads(row["facets_json"])),
            evidence=evidence,
            title=row["title"],
            semantic_text=row["semantic_text"],
            lexical_text=row["lexical_text"],
            created_at=datetime_from_str(row["created_at"]),
            updated_at=datetime_from_str(row["updated_at"]),
            reference_count=row["reference_count"],
        )

    def _mem_id(self, candidate: MemoryCandidate) -> str:
        return "mem_" + self._candidate_digest(candidate)

    def _candidate_digest(self, candidate: MemoryCandidate) -> str:
        payload = json.dumps(
            {
                "text": candidate.text.strip(),
                "facets": candidate.facets.to_dict(),
                "evidence": [ref.to_dict() for ref in candidate.evidence],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _semantic_text(self, candidate: MemoryCandidate) -> str:
        return candidate.text.strip()

    def _lexical_text(self, title: str, candidate: MemoryCandidate) -> str:
        sections = [
            f"title: {title}",
            candidate.facets.lexical_text(),
            f"body: {candidate.text.strip()}",
        ]
        return "\n".join(section for section in sections if section)

    def _title_for(self, text: str) -> str:
        stripped = " ".join(text.strip().split())
        return stripped[:80]

    def _norm(self, value: str) -> str:
        return " ".join(value.casefold().split())

    def _fragment_text(self, items: list[MemoryItem]) -> str:
        sections: list[str] = []
        for item in items:
            links = self.links_for(item.mem_id)
            sections.append(
                "\n".join(
                    part
                    for part in [
                        f"mem_id: {item.mem_id}",
                        f"title: {item.title or ''}",
                        f"text: {item.text}",
                        item.facets.lexical_text(),
                        f"reference_count: {item.reference_count}",
                        "links: "
                        + ", ".join(f"{link.relation.value}:{link.to_mem_id}" for link in links),
                    ]
                    if part.strip()
                )
            )
        return "\n\n---\n\n".join(sections)
