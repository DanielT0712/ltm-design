from __future__ import annotations

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

            CREATE TABLE IF NOT EXISTS memory_embeddings (
              mem_id TEXT NOT NULL,
              model TEXT NOT NULL,
              dimensions INTEGER NOT NULL,
              vector_json TEXT NOT NULL,
              created_at TEXT NOT NULL,
              PRIMARY KEY (mem_id, model),
              FOREIGN KEY (mem_id) REFERENCES memory_items(mem_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS id_counters (
              name TEXT PRIMARY KEY,
              next_value INTEGER NOT NULL
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
        facets_json = json.dumps(candidate.facets.to_dict(), sort_keys=True)
        evidence_json = json.dumps([ref.to_dict() for ref in candidate.evidence], sort_keys=True)
        existing = self._conn.execute(
            """
            SELECT candidate_id FROM pending_memory_candidates
            WHERE text = ? AND facets_json = ? AND evidence_json = ?
            ORDER BY created_at LIMIT 1
            """,
            (candidate.text, facets_json, evidence_json),
        ).fetchone()
        if existing:
            return existing["candidate_id"]
        candidate_id = self._next_id("c")
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
                    facets_json,
                    evidence_json,
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
        title = candidate.title or self._title_for(candidate.text)
        semantic_text = self._semantic_text(candidate)
        lexical_text = self._lexical_text(title, candidate)
        evidence_json = json.dumps([ref.to_dict() for ref in candidate.evidence], sort_keys=True)
        facets_json = json.dumps(candidate.facets.to_dict(), sort_keys=True)
        existing_row = self._conn.execute(
            """
            SELECT mem_id FROM memory_items
            WHERE text = ? AND facets_json = ? AND evidence_json = ?
            ORDER BY created_at LIMIT 1
            """,
            (candidate.text, facets_json, evidence_json),
        ).fetchone()
        if existing_row:
            existing = self.get(existing_row["mem_id"])
            if existing is not None:
                return existing

        mem_id = self._next_id("m")
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

    def store_embedding(
        self,
        mem_id: str,
        *,
        model: str,
        dimensions: int,
        vector: list[float],
        now: datetime | None = None,
    ) -> None:
        if len(vector) != dimensions:
            raise ValueError("vector length does not match dimensions")
        if self.get(mem_id) is None:
            raise KeyError(mem_id)
        with self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO memory_embeddings (
                  mem_id, model, dimensions, vector_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (mem_id, model, dimensions, json.dumps(vector), datetime_to_str(now or utc_now())),
            )

    def list_embeddings(self, *, model: str) -> list[tuple[str, list[float]]]:
        rows = self._conn.execute(
            "SELECT mem_id, vector_json FROM memory_embeddings WHERE model = ? ORDER BY mem_id",
            (model,),
        ).fetchall()
        return [(row["mem_id"], json.loads(row["vector_json"])) for row in rows]

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
        existing = self._conn.execute(
            """
            SELECT fragment_id FROM memory_fragments
            WHERE kind = ? AND mem_ids_json = ?
            ORDER BY created_at LIMIT 1
            """,
            (kind.value, json.dumps(list(mem_id_tuple))),
        ).fetchone()
        fragment_id = existing["fragment_id"] if existing else self._next_id("f")
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

    def assign_to_recall_fragment(
        self,
        mem_id: str,
        *,
        max_tokens: int = 20_000,
        now: datetime | None = None,
    ) -> MemoryFragment:
        item = self.get(mem_id)
        if item is None:
            raise KeyError(mem_id)

        fragments = self._list_fragments(kind=FragmentKind.RECALL)
        connected_ids = set(self.connected_mem_ids(mem_id, relation_depth=2))
        item_tokens = self._item_token_estimate(item)
        fitting_fragments = [
            fragment for fragment in fragments
            if fragment.token_count_estimate + item_tokens <= max_tokens
        ]

        chosen: MemoryFragment | None = None
        if connected_ids and fitting_fragments:
            connected_ranked = sorted(
                fitting_fragments,
                key=lambda fragment: (
                    len(set(fragment.mem_ids).intersection(connected_ids)),
                    max_tokens - fragment.token_count_estimate,
                ),
                reverse=True,
            )
            if connected_ranked and set(connected_ranked[0].mem_ids).intersection(connected_ids):
                chosen = connected_ranked[0]

        if chosen is None and fitting_fragments:
            chosen = min(fitting_fragments, key=lambda fragment: fragment.token_count_estimate)

        if chosen is None:
            return self.build_fragment(
                kind=FragmentKind.RECALL,
                mem_ids=(mem_id,),
                title=item.title or self._title_for(item.text),
                now=now,
            )

        next_mem_ids = tuple(dict.fromkeys((*chosen.mem_ids, mem_id)))
        self._delete_fragment(chosen.fragment_id)
        return self.build_fragment(
            kind=FragmentKind.RECALL,
            mem_ids=next_mem_ids,
            title=chosen.title,
            now=now,
        )

    def _delete_fragment(self, fragment_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM memory_fragments WHERE fragment_id = ?", (fragment_id,))

    def list_fragments(self, *, kind: FragmentKind | None = None) -> list[MemoryFragment]:
        return self._list_fragments(kind=kind)

    def fragments_containing_mem_ids(
        self,
        mem_ids: Iterable[str],
        *,
        kind: FragmentKind | None = None,
        limit: int | None = None,
    ) -> list[MemoryFragment]:
        wanted = set(mem_ids)
        if not wanted:
            return []
        fragments: list[MemoryFragment] = []
        for fragment in self._list_fragments(kind=kind):
            if wanted.intersection(fragment.mem_ids):
                fragments.append(fragment)
        fragments = self._drop_shadowed_fragments(fragments)
        return fragments[:limit] if limit is not None else fragments

    def _drop_shadowed_fragments(self, fragments: list[MemoryFragment]) -> list[MemoryFragment]:
        kept: list[MemoryFragment] = []
        for fragment in sorted(fragments, key=lambda item: (item.updated_at, item.fragment_id), reverse=True):
            mem_ids = set(fragment.mem_ids)
            if any(mem_ids.issubset(set(existing.mem_ids)) for existing in kept):
                continue
            kept.append(fragment)
        return list(reversed(kept))

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

    def _next_id(self, prefix: str) -> str:
        with self._conn:
            row = self._conn.execute(
                "SELECT next_value FROM id_counters WHERE name = ?",
                (prefix,),
            ).fetchone()
            value = row["next_value"] if row else 1
            self._conn.execute(
                """
                INSERT INTO id_counters (name, next_value) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET next_value = excluded.next_value
                """,
                (prefix, value + 1),
            )
        return f"{prefix}{self._base62_int(value)}"

    def _base62_int(self, number: int) -> str:
        alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        if number == 0:
            return alphabet[0]
        chars: list[str] = []
        while number:
            number, rem = divmod(number, len(alphabet))
            chars.append(alphabet[rem])
        return "".join(reversed(chars))

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

    def _list_fragments(self, *, kind: FragmentKind | None = None) -> list[MemoryFragment]:
        if kind is None:
            rows = self._conn.execute(
                "SELECT * FROM memory_fragments ORDER BY created_at, fragment_id"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memory_fragments WHERE kind = ? ORDER BY created_at, fragment_id",
                (kind.value,),
            ).fetchall()
        return [self._row_to_fragment(row) for row in rows]

    def _row_to_fragment(self, row: sqlite3.Row) -> MemoryFragment:
        return MemoryFragment(
            fragment_id=row["fragment_id"],
            kind=FragmentKind(row["kind"]),
            mem_ids=tuple(json.loads(row["mem_ids_json"])),
            title=row["title"],
            text=row["text"],
            token_count_estimate=row["token_count_estimate"],
            created_at=datetime_from_str(row["created_at"]),
            updated_at=datetime_from_str(row["updated_at"]),
        )

    def _item_token_estimate(self, item: MemoryItem) -> int:
        return max(1, len(self._fragment_text([item]).split()))

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
