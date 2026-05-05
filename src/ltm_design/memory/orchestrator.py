from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from ltm_design.llm.client import ChatMessage, parse_json_object
from ltm_design.memory.embeddings import EmbeddingProvider
from ltm_design.memory.models import EvidenceRef, Facets, MemoryCandidate, MemoryLink, MemoryRelation
from ltm_design.memory.models import FragmentKind
from ltm_design.memory.prompts import (
    MAIN_MEMORY_DECISION_PROMPT,
    MEMORY_CATALOGER_PROMPT,
    MEMORY_FRAGMENT_AGENT_PROMPT,
    MEMORY_SUMMARIZER_PROMPT,
    STORAGE_CONNECTION_FRAGMENT_PROMPT,
    STORAGE_CONNECTION_SUMMARIZER_PROMPT,
)
from ltm_design.memory.search import MemoryIndexer, MemorySearch
from ltm_design.memory.store import MemoryStore
from ltm_design.memory.tools import MemoryTools


class ChatClient(Protocol):
    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str: ...


@dataclass(frozen=True)
class MemoryModels:
    main: str = "deepseek-v4-pro"
    worker: str = "deepseek-v4-flash"


@dataclass(frozen=True)
class MemorySkillResult:
    approved_mem_ids: tuple[str, ...]
    raw_catalog: dict
    raw_connections: str
    raw_decision: dict


@dataclass(frozen=True)
class RetrievedMemoryContext:
    text: str
    mem_ids: tuple[str, ...]


class MemoryOrchestrator:
    def __init__(
        self,
        *,
        store: MemoryStore,
        embeddings: EmbeddingProvider,
        chat_client: ChatClient,
        models: MemoryModels = MemoryModels(),
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.chat_client = chat_client
        self.models = models
        self.indexer = MemoryIndexer(store, embeddings)
        self.search = MemorySearch(store, embeddings)
        self.tools = MemoryTools(store)

    def process_memory_skill(
        self,
        *,
        processed_through: str,
        evidence_range: str,
        recent_context: str = "",
        conversation_messages: list[ChatMessage] | None = None,
    ) -> MemorySkillResult:
        catalog = self._catalog_memories(
            processed_through=processed_through,
            evidence_range=evidence_range,
            recent_context=recent_context,
            conversation_messages=conversation_messages,
        )
        candidates = self._candidates_from_catalog(catalog)
        candidate_ids = [
            self.store.propose_candidate(candidate, trigger="active_model")
            for candidate in candidates
        ]
        connections = self._discover_connections(candidates)
        decision_raw = self._decide_memories(catalog=catalog, connections=connections)
        approved_by_index: dict[int, str] = {}
        rejected_indices = {
            item["new_memory_index"]
            for item in decision_raw.get("reject_memories", [])
            if item.get("new_memory_index", -1) < len(candidate_ids)
        }

        for index, candidate_id in enumerate(candidate_ids):
            if index in rejected_indices:
                continue
            approved = self.store.approve_candidate(candidate_id)
            approved_by_index[index] = approved.mem_id
        for index, candidate_id in enumerate(candidate_ids):
            if index in rejected_indices:
                self.store.reject_candidate(candidate_id)

        for link in decision_raw.get("approve_links", []):
            index = self._link_new_index(link)
            relation = link.get("relation")
            if relation == MemoryRelation.SAME_AS.value or index not in approved_by_index:
                continue
            self.store.add_links(
                approved_by_index[index],
                (MemoryLink(to_mem_id=link["to_mem_id"], relation=MemoryRelation(relation)),),
            )

        approved_mem_ids: list[str] = []
        for mem_id in approved_by_index.values():
            item = self.store.get(mem_id)
            if item is not None:
                self.indexer.index_item(item)
                self.store.assign_to_recall_fragment(mem_id)
                approved_mem_ids.append(mem_id)
        return MemorySkillResult(
            approved_mem_ids=tuple(approved_mem_ids),
            raw_catalog=catalog,
            raw_connections=connections,
            raw_decision=decision_raw,
        )

    def retrieve_context(
        self,
        *,
        user_message: str,
        conversation_context: str,
        max_candidates: int = 24,
        group_size: int = 8,
    ) -> RetrievedMemoryContext:
        result = self.search.search(text=f"{user_message}\n{conversation_context}")
        candidate_ids = [candidate.mem_id for candidate in result.candidates[:max_candidates]]
        fragments = self.store.fragments_containing_mem_ids(
            candidate_ids,
            kind=FragmentKind.RECALL,
            limit=max(1, max_candidates // max(1, group_size)),
        )
        relevant_ids: list[str] = []
        if fragments:
            for fragment in fragments:
                reply = self._check_relevance(
                    mem_ids=list(fragment.mem_ids),
                    user_message=user_message,
                    conversation_context=conversation_context,
                    fragment_id=fragment.fragment_id,
                )
                for memory in reply.get("memories", []):
                    if memory.get("relevant") and memory.get("mem_id"):
                        relevant_ids.append(memory["mem_id"])
        else:
            for group in self._chunks(candidate_ids, group_size):
                reply = self._check_relevance(
                    mem_ids=group,
                    user_message=user_message,
                    conversation_context=conversation_context,
                )
                for memory in reply.get("memories", []):
                    if memory.get("relevant") and memory.get("mem_id"):
                        relevant_ids.append(memory["mem_id"])
        relevant_ids = list(dict.fromkeys(relevant_ids))
        if not relevant_ids:
            return RetrievedMemoryContext(text="", mem_ids=())
        summary = self._summarize(
            mem_ids=tuple(relevant_ids),
            user_message=user_message,
            conversation_context=conversation_context,
        )
        return RetrievedMemoryContext(text=summary, mem_ids=tuple(relevant_ids))

    def _catalog_memories(
        self,
        *,
        processed_through: str,
        evidence_range: str,
        recent_context: str,
        conversation_messages: list[ChatMessage] | None = None,
    ) -> dict:
        skill_prompt = "\n\n".join(
            [
                "MEMORY CATALOGER SKILL PROMPT",
                MEMORY_CATALOGER_PROMPT,
                f"processed_through: {processed_through}",
            ]
        )
        messages = list(conversation_messages) if conversation_messages is not None else [
            ChatMessage(role="user", content=evidence_range),
        ]
        messages.append(ChatMessage(role="developer", content=skill_prompt))
        text = self.chat_client.chat(
            model=self.models.main,
            messages=messages,
        )
        return parse_json_object(text)

    def _discover_connections(self, candidates: list[MemoryCandidate]) -> str:
        if not candidates:
            return ""
        max_fragments = 24
        hit_mem_ids: dict[str, None] = {}
        for candidate in candidates:
            nearby = self.search.search(
                text=candidate.text,
                facets=candidate.facets,
                vector_top_k=12,
                fts_top_k=12,
                facet_top_k=12,
            ).candidates
            for nearby_candidate in nearby:
                hit_mem_ids[nearby_candidate.mem_id] = None

        fragments = self.store.fragments_containing_mem_ids(
            hit_mem_ids.keys(),
            kind=FragmentKind.RECALL,
            limit=max_fragments,
        )
        new_memories_text = self._format_new_memories(candidates)
        if not fragments:
            return ""

        fragment_outputs: list[dict] = []
        for fragment in fragments:
            prompt_text = "\n\n".join(
                [
                    f"EXISTING MEMORIES ({fragment.fragment_id})",
                    self._format_existing_memories(fragment.mem_ids),
                    "NEW MEMORIES",
                    new_memories_text,
                ]
            )
            text = self.chat_client.chat(
                model=self.models.worker,
                messages=[
                    ChatMessage(role="system", content=STORAGE_CONNECTION_FRAGMENT_PROMPT),
                    ChatMessage(role="user", content=prompt_text),
                ],
            )
            output = parse_json_object(text)
            output["fragment_id"] = fragment.fragment_id
            fragment_outputs.append(output)

        connected_ids = {
            connection["existing_mem_id"]
            for output in fragment_outputs
            for connection in output.get("connections", [])
            if connection.get("existing_mem_id")
        }
        return self._summarize_storage_connections(
            new_memories_text=new_memories_text,
            fragment_outputs=fragment_outputs,
            connected_ids=connected_ids,
        )

    def _summarize_storage_connections(
        self,
        *,
        new_memories_text: str,
        fragment_outputs: list[dict],
        connected_ids: set[str],
    ) -> str:
        if not connected_ids:
            return ""
        prompt_text = "\n\n".join(
            [
                "NEW MEMORIES",
                new_memories_text,
                "PROPOSED CONNECTIONS",
                self._format_proposed_connection_memories(fragment_outputs, connected_ids),
            ]
        )
        text = self.chat_client.chat(
            model=self.models.worker,
            messages=[
                ChatMessage(role="system", content=STORAGE_CONNECTION_SUMMARIZER_PROMPT),
                ChatMessage(role="user", content=prompt_text),
            ],
        )
        return text.strip()

    def _decide_memories(self, *, catalog: dict, connections: str) -> dict:
        payload = connections or "No storage-time connections were found."
        text = self.chat_client.chat(
            model=self.models.main,
            messages=[
                ChatMessage(role="system", content=MAIN_MEMORY_DECISION_PROMPT),
                ChatMessage(role="user", content=payload),
            ],
        )
        return parse_json_object(text)

    def _check_relevance(
        self,
        *,
        mem_ids: list[str],
        user_message: str,
        conversation_context: str,
        fragment_id: str | None = None,
    ) -> dict:
        prompt_text = "\n\n".join(
            [
                f"EXISTING MEMORIES ({fragment_id or 'ad_hoc_group'})",
                self._format_existing_memories(mem_ids),
                "CURRENT REQUEST",
                f"user_message: {user_message}",
                f"conversation_context: {conversation_context}",
            ]
        )
        text = self.chat_client.chat(
            model=self.models.worker,
            messages=[
                ChatMessage(role="system", content=MEMORY_FRAGMENT_AGENT_PROMPT),
                ChatMessage(role="user", content=prompt_text),
            ],
        )
        return parse_json_object(text)

    def _summarize(self, *, mem_ids: tuple[str, ...], user_message: str, conversation_context: str) -> str:
        expanded: list[dict] = []
        seen: set[str] = set()
        for mem_id in mem_ids:
            record = self.tools.load_full_memory(mem_id, relation_depth=2)
            for item in record.records:
                if item["mem_id"] in seen:
                    continue
                seen.add(item["mem_id"])
                expanded.append(item)
        payload = {
            "user_message": user_message,
            "conversation_context": conversation_context,
            "candidate_memory_items": expanded,
        }
        return self.chat_client.chat(
            model=self.models.worker,
            messages=[
                ChatMessage(role="system", content=MEMORY_SUMMARIZER_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, indent=2)),
            ],
        )

    def _candidates_from_catalog(self, catalog: dict) -> list[MemoryCandidate]:
        candidates: list[MemoryCandidate] = []
        for item in catalog.get("memories", []):
            evidence = tuple(
                EvidenceRef(
                    source_id=ref["source_id"],
                    locator={key: value for key, value in ref.items() if key != "source_id"},
                )
                for ref in item.get("references", [])
            )
            candidates.append(
                MemoryCandidate(
                    text=item["text"],
                    facets=Facets.from_mapping(item.get("facets")),
                    evidence=evidence,
                )
            )
        return candidates

    def _memory_snippet(self, mem_id: str) -> dict:
        item = self.store.get(mem_id)
        if item is None:
            return {"mem_id": mem_id, "missing": True}
        return {
            "mem_id": item.mem_id,
            "text": item.text,
            "facets": item.facets.to_dict(),
            "reference_count": item.reference_count,
            "links": [link.to_dict() for link in self.store.links_for(item.mem_id)],
        }

    def _format_new_memories(self, candidates: list[MemoryCandidate]) -> str:
        sections: list[str] = []
        for index, candidate in enumerate(candidates):
            sections.append(
                "\n".join(
                    [
                        f"new_memory_index: {index}",
                        f"text: {candidate.text}",
                    ]
                )
            )
        return "\n\n".join(sections)

    def _format_existing_memories(self, mem_ids: Iterable[str]) -> str:
        sections: list[str] = []
        for mem_id in mem_ids:
            item = self.store.get(mem_id)
            if item is None:
                continue
            sections.append(
                "\n".join(
                    part
                    for part in [
                        f"mem_id: {item.mem_id}",
                        f"text: {item.text}",
                        self._format_links(item.mem_id),
                    ]
                    if part.strip()
                )
            )
        return "\n\n".join(sections)

    def _format_facets(self, facets: Facets) -> str:
        parts = [
            f"{key}: {', '.join(values)}"
            for key, values in facets.to_dict().items()
            if values
        ]
        return "; ".join(parts) if parts else "none"

    def _format_links(self, mem_id: str) -> str:
        links = self.store.links_for(mem_id)
        if not links:
            return "links: none"
        return "links: " + ", ".join(
            f"{link.relation.value} {link.to_mem_id}" for link in links
        )

    def _format_proposed_connection_memories(
        self,
        fragment_outputs: list[dict],
        connected_ids: set[str],
    ) -> str:
        connections_by_mem_id: dict[str, list[dict]] = {mem_id: [] for mem_id in sorted(connected_ids)}
        for output in fragment_outputs:
            for connection in output.get("connections", []):
                mem_id = connection.get("existing_mem_id")
                if mem_id in connections_by_mem_id:
                    connections_by_mem_id[mem_id].append(connection)

        sections: list[str] = []
        for mem_id, connections in connections_by_mem_id.items():
            item = self.store.get(mem_id)
            if item is None:
                continue
            proposed_connections = ", ".join(
                f"new_memory_index {connection.get('new_memory_index')}: {connection.get('relation')}"
                for connection in connections
            )
            sections.append(
                "\n".join(
                    part
                    for part in [
                        f"mem_id: {item.mem_id}",
                        f"proposed_connections: {proposed_connections or 'none'}",
                        f"text: {item.text}",
                        self._format_links(item.mem_id),
                    ]
                    if part.strip()
                )
            )
        return "\n\n".join(sections) if sections else "none"

    def _chunks(self, values: list[str], size: int) -> list[list[str]]:
        return [values[index:index + size] for index in range(0, len(values), size)]

    def _link_new_index(self, link: dict) -> int:
        return link.get("from_new_memory_index", link.get("new_memory_index", -1))
