from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from ltm_design.llm.client import ChatMessage, parse_json_object
from ltm_design.memory.embeddings import EmbeddingProvider
from ltm_design.memory.models import EvidenceRef, Facets, MemoryCandidate, MemoryLink, MemoryRelation
from ltm_design.memory.prompts import (
    CONNECTION_FRAGMENT_PROMPT,
    MAIN_MEMORY_DECISION_PROMPT,
    MEMORY_CATALOGER_PROMPT,
    MEMORY_FRAGMENT_AGENT_PROMPT,
    MEMORY_SUMMARIZER_PROMPT,
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
    raw_connections: list[dict]
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
    ) -> MemorySkillResult:
        catalog = self._catalog_memories(
            processed_through=processed_through,
            evidence_range=evidence_range,
            recent_context=recent_context,
        )
        candidates = self._candidates_from_catalog(catalog)
        candidate_ids = [
            self.store.propose_candidate(candidate, trigger="active_model")
            for candidate in candidates
        ]
        connections = [
            self._discover_connections(index, candidate)
            for index, candidate in enumerate(candidates)
        ]
        decision_raw = self._decide_memories(catalog=catalog, connections=connections)
        approved_by_index: dict[int, str] = {}
        rejected_indices = {
            item["new_memory_index"]
            for item in decision_raw.get("reject_memories", [])
            if item.get("new_memory_index", -1) < len(candidate_ids)
        }

        for item in decision_raw.get("approve_memories", []):
            index = item.get("new_memory_index", -1)
            if index < 0 or index >= len(candidate_ids) or index in rejected_indices:
                continue
            approved = self.store.approve_candidate(candidate_ids[index])
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
        relevant_ids: list[str] = []
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

    def _catalog_memories(self, *, processed_through: str, evidence_range: str, recent_context: str) -> dict:
        payload = {
            "processed_through": processed_through,
            "evidence_range": evidence_range,
            "recent_context": recent_context,
        }
        text = self.chat_client.chat(
            model=self.models.main,
            messages=[
                ChatMessage(role="system", content=MEMORY_CATALOGER_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, indent=2)),
            ],
        )
        return parse_json_object(text)

    def _discover_connections(self, index: int, candidate: MemoryCandidate) -> dict:
        nearby = self.search.search(text=candidate.text, facets=candidate.facets, vector_top_k=12, fts_top_k=12, facet_top_k=12).candidates
        existing = [self._memory_snippet(candidate.mem_id) for candidate in nearby]
        payload = {
            "new_memories": [{"index": index, "text": candidate.text, "facets": candidate.facets.to_dict()}],
            "existing_memories": existing,
        }
        text = self.chat_client.chat(
            model=self.models.worker,
            messages=[
                ChatMessage(role="system", content=CONNECTION_FRAGMENT_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, indent=2)),
            ],
        )
        return parse_json_object(text)

    def _decide_memories(self, *, catalog: dict, connections: list[dict]) -> dict:
        payload = {"catalog": catalog, "connections": connections}
        text = self.chat_client.chat(
            model=self.models.main,
            messages=[
                ChatMessage(role="system", content=MAIN_MEMORY_DECISION_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, indent=2)),
            ],
        )
        return parse_json_object(text)

    def _check_relevance(self, *, mem_ids: list[str], user_message: str, conversation_context: str) -> dict:
        payload = {
            "user_message": user_message,
            "conversation_context": conversation_context,
            "memory_items": [self._memory_snippet(mem_id) for mem_id in mem_ids],
        }
        text = self.chat_client.chat(
            model=self.models.worker,
            messages=[
                ChatMessage(role="system", content=MEMORY_FRAGMENT_AGENT_PROMPT),
                ChatMessage(role="user", content=json.dumps(payload, indent=2)),
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

    def _chunks(self, values: list[str], size: int) -> list[list[str]]:
        return [values[index:index + size] for index in range(0, len(values), size)]

    def _link_new_index(self, link: dict) -> int:
        return link.get("from_new_memory_index", link.get("new_memory_index", -1))
