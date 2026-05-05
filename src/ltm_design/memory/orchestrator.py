from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from ltm_design.llm.client import ChatMessage, parse_json_object
from ltm_design.memory.embeddings import EmbeddingProvider
from ltm_design.memory.models import EvidenceRef, Facets, MemoryCandidate, MemoryLink, MemoryRelation
from ltm_design.memory.models import FragmentKind
from ltm_design.memory.prompts import (
    ANCHOR_DISAMBIGUATION_PROMPT,
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
        evidence_range: str,
        recent_context: str = "",
        conversation_messages: list[ChatMessage] | None = None,
        uncommitted_messages: list[ChatMessage] | None = None,
    ) -> MemorySkillResult:
        catalog = self._catalog_memories(
            evidence_range=evidence_range,
            recent_context=recent_context,
            conversation_messages=conversation_messages,
            uncommitted_messages=uncommitted_messages,
        )
        uncommitted_source_texts = self._source_texts_by_marker(uncommitted_messages or [])
        catalog = self._filter_catalog_to_uncommitted_sources(
            catalog,
            uncommitted_source_texts,
        )
        catalog = self._repair_ambiguous_anchors(
            catalog,
            uncommitted_source_texts,
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
        evidence_range: str,
        recent_context: str,
        conversation_messages: list[ChatMessage] | None = None,
        uncommitted_messages: list[ChatMessage] | None = None,
    ) -> dict:
        catalog_prompt = self._catalog_prompt(uncommitted_messages or [])
        skill_prompt = "\n\n".join(
            [
                "MEMORY CATALOGER SKILL PROMPT",
                catalog_prompt,
            ]
        )
        messages = list(conversation_messages) if conversation_messages is not None else [
            ChatMessage(role="user", content=evidence_range),
        ]
        messages.append(ChatMessage(role="developer", content=skill_prompt))
        if uncommitted_messages:
            messages.extend(uncommitted_messages)
        text = self.chat_client.chat(
            model=self.models.main,
            messages=messages,
        )
        return parse_json_object(text)

    def _filter_catalog_to_uncommitted_sources(
        self,
        catalog: dict,
        source_texts: dict[str, str],
    ) -> dict:
        if not source_texts:
            return catalog
        memories: list[dict] = []
        for item in catalog.get("memories", []):
            references = [
                ref
                for ref in item.get("references", [])
                if self._reference_matches_uncommitted_source(ref, source_texts)
            ]
            if not references:
                continue
            item = dict(item)
            item["references"] = references
            memories.append(item)
        catalog = dict(catalog)
        catalog["memories"] = memories
        return catalog

    def _reference_matches_uncommitted_source(self, ref: dict, source_texts: dict[str, str]) -> bool:
        source_text = source_texts.get(self._reference_source_id(ref))
        if not source_text:
            return False
        start_anchor = ref.get("start_anchor")
        end_anchor = ref.get("end_anchor")
        if not isinstance(start_anchor, str) or not isinstance(end_anchor, str):
            return False
        if not start_anchor or not end_anchor:
            return False
        start = source_text.find(start_anchor)
        if start == -1:
            return False
        return source_text.find(end_anchor, start) != -1

    def _repair_ambiguous_anchors(
        self,
        catalog: dict,
        source_texts: dict[str, str],
    ) -> dict:
        tasks: list[dict] = []
        for memory_position, item in enumerate(catalog.get("memories", [])):
            memory_index = int(item.get("new_memory_index", memory_position))
            for reference_index, ref in enumerate(item.get("references", [])):
                source_id = self._reference_source_id(ref)
                source_text = source_texts.get(source_id)
                if not source_text:
                    continue
                ref["source_marker"] = f"[{source_id}]"
                for anchor_name in ("start_anchor", "end_anchor"):
                    anchor = ref.get(anchor_name)
                    if not isinstance(anchor, str) or not anchor:
                        continue
                    occurrences = [match.span() for match in re.finditer(re.escape(anchor), source_text)]
                    if len(occurrences) <= 1:
                        continue
                    options = self._unique_anchor_options(source_text, occurrences)
                    if len(options) > 1:
                        tasks.append(
                            {
                                "new_memory_index": memory_index,
                                "reference_index": reference_index,
                                "anchor": anchor_name,
                                "options": options,
                            }
                        )
        if not tasks:
            return catalog

        prompt = self._format_anchor_disambiguation_tasks(tasks)
        text = self.chat_client.chat(
            model=self.models.main,
            messages=[
                ChatMessage(role="system", content=ANCHOR_DISAMBIGUATION_PROMPT),
                ChatMessage(role="user", content=prompt),
            ],
        )
        choices = parse_json_object(text).get("choices", [])
        memories = catalog.get("memories", [])
        by_key = {
            (task["new_memory_index"], task["reference_index"], task["anchor"]): task
            for task in tasks
        }
        memory_by_index = {
            int(item.get("new_memory_index", position)): item
            for position, item in enumerate(memories)
        }
        for choice in choices:
            key = (
                int(choice.get("new_memory_index", -1)),
                int(choice.get("reference_index", -1)),
                choice.get("anchor"),
            )
            task = by_key.get(key)
            memory = memory_by_index.get(key[0])
            if task is None or memory is None:
                continue
            selected = int(choice.get("choice", 0))
            if selected < 1 or selected > len(task["options"]):
                continue
            memory["references"][key[1]][key[2]] = task["options"][selected - 1]
        return catalog

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
                    source_id=self._reference_source_id(ref),
                    locator={
                        key: value
                        for key, value in ref.items()
                        if key not in {"source_id", "source_marker"}
                    },
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

    def _source_texts_by_marker(self, messages: list[ChatMessage]) -> dict[str, str]:
        texts: dict[str, str] = {}
        marker_pattern = re.compile(r"^source_marker:\s*\[([A-Za-z0-9]+)\]\s*$", re.MULTILINE)
        for message in messages:
            matches = list(marker_pattern.finditer(message.content))
            for index, match in enumerate(matches):
                start = match.end()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(message.content)
                texts[match.group(1)] = message.content[start:end].strip()
        return texts

    def _normalize_source_id(self, value: str) -> str:
        match = re.search(r"\[([A-Za-z0-9]+)\]", value)
        return match.group(1) if match else value

    def _reference_source_id(self, ref: dict) -> str:
        return self._normalize_source_id(str(ref.get("source_marker") or ref.get("source_id") or ""))

    def _catalog_start_marker_instruction(self, uncommitted_messages: list[ChatMessage]) -> str:
        source_texts = self._source_texts_by_marker(uncommitted_messages)
        if not source_texts:
            return ""
        start_source_id = next(iter(source_texts))
        excerpt = self._first_two_sentences(self._source_body_text(source_texts[start_source_id]))
        parts = [
            f"The starting source marker is [{start_source_id}]. Never create memories with a source marker earlier "
            f"than this. If a memory you were planning to include has a source marker smaller than [{start_source_id}], "
            "be absolutely certain you do not include it. It is important that you ensure all the memories you create "
            f"have source markers greater than or equal to [{start_source_id}].",
        ]
        if excerpt:
            parts.append(
                f'The text you should process starts here: "{excerpt}" '
                "Do not commit anything before this starting text to memory; earlier conversation is context only "
                "and must not become a new memory. This is a common mistake and you should take great care in preventing it."
            )
        return " ".join(parts)

    def _catalog_prompt(self, uncommitted_messages: list[ChatMessage]) -> str:
        instruction = self._catalog_start_marker_instruction(uncommitted_messages)
        if not instruction:
            return MEMORY_CATALOGER_PROMPT
        return MEMORY_CATALOGER_PROMPT.replace(
            "\nOutput JSON:",
            f"\n{instruction}\n\nOutput JSON:",
            1,
        )

    def _source_body_text(self, source_text: str) -> str:
        lines = source_text.strip().splitlines()
        if lines and re.fullmatch(r"User journal entry dated .+\.", lines[0].strip()):
            lines = lines[1:]
        return "\n".join(lines).strip()

    def _first_two_sentences(self, text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text).strip()
        if not collapsed:
            return ""
        matches = list(re.finditer(r"(?<=[.!?])\s+", collapsed))
        if len(matches) >= 2:
            return collapsed[:matches[1].start()].strip()
        return collapsed

    def _format_anchor_disambiguation_tasks(self, tasks: list[dict]) -> str:
        sections: list[str] = []
        for task in tasks:
            options = "\n".join(
                f"{index}. \"{option}\""
                for index, option in enumerate(task["options"], start=1)
            )
            sections.append(
                "\n".join(
                    [
                        f"Your {task['anchor']} for memory number {task['new_memory_index']} was not unique.",
                        f"reference_index: {task['reference_index']}",
                        "Did you mean:",
                        options,
                        "Select the correct anchor for your memory by its number.",
                    ]
                )
            )
        return "\n\n".join(sections)

    def _unique_anchor_options(self, text: str, occurrences: list[tuple[int, int]]) -> list[str]:
        token_spans = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        options = [text[start:end] for start, end in occurrences]
        radius = 0
        while len(set(options)) < len(options) and radius < 16:
            radius += 1
            options = [
                self._expanded_anchor_text(text, token_spans, occurrence, radius)
                for occurrence in occurrences
            ]
        return options

    def _expanded_anchor_text(
        self,
        text: str,
        token_spans: list[tuple[int, int]],
        occurrence: tuple[int, int],
        radius: int,
    ) -> str:
        start, end = occurrence
        covered = [
            index
            for index, (token_start, token_end) in enumerate(token_spans)
            if token_start < end and token_end > start
        ]
        if not covered:
            return text[start:end]
        first = max(0, covered[0] - radius)
        last = min(len(token_spans) - 1, covered[-1] + radius)
        expanded_start = token_spans[first][0]
        expanded_end = token_spans[last][1]
        return text[expanded_start:expanded_end]
