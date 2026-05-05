import json
import re

from ltm_design.llm.client import ChatMessage
from ltm_design.memory.models import MemoryCandidate
from ltm_design.memory.orchestrator import MemoryModels, MemoryOrchestrator
from ltm_design.memory.store import MemoryStore
from conftest import FakeEmbeddingProvider


class FakeChatClient:
    def __init__(self):
        self.calls = []

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        self.calls.append((model, messages))
        prompt_text = "\n\n".join(message.content for message in messages)
        if "memory cataloger" in prompt_text:
            assert "source_id" not in messages[1].content
            assert "whole_source" not in messages[1].content
            assert messages[0].role == "user"
            assert messages[1].role == "developer"
            assert "MEMORY CATALOGER SKILL PROMPT" in messages[1].content
            return json.dumps(
                {
                    "memories": [
                        {
                            "new_memory_index": 0,
                            "text": "The user wants same_as memories thrown away instead of written.",
                            "facets": {"people": [], "topics": ["memory deduplication"], "emotions": [], "events": [], "places": [], "objects": [], "times": []},
                            "references": [{"source_marker": "[s1]", "speaker": "user", "start_anchor": "same_as", "end_anchor": "thrown away"}],
                        }
                    ],
                }
            )
        if "repairing source anchors" in prompt_text:
            return json.dumps({"choices": [{"new_memory_index": 0, "reference_index": 0, "anchor": "start_anchor", "choice": 2}]})
        if "memory connection checker" in prompt_text:
            existing_mem_id = re.search(r"^mem_id: ([^\n]+)", messages[1].content, re.MULTILINE).group(1)
            return json.dumps({"connections": [{"new_memory_index": 0, "existing_mem_id": existing_mem_id, "relation": "same_as"}]})
        if "memory connection summarizer" in prompt_text:
            existing_mem_id = re.search(r"^[- ]*existing_mem_id: ([^\n]+)", messages[1].content, re.MULTILINE).group(1)
            assert "CONNECTED EXISTING MEMORIES" not in messages[1].content
            assert "PROPOSED CONNECTIONS BY NEW MEMORY" in messages[1].content
            assert "new_memory_index: 0" in messages[1].content
            assert "relation: same_as" in messages[1].content
            return f"0:\nThis is redundant with the existing memory that the user wants \"same_as memories thrown away instead of written\" (id: {existing_mem_id}, same_as)."
        if "continuing the memory cataloging task" in prompt_text:
            assert re.match(
                r"^0:\nThis is redundant with the existing memory .*\(id: [^,]+, same_as\)\.$",
                messages[1].content,
            )
            assert "CATALOG" not in messages[1].content
            assert "NEW MEMORIES" not in messages[1].content
            return json.dumps(
                {
                    "reject_memories": [{"new_memory_index": 0, "reason": "same_as"}],
                    "approve_links": [],
                }
            )
        if "checking a group of memory items" in prompt_text:
            mem_id = re.search(r"^mem_id: ([^\n]+)", messages[1].content, re.MULTILINE).group(1)
            return json.dumps({"memories": [{"mem_id": mem_id, "relevant": True, "memory_tags": ["dedupe"], "descriptors": ["relevant"], "reason": "matches"}]})
        if "selecting useful memory context" in prompt_text:
            return "The user wants same_as memories thrown away [mem_existing]."
        raise AssertionError(prompt_text)


def test_orchestrator_uses_pro_for_main_and_flash_for_workers(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    existing = store.remember(MemoryCandidate(text="The user wants same_as memories thrown away instead of written."))
    # Force deterministic id used by fake connection.
    assert existing.mem_id != "mem_existing"
    fake = FakeChatClient()
    orchestrator = MemoryOrchestrator(
        store=store,
        embeddings=FakeEmbeddingProvider(),
        chat_client=fake,
        models=MemoryModels(main="deepseek-v4-pro", worker="deepseek-v4-flash"),
    )
    orchestrator.indexer.index_item(existing)
    store.assign_to_recall_fragment(existing.mem_id)

    result = orchestrator.process_memory_skill(
        evidence_range="[src:thread_1 msg:u_1] same_as should be thrown away",
    )

    assert result.approved_mem_ids == ()
    assert [call[0] for call in fake.calls] == [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]
    fragment_input = fake.calls[1][1][1].content
    assert "NEW MEMORIES" in fragment_input
    assert "new_memory_index: 0" in fragment_input
    assert "existing_fragment" not in fragment_input
    assert "\"new_memories\"" not in fragment_input
    summary_input = fake.calls[2][1][1].content
    assert "PROPOSED CONNECTIONS" in summary_input
    assert "\"fragment_outputs\"" not in summary_input
    decision_input = fake.calls[3][1][1].content
    assert "same_as memories thrown away instead of written" in decision_input
    assert "STORAGE CONNECTION SUMMARY" not in decision_input
    assert "NEW MEMORIES" not in decision_input


def test_orchestrator_approves_cataloged_memories_by_default(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    class DefaultApproveChatClient(FakeChatClient):
        def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
            prompt_text = "\n\n".join(message.content for message in messages)
            if "continuing the memory cataloging task" in prompt_text:
                return json.dumps({"reject_memories": [], "approve_links": []})
            if "memory connection checker" in prompt_text or "memory connection summarizer" in prompt_text:
                raise AssertionError("no existing fragments should mean no connection workers")
            return super().chat(model=model, messages=messages, temperature=temperature)

    fake = DefaultApproveChatClient()
    orchestrator = MemoryOrchestrator(
        store=store,
        embeddings=FakeEmbeddingProvider(),
        chat_client=fake,
        models=MemoryModels(main="deepseek-v4-pro", worker="deepseek-v4-flash"),
    )

    result = orchestrator.process_memory_skill(
        evidence_range="[src:thread_1 msg:u_1] remember this",
    )

    assert len(result.approved_mem_ids) == 1


def test_orchestrator_retrieves_and_summarizes_with_worker_model(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    item = store.remember(MemoryCandidate(text="The user wants summarizers to cite mem ids only."))
    embeddings = FakeEmbeddingProvider()
    fake = FakeChatClient()
    orchestrator = MemoryOrchestrator(store=store, embeddings=embeddings, chat_client=fake)
    orchestrator.indexer.index_item(item)

    context = orchestrator.retrieve_context(user_message="how should summaries cite?", conversation_context="")

    assert "mem_existing" in context.text
    assert context.mem_ids == (item.mem_id,)


def test_orchestrator_repairs_ambiguous_catalog_anchors(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    class AmbiguousAnchorChatClient(FakeChatClient):
        def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
            self.calls.append((model, messages))
            prompt_text = "\n\n".join(message.content for message in messages)
            if "memory cataloger" in prompt_text:
                dev_prompt = next(message.content for message in messages if message.role == "developer")
                assert "The starting source marker is [s1]" in dev_prompt
                assert 'The text you should process starts here: "alpha cats and dogs. beta cats and salmon."' in dev_prompt
                assert dev_prompt.index("The starting source marker is [s1]") < dev_prompt.index("Output JSON:")
                return json.dumps(
                    {
                        "memories": [
                            {
                                "new_memory_index": 0,
                                "text": "The user mentioned the second cats example.",
                                "facets": {},
                                "references": [
                                    {
                                        "source_marker": "[s1]",
                                        "speaker": "user",
                                        "start_anchor": "cats",
                                        "end_anchor": "salmon",
                                    }
                                ],
                            }
                        ]
                    }
                )
            if "repairing source anchors" in prompt_text:
                assert '1. "alpha cats and"' in messages[1].content
                assert '2. "beta cats and"' in messages[1].content
                return json.dumps(
                    {
                        "choices": [
                            {
                                "new_memory_index": 0,
                                "reference_index": 0,
                                "anchor": "start_anchor",
                                "choice": 2,
                            }
                        ]
                    }
                )
            if "continuing the memory cataloging task" in prompt_text:
                return json.dumps({"reject_memories": [], "approve_links": []})
            raise AssertionError(prompt_text)

    fake = AmbiguousAnchorChatClient()
    orchestrator = MemoryOrchestrator(
        store=store,
        embeddings=FakeEmbeddingProvider(),
        chat_client=fake,
    )

    result = orchestrator.process_memory_skill(
        evidence_range="",
        conversation_messages=[],
        uncommitted_messages=[
            ChatMessage(role="user", content="source_marker: [s1]\nalpha cats and dogs. beta cats and salmon."),
        ],
    )

    item = store.get(result.approved_mem_ids[0])
    assert item is not None
    assert item.evidence[0].source_id == "s1"
    assert item.evidence[0].locator["start_anchor"] == "beta cats and"


def test_orchestrator_drops_catalog_memories_from_committed_context(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")

    class BoundaryViolatingChatClient(FakeChatClient):
        def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
            self.calls.append((model, messages))
            prompt_text = "\n\n".join(message.content for message in messages)
            if "memory cataloger" in prompt_text:
                dev_prompt = next(message.content for message in messages if message.role == "developer")
                assert "The starting source marker is [s4]" in dev_prompt
                assert 'The text you should process starts here: "new uncommitted fact"' in dev_prompt
                assert dev_prompt.index("The starting source marker is [s4]") < dev_prompt.index("Output JSON:")
                return json.dumps(
                    {
                        "memories": [
                            {
                                "new_memory_index": 0,
                                "text": "The user mentioned an old committed fact.",
                                "facets": {},
                                "references": [
                                    {
                                        "source_marker": "[s1]",
                                        "speaker": "user",
                                        "start_anchor": "old committed",
                                        "end_anchor": "old committed",
                                    }
                                ],
                            },
                            {
                                "new_memory_index": 1,
                                "text": "The user mentioned old content mislabeled with the new source marker.",
                                "facets": {},
                                "references": [
                                    {
                                        "source_marker": "[s4]",
                                        "speaker": "user",
                                        "start_anchor": "old committed",
                                        "end_anchor": "old committed",
                                    }
                                ],
                            },
                            {
                                "new_memory_index": 2,
                                "text": "The user mentioned a new uncommitted fact.",
                                "facets": {},
                                "references": [
                                    {
                                        "source_marker": "[s4]",
                                        "speaker": "user",
                                        "start_anchor": "new uncommitted",
                                        "end_anchor": "new uncommitted",
                                    }
                                ],
                            },
                        ]
                    }
                )
            if "continuing the memory cataloging task" in prompt_text:
                return json.dumps({"reject_memories": [], "approve_links": []})
            raise AssertionError(prompt_text)

    fake = BoundaryViolatingChatClient()
    orchestrator = MemoryOrchestrator(
        store=store,
        embeddings=FakeEmbeddingProvider(),
        chat_client=fake,
    )

    result = orchestrator.process_memory_skill(
        evidence_range="",
        conversation_messages=[
            ChatMessage(role="user", content="source_marker: [s1]\nold committed fact"),
        ],
        uncommitted_messages=[
            ChatMessage(role="user", content="source_marker: [s4]\nnew uncommitted fact"),
        ],
    )

    assert len(result.approved_mem_ids) == 1
    item = store.get(result.approved_mem_ids[0])
    assert item is not None
    assert item.text == "The user mentioned a new uncommitted fact."
    assert item.evidence[0].source_id == "s4"
