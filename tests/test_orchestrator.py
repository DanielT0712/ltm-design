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
        system = messages[0].content
        if "memory cataloger" in system:
            return json.dumps(
                {
                    "memories": [
                        {
                            "text": "The user wants same_as memories thrown away instead of written.",
                            "facets": {"people": [], "topics": ["memory deduplication"], "emotions": [], "events": [], "places": [], "objects": [], "times": []},
                            "references": [{"source_id": "thread_1", "message_id": "u_1", "event_id": None, "speaker": "user", "whole_source": True, "start_anchor": None, "end_anchor": None}],
                            "connection_hints": [],
                        }
                    ],
                    "processed_through": "[src:thread_1 msg:u_1]",
                }
            )
        if "memory connection checker" in system:
            existing_mem_id = re.search(r"^mem_id: (mem_[^\n]+)", messages[1].content, re.MULTILINE).group(1)
            return json.dumps({"connections": [{"new_memory_index": 0, "existing_mem_id": existing_mem_id, "relation": "same_as"}]})
        if "memory connection summarizer" in system:
            existing_mem_id = re.search(r"^mem_id: (mem_[^\n]+)", messages[1].content, re.MULTILINE).group(1)
            assert "CONNECTED EXISTING MEMORIES" not in messages[1].content
            assert "proposed_connections: new_memory_index 0: same_as" in messages[1].content
            return f"0:\nThis is redundant with the existing memory that the user wants \"same_as memories thrown away instead of written\" (id: {existing_mem_id}, same_as)."
        if "continuing the memory cataloging task" in system:
            assert re.match(
                r"^0:\nThis is redundant with the existing memory .*\(id: mem_[^,]+, same_as\)\.$",
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
        if "checking a group of memory items" in system:
            payload = json.loads(messages[1].content)
            return json.dumps({"memories": [{"mem_id": payload["memory_items"][0]["mem_id"], "relevant": True, "memory_tags": ["dedupe"], "descriptors": ["relevant"], "reason": "matches"}]})
        if "selecting useful memory context" in system:
            return "The user wants same_as memories thrown away [mem_existing]."
        raise AssertionError(system)


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
        processed_through="[src:thread_1 msg:u_0]",
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
            system = messages[0].content
            if "continuing the memory cataloging task" in system:
                return json.dumps({"reject_memories": [], "approve_links": []})
            if "memory connection checker" in system or "memory connection summarizer" in system:
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
        processed_through="[src:thread_1 msg:u_0]",
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
