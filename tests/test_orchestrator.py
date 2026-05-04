import json

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
        if "directly connect" in system:
            return json.dumps({"connections": [{"new_memory_index": 0, "existing_mem_id": "mem_existing", "relation": "same_as"}]})
        if "continuing the memory cataloging task" in system:
            return json.dumps(
                {
                    "approve_memories": [],
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

    result = orchestrator.process_memory_skill(
        processed_through="[src:thread_1 msg:u_0]",
        evidence_range="[src:thread_1 msg:u_1] same_as should be thrown away",
    )

    assert result.approved_mem_ids == ()
    assert [call[0] for call in fake.calls] == [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-pro",
    ]


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
