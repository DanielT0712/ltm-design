# ltm-design

Long-term memory retrieval experiments for LLMs.

This repo is intended to prototype a memory-item retrieval architecture:

- store extracted textual memory items as the hot memory layer
- keep raw episodes/input records as cold evidence and cache material
- precompute reusable memory state at write time
- retrieve with vector prefiltering plus per-memory relevance agents
- cluster affirmative memories into topic summaries
- keep retrieval scaffolding out of the user-facing conversation thread

See [ARCHITECTURE.md](ARCHITECTURE.md) for the first concrete build plan.

## Implemented v0

- `MemoryStore`: SQLite-backed `memory_items`, `memory_links`, and FTS5 storage.
- `ActiveMemorizationSkill`: the callable surface an agent uses when it decides something should be remembered.
- `IdleAutoMemorizationTrigger`: invokes the same active memorization skill after a quiet period with model-proposed memory candidates.
- `reference_count`: incremented when the final main conversation model accepts a memory as important, so later fragment and summary stages can treat repeated use as a significance signal.
- `MemoryOrchestrator`: model-call harness for memory cataloging, connection checks, main memory decisions, relevance checks, and summarization.
- `DeepSeekClient`: reads `DEEPSEEK_API_KEY` from the environment or repo `.env`; defaults are `deepseek-v4-pro` for main decisions and `deepseek-v4-flash` for relevance/connection/summarization workers.
- `QwenEmbeddingProvider`: Qwen dense embeddings for retrieval.
- Hybrid retrieval stack: Qwen vector search over `semantic_text`, SQLite FTS5 over `lexical_text`, exact facet matching, sparse relation expansion, then model relevance workers.

Minimal wiring:

```python
from ltm_design.llm.client import DeepSeekClient
from ltm_design.memory.embeddings import QwenEmbeddingProvider
from ltm_design.memory.orchestrator import MemoryOrchestrator
from ltm_design.memory.store import MemoryStore

store = MemoryStore("data/memory.db")
orchestrator = MemoryOrchestrator(
    store=store,
    embeddings=QwenEmbeddingProvider(),
    chat_client=DeepSeekClient(),
)
```

Install embedding dependencies when running real retrieval:

```bash
uv sync --extra embeddings
```
