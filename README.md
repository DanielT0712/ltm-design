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
