# LTM Architecture Plan

## Goal

Prototype a long-term memory system where extracted textual memory items are the hot memory layer, raw conversation episodes are preserved as cold evidence, and retrieval is mediated by small memory agents instead of flattening everything into one lossy summary store.

The experimental question is:

> Can cached memory-item fragments plus evidence-backed raw episodes provide better recall and provenance than conventional vector-RAG, while keeping the live conversation clean and low-latency?

## Current Model/API Assumptions

DeepSeek's official API docs currently describe an OpenAI/Anthropic-compatible API with OpenAI-style `base_url` set to `https://api.deepseek.com`. Current model IDs include:

- `deepseek-v4-flash`
- `deepseek-v4-pro`

Compatibility aliases `deepseek-chat` and `deepseek-reasoner` are documented as deprecated after 2026-07-24, so new code should keep model names in config and default to `deepseek-v4-flash` for cheap agent calls.

## Repository Shape

Start as a Python research harness before turning it into a service.

```text
ltm-design/
  README.md
  ARCHITECTURE.md
  .env.example
  pyproject.toml
  src/ltm_design/
    config.py
    llm/
      client.py
      prompts.py
      schemas.py
    memory/
      models.py
      chunking.py
      store.py
      embeddings.py
      full_text.py
      cache.py
      cache_warmer.py
    retrieval/
      prefilter.py
      memory_agent.py
      request_tag_grouping.py
      summarizer.py
      orchestrator.py
    conversation/
      session.py
      branch_state.py
    evals/
      datasets.py
      metrics.py
      scenarios.py
  tests/
  examples/
  scripts/
```

## Data Model

`MemoryItem`

- `mem_id`: stable identifier
- `text`: the remembered user/context-specific claim, preference, observation, correction, or event
- `facets`: typed recall handles used by BM25, exact facet matching, and memory-fragment agents
  - `people`
  - `topics`
  - `emotions`
  - `events`
  - `places`
  - `objects`
  - `times`
- `evidence`: pointers to original transcript/input records, loaded only when a model asks for deeper context
- `created_at`
- `updated_at`
- `semantic_text`: clean text embedded for dense retrieval, normally derived from `text`
- `lexical_text`: denormalized text indexed for full-text search, built from `text` plus facets
- `cache_ref`: pointer to provider/local cache artifact for the serialized memory fragment

`EvidenceRecord`

- `source_id`: stable id for a raw conversation episode, message bundle, multimodal input record, or imported source
- `conversation_id`
- `text`: transcript text, OCR text, model-readable observation text, or pointer description
- `metadata`: source-specific locator data
- `created_at`

Evidence records are long-term storage, not a temporary buffer. Memory items point to evidence; raw evidence is rehydrated only when extraction, connection discovery, recall, or repair needs more context.

`MemoryLink`

- `from_mem_id`
- `to_mem_id`
- `relation`: one of `updates`, `contradicts`, `supports`, `elaborates`, `caused_by`, `part_of`, `same_as`

Links should be sparse. Do not create generic relatedness edges; add a link only when the fixed relation describes a meaningful recall or reconciliation path.

Raw episodes and multimodal input records remain durable evidence. They are not the normal retrieval unit. They are used for audit, repair, re-indexing, contradiction resolution, cached memory-fragment context, and occasional evidence rehydration when the conversation model explicitly needs more context.

`MemoryAgentReply`

- `mem_id`
- `relevant`: boolean
- `topic`: short label
- `rationale`: terse internal-facing explanation
- `evidence_refs`: memory item or source evidence pointers
- `claims`: structured facts the memory item supports
- `memory_tags`: request-specific labels produced only for the current retrieval and passed toward the main conversation model

`ClusterSummary`

- `cluster_id`
- `topic`
- `summary`
- `evidence_refs`
- `mem_ids`
- `contradictions`

## Write Path

1. Persist raw conversation messages and multimodal input references as durable `EvidenceRecord`s.
2. Active or idle extraction starts a memory extraction branch with the evidence range and a cursor for what has already been processed into memory.
3. The extraction branch returns all atomic memories worth preserving from that evidence range, not just one main memory. Extracted memories are not committed immediately.
4. Run a connection-discovery branch over each extracted memory. This mirrors recall: vector/full-text/facet/link prefiltering finds nearby memory fragments, then connection-fragment agents propose fixed-relation links or identify already-existing memories.
5. Surface extracted memories, evidence refs, nearby memory snippets, and proposed connections to the main memory decision model.
6. The main memory decision model approves/rejects memory writes and approves/rejects links. It can call `load_full_memory(mem_id)` when snippets are insufficient.
7. Approved memories become `MemoryItem`s. Derive facets, `semantic_text`, and `lexical_text`; insert full-text/vector rows.
8. Approved links become sparse `MemoryLink`s. Do not auto-create generic relatedness.
9. Build or update reusable memory fragments:
   - system instruction for memory-agent behavior
   - one or more serialized memory items
   - facets, sparse links, and `reference_count`
   - evidence pointer summary, not the full source transcript by default
   - strict relevance output schema
10. Persist memory items, evidence pointers, indexes, links, fragments, and cache metadata.

### Fragment Blocking

Memory agents operate on cached fragments, while the rest of retrieval works by `mem_id`. A fragment is a stable serialized block containing one or more memory items.

First-pass blocking rules:

- create single-item fragments for high-reference or unusually detailed memories
- group low-token items with overlapping facets into recall fragments
- create connection fragments around linked neighborhoods, such as a memory plus `updates`/`contradicts`/`same_as`
- target small stable fragments first, then experiment with 2K/8K/20K token budgets
- never use fragment ids as canonical memory ids; fragments are cache/index artifacts over `mem_id`s

Important caveat: most hosted chat APIs do not expose portable raw KV-cache objects. The first implementation should use DeepSeek automatic context caching as the baseline and model the rest behind `cache.py`. The retriever should monitor returned cache usage stats, when available, and optionally run a cache warmer to keep high-priority memory-item fragments hot.

## Retrieval Path

Branch 1: private memory retrieval.

1. User asks or conversation state implies a memory need.
2. Main model `M` emits a compact `MemoryQuery`:
   - natural-language query
   - facets
   - time hints
   - desired evidence type
   - max memories / latency budget
3. Multi-channel prefilter returns candidate `mem_id`s from vector, full-text, facet, relation, and policy channels.
4. Merge duplicate candidates by `mem_id` and fan out selected ids to `m_i` memory-fragment agents.
5. Keep affirmative replies containing `mem_id`, request-specific memory tags, descriptors, and structured refs.
6. Group similar request-specific memory tags/descriptors.
7. Resolve grouped memory content by `mem_id`; follow evidence pointers only when deeper source context is explicitly needed.
8. Send each group to summarization agent `S`.
9. Have `M` select the useful memory output for Branch 2.
10. Return structured memory output:

```json
{
  "summaries": [
    {
      "topic": "string",
      "summary": "string",
      "evidence_refs": [],
      "mem_ids": []
    }
  ]
}
```

See [RETRIEVER_PLAN.md](RETRIEVER_PLAN.md) for the concrete retriever contract, trace schema, cache-stat instrumentation, and keep-warm experiment.

Active search fallback: if `M` does not find what it needs from the thresholded vector pass, it can trigger a second retrieval pass that queries all memory-fragment agents directly, bypassing the vector threshold.

Branch 2: clean user-facing conversation.

1. Continue the visible conversation from the original user-visible state.
2. Inject only the extracted memory output, not the prefilter prompts, agent deliberation, cluster scaffolding, or rejected memories.
3. Optionally expose citations/provenance when the response uses memory-backed facts.

## First Milestone

Build a deterministic local harness:

1. `memory.store`: persist `MemoryItem`s, facets, evidence refs, sparse links, FTS rows, and `reference_count`.
2. `memory.memorization`: expose `ActiveMemorizationSkill` for model-triggered memory writes and `IdleAutoMemorizationTrigger` for automatic idle-time writes through the same skill surface.
3. `ingest`: load example conversations and extract `MemoryItem`s from memorable hotspots.
4. `index`: create facets, `semantic_text`, `lexical_text`, Qwen embeddings, and full-text rows.
5. `prefilter`: union `mem_id`s from Qwen vector search, full-text search, facet matching, relation expansion, and policy channels.
6. `memory_agent`: call DeepSeek with a strict JSON schema prompt returning relevance, `mem_id`, request-specific memory tags, and descriptors.
7. `request_tag_grouping`: group similar request-specific memory tags/descriptors.
8. `summarize`: resolve memory content by id and produce `{summary, evidence_refs, mem_ids, memory_tags}`.
9. `select`: have the main model choose the useful memory for Branch 2 and increment `reference_count` for accepted mem ids.
10. `answer`: call the main model using the clean memory bundle.
11. `eval`: compare against hand-written recall questions.

## Key Experiments

- Fragment size: single-item vs facet-grouped vs linked-neighborhood fragments, then 2K/8K/20K token budgets.
- Prefilter K: latency/recall curve.
- Memory-agent model: cheap model for `m_i`, stronger model for `S` and `M`.
- Summary shape: terse facts vs narrative summaries vs evidence tables.
- Cache strategy: DeepSeek automatic context cache, optional keep-warm pings, later local open-weight KV cache if needed.
- Provenance quality: exact spans, offsets, and contradiction detection.

## Implementation Notes

- Keep all prompts versioned in code.
- Treat raw episodes/input records as immutable cold evidence once written.
- Treat memory items as the normal retrieval and storage unit.
- Use `tags` only for request-specific labels produced by memory-fragment agents for the main conversation model; stored labels are called facets.
- Never let retrieval scaffolding enter the visible chat transcript.
- Log every retrieval run as an inspectable trace.
- Prefer async fanout with concurrency limits.
- Make model/provider configurable from `.env`.
- Separate "memory found" from "memory safe to use"; privacy and recency policies should gate final injection.
