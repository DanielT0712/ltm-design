# Retriever Plan

## Purpose

Build the private retrieval branch first. Its job is to turn the latest user prompt plus a compact context description into a provenance-backed memory bundle that the user-facing branch can consume without seeing retrieval scaffolding.

The retriever should answer:

- which memory items are probably relevant?
- which items affirm relevance after reading their stable memory fragment?
- what topics do those blocks support?
- what exact spans justify the final memory bundle?
- how much did DeepSeek automatic context caching help?

## Intended Flow

The retrieval branch can be triggered by:

- an automatic scheduler/timer
- the current model deciding memory would help
- a skill/tool call from the model
- an explicit user request to remember/search/recall
- a background refresh after new memories are written

Once triggered:

1. The current main model sees the latest user prompt and visible conversation context.
2. It creates:
   - a memory-search prompt, paraphrased if the user prompt is too long
   - a much shorter context note describing why this prompt matters
3. The retriever embeds the memory-search prompt.
4. Hybrid prefiltering selects memory items above a similarity threshold, then optionally reranks or boosts by facets, links, and policy.
5. Each selected memory fragment model receives:
   - its stable memory fragment prefix
   - the memory-search prompt
   - the compact context note
   - a structured-output instruction
6. Each memory fragment model returns yes/no relevance. If yes, it returns:
   - `mem_id`
   - relevant memory tags, generated only for this retrieval request
   - 2-3 adjectives/descriptors for how the memory relates
   - memory-item citations or evidence refs when available
7. The main model groups similar request-specific memory tags/descriptors.
8. The retriever calls summarizer models for each group.
9. Summarizer models receive the grouped memory references, the underlying structured memory content, and the same compact context note.
10. Summarizers organize useful memory for the main model, preserving and quoting as much relevant detail as possible.
11. The main model selects useful memory from the summaries.
12. Branch 2 continues the original conversation fresh with the selected memory injected.

The memory fragment models should only decide relevance and produce routing metadata. They may produce request-specific `memory_tags` for grouping and for the main conversation model, but tags are not stored as permanent memory labels. Stored labels are typed facets.

## Memorization And Connection Pipeline

Memory writing uses the same discipline as recall: extracted memories and possible connections are surfaced to the main memory decision model, not silently committed.

1. Persist incoming conversation or multimodal inputs as durable evidence records.
2. The live conversation model starts an active memory-cataloging branch when the memory trigger policy fires, or the idle scheduler starts the same branch after a quiet period.
3. Start the memory extraction branch with:
   - the durable evidence range
   - a cursor describing what evidence has already been processed into memory
   - optional recent visible context
   - optional existing memory snippets if they are cheap to provide
4. The extraction branch returns all worthwhile atomic memories from the evidence range. It does not need to know that these are "candidates" or why it was triggered.
5. Store extracted memories in `pending_memory_candidates`.
6. Automatically run connection discovery for each pending memory:
   - use memory text/facets as a query
   - retrieve nearby existing `mem_id`s with vector, FTS, facet matching, and sparse link expansion
   - ask connection-discovery fragment agents whether the memory already exists in their fragment or should connect to fragment memories
   - proposed links use only `updates`, `contradicts`, `supports`, `elaborates`, `caused_by`, `part_of`, or `same_as`
7. Surface extracted memories, nearby memory snippets, connection proposals, and evidence refs to the main memory decision model.
8. The main memory decision model approves/rejects memory writes and approves/rejects links. It may use `load_full_memory(mem_id)` when snippets are insufficient.
9. Approved memory items are indexed and placed into memory item groups.
10. When placing an approved memory into a recall group, choose the group with the most direct/nearby connections that still has space. Target max group size is 20K tokens. If all connected groups are full, or the memory has no connections, choose the group with the most empty space. If no group can fit the memory, create a new group.

The active memorization skill and idle auto-memorization path share this pipeline. Active memorization starts when the live conversation wants memory processing now. Idle auto-memorization starts the same processing after a quiet period and supplies the next unprocessed evidence range.

### Live Conversation Trigger Policy

The live conversation model should start active memory cataloging when:

- the user explicitly asks to remember, save, note, keep in mind, or update something
- the user corrects a previous assumption or stored understanding
- the user states a durable preference, constraint, goal, commitment, decision, relationship, identity/context fact, or recurring pattern
- the user provides important project/state/context information that future conversations should know
- the user expresses a grounded emotional or multimodal signal useful for future interaction
- the conversation accumulates enough memory-worthy details that waiting for idle processing risks losing structure

It should not start active cataloging for generic facts, ordinary task mechanics, filler, politeness, assistant explanations, or weak guesses.

### Memory Extraction Branch Prompt Contract

The extraction branch should remember everything worth remembering, not compress a conversation into one main memory.

Input:

```json
{
  "processed_cursor": "string",
  "evidence_records": [],
  "recent_context_note": "string",
  "existing_memory_snippets": []
}
```

Output:

```json
{
  "memories": [
    {
      "text": "standalone memory item",
      "facets": {
        "people": [],
        "topics": [],
        "emotions": [],
        "events": [],
        "places": [],
        "objects": [],
        "times": []
      },
      "evidence": [],
      "connection_hints": []
    }
  ],
  "processed_through": "string"
}
```

### Full Memory Tool

Models that receive only a memory snippet can request the full record by `mem_id`.

`load_full_memory(mem_id)` returns:

- memory text
- facets
- evidence refs
- sparse links
- reference count
- fragment memberships when available
- source evidence excerpts only when explicitly requested and permitted

Use this tool for redundancy decisions, relation approval, contradiction checks, evidence-quality checks, and summaries that need exact detail. Do not load every memory by default.

## DeepSeek Caching Position

Use DeepSeek automatic context caching as the default cache layer. Do not build custom KV-cache persistence in the first version.

The memory-agent prompt should be shaped so the stable prefix is maximally reusable:

1. static memory-agent system instruction
2. stable serialized memory item fragment
3. stable output schema
4. volatile query appended last

This gives DeepSeek's automatic cache the best chance to reuse the large memory-block prefix while the query changes per retrieval.

Track cache behavior from every API response. The client wrapper should persist raw usage fields plus normalized metrics:

- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `cached_tokens`, when reported
- `cache_hit_ratio = cached_tokens / prompt_tokens`, when available
- `latency_ms`
- `model`
- `mem_id`, for memory-agent calls
- `request_kind`: `memory_agent`, `summarizer`, `query_planner`, `answer`

If a response does not expose cache stats, store `null` rather than guessing.

## Keep-Warm Job

Add keep-warm as an optional experiment, not a correctness requirement.

`memory/cache_warmer.py`

- selects memory items likely to be retrieved soon
- sends a cheap no-op relevance prompt against each selected item fragment
- records cache stats and latency
- backs off when cache hit ratio is already high or spend budget is reached

Initial policy:

- run manually from `scripts/warm_cache.py`
- later support a scheduled loop
- warm only top-priority item fragments:
  - recently written memory items
  - frequently retrieved memory items
  - user-pinned blocks
  - blocks with poor latency but high reuse
- configurable interval, default 30 minutes
- configurable max blocks per run, default 25
- configurable daily spend ceiling

The warm prompt must not create user-visible memory output. It should use the same stable prefix as the real memory-agent calls and append a trivial query such as: "Warm this memory context. Return not relevant."

## Embeddings And Prefiltering

DeepSeek's official API docs currently document chat/completion models and OpenAI-compatible chat usage, but do not clearly document a first-party embeddings model. Treat embeddings as a swappable local/provider component rather than binding them to DeepSeek.

Embedding implementation:

- local/open embeddings with `sentence-transformers`
- default model: `Qwen/Qwen3-Embedding-8B`
- use Qwen's `query:` prefix for query embeddings
- store vectors in SQLite `memory_embeddings` rows keyed by `(mem_id, model)`
- search vectors with cosine similarity over normalized vectors
- combine vector candidates with SQLite FTS5 and exact facet matching by `mem_id`

Why Qwen first:

- open weights under Apache 2.0
- still-strong public benchmark results for multilingual retrieval and related embedding tasks
- 32K context, useful for larger memory fragments
- instruction-aware query embeddings
- Matryoshka-style flexible output dimensions up to 4096
- cheap enough for repeated memory indexing
- deterministic and inspectable
- avoids coupling retrieval experiments to one chat provider

Qwen model options:

- `Qwen/Qwen3-Embedding-8B`: highest quality default
- `Qwen/Qwen3-Embedding-4B`: lighter fallback
- `Qwen/Qwen3-Embedding-0.6B`: local/dev fallback

Currentness note as of 2026-05-04:

- `Qwen/Qwen3-Embedding-8B` was released in 2025, so it is not the newest embedding model.
- It remains a good default for this repo because it is open/local, high quality, long-context, and easy to cache.
- Newer 2026 hosted/multimodal candidates such as Gemini Embedding 2 should be treated as A/B providers, not as the default dependency.
- If open/local quality becomes the bottleneck, evaluate newer open models and Qwen rerankers against our own memory-recall evals rather than trusting one public leaderboard.

Interface:

`EmbeddingProvider`

- `embed_query(text: str) -> list[float]`
- `embed_documents(texts: list[str]) -> list[list[float]]`
- `model_name`
- `dimensions`

Later provider options:

- OpenAI embeddings
- Voyage/Cohere/Jina embeddings
- Gemini embeddings
- BGE/E5/GTE models
- Qwen reranker after vector prefiltering

Vector similarity alone is not the whole retrieval decision. It is the first recall gate.

Best prefilter path for this architecture:

1. dense embedding search for broad semantic recall
2. full-text search for exact words, names, ids, rare phrases, and facet values
3. metadata filters for namespace/time/user scope
4. facet lexical boosts for exact hooks
5. optional reranker over the top candidates
6. memory-fragment agents for final yes/no relevance and request-specific `memory_tags`
7. direct-all-fragments fallback for active searches that miss on the first pass

This keeps vector search fast while avoiding the common failure mode where semantically relevant but lexically odd memories fall below threshold.

## Full-Text Search

Full-text search should mean an inverted index over normalized lexical fields, not a hand-rolled keyword scan.

Recommended v0:

- SQLite for the local prototype's metadata and memory storage, with the storage interface kept portable to hosted relational stores such as PlanetScale/MySQL or Postgres later
- SQLite FTS5 virtual table for full-text search
- Qwen embeddings stored beside the same memory items

FTS5 may expose BM25 ranking, but the retriever should not collapse every channel into one blended score. Each channel gets its own threshold and contributes candidate `mem_id`s independently.

What FTS searches:

- exact names
- tool names
- file paths
- unusual phrases
- facets
- short memory titles
- structured memory values serialized as text

What FTS is good at:

- "that conversation about sqlite-vec"
- "DeepSeek cache stats"
- "the API key env issue"
- "Qwen3-Embedding-8B"
- exact phrase and rare-token recall

What FTS is not good at:

- paraphrased semantic meaning
- fuzzy conceptual similarity
- memories where none of the words overlap

That is why it pairs with Qwen dense embeddings.

Storage-stage requirement:

Every memory item should produce two search documents:

1. `semantic_text`: natural language text for embedding
2. `lexical_text`: denormalized text for full-text search

`semantic_text` should be clean prose:

- the memory item text or a compact normalized memory statement
- enough context to represent meaning
- no excessive boilerplate

`lexical_text` should be search-friendly:

- memory item text
- memory title
- facets repeated plainly by category
- ids
- source names
- important exact strings

Example:

```text
title: DeepSeek API key and cache plan
facets.people: Daniel
facets.topics: DeepSeek API key cache prompt-cache Qwen embeddings retriever
facets.objects: .env cached_tokens SQLite FTS5
facets.events: memory storage design discussion
body: Daniel wanted memory storage to keep extracted user-relevant hotspots instead of literal transcript chunks.
```

SQLite sketch:

```sql
CREATE TABLE memory_items (
  mem_id TEXT PRIMARY KEY,
  title TEXT,
  text TEXT NOT NULL,
  facets_json TEXT NOT NULL,
  evidence_json TEXT NOT NULL,
  semantic_text TEXT NOT NULL,
  lexical_text TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE evidence_records (
  source_id TEXT PRIMARY KEY,
  conversation_id TEXT,
  text TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE pending_memory_candidates (
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

CREATE TABLE memory_links (
  from_mem_id TEXT NOT NULL,
  to_mem_id TEXT NOT NULL,
  relation TEXT NOT NULL CHECK (
    relation IN ('updates', 'contradicts', 'supports', 'elaborates', 'caused_by', 'part_of', 'same_as')
  ),
  created_at TEXT NOT NULL,
  PRIMARY KEY (from_mem_id, to_mem_id, relation)
);

CREATE TABLE memory_fragments (
  fragment_id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('recall', 'connection')),
  title TEXT NOT NULL,
  mem_ids_json TEXT NOT NULL,
  text TEXT NOT NULL,
  token_count_estimate INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE memory_items_fts USING fts5(
  mem_id UNINDEXED,
  title,
  lexical_text,
  facets,
  tokenize = 'unicode61 remove_diacritics 2'
);
```

At write time:

1. persist evidence records
2. propose memory candidates from active model calls or idle extraction
3. run connection discovery against existing memories
4. surface candidates and proposed links to the main model
5. approve/reject candidates and links
6. create stable `mem_id`s for approved candidates
7. derive facets
8. build `semantic_text`
9. build `lexical_text`
10. embed `semantic_text` with Qwen
11. insert memory rows, FTS rows, vector rows, approved links, and fragment rows

At retrieval time:

1. run Qwen vector search over `semantic_text` embeddings
2. run FTS5 search over `lexical_text`, `title`, and serialized facets
3. run exact facet/id matching
4. optionally expand from strong candidates through sparse memory links
4. apply each channel's own threshold
5. union candidate ids by `mem_id`
6. remove overlap by merging channel evidence
7. pass only selected `mem_id`s to memory-fragment agents

Candidate merging must track ids, not files or raw memory payloads:

```json
{
  "mem_id": "mem_123",
  "channels": ["vector", "full_text", "facet"],
  "evidence": {
    "vector": {"similarity": 0.71},
    "full_text": {"matched_terms": ["DeepSeek", "cache"], "rank": 3},
    "facet": {"matched_facets": {"topics": ["memory storage"], "objects": ["DeepSeek"]}}
  }
}
```

The prefilter can use per-channel caps to control fanout:

- vector: keep if `similarity >= vector_threshold`, then cap to `vector_top_k`
- full-text: keep if `matched_terms >= min_term_matches` or rank within `fts_top_k`
- exact facets: keep if any strong people/topic/emotion/event/place/object/time/id facet matches
- relation links: expand sparingly from strong candidates using `updates`, `contradicts`, `supports`, `elaborates`, `caused_by`, `part_of`, or `same_as`
- pinned/recent/manual: keep by policy

Selection into memory-fragment agents should preserve channel diversity instead of sorting by one universal number.

If the query contains exact-looking tokens, expand the full-text channel:

- quoted text
- file paths
- code symbols
- model names
- proper nouns
- ids
- dates

For those queries, do not weight BM25 into a blended score. Instead, relax or expand the full-text channel threshold/cap while leaving vector thresholds unchanged.

## Retriever Inputs

`RetrievalRequest`

- `conversation_id`
- `turn_id`
- `visible_messages`: current clean thread
- `latest_user_message`
- `retrieval_hint`: optional caller-provided query
- `top_k`: default 32
- `max_agent_calls`: default 16
- `max_clusters`: default 5
- `latency_budget_ms`: optional
- `include_debug`: default false

`ComposedMemoryQuery`

- `search_text`: paraphrased latest prompt for semantic search
- `context_note`: concise description of the current conversation context
- `original_user_prompt`
- `trigger_kind`: automatic, model_skill, timer, explicit_user, active_search
- `search_mode`: thresholded, direct_all_fragments
- `similarity_threshold`: default 0.35 for local cosine embeddings, to be calibrated

## Retriever Outputs

`RetrievalResult`

- `memory_bundle`: safe object for the user-facing branch
- `trace_id`: pointer to full private trace
- `stats`: compact timings and cache metrics
- `empty_reason`: when no memory is returned

`MemoryBundle`

```json
{
  "query": "string",
  "summaries": [
    {
      "topic": "string",
      "summary": "string",
      "mem_ids": ["string"],
      "evidence_refs": [
        {
          "mem_id": "string",
          "source_id": "string",
          "path": "string"
        }
      ],
      "contradictions": []
    }
  ]
}
```

The user-facing branch receives only `memory_bundle`, never the raw retrieval trace.

## Pipeline

### 1. Trigger And Query Composition

Convert visible conversation state into `ComposedMemoryQuery`.

`ComposedMemoryQuery`

- `search_text`: a paraphrase of the latest user prompt optimized for memory search
- `context_note`: an even shorter note about the local context for the prompt
- `facets`: optional extracted query facets
- `time_hints`: optional temporal hints
- `intent`: preference, fact, relationship, ongoing_context, decision, correction, other
- `must_have_evidence`: boolean

First implementation can be heuristic:

- if the latest user message is short, use it as `search_text`
- if long, call the main model to paraphrase it into a memory-search query
- create `context_note` from the previous 1-3 visible turns or a rolling clean-thread summary
- skip deeper model-based planning until the rest of the retriever works

### 2. Multi-Channel Prefilter

Embed `search_text` with Qwen, run full-text search, run facet/id matching, optionally expand through sparse links, and return a union of candidate `mem_id`s. Each retrieval channel has its own threshold and cap.

Channels:

- `vector`: cosine similarity against memory-fragment embedding
- `full_text`: lexical match against title/facets/lexical text
- `facet`: exact or normalized facet/id match
- `relation`: sparse expansion through fixed memory links
- `policy`: pinned, recent, frequently retrieved, or user-prioritized memories
- `reranker`: optional second-stage rerank over already selected candidate ids

Output `CandidateMemory`

- `mem_id`
- `channels`
- `channel_evidence`
- `token_count`
- `cache_priority`
- `reference_count`
- `matched_facets`
- `matched_terms`

### 3. Candidate Selection

Choose which candidates become memory-agent calls.

Rules:

- keep any fragment that crosses at least one channel threshold
- merge duplicate candidates by `mem_id`
- cap at `max_agent_calls` only after channel union
- skip blocks above token budget
- prefer diversity by conversation/time/topic when scores are close
- include pinned memories even if vector rank is mediocre
- resolve memory-item content only after ids are selected
- resolve raw evidence only if the summarizer or main model explicitly needs deeper context

### 4. Memory-Agent Fanout

For each selected memory fragment, call its memory fragment model with:

- stable system instruction
- stable serialized memory fragment
- stable JSON schema
- volatile `search_text` and `context_note` last

Expected JSON:

```json
{
  "mem_id": "string",
  "relevant": true,
  "memory_tags": ["string"],
  "descriptors": ["string"],
  "reason": "string",
  "evidence_refs": [
    {
      "source_id": "string",
      "path": "string"
    }
  ],
  "structured_refs": [
    {
      "mem_id": "string",
      "facet_path": "string",
      "path": "string"
    }
  ]
}
```

Acceptance threshold:

- keep `relevant = true`
- require at least one request-specific memory tag or structured ref
- prefer memory-item text and evidence pointers for factual claims, but allow structured refs when the summary model can load memory by id/facet

Run with async concurrency, default 8.

### 5. Request-Tag Grouping

Group affirmative replies by similar request-specific memory tags/descriptors.

First implementation:

- normalize tags and descriptors
- embed `memory_tags + descriptors + reason`
- group above a similarity threshold
- cap at `max_clusters`

Fallback:

- group by exact normalized request-specific memory tag

### 6. Summarization Agents

Each cluster goes to summarizer `S`.

Input:

- `search_text`
- `context_note`
- grouped request-specific memory tags/descriptors
- structured memory content resolved by `mem_id`
- raw evidence excerpts only when explicitly requested by the conversation model or summarizer

Output:

- concise summary
- mem_ids
- request-specific memory tags / citations
- evidence refs or short quotes
- contradictions

Summarizer must preserve uncertainty, quote relevant details when useful, cite memory ids and request-specific tags, and avoid filling gaps from general model knowledge.

### 7. Main-Model Memory Selection

The main model receives the grouped summaries and picks only the useful information for Branch 2.

This step should:

- discard merely adjacent memories
- preserve exact details that matter
- keep citations to memory ids and request-specific tags
- stay under the Branch 2 memory-injection token budget

### 8. Final Memory Extraction

Normalize and filter cluster summaries into `MemoryBundle`.

Final checks:

- remove summaries with no spans
- deduplicate repeated claims
- sort by relevance
- enforce token budget for Branch 2
- redact private/tool/internal fields

### 9. Active Search Fallback

The model can actively search for memories.

First attempt:

- same thresholded flow as above
- vector prefilter selects only memory fragments above threshold

If the model does not find what it wants:

- second attempt switches `search_mode` to `direct_all_fragments`
- query all memory fragment models directly, bypassing the vector threshold
- keep stricter per-agent acceptance threshold to control noise
- optionally limit to memory namespaces/time ranges if the model can specify them

This fallback is expensive, so trace it distinctly and include token/cost stats.

## Trace Schema

Every retrieval run writes one JSONL or SQLite trace.

`RetrievalTrace`

- `trace_id`
- `conversation_id`
- `turn_id`
- `created_at`
- `request`
- `memory_query`
- `composed_memory_query`
- `prefilter_candidates`
- `selected_candidates`
- `agent_replies`
- `request_tag_groups`
- `summaries`
- `final_bundle`
- `timings`
- `cache_stats`
- `errors`

Trace files live under `runs/retrieval/` and are gitignored.

## Metrics

Core metrics:

- recall@K against eval questions
- accepted memories per query
- answer accuracy with vs without retriever
- raw-span precision
- contradiction rate
- p50/p95 latency
- token cost by stage
- cache hit ratio by memory-item fragment
- cache warm effectiveness
- active-search second-pass frequency

Cache-specific metrics:

- memory-agent latency with high vs low cached-token ratio
- hit ratio decay over time
- warm-job cost vs saved latency/cost
- best warm interval by block reuse pattern

## Implementation Order

1. Define Pydantic schemas for requests, replies, bundles, traces, and stats.
2. Build DeepSeek client wrapper with response usage/caching stats capture.
3. Build local JSON/SQLite memory store.
4. Build storage-stage indexing: facets, `semantic_text`, `lexical_text`, Qwen vector, and SQLite FTS5.
5. Build deterministic multi-channel prefilter using vector threshold, full-text threshold, facet match, relation expansion, and id union.
6. Build memory-fragment agent prompt and parser.
7. Build async fanout and trace logging.
8. Add request-tag grouping.
9. Add summarizer.
10. Add main-model memory selection and record final accepted mem ids through `MemoryStore.record_final_acceptance`.
11. Add final bundle sanitizer.
12. Add active-search fallback.
13. Add manual cache warmer script.
14. Add eval scenarios.

## Open Decisions

- Embeddings provider: Qwen3-Embedding-8B first, with 4B/0.6B fallbacks for constrained machines.
- Store: SQLite first, or JSONL until retrieval logic settles.
- Tag grouping library: sklearn, scipy, or no dependency for v0.
- Whether query composition uses `M` immediately or starts heuristic-only.
- Whether Branch 2 answer generation lives in this repo's first milestone or waits until retriever quality is measurable.
