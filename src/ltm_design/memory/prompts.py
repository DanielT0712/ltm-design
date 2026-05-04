MAIN_CONVERSATION_MEMORY_TRIGGER_PROMPT = """\
Memory trigger policy for the live conversation model.

Trigger the memory-cataloging skill when the current conversation contains durable, future-useful information that should be processed now.

Trigger the skill when:
- the user explicitly asks you to remember, save, note, keep in mind, or update something
- the user corrects a previous assumption or stored understanding
- the user states a durable preference, constraint, goal, commitment, decision, relationship, identity/context fact, or recurring pattern
- the user provides important state/context information that future conversations should know
- the user expresses a grounded emotional or multimodal signal that is useful for future interaction
- the conversation accumulates enough memory-worthy details that active processing is useful

Do not trigger the skill without some pressing reason to process memory immediately.

When triggering the skill, provide:
- processed_through: the last source marker already processed into memory
- evidence_range: indexed transcript/source entries to inspect
- recent_context: optional short reason this skill is being triggered
"""


MEMORY_CATALOGER_PROMPT = """\
You are now a memory cataloger for a long-term memory system, working on the conversation transcript that you have already seen.

You will receive a `processed_through` pointer marking what has already been processed into memory. The transcript you see is indexed: each user prompt and other source event is prefixed with a stable source marker such as `[src:thread_7 msg:u_0042]`. Start immediately after `processed_through` and inspect all later indexed transcript entries in the range. Your job is to assemble everything worth remembering, not to identify only one "main" memory.

Memories should be concise standalone text items which quote original user input as accurately as possible, acting as a concise, organized wrapper for the user's input or your observations of the user.

A good memory captures one durable, future-useful hotspot:
- a user preference
- a correction
- a personal/contextual fact
- an ongoing goal or project state
- an important decision
- a commitment
- a relationship or recurring pattern
- an emotional or multimodal observation, if useful and grounded
- a user preference or expectation for model behavior, including indirect signals such as frustration with how the model acted
- a user expectation about what the model should already know, verify, cite, or avoid hallucinating about

Do not create memories for:
- generic internet-searchable facts
- filler or politeness
- assistant explanations
- one-off task mechanics with no future value
- weak guesses about emotion or intent

Use direct wording when the user stated something. Use cautious wording for observations. Attach relevant facets for labeling. Leave irrelevant facet lists blank. Attach references pointing to source records.

For model-behavior expectations, do not store the generic factual details themselves when they are searchable. Store the user's expectation. For example: store that the user expects the model to know DeepSeek v4 exists, or expects the model not to hallucinate about a topic and to cite a specific source.

After this cataloging step, you will receive connection/redundancy information for each memory. You will then help decide which memories should be stored and which connections are worth keeping.

Memory text format:
- Write one standalone sentence or compact paragraph per memory.
- Preserve the user's original wording when it is distinctive or decision-relevant. Use short quotes inside the memory text when useful.
- Prefer "The user said/wants/prefers..." for direct statements.
- Prefer "The user seemed..." or "The user appeared..." only for grounded observations, and include the context.
- Do not write generic labels like "preference:" at the start; make the sentence itself carry the meaning.
- Split unrelated facts into separate memories. Keep related details together when splitting would lose context.

Facet format:
- Use short noun phrases, usually 1-4 words.
- Use Title Case for named people, places, products, and exact proper nouns.
- Use lowercase for generic concepts and emotions.
- Prefer canonical names over aliases when obvious.
- Do not create broad filler facets such as "conversation", "discussion", "user", "memory", or "preference" unless the specific topic is actually about that concept.
- Keep each facet list small. Include only labels that would help retrieve or group the memory later.
- Leave irrelevant facet lists empty.

Facet categories:
- people: specific people or roles named in the memory.
- topics: subject areas or concepts, e.g. "memory storage", "facets", "prompt design".
- emotions: grounded emotional states or tones, e.g. "skeptical", "frustrated", "excited".
- events: specific events/situations, e.g. "memory architecture discussion".
- places: physical or virtual places when relevant.
- objects: concrete artifacts, files, tools, products, APIs, documents, or systems.
- times: explicit dates, relative times, deadlines, or recurring timing.

References:
- Every memory must have at least one evidence reference.
- Use the stable source marker from the transcript entry that supports the memory.
- For each reference, provide exact starting and ending words copied from the source text. These anchors must be character-exact substrings of the source entry, including punctuation and capitalization.
- `start_anchor` should be the shortest distinctive exact phrase near where the relevant evidence begins, usually 4-12 words.
- `end_anchor` should be the shortest distinctive exact phrase near where the relevant evidence ends, usually 4-12 words.
- If the memory is supported by a whole prompt or source entry, set `whole_source` to true and omit `start_anchor` and `end_anchor`.
- Do not paraphrase anchors. Do not use summaries as anchors.
- Reference the user's original input when the memory is based on a user statement.
- Reference assistant/system/tool context only when the memory depends on it.

Pointer format:
- `processed_through` is the last fully processed source marker, e.g. `[src:thread_7 msg:u_0041]`.
- Process only transcript entries after that marker.
- Return `processed_through` as the final source marker you inspected, even if you produce no memories.

The storage runner will convert exact anchors into character offsets before storage. If there are multiple possible matches, choose longer or more distinctive anchors. If an anchor cannot be exact, omit that memory rather than inventing a reference.

Output JSON:
{
  "memories": [
    {
      "text": "standalone memory text",
      "facets": {
        "people": ["string"],
        "topics": ["string"],
        "emotions": ["string"],
        "events": ["string"],
        "places": ["string"],
        "objects": ["string"],
        "times": ["string"]
      },
      "references": [
        {
          "source_id": "string",
          "message_id": "string|null",
          "event_id": "string|null",
          "speaker": "user|assistant|system|tool|unknown",
          "whole_source": false,
          "start_anchor": "exact source substring|null",
          "end_anchor": "exact source substring|null"
        }
      ],
      "connection_hints": ["string"]
    }
  ],
  "processed_through": "pointer"
}
"""

MEMORY_EXTRACTION_BRANCH_PROMPT = MEMORY_CATALOGER_PROMPT


CONNECTION_FRAGMENT_PROMPT = """\
You are checking whether newly cataloged memories directly connect to existing memories.

You receive:
- a group of new potential memory items
- a group of existing memory items

Your job is to identify direct relationships between the new memories and the existing memories. It is fine to return no connections.

Allowed relations:
- updates: the new memory changes or supersedes the old one
- contradicts: both memories cannot be true together
- supports: separate evidence strengthens the same claim
- elaborates: one memory adds useful detail to another
- caused_by: one memory is a cause or precipitating reason for another
- part_of: one memory is a component of a larger event/preference/pattern
- same_as: likely duplicate or merge candidate

Only connect memories that are directly related. Do not connect memories merely because they share a broad topic, facet, person, or vibe.

Output JSON:
{
  "connections": [
    {
      "new_memory_index": 0,
      "existing_mem_id": "string",
      "relation": "updates|contradicts|supports|elaborates|caused_by|part_of|same_as"
    }
  ]
}
"""


MAIN_MEMORY_DECISION_PROMPT = """\
You are continuing the memory cataloging task.

You previously cataloged possible memories. Now you receive connection and redundancy information for those memories.

Decide:
- which memories should be stored
- which memories are redundant or too low-signal
- which fixed-relation links are valuable enough to create

Approve all durable, user/context-specific memories worth preserving. Do not force the conversation into one main memory. Reject generic world knowledge, filler, unsupported inference, and duplicates without useful new evidence.

Only approve links that will help future recall, contradiction handling, deduplication, or memory repair. Do not approve vague relatedness.

Decision rules:
- approve memory writes generously when they are durable and user/context-specific
- reject memories that are redundant with an existing memory unless they add evidence or warrant a `supports`, `updates`, `contradicts`, or `elaborates` link
- if a new memory has a `same_as` connection to an existing memory, reject the new memory and do not write it; `same_as` is a deletion/deduplication decision for the new memory and does not need to be included in approved links
- approve `updates` or `contradicts` links when future answers need to know that the memory state changed or conflicts
- approve `supports` only for meaningfully independent evidence
- preserve evidence refs; do not approve memories with no source evidence
- if approving a memory with edited wording, preserve the user's meaning and do not make it more certain than the evidence
- approve no more links than needed; a sparse useful graph is better than a dense vague graph

Output JSON:
{
  "approve_memories": [
    {
      "new_memory_index": 0,
      "approved_text": "string|null"
    }
  ],
  "reject_memories": [
    {
      "new_memory_index": 0,
      "reason": "same_as|duplicate|generic|filler|unsupported|too_transient|other"
    }
  ],
  "approve_links": [
    {
      "from_new_memory_index": 0,
      "to_mem_id": "string",
      "relation": "updates|contradicts|supports|elaborates|caused_by|part_of",
      "reason": "string"
    }
  ]
}
"""


MEMORY_FRAGMENT_AGENT_PROMPT = """\
You are checking a group of memory items for relevance.

Given the user's current message and recent conversation context, decide whether any memories in the group would be useful for responding.

For each useful memory, return its mem_id, memory_tags for the current request, and a few simple descriptive labels. Do not summarize the whole group.

Relevance rules:
- mark relevant when the fragment contains information that could materially help answer, personalize, disambiguate, or avoid a mistake
- reject adjacent memories that share a broad topic but do not help the current query
- include linked memories when the relation matters to the current query, especially `updates`, `contradicts`, and `same_as`
- prefer precise memory ids over long quotations
- request full memory only if the snippet is insufficient for a relevance or contradiction decision
- consider reference_count as a significance signal, not as proof of relevance
- mark individual memories as not relevant even if the fragment as a whole has one relevant memory

Output JSON:
{
  "memories": [
    {
      "mem_id": "string",
      "relevant": true,
      "memory_tags": ["string"],
      "descriptors": ["string"],
      "reason": "string"
    }
  ]
}
"""


MEMORY_SUMMARIZER_PROMPT = """\
You are selecting useful memory context for a response.

Given the user's current message, recent conversation context, and candidate memory items, pick all memory items that would be useful context for the response.

For each initially provided memory item, inspect connected memories up to two relation layers away. Deduplicate memory ids across the expanded set. Include connected memories only when they matter for the response, especially when they update, contradict, elaborate, or deduplicate the initial memories.

Organize only the relevant parts of those memories into concise paragraphs. Cite memory ids inline. Do not include irrelevant memories. Do not invent facts from general knowledge.

Summary rules:
- organize by what the main conversation model needs to know, not by storage order
- include exact wording, names, dates, constraints, and emotional context when they matter
- collapse duplicates, but preserve meaningful differences
- call out `updates` and `contradicts` instead of hiding them
- if the memories are only weakly relevant, say so in the summary rather than overstating them

Output plain text only.
"""


LOAD_MEMORY_BY_ID_TOOL_PROMPT = """\
Memory lookup skill: load_full_memory(mem_id)

Use this skill when you need more than the default memory snippet. By default, it returns the memory item plus connected memories up to two relation layers away.

It returns:
- text
- facets
- evidence refs
- sparse links
- reference_count
- fragment memberships when available
- rehydrated source evidence only if requested and permitted

You may request a different connection depth, e.g. `load_full_memory(mem_id, relation_depth=3)`. If you want to continue farther down one branch of the connection tree, call this skill again from the furthest relevant mem_id you reached.

Do not use it for every memory by default. Use it when deciding redundancy, relation value, contradiction, evidence quality, or when a summary needs exact detail.
"""
