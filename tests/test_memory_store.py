from datetime import timedelta

from ltm_design.memory.memorization import (
    ActiveMemorizationSkill,
    IdleAutoMemorizationTrigger,
    MainModelMemoryDecision,
    MemorizationRequest,
    MemoryWriter,
)
from ltm_design.memory.models import (
    EvidenceRecord,
    EvidenceRef,
    Facets,
    FragmentKind,
    MemoryCandidate,
    MemoryLink,
    MemoryRelation,
    SourceAnchorRef,
    utc_now,
)
from ltm_design.memory.source_location import locate_source_span, resolve_source_anchors
from ltm_design.memory.store import MemoryStore


def test_active_memorization_persists_text_facets_and_evidence(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    writer = MemoryWriter(store)
    skill = ActiveMemorizationSkill(writer)

    result = skill.call(
        MemorizationRequest(
            text="Daniel wants memory tags reserved for fragment-agent output only.",
            facets=Facets(
                people=("Daniel",),
                topics=("memory storage", "fragment agents"),
                events=("storage design discussion",),
            ),
            evidence=(EvidenceRef(source_id="episode_1", locator={"message_id": "msg_1"}),),
        )
    )

    assert len(result.mem_ids) == 1
    item = store.get(result.mem_ids[0])
    assert item is not None
    assert item.reference_count == 0
    assert item.facets.people == ("Daniel",)
    assert item.evidence[0].source_id == "episode_1"
    assert store.search_fts("fragment")[0].mem_id == item.mem_id


def test_reference_count_tracks_final_model_acceptance(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    item = store.remember(MemoryCandidate(text="Daniel prefers sparse fixed memory relations."))

    store.record_reference(item.mem_id)
    store.record_reference(item.mem_id, amount=2)
    store.record_final_acceptance((item.mem_id,))

    assert store.get(item.mem_id).reference_count == 4


def test_auto_memorization_runs_after_idle_period(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    skill = ActiveMemorizationSkill(MemoryWriter(store))
    trigger = IdleAutoMemorizationTrigger(skill, idle_after=timedelta(minutes=10))
    now = utc_now()

    early = trigger.run(
        last_interaction_at=now - timedelta(minutes=5),
        now=now,
        candidates=(MemoryCandidate(text="Too soon."),),
    )
    later = trigger.run(
        last_interaction_at=now - timedelta(minutes=11),
        now=now,
        candidates=(MemoryCandidate(text="Daniel wants idle-time automatic memory extraction."),),
    )

    assert early is None
    assert later is not None
    assert later.trigger == "auto_idle"
    assert len(later.candidate_ids) == 1
    assert len(store.list_items()) == 0

    skill.apply_main_model_decision(
        MainModelMemoryDecision(approve_candidate_ids=later.candidate_ids)
    )

    assert len(store.list_items()) == 1


def test_sparse_fixed_links_are_stored(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    base = store.remember(MemoryCandidate(text="Daniel dislikes generic related_to memory links."))
    elaboration = store.remember(
        MemoryCandidate(
            text="Daniel wants memory links to use a fixed relation set.",
            links=(MemoryLink(to_mem_id=base.mem_id, relation=MemoryRelation.ELABORATES),),
        )
    )

    links = store.links_for(elaboration.mem_id)
    assert links == [MemoryLink(to_mem_id=base.mem_id, relation=MemoryRelation.ELABORATES)]


def test_connected_mem_ids_walks_two_layers_by_default(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.remember(MemoryCandidate(text="First memory."))
    second = store.remember(MemoryCandidate(text="Second memory."))
    third = store.remember(MemoryCandidate(text="Third memory."))
    store.add_links(first.mem_id, (MemoryLink(to_mem_id=second.mem_id, relation=MemoryRelation.ELABORATES),))
    store.add_links(second.mem_id, (MemoryLink(to_mem_id=third.mem_id, relation=MemoryRelation.SUPPORTS),))

    assert store.connected_mem_ids(first.mem_id) == (second.mem_id, third.mem_id)
    assert store.connected_mem_ids(first.mem_id, relation_depth=1) == (second.mem_id,)


def test_evidence_records_are_durable(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    record = EvidenceRecord(
        source_id="episode_1",
        conversation_id="thread_1",
        text="Daniel said memory evidence should be long-term storage, not a temporary buffer.",
        metadata={"kind": "conversation"},
    )

    store.add_evidence(record)

    fetched = store.get_evidence("episode_1")
    assert fetched is not None
    assert fetched.text == record.text
    assert fetched.metadata == {"kind": "conversation"}


def test_memory_fragment_blocks_multiple_memory_items(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    first = store.remember(MemoryCandidate(text="Daniel wants sparse memory links."))
    second = store.remember(MemoryCandidate(text="Daniel wants fragment blocking planned explicitly."))

    fragment = store.build_fragment(
        kind=FragmentKind.RECALL,
        mem_ids=(first.mem_id, second.mem_id),
        title="Memory architecture preferences",
    )

    assert fragment.mem_ids == (first.mem_id, second.mem_id)
    assert first.mem_id in fragment.text
    assert second.mem_id in fragment.text


def test_locate_source_span_returns_stable_coordinates():
    record = EvidenceRecord(
        source_id="episode_1",
        text="User: I want facets to be typed recall handles, not tags.",
    )

    span = locate_source_span(record, "facets to be typed recall handles", speaker_hint="user")

    assert span is not None
    assert span.source_id == "episode_1"
    assert span.speaker == "user"
    assert span.matched_text == "facets to be typed recall handles"
    assert span.to_evidence_ref().locator["char_start"] == span.char_start


def test_resolve_source_anchors_uses_exact_start_and_end_words():
    record = EvidenceRecord(
        source_id="thread_1",
        text="[src:thread_1 msg:u_0042] User: I want facets to be typed recall handles, not tags.",
    )
    ref = SourceAnchorRef(
        source_id="thread_1",
        message_id="u_0042",
        speaker="user",
        start_anchor="I want facets",
        end_anchor="not tags.",
    )

    span = resolve_source_anchors(record, ref)

    assert span is not None
    assert span.matched_text == "I want facets to be typed recall handles, not tags."


def test_resolve_source_anchors_can_reference_whole_source():
    record = EvidenceRecord(
        source_id="thread_1",
        text="[src:thread_1 msg:u_0042] User: Remember this whole prompt.",
    )
    ref = SourceAnchorRef(
        source_id="thread_1",
        message_id="u_0042",
        speaker="user",
        whole_source=True,
    )

    span = resolve_source_anchors(record, ref)

    assert span is not None
    assert span.char_start == 0
    assert span.char_end == len(record.text)
    assert span.matched_text == record.text
