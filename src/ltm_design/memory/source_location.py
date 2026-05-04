from __future__ import annotations

from ltm_design.memory.models import EvidenceRecord, SourceAnchorRef, SourceSpan


def locate_source_span(
    record: EvidenceRecord,
    quote_or_summary: str,
    *,
    speaker_hint: str | None = None,
) -> SourceSpan | None:
    """Locate exact text inside an evidence record when a cataloger supplies identifying text."""
    needle = quote_or_summary.strip()
    if not needle:
        return None

    start = record.text.find(needle)
    if start == -1:
        folded_start = record.text.casefold().find(needle.casefold())
        if folded_start == -1:
            return None
        start = folded_start

    end = start + len(needle)
    return SourceSpan(
        source_id=record.source_id,
        speaker=speaker_hint or "unknown",
        char_start=start,
        char_end=end,
        matched_text=record.text[start:end],
    )


def resolve_source_anchors(record: EvidenceRecord, ref: SourceAnchorRef) -> SourceSpan | None:
    """Resolve exact start/end anchors into source coordinates."""
    if ref.source_id != record.source_id:
        return None
    if ref.whole_source:
        return SourceSpan(
            source_id=record.source_id,
            message_id=ref.message_id,
            event_id=ref.event_id,
            speaker=ref.speaker,
            char_start=0,
            char_end=len(record.text),
            matched_text=record.text,
        )
    if ref.start_anchor is None or ref.end_anchor is None:
        return None
    start = record.text.find(ref.start_anchor)
    if start == -1:
        return None
    end_anchor_start = record.text.find(ref.end_anchor, start)
    if end_anchor_start == -1:
        return None
    end = end_anchor_start + len(ref.end_anchor)
    return SourceSpan(
        source_id=record.source_id,
        message_id=ref.message_id,
        event_id=ref.event_id,
        speaker=ref.speaker,
        char_start=start,
        char_end=end,
        matched_text=record.text[start:end],
    )
