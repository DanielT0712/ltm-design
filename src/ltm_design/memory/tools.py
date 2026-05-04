from __future__ import annotations

from dataclasses import dataclass

from ltm_design.memory.store import MemoryStore


@dataclass(frozen=True)
class FullMemoryRecord:
    root_mem_id: str
    records: list[dict]


class MemoryTools:
    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def load_full_memory(self, mem_id: str, *, relation_depth: int = 2) -> FullMemoryRecord:
        mem_ids = (mem_id, *self.store.connected_mem_ids(mem_id, relation_depth=relation_depth))
        records: list[dict] = []
        for current_mem_id in dict.fromkeys(mem_ids):
            item = self.store.get(current_mem_id)
            if item is None:
                continue
            records.append(
                {
                    "mem_id": item.mem_id,
                    "title": item.title,
                    "text": item.text,
                    "facets": item.facets.to_dict(),
                    "evidence": [ref.to_dict() for ref in item.evidence],
                    "links": [link.to_dict() for link in self.store.links_for(item.mem_id)],
                    "reference_count": item.reference_count,
                }
            )
        return FullMemoryRecord(root_mem_id=mem_id, records=records)
