from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltm_design.llm.client import ChatMessage, DeepSeekClient  # noqa: E402
from ltm_design.memory.models import EvidenceRecord  # noqa: E402
from ltm_design.memory.orchestrator import MemoryModels, MemoryOrchestrator  # noqa: E402
from ltm_design.memory.store import MemoryStore  # noqa: E402


BASE_URL = "https://journal.derikbadman.com"
DATA_DIR = ROOT / "data" / "journal_derik"
RAW_DIR = DATA_DIR / "raw_html"
ENTRIES_JSONL = DATA_DIR / "entries.jsonl"
RUN_STATE_JSON = DATA_DIR / "run_state.json"
DB_PATH = DATA_DIR / "memory.db"

EXPERIMENT_INSTRUCTION = """\
Temporary experiment instruction for the main conversation model:
Treat each journal entry as if the user has just shared it in a conversation. For this test only,
call the memory-cataloging agent after each journal entry so durable, future-useful details can
be stored. When several journal entries appear in one conversation chunk, do not add extra
intervention or special branching logic; process the transcript normally using processed_through
and evidence_range.
"""


def configure_data_dir(path: Path) -> None:
    global DATA_DIR, RAW_DIR, ENTRIES_JSONL, RUN_STATE_JSON, DB_PATH
    DATA_DIR = path
    RAW_DIR = DATA_DIR / "raw_html"
    ENTRIES_JSONL = DATA_DIR / "entries.jsonl"
    RUN_STATE_JSON = DATA_DIR / "run_state.json"
    DB_PATH = DATA_DIR / "memory.db"


@dataclass(frozen=True)
class JournalEntry:
    source_id: str
    message_id: str
    title: str
    url: str
    published_at: str
    text: str
    html_path: str

    @property
    def source_marker(self) -> str:
        return f"[src:{self.source_id} msg:{self.message_id}]"


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        values = dict(attrs)
        href = values.get("href")
        if href:
            self.links.append(href)


class ArticleTextParser(HTMLParser):
    BLOCK_TAGS = {"p", "div", "header", "h1", "h2", "h3", "li", "blockquote", "br"}

    def __init__(self) -> None:
        super().__init__()
        self.in_article = False
        self.depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self.in_article = True
            self.depth = 1
            return
        if not self.in_article:
            return
        self.depth += 1
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if not self.in_article:
            return
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        self.depth -= 1
        if tag == "article" or self.depth <= 0:
            self.in_article = False

    def handle_data(self, data: str) -> None:
        if self.in_article:
            self.parts.append(data)

    def text(self) -> str:
        value = html.unescape("".join(self.parts))
        value = re.sub(r"[ \t\r\f\v]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


class HashEmbeddingProvider:
    model_name = "experiment-hash-embedding"
    dimensions = 128

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[A-Za-z0-9_]+", text.casefold()):
            bucket = int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % self.dimensions
            vector[bucket] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return vector if norm == 0 else [value / norm for value in vector]


class ExperimentDeterministicChatClient:
    """Local stand-in that exercises storage flow when external model auth is unavailable."""

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        system = messages[0].content
        payload = json.loads(messages[-1].content)
        if "memory cataloger" in system:
            return json.dumps(self._catalog(payload))
        if "checking whether newly cataloged memories" in system:
            return json.dumps({"connections": []})
        if "continuing the memory cataloging task" in system:
            memories = payload["catalog"].get("memories", [])
            return json.dumps({
                "approve_memories": [
                    {"new_memory_index": index, "approved_text": None}
                    for index in range(len(memories))
                ],
                "reject_memories": [],
                "approve_links": [],
            })
        if "checking a group of memory items" in system:
            return json.dumps({
                "memories": [
                    {
                        "mem_id": item["mem_id"],
                        "relevant": True,
                        "memory_tags": ["journal"],
                        "descriptors": ["deterministic"],
                        "reason": "deterministic experiment mode",
                    }
                    for item in payload.get("memory_items", [])
                    if item.get("mem_id")
                ]
            })
        if "selecting useful memory context" in system:
            return "\n".join(
                f"- {item['text']} ({item['mem_id']})"
                for item in payload.get("candidate_memory_items", [])
            )
        raise RuntimeError("unhandled deterministic chat prompt")

    def _catalog(self, payload: dict) -> dict:
        evidence_range = payload["evidence_range"]
        markers = list(re.finditer(r"^\[src:([^\] ]+) msg:([^\]]+)\]$", evidence_range, flags=re.M))
        memories = []
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(evidence_range)
            block = evidence_range[start:end].strip()
            date_match = re.search(r"User journal entry dated ([^.]+)\.", block)
            title = date_match.group(1) if date_match else marker.group(2)
            body = block[date_match.end():].strip() if date_match else block
            body = re.sub(r"\s+", " ", body).strip()
            if not body:
                continue
            excerpt = body[:700].rstrip()
            if len(body) > len(excerpt):
                excerpt += "..."
            memories.append(
                {
                    "text": f"The journal entry for {title} says: {excerpt}",
                    "facets": {
                        "people": [],
                        "topics": ["journal entry"],
                        "emotions": [],
                        "events": ["journal memory experiment"],
                        "places": [],
                        "objects": ["Derik Badman's Journal"],
                        "times": [title.split()[0]],
                    },
                    "references": [
                        {
                            "source_id": marker.group(1),
                            "message_id": marker.group(2),
                            "event_id": None,
                            "speaker": "user",
                            "whole_source": True,
                        }
                    ],
                    "connection_hints": [],
                }
            )
        processed = markers[-1].group(0) if markers else payload.get("processed_through", "")
        return {"memories": memories, "processed_through": processed}


def fetch_url(url: str, *, retries: int = 3, sleep_s: float = 0.25) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ltm-design-memory-experiment/0.1"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # pragma: no cover - diagnostic script
            last_error = exc
            time.sleep(sleep_s * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_error}") from last_error


def cache_path_for_url(url: str) -> Path:
    parsed = urllib.parse.urlparse(url)
    slug = parsed.path.strip("/").replace("/", "__").replace(":", "-") or "index"
    return RAW_DIR / f"{slug}.html"


def read_or_fetch(url: str) -> tuple[str, Path, bool]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path_for_url(url)
    if path.exists():
        return path.read_text(encoding="utf-8"), path, True
    text = fetch_url(url)
    path.write_text(text, encoding="utf-8")
    return text, path, False


def extract_links(markup: str, pattern: str) -> list[str]:
    parser = LinkParser()
    parser.feed(markup)
    seen: dict[str, None] = {}
    for href in parser.links:
        if re.fullmatch(pattern, href):
            seen[href] = None
    return list(seen)


def parse_entry(url: str) -> JournalEntry:
    markup, path, _ = read_or_fetch(url)
    parser = ArticleTextParser()
    parser.feed(markup)
    text = parser.text()
    raw_slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug_match = re.fullmatch(r"(\d{4}-\d{2}-\d{2})-(\d{2}):(\d{2})", raw_slug)
    title = f"{slug_match.group(1)} {slug_match.group(2)}:{slug_match.group(3)}" if slug_match else raw_slug
    stamp = title.replace(" ", "T")
    try:
        published = datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        published = datetime.now(timezone.utc).isoformat()
    slug = raw_slug.replace(":", "-")
    return JournalEntry(
        source_id=f"journal_derik_{slug}",
        message_id=f"u_{slug}",
        title=title,
        url=url,
        published_at=published,
        text=text,
        html_path=str(path.relative_to(ROOT)),
    )


def scrape(workers: int) -> list[JournalEntry]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archives_html, _, _ = read_or_fetch(f"{BASE_URL}/archives")
    month_paths = extract_links(archives_html, r"/archives/\d{4}-\d{2}")
    month_urls = [urllib.parse.urljoin(BASE_URL, path) for path in month_paths]
    print(f"discovered {len(month_urls)} archive months")

    def entry_paths_for_month(month_url: str) -> list[str]:
        markup, _, cached = read_or_fetch(month_url)
        status = "cached" if cached else "fetched"
        print(f"{status} {month_url}")
        return extract_links(markup, r"/entries/\d{4}-\d{2}-\d{2}-\d{2}:\d{2}")

    entry_paths: dict[str, None] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for paths in pool.map(entry_paths_for_month, month_urls):
            for path in paths:
                entry_paths[path] = None

    entry_urls = [urllib.parse.urljoin(BASE_URL, path) for path in entry_paths]
    print(f"discovered {len(entry_urls)} entries")
    entries: list[JournalEntry] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for entry in pool.map(parse_entry, entry_urls):
            if entry.text:
                entries.append(entry)
                print(f"cached entry {entry.title}")

    entries.sort(key=lambda item: item.published_at)
    with ENTRIES_JSONL.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(entries)} normalized entries to {ENTRIES_JSONL.relative_to(ROOT)}")
    return entries


def load_entries() -> list[JournalEntry]:
    if not ENTRIES_JSONL.exists():
        raise SystemExit(f"{ENTRIES_JSONL} does not exist; run scrape first")
    entries = []
    for line in ENTRIES_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(JournalEntry(**json.loads(line)))
    return entries


def transcript_for(entries: list[JournalEntry]) -> str:
    chunks = []
    for entry in entries:
        chunks.append(
            "\n".join(
                [
                    entry.source_marker,
                    f"User journal entry dated {entry.title}.",
                    entry.text,
                ]
            )
        )
    return "\n\n".join(chunks)


def add_evidence(store: MemoryStore, entry: JournalEntry) -> None:
    store.add_evidence(
        EvidenceRecord(
            source_id=entry.source_id,
            conversation_id="journal_derik_memory_experiment",
            text=entry.text,
            created_at=datetime.fromisoformat(entry.published_at),
            metadata={
                "kind": "public_journal_entry",
                "url": entry.url,
                "title": entry.title,
                "message_id": entry.message_id,
                "temporary_experiment_instruction": EXPERIMENT_INSTRUCTION,
            },
        )
    )


def load_state() -> dict:
    if RUN_STATE_JSON.exists():
        return json.loads(RUN_STATE_JSON.read_text(encoding="utf-8"))
    return {"processed_through": "", "completed_source_ids": []}


def save_state(state: dict) -> None:
    RUN_STATE_JSON.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def run_memory(batch_size: int, max_entries: int | None, chat: str, main_model: str, worker_model: str) -> None:
    entries = load_entries()
    if max_entries is not None:
        entries = entries[:max_entries]
    state = load_state()
    completed = set(state.get("completed_source_ids", []))
    pending = [entry for entry in entries if entry.source_id not in completed]
    print(f"loaded {len(entries)} entries; {len(pending)} pending", flush=True)

    store = MemoryStore(DB_PATH)
    chat_client = (
        DeepSeekClient(timeout_s=120, stream=True, log_stream_progress=True, thinking="disabled")
        if chat == "deepseek"
        else ExperimentDeterministicChatClient()
    )
    orchestrator = MemoryOrchestrator(
        store=store,
        embeddings=HashEmbeddingProvider(),
        chat_client=chat_client,
        models=MemoryModels(main=main_model, worker=worker_model),
    )
    processed_through = state.get("processed_through", "")

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        for entry in batch:
            add_evidence(store, entry)
        evidence_range = transcript_for(batch)
        print(f"processing {batch[0].title} .. {batch[-1].title} ({len(batch)} entries)", flush=True)
        result = orchestrator.process_memory_skill(
            processed_through=processed_through,
            evidence_range=evidence_range,
            recent_context=EXPERIMENT_INSTRUCTION,
        )
        processed_through = batch[-1].source_marker
        completed.update(entry.source_id for entry in batch)
        state = {
            "processed_through": processed_through,
            "completed_source_ids": sorted(completed),
            "last_approved_mem_ids": list(result.approved_mem_ids),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        save_state(state)
        print(f"approved {len(result.approved_mem_ids)} memories; processed_through={processed_through}", flush=True)


def stats() -> None:
    if not DB_PATH.exists():
        print(f"no DB at {DB_PATH}")
        return
    conn = sqlite3.connect(DB_PATH)
    for table in ["evidence_records", "pending_memory_candidates", "memory_items", "memory_links", "memory_embeddings", "memory_fragments"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {count}")
    if RUN_STATE_JSON.exists():
        state = json.loads(RUN_STATE_JSON.read_text(encoding="utf-8"))
        print(f"processed entries: {len(state.get('completed_source_ids', []))}")
        print(f"processed_through: {state.get('processed_through', '')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    sub = parser.add_subparsers(dest="command", required=True)
    scrape_parser = sub.add_parser("scrape")
    scrape_parser.add_argument("--workers", type=int, default=8)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--batch-size", type=int, default=1)
    run_parser.add_argument("--max-entries", type=int)
    run_parser.add_argument("--chat", choices=["deepseek", "deterministic"], default="deepseek")
    run_parser.add_argument("--main-model", default="deepseek-v4-flash")
    run_parser.add_argument("--worker-model", default="deepseek-v4-flash")
    sub.add_parser("stats")
    args = parser.parse_args()
    configure_data_dir(args.data_dir)

    if args.command == "scrape":
        scrape(args.workers)
    elif args.command == "run":
        run_memory(args.batch_size, args.max_entries, args.chat, args.main_model, args.worker_model)
    elif args.command == "stats":
        stats()


if __name__ == "__main__":
    main()
