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
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ltm_design.llm.client import ChatMessage, DeepSeekClient, parse_json_object  # noqa: E402
from ltm_design.memory.models import EvidenceRecord  # noqa: E402
from ltm_design.memory.orchestrator import MemoryModels, MemoryOrchestrator  # noqa: E402
from ltm_design.memory.prompts import MAIN_CONVERSATION_MEMORY_TRIGGER_PROMPT  # noqa: E402
from ltm_design.memory.store import MemoryStore  # noqa: E402


BASE_URL = "https://journal.derikbadman.com"
DATA_DIR = ROOT / "data" / "journal_derik"
RAW_DIR = DATA_DIR / "raw_html"
ENTRIES_JSONL = DATA_DIR / "entries.jsonl"
RUN_STATE_JSON = DATA_DIR / "run_state.json"
DB_PATH = DATA_DIR / "memory.db"
MODEL_IO_JSONL = DATA_DIR / "model_io.jsonl"
REPORT_HTML = DATA_DIR / "browser.html"

EXPERIMENT_INSTRUCTION = """\
Temporary experiment instruction for the main conversation model:
Treat each journal entry as if the user has just shared it in a conversation. For this test only,
call the memory-cataloging agent after each journal entry so durable, future-useful details can
be stored. For experiment observability, respond with JSON:
{"call_memory_storage": true, "processed_through": "the last source marker already processed before this call"}
"""


def configure_data_dir(path: Path) -> None:
    global DATA_DIR, RAW_DIR, ENTRIES_JSONL, RUN_STATE_JSON, DB_PATH, MODEL_IO_JSONL, REPORT_HTML
    DATA_DIR = path
    RAW_DIR = DATA_DIR / "raw_html"
    ENTRIES_JSONL = DATA_DIR / "entries.jsonl"
    RUN_STATE_JSON = DATA_DIR / "run_state.json"
    DB_PATH = DATA_DIR / "memory.db"
    MODEL_IO_JSONL = DATA_DIR / "model_io.jsonl"
    REPORT_HTML = DATA_DIR / "browser.html"


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
        prompt_text = "\n\n".join(message.content for message in messages)
        if "memory cataloger" in prompt_text:
            payload = {
                "evidence_range": "\n\n".join(message.content for message in messages if message.role == "user"),
                "processed_through": self._field_from_skill_prompt(prompt_text, "processed_through"),
            }
            return json.dumps(self._catalog(payload))
        if "Memory trigger policy for the live conversation model" in prompt_text:
            return json.dumps({"call_memory_storage": True, "processed_through": ""})
        if "memory connection checker" in prompt_text:
            return json.dumps({"connections": []})
        if "memory connection summarizer" in prompt_text:
            return ""
        if "continuing the memory cataloging task" in prompt_text:
            return json.dumps({"reject_memories": [], "approve_links": []})
        if "checking a group of memory items" in prompt_text:
            payload = json.loads(messages[-1].content)
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
        if "selecting useful memory context" in prompt_text:
            payload = json.loads(messages[-1].content)
            return "\n".join(
                f"- {item['text']} ({item['mem_id']})"
                for item in payload.get("candidate_memory_items", [])
            )
        raise RuntimeError("unhandled deterministic chat prompt")

    def _field_from_skill_prompt(self, text: str, field: str) -> str:
        match = re.search(rf"^{re.escape(field)}: (.*)$", text, flags=re.M)
        return match.group(1) if match else ""

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


class RecordingChatClient:
    def __init__(self, inner: object, path: Path) -> None:
        self.inner = inner
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.context: dict = {}

    def set_context(self, **values: object) -> None:
        self.context = dict(values)

    def chat(self, *, model: str, messages: list[ChatMessage], temperature: float = 0.0) -> str:
        self.counter += 1
        started = datetime.now(timezone.utc)
        role = prompt_role("\n\n".join(message.content for message in messages))
        record = {
            "call_id": f"call_{self.counter:04d}",
            "started_at": started.isoformat(),
            "model": model,
            "temperature": temperature,
            "role": role,
            "branch": "memory_storage_skill" if role in {"cataloger", "connection_fragment", "connection_summarizer", "storage_decision"} else "main",
            "context": self.context,
            "messages": [asdict(message) for message in messages],
        }
        try:
            response = self.inner.chat(model=model, messages=messages, temperature=temperature)
            record["response"] = response
            record["ok"] = True
            return response
        except Exception as exc:
            record["ok"] = False
            record["error"] = repr(exc)
            raise
        finally:
            ended = datetime.now(timezone.utc)
            record["ended_at"] = ended.isoformat()
            record["duration_s"] = round((ended - started).total_seconds(), 3)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def prompt_role(system_prompt: str) -> str:
    if "memory cataloger" in system_prompt:
        return "cataloger"
    if "Memory trigger policy for the live conversation model" in system_prompt:
        return "main_trigger"
    if "memory connection checker" in system_prompt:
        return "connection_fragment"
    if "memory connection summarizer" in system_prompt:
        return "connection_summarizer"
    if "continuing the memory cataloging task" in system_prompt:
        return "storage_decision"
    if "checking a group of memory items" in system_prompt:
        return "retrieval_fragment"
    if "selecting useful memory context" in system_prompt:
        return "retrieval_summarizer"
    return "unknown"


def parse_json_loose(text: str) -> dict:
    try:
        return parse_json_object(text)
    except Exception:
        return {}


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


def main_user_message_for(entry: JournalEntry) -> ChatMessage:
    return ChatMessage(
        role="user",
        content="\n".join(
            [
                entry.source_marker,
                f"User journal entry dated {entry.title}.",
                entry.text,
            ]
        ),
    )


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
    base_chat_client = (
        DeepSeekClient(timeout_s=120, stream=True, log_stream_progress=True, thinking="disabled")
        if chat == "deepseek"
        else ExperimentDeterministicChatClient()
    )
    chat_client = RecordingChatClient(base_chat_client, MODEL_IO_JSONL)
    orchestrator = MemoryOrchestrator(
        store=store,
        embeddings=HashEmbeddingProvider(),
        chat_client=chat_client,
        models=MemoryModels(main=main_model, worker=worker_model),
    )
    processed_through = state.get("processed_through", "")
    main_messages: list[ChatMessage] = [
        ChatMessage(
            role="system",
            content="\n\n".join([MAIN_CONVERSATION_MEMORY_TRIGGER_PROMPT, EXPERIMENT_INSTRUCTION]),
        )
    ]
    for entry in entries:
        if entry.source_id in completed:
            main_messages.append(main_user_message_for(entry))

    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        for entry in batch:
            add_evidence(store, entry)
        print(f"processing {batch[0].title} .. {batch[-1].title} ({len(batch)} entries)", flush=True)
        for entry in batch:
            main_messages.append(main_user_message_for(entry))
        chat_client.set_context(
            entry_titles=[entry.title for entry in batch],
            entry_source_ids=[entry.source_id for entry in batch],
            conversation_entries=sum(1 for message in main_messages if message.role == "user"),
            processed_through_before=processed_through,
        )
        trigger_text = chat_client.chat(
            model=main_model,
            messages=list(main_messages),
        )
        trigger = parse_json_loose(trigger_text)
        if not trigger.get("call_memory_storage", True):
            print("main model did not call memory storage; skipping storage branch", flush=True)
            continue
        result = orchestrator.process_memory_skill(
            processed_through=trigger.get("processed_through") or processed_through,
            evidence_range="",
            recent_context="",
            conversation_messages=list(main_messages),
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
    if MODEL_IO_JSONL.exists():
        calls = [json.loads(line) for line in MODEL_IO_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"model calls: {len(calls)}")
        print("model call roles: " + ", ".join(f"{key}={value}" for key, value in sorted(Counter(call.get("role", "unknown") for call in calls).items())))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def table_rows(conn: sqlite3.Connection, table: str) -> list[dict]:
    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    return [dict(row) for row in rows]


def generate_report() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    entries = [asdict(entry) for entry in load_entries()] if ENTRIES_JSONL.exists() else []
    calls = read_jsonl(MODEL_IO_JSONL)
    state = json.loads(RUN_STATE_JSON.read_text(encoding="utf-8")) if RUN_STATE_JSON.exists() else {}
    db = {
        "memory_items": [],
        "memory_links": [],
        "memory_fragments": [],
        "pending_memory_candidates": [],
        "evidence_records": [],
    }
    if DB_PATH.exists():
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        for table in db:
            db[table] = table_rows(conn, table)
        conn.close()
    data = {
        "entries": entries,
        "calls": calls,
        "state": state,
        "db": db,
        "meta": {
            "main_trigger_prompt": MAIN_CONVERSATION_MEMORY_TRIGGER_PROMPT,
            "experiment_instruction": EXPERIMENT_INSTRUCTION,
            "main_model_call_simulated": False,
        },
    }
    REPORT_HTML.write_text(render_report_html(data), encoding="utf-8")
    print(f"wrote report to {REPORT_HTML}")


def render_report_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory Storage Timeline</title>
<style>
:root {{ color-scheme: light; --ink:#1f2933; --muted:#667085; --line:#d5dbe7; --panel:#f7f8fb; --main:#155e75; --skill:#7c2d12; --worker:#365314; --merge:#5b21b6; --decision:#9f1239; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color:var(--ink); background:#fbfcfe; }}
header {{ position:sticky; top:0; z-index:10; display:flex; justify-content:space-between; gap:20px; align-items:center; padding:14px 22px; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); backdrop-filter:blur(8px); }}
h1 {{ margin:0; font-size:20px; }}
.toolbar {{ display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
input {{ width:280px; max-width:44vw; border:1px solid var(--line); border-radius:6px; padding:8px 10px; font:inherit; }}
button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; font:inherit; cursor:pointer; }}
button.active {{ background:#111827; color:#fff; border-color:#111827; }}
main {{ max-width:1280px; margin:0 auto; padding:18px 22px 80px; }}
.summary {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin-bottom:20px; }}
.stat {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:10px; }}
.stat b {{ display:block; font-size:22px; }}
.muted {{ color:var(--muted); font-size:12px; }}
.timeline {{ position:relative; margin-left:18px; }}
.timeline::before {{ content:""; position:absolute; top:0; bottom:0; left:18px; border-left:2px solid #cbd5e1; }}
.turn {{ position:relative; margin:0 0 28px 54px; }}
.turn-marker {{ position:absolute; left:-46px; top:12px; width:28px; height:28px; border-radius:50%; background:#fff; border:3px solid var(--main); }}
.turn-card {{ background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; box-shadow:0 1px 2px rgba(15,23,42,.04); }}
.turn-head {{ padding:12px 14px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; align-items:baseline; }}
.turn-head h2 {{ margin:0; font-size:17px; }}
.section {{ padding:12px 14px; border-bottom:1px solid #eef2f7; }}
.section:last-child {{ border-bottom:0; }}
.label {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:#475467; margin-bottom:8px; }}
.branch {{ margin:12px 0 0 18px; padding-left:20px; border-left:3px solid var(--skill); }}
.branch-title {{ margin:0 0 10px; color:var(--skill); font-weight:800; }}
.node {{ background:#fff; border:1px solid var(--line); border-radius:8px; margin:10px 0; overflow:hidden; }}
.node.cataloger {{ border-left:5px solid var(--skill); }}
.node.fragment {{ border-left:5px solid var(--worker); }}
.node.summarizer {{ border-left:5px solid var(--merge); }}
.node.decision {{ border-left:5px solid var(--decision); }}
.node-head {{ padding:10px 12px; background:#f8fafc; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; align-items:baseline; }}
.node-head h3 {{ margin:0; font-size:15px; }}
pre {{ margin:0; white-space:pre-wrap; overflow:auto; max-height:720px; background:#101828; color:#f8fafc; border-radius:6px; padding:10px; font-size:12px; line-height:1.45; }}
.call-body {{ padding:10px; }}
.synthetic {{ border-left:5px solid var(--main); }}
.fragments {{ display:block; }}
.connector {{ margin:8px 0 2px; color:#667085; font-size:12px; text-align:center; }}
.flow {{ padding:12px 14px; }}
.flow pre {{ max-height:680px; background:#0f172a; }}
.branch-marker {{ margin:12px 0; padding:10px 12px; border:1px dashed var(--skill); border-radius:8px; background:#fff7ed; color:var(--skill); font-weight:800; }}
.hidden {{ display:none !important; }}
@media (max-width:900px) {{ .turn {{ margin-left:42px; }} input {{ max-width:100%; width:100%; }} header {{ align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body>
<header>
  <div>
    <h1>Memory Storage Timeline</h1>
    <div class="muted" id="state"></div>
  </div>
  <div class="toolbar">
    <input id="filter" placeholder="Filter timeline text">
    <button id="collapseFragments">Collapse Fragments</button>
    <button id="expandAll">Expand All</button>
  </div>
</header>
<main>
  <section class="summary" id="summary"></section>
  <section class="timeline" id="timeline"></section>
</main>
<script>
const DATA = {payload};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>]/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;"}}[ch]));
const clip = (value, n=120) => {{
  value = String(value ?? "").replace(/\\s+/g, " ").trim();
  return value.length > n ? value.slice(0, n - 1) + "…" : value;
}};
function roleCounts() {{
  return DATA.calls.reduce((acc, call) => (acc[call.role]=(acc[call.role]||0)+1, acc), {{}});
}}
function renderSummary() {{
  const roleCounts = DATA.calls.reduce((acc, call) => (acc[call.role]=(acc[call.role]||0)+1, acc), {{}});
  const stats = [
    ["entries", DATA.entries.length],
    ["model calls", DATA.calls.length],
    ["memories", DATA.db.memory_items.length],
    ["links", DATA.db.memory_links.length],
    ["fragments", DATA.db.memory_fragments.length],
    ["processed", (DATA.state.completed_source_ids || []).length],
  ];
  $("summary").innerHTML = stats.map(([k,v]) => `<div class="stat"><b>${{v}}</b><span class="muted">${{k}}</span></div>`).join("")
    + `<div class="stat"><b>${{Object.entries(roleCounts).map(([k,v]) => `${{k}}:${{v}}`).join(" ") || "none"}}</b><span class="muted">call roles</span></div>`;
  $("state").textContent = DATA.state.processed_through || "no processed pointer";
}}
function groupBranches() {{
  const groups = [];
  const byKey = new Map();
  for (const call of DATA.calls) {{
    const ctx = call.context || {{}};
    const key = (ctx.entry_source_ids || ["unknown"]).join(",");
    if (!byKey.has(key)) {{
      const title = (ctx.entry_titles || [key])[0] || key;
      byKey.set(key, {{title, context: ctx, calls: []}});
      groups.push(byKey.get(key));
    }}
    byKey.get(key).calls.push(call);
  }}
  return groups;
}}
function messageContent(call, role) {{
  const match = (call.messages || []).find(message => message.role === role);
  return match ? match.content : "";
}}
function mainConversationThrough(group) {{
  const wanted = new Set(group.context.entry_source_ids || []);
  const currentIndex = DATA.entries.findIndex(entry => wanted.has(entry.source_id));
  const visible = currentIndex < 0 ? DATA.entries : DATA.entries.slice(0, currentIndex + 1);
  return visible.map((entry, index) => [
    `Prompt ${{index + 1}}`,
    entry.source_marker,
    `User journal entry dated ${{entry.title}}.`,
    entry.text,
    "(main model response: temporary experiment instruction calls memory storage skill after the journal entry)"
  ].join("\\n")).join("\\n\\n");
}}
function callTranscript(call) {{
  const parts = [];
  for (const [index, message] of (call.messages || []).entries()) {{
    parts.push(`>>> MESSAGE ${{index + 1}} · role=${{message.role}}\\n${{message.content || ""}}`);
  }}
  parts.push(`<<< RESPONSE · ${{call.ok ? "ok" : "error"}}\\n${{call.response || call.error || ""}}`);
  return parts.join("\\n\\n");
}}
function syntheticMainTranscript(group, index) {{
  const entry = DATA.entries.find(item => (group.context.entry_source_ids || []).includes(item.source_id));
  return [
    "!!! HARNESS-SIMULATED MAIN MODEL TURN",
    "No real main model trigger call was made in this run. The harness directly invoked memory storage after each journal entry.",
    "",
    ">>> SYSTEM / TRIGGER POLICY THAT SHOULD CONTROL SKILL CALLING",
    DATA.meta.main_trigger_prompt || "",
    "",
    ">>> TEMPORARY EXPERIMENT INSTRUCTION",
    DATA.meta.experiment_instruction || "",
    "",
    `>>> MAIN CONVERSATION HISTORY THROUGH PROMPT ${{index + 1}}`,
    mainConversationThrough(group),
    "",
    "<<< SIMULATED MAIN RESPONSE",
    "call memory-cataloging skill"
  ].join("\\n");
}}
function renderCallNode(call) {{
  return `<div class="node ${{call.role === "main_trigger" ? "synthetic" : call.role === "cataloger" ? "cataloger" : call.role === "connection_fragment" ? "fragment" : call.role === "connection_summarizer" ? "summarizer" : "decision"}}">
    <div class="node-head"><h3>${{esc(call.call_id)}} · ${{esc(call.role)}}</h3><span class="muted">${{esc(call.model)}} · ${{esc(call.duration_s)}}s</span></div>
    <div class="call-body"><pre>${{esc(callTranscript(call))}}</pre></div>
  </div>`;
}}
function renderSyntheticMainNode(group, index) {{
  return `<div class="node synthetic"><div class="node-head"><h3>main prompt ${{index + 1}} · simulated trigger</h3><span class="muted">not a recorded model call</span></div><div class="call-body"><pre>${{esc(syntheticMainTranscript(group, index))}}</pre></div></div>`;
}}
function renderTurn(group, index) {{
  const entry = DATA.entries.find(item => (group.context.entry_source_ids || []).includes(item.source_id));
  const calls = group.calls;
  const mainTrigger = calls.find(call => call.role === "main_trigger");
  const cataloger = calls.find(call => call.role === "cataloger");
  const fragments = calls.filter(call => call.role === "connection_fragment");
  const summarizer = calls.find(call => call.role === "connection_summarizer");
  const decision = calls.find(call => call.role === "storage_decision");
  const catalogerHtml = cataloger ? renderCallNode(cataloger) : "";
  const fragmentHtml = fragments.map(call => renderCallNode(call)).join("");
  const summarizerHtml = summarizer ? `<div class="connector">fragment branches merge into summarizer</div>${{renderCallNode(summarizer)}}` : `<div class="connector">no summarizer call: no connected ids</div>`;
  const decisionHtml = decision ? `<div class="connector">summarizer output returns to storage decision</div>${{renderCallNode(decision)}}` : "";
  return `<article class="turn" data-search="${{esc(JSON.stringify(group).toLowerCase())}}">
    <div class="turn-marker"></div>
    <div class="turn-card">
      <div class="turn-head"><h2>Prompt ${{index + 1}} · ${{esc(group.title)}}</h2><span class="muted">${{calls.length}} model calls</span></div>
      <div class="section flow">
        <span class="label">Main Conversation / Skill Trigger</span>
        ${{mainTrigger ? renderCallNode(mainTrigger) : renderSyntheticMainNode(group, index)}}
        <div class="branch-marker">↳ storage skill branch starts here</div>
      </div>
      <div class="section branch">
        <p class="branch-title">Branch: memory storage skill called from prompt ${{index + 1}}</p>
        ${{catalogerHtml}}
        <div class="connector">cataloger output fans out to ${{fragments.length}} fixed-fragment checker branches</div>
        <div class="fragments">${{fragmentHtml || "<p class='muted'>No fragment branches for this turn.</p>"}}</div>
        ${{summarizerHtml}}
        ${{decisionHtml}}
      </div>
      <div class="section">
        <span class="label">Return To Original Branch</span>
        <pre>Storage branch complete. The next journal entry is appended to the original conversation branch, not to the storage skill branch.</pre>
      </div>
    </div>
  </article>`;
}}
function renderTimeline() {{
  $("timeline").innerHTML = groupBranches().map(renderTurn).join("");
}}
function applyFilter() {{
  const q = $("filter").value.trim().toLowerCase();
  document.querySelectorAll(".turn").forEach(turn => {{
    turn.classList.toggle("hidden", q && !turn.dataset.search.includes(q));
  }});
}}
function render() {{ renderSummary(); renderTimeline(); applyFilter(); }}
$("filter").addEventListener("input", applyFilter);
$("collapseFragments").addEventListener("click", () => document.body.classList.toggle("fragments-collapsed"));
$("expandAll").addEventListener("click", () => document.querySelectorAll("pre").forEach(pre => pre.style.maxHeight = "none"));
render();
</script>
<style>
body.fragments-collapsed .node.fragment .call-body {{ display:none; }}
</style>
</body>
</html>
"""


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
    sub.add_parser("report")
    args = parser.parse_args()
    configure_data_dir(args.data_dir)

    if args.command == "scrape":
        scrape(args.workers)
    elif args.command == "run":
        run_memory(args.batch_size, args.max_entries, args.chat, args.main_model, args.worker_model)
    elif args.command == "stats":
        stats()
    elif args.command == "report":
        generate_report()


if __name__ == "__main__":
    main()
