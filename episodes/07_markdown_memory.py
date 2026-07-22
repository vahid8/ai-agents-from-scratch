"""Episode 7 — Memory, the OTHER way: markdown files + a text index (not vectors).

In Ep5 we gave the agent long-term memory with EMBEDDINGS: every fact became a
vector, and `recall` searched by MEANING (cosine similarity). That's one whole school
of memory. This episode builds the OTHER school, the one that runs most coding agents
you already use:

    MARKDOWN FILES + AN INDEX.

The agent writes plain `.md` files to disk -- one per topic -- and keeps a tiny
`index.md` ROUTER that lists each file with a one-line description. To recall, it does
NOT embed anything. It searches the TEXT: a full-text index (SQLite FTS5, straight from
the standard library) finds the right file by keyword, and the agent reads it.

Why bother, when Ep5's vectors already worked?

    - CHEAP        no embedding calls, no vector store -- just files and a text index.
    - READABLE     the memory IS `notes/plan.md`; you can open it, edit it, diff it.
    - AGENT-WRITABLE the model authors its own memory in a format humans also edit.

This is exactly how Claude Code's `CLAUDE.md`, the open `AGENTS.md` standard, and
Nous Research's Hermes agent (MEMORY.md + an FTS text index) remember things: markdown
on disk, retrieved by structure and keyword -- not by meaning.

Neither school is "correct". Vectors win when the wording won't match ("which day do I
rotate keys?" -> "...every Monday"); markdown+index wins when memory should be legible,
editable, and free. Real agents often run BOTH.

The loop is the SAME ReAct loop from Ep5/Ep6, under the SAME tracer from Ep6. Only the
two memory tools changed: remember[topic | fact] writes a markdown file, recall[query]
searches the text index.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/07_markdown_memory.py
"""

import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

from openai import OpenAI

from trace import Tracer  # the tiny run tracer from Ep6 (episodes/trace.py)


def get_gemini_key() -> str:
    """Read the key from GEMINI_API_KEY, or from the file GEMINI_API_KEY_FILE points to."""
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    if key_file := os.environ.get("GEMINI_API_KEY_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEY_FILE in your .env")


# Same OpenAI SDK pointed at Gemini's free endpoint -- unchanged since Ep1.
client = OpenAI(
    api_key=get_gemini_key(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"
# NOTE: no EMBED_MODEL this episode. That's the whole point -- memory here is text, not
# vectors, so there are zero embedding calls.


# ======================= MARKDOWN MEMORY ON DISK =============================
# Long-term memory is a folder of plain markdown files plus a router (index.md) and a
# full-text index (SQLite FTS5). You can open any of these in a text editor -- that's
# the feature. Ep5's memory was an opaque list of vectors; this one you can READ.
MEM_DIR = Path(__file__).with_name("md_memory")
INDEX_MD = MEM_DIR / "index.md"          # the human-readable ROUTER (topic -> one-liner)
FTS_DB = MEM_DIR / "fts.db"              # the machine index: full-text search over the files


def _slug(topic: str) -> str:
    """Turn a topic into a safe file name, e.g. 'API keys' -> 'api-keys'."""
    s = re.sub(r"[^a-z0-9]+", "-", topic.strip().lower()).strip("-")
    return s or "note"


def _fts() -> sqlite3.Connection:
    """Open the full-text index, creating the table the first time."""
    db = sqlite3.connect(FTS_DB)
    db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS notes USING fts5(topic, file, body)")
    return db


def remember(entry: str) -> str:
    """Save a fact as MARKDOWN. Format: 'topic | the fact'.

    Three things happen, all in plain text: (1) write/append the fact to
    md_memory/<topic>.md, (2) add a line to the index.md router, (3) index the file's
    text in FTS5 so recall can find it by keyword. No embeddings anywhere.
    """
    topic, _, fact = entry.partition("|")
    topic, fact = topic.strip() or "note", fact.strip() or entry.strip()

    path = MEM_DIR / f"{_slug(topic)}.md"
    if not path.exists():
        path.write_text(f"# {topic}\n\n- {fact}\n")
        # add a router line so a human (or the agent) can see what files exist
        with INDEX_MD.open("a") as f:
            f.write(f"- [{topic}]({path.name}) - {fact[:60]}\n")
    else:
        with path.open("a") as f:
            f.write(f"- {fact}\n")

    # (re)index this file's full text for keyword search
    db = _fts()
    db.execute("DELETE FROM notes WHERE file = ?", (path.name,))
    db.execute("INSERT INTO notes(topic, file, body) VALUES (?,?,?)",
               (topic, path.name, path.read_text()))
    db.commit()
    db.close()
    return f'wrote md_memory/{path.name}  (topic: "{topic}")'


def recall(query: str, k: int = 3) -> str:
    """Look facts up by TEXT, not meaning: full-text search the markdown files.

    FTS5 ranks files by keyword match (BM25). We return the best files' contents --
    the same markdown a human would read. Because it's keyword-based it's fast and
    free, but it only finds what the WORDS share (contrast Ep5's meaning search).
    """
    if not FTS_DB.exists():
        return "long-term memory is empty"
    db = _fts()
    # match any of the query words; order by FTS5's built-in relevance ranking
    terms = " OR ".join(re.findall(r"[a-z0-9]+", query.lower())) or query
    rows = db.execute(
        "SELECT file, body FROM notes WHERE notes MATCH ? ORDER BY rank LIMIT ?",
        (terms, k)).fetchall()
    db.close()
    if not rows:
        return f'no markdown notes matched "{query}"'
    return "recalled from markdown:\n" + "\n".join(
        f"--- {file} ---\n{body.strip()}" for file, body in rows)


TOOLS = {"remember": remember, "recall": recall}


# --- THE ReAct PROMPT: same format as Ep5, two markdown-memory tools. -----------
SYSTEM = """You are a helpful assistant whose long-term memory is a set of MARKDOWN
files on disk. You hold a normal conversation and can save or look up durable facts
using this EXACT format when useful:

Thought: decide whether to save a fact or look one up
Action: tool_name[input]

After an Action you get an Observation. Then take another Action, or reply:

Thought: reason about your reply
Answer: your reply to the user

Your memory tools:
- remember[topic | fact]  save a durable fact to a markdown file, grouped by topic,
                          e.g. remember[billing | Vahid is on the Nimbus Pro plan]
- recall[query]           keyword-search the markdown files for something saved earlier,
                          e.g. recall[which plan]

Rules: ALWAYS begin with a Thought. When the user states a durable fact about
themselves, save it with remember[topic | fact], choosing a short topic. When they ask
about something they may have told you before but that ISN'T in this conversation, use
recall[...] first and answer from what it returns. If the answer is already in this
conversation, just answer. Output ONE Thought then ONE Action, or ONE Thought then ONE
Answer."""

MAX_STEPS = 6
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def agent_reply(messages: list[dict], tracer: Tracer) -> None:
    """Run the ReAct loop for ONE user turn (already appended) until an Answer.

    Identical machinery to Ep5/Ep6 -- the tools just happen to read and write markdown
    now. Every model call and tool call is recorded on the tracer.
    """
    for _ in range(MAX_STEPS):
        t = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
        tracer.llm(resp, time.perf_counter() - t)
        text = resp.choices[0].message.content.strip()

        action = ACTION_RE.search(text)
        answer_at = text.find("Answer:")

        # Stop condition: the model produced its final Answer.
        if answer_at != -1 and (action is None or answer_at < action.start()):
            print(text)
            messages.append({"role": "assistant", "content": text})
            return

        if action is None:  # no tool, no answer -- nudge it back to the format
            print(text)
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "Observation: reply with remember[...], recall[...] or an Answer."})
            continue

        # Truncate at the Action so the model can't hallucinate its own Observation.
        turn = text[: action.end()]
        print(turn)
        messages.append({"role": "assistant", "content": turn})

        name, arg = action.group(1), action.group(2).strip()
        t = time.perf_counter()
        observation = TOOLS[name](arg) if name in TOOLS else f"unknown tool: {name}"
        tracer.tool(name, arg, observation, time.perf_counter() - t)
        print(f"Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    else:
        print("gave up -- no answer.\n")


def chat(user_messages: list[str], label: str) -> None:
    """Feed a scripted sequence of user turns through one fresh conversation."""
    messages = [{"role": "system", "content": SYSTEM}]   # a NEW short-term scratchpad
    tracer = Tracer(task=label, model=MODEL)
    for user in user_messages:
        print(f"🧑 {user}")
        messages.append({"role": "user", "content": user})
        agent_reply(messages, tracer)
    tracer.finish("(session end)")


# ============================ THE DEMO =======================================
# Same two-session shape as Ep5, so you can compare the two memory schools directly.
# SESSION 1 writes markdown files; we "restart" (short-term memory gone, files remain);
# SESSION 2 is a fresh conversation that can only answer from the markdown on disk.
SESSION_1 = [
    "Hi! I'm Vahid. For the record, I'm on the Nimbus Pro plan.",
    "One more thing to note: I rotate my API keys every Monday.",
    "Quick check before you forget -- what plan did I just say I'm on?",
]
SESSION_2 = [
    "Hey, I'm back. Remind me -- what did I say about rotating my API keys?",
]

if __name__ == "__main__":
    # Fresh start each run for a clean, repeatable demo (real use would KEEP the folder).
    if MEM_DIR.exists():
        shutil.rmtree(MEM_DIR)
    MEM_DIR.mkdir()
    INDEX_MD.write_text("# Memory index (router)\n\n")

    print("=== SESSION 1 — you tell the agent things (saved as markdown files) ===\n")
    chat(SESSION_1, "session 1: save facts to markdown")

    print("=== the program 'closes' — short-term memory is wiped, the .md files stay ===")
    files = sorted(p.name for p in MEM_DIR.glob("*.md"))
    print(f"📁 md_memory/ now holds: {', '.join(files)}\n")
    print("index.md router:")
    print(INDEX_MD.read_text())

    print("=== SESSION 2 — a brand-new conversation; only the markdown survived ===\n")
    chat(SESSION_2, "session 2: recall from markdown")
