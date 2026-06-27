"""Episode 4 — A toolbox: give the agent several tools and let it CHOOSE.

So far our agent's tools came one at a time — a calculator in Ep1, a web search
in Ep3. But a real agent has a *drawer* of tools, and the smart part isn't any one
tool: it's picking the RIGHT one for each step. Ask it some math and it should
reach for the calculator. Ask about your own product and it should read your docs,
not guess. Ask about the world and it should search the web.

This episode we hand it three tools at once and let it route between them:

    calculator[expr]    -> arithmetic, done by real Python (LLMs fumble big numbers)
    web_search[query]   -> facts from the open web         (Tavily, DuckDuckGo fallback)
    search_docs[query]  -> OUR OWN handbook, by MEANING     (RAG retrieval, from S1)

`search_docs` is the interesting one: it's retrieval-augmented generation reused as
a TOOL. We embed a tiny local handbook once at startup, and when the agent calls it
we embed the query and return the closest passages by cosine similarity — so the
agent can answer from *your* documents, not its training data.

The loop is the SAME ReAct loop from Ep2/Ep3. The only new idea is the prompt: it
describes three tools and tells the agent to match the tool to the question. Watch
the demo task — it needs all three, and the agent routes to each in turn.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/04_toolbox.py
"""

import math
import os
import re
import time
from pathlib import Path

import httpx
from ddgs import DDGS
from openai import OpenAI


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
EMBED_MODEL = "gemini-embedding-001"  # same family, free key; we ask for 768-d vectors
DIMS = 768


# --- TOOL 1: calculator — arithmetic, done by real Python. ---------------------
# Identical to Ep1: LLMs are unreliable at big-number math, so we hand it to Python.
# Locked to digits and + - * / ( ) . only -- no names, no calls -- so eval is safe.
def calculator(expression: str) -> str:
    if "**" in expression or not set(expression) <= set("0123456789+-*/(). "):
        return "error: only numbers and + - * / ( ) are allowed"
    try:
        return str(eval(expression))  # safe: input restricted to arithmetic chars
    except ArithmeticError as e:
        return f"error: {e}"


# --- TOOL 2: web_search — facts from the open web. -----------------------------
# Ep3's tool, now locked to Tavily for quality. Tavily is an agent-tuned search:
# cleaner, more relevant results than raw DuckDuckGo. It needs a (free-tier) key,
# so when no key is set we fall back to keyless DuckDuckGo -- the repo still runs
# with just a Gemini key, you just get better search once you add a Tavily key.
MAX_RESULTS = 4


def get_tavily_key() -> str | None:
    """Read TAVILY_API_KEY, or the file TAVILY_API_KEY_FILE points to (else None).

    Same paths-not-secrets rule as the Gemini key: keep the real key in ~/.secrets/
    and point TAVILY_API_KEY_FILE at it (free tier: https://tavily.com). None here
    just means we fall back to DuckDuckGo.
    """
    if key := os.environ.get("TAVILY_API_KEY"):
        return key
    if key_file := os.environ.get("TAVILY_API_KEY_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    return None


def _format(results: list[dict]) -> str:
    """Both backends hand back title/body/href rows -- render them the same way."""
    if not results:
        return "no results found"
    lines = []
    for i, r in enumerate(results, 1):
        snippet = " ".join((r.get("body") or "").split())[:200]
        lines.append(f"[{i}] {r.get('title', '')}\n    {snippet}\n    {r.get('href', '')}")
    return "\n".join(lines)


def _tavily(query: str, key: str) -> list[dict]:
    """Tavily REST -- one HTTP POST, normalised to our row shape (no extra library)."""
    resp = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": MAX_RESULTS},
        timeout=20,
    )
    resp.raise_for_status()
    return [{"title": r.get("title"), "body": r.get("content"), "href": r.get("url")}
            for r in resp.json().get("results", [])]


def _duckduckgo(query: str) -> list[dict]:
    """Keyless fallback. Free search rate-limits, so RETRY with backoff."""
    for attempt in range(3):
        try:
            if results := DDGS().text(query, max_results=MAX_RESULTS):
                return results
        except Exception:  # network/rate-limit hiccups shouldn't kill the loop
            pass
        time.sleep(1.5 * (attempt + 1))
    return []


def web_search(query: str) -> str:
    """Search the live web; return the top results as title + snippet + url.

    Prefers Tavily (agent-tuned, higher quality) when a key is set; otherwise -- or
    if Tavily errors -- falls back to keyless DuckDuckGo so the loop never stalls.
    """
    query = query.strip()
    if key := get_tavily_key():
        try:
            return _format(_tavily(query, key))
        except Exception as e:  # a flaky tool is no reason to crash -- try DuckDuckGo
            if rows := _duckduckgo(query):
                return _format(rows)
            return f"search error (tavily): {e}"
    return _format(_duckduckgo(query))


# --- TOOL 3: search_docs — retrieval over OUR handbook (RAG as a tool). --------
# Our tiny "knowledge base": a few facts about a made-up product, the Nimbus API.
# In real life these are your files, wiki, or support docs -- the pipeline is identical.
DOCS = {
    "billing.md": (
        "Nimbus API usage is billed per token, counting both your input and the "
        "model's output. Invoices go out on the first of each month. You can set a "
        "hard monthly spending cap per API key from the dashboard to avoid surprises."
    ),
    "limits.md": (
        "Every Nimbus API key is rate limited to sixty requests per minute. Go over "
        "and the API returns a 429 Too Many Requests error until the window resets. "
        "The limit is per key, so one busy key never slows down anyone else's."
    ),
    "support.md": (
        "Nimbus support runs Monday to Friday, nine to five Central European Time. "
        "Paid plans get a four hour response target; the free plan is best effort."
    ),
}


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


def embed(texts: list[str]) -> list[list[float]]:
    """One batched call to Gemini's embeddings endpoint -> one vector per text."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


# INGEST once at startup: embed every doc so the tool can search them by meaning.
# (Tiny corpus, so one doc = one chunk. S1's RAG episodes show real chunking.)
_store = [{"doc": name, "text": text} for name, text in DOCS.items()]
for item, vec in zip(_store, embed([it["text"] for it in _store])):
    item["vec"] = vec


def search_docs(query: str) -> str:
    """Retrieve the handbook passages closest in MEANING to the query (top 2).

    This is RAG used as a tool: embed the query, score every passage by cosine, and
    hand back the best ones. It finds the right doc even when no words overlap --
    'how many calls before I'm blocked' lands on the '60 / minute' passage.
    """
    qv = embed([query.strip()])[0]
    scored = sorted(((cosine(qv, it["vec"]), it["doc"], it["text"]) for it in _store), reverse=True)
    return "\n".join(f"[{doc}] {text}" for _, doc, text in scored[:2])


TOOLS = {"calculator": calculator, "web_search": web_search, "search_docs": search_docs}


# --- THE ReAct PROMPT: same format as Ep3, now describing a TOOLBOX. -----------
# The whole new lesson is here: three tools, and one instruction to match the tool
# to the question. The loop below doesn't change at all.
SYSTEM = """You answer questions step by step, CHOOSING the right tool for each step,
using this EXACT format:

Thought: decide what you need next and WHICH tool fits
Action: tool_name[input]

After each Action you are given an Observation with the result. Take as many steps
as you need. When you can answer, write:

Thought: reason about the final answer
Answer: a concise answer

Your toolbox -- pick the tool that matches the need:
- calculator[expression]  arithmetic only, e.g. calculator[150-60]. NEVER do math in your head.
- search_docs[query]      OUR internal Nimbus handbook (billing, limits, support), e.g. search_docs[rate limit]
- web_search[query]       general facts from the open web, e.g. web_search[Transformer paper year]

Rules: ALWAYS begin with a Thought. Output ONE Thought and then ONE Action (or ONE
Answer). Never write an Observation yourself -- it is given to you. Use calculator for
ANY arithmetic, search_docs for anything about OUR product, and web_search for world
facts. Base your Answer only on what the Observations actually say."""

TASK = ("Two questions about running on the Nimbus API. First: our handbook sets a "
        "per-key rate limit -- if a single key fires 150 requests in one minute, how "
        "many of them are rejected, and what error do those get? Second, unrelated: in "
        "what year was the Transformer paper, 'Attention Is All You Need', published? "
        "Answer both.")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task: {TASK}"}]

MAX_STEPS = 8
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)

print(f"📚 ingested {len(DOCS)} docs into search_docs ({DIMS}-d vectors)")
print(f"🎯 task: {TASK}\n")

# ============================ THE ReAct LOOP ==================================
# Identical to Ep2/Ep3 -- the agent doesn't care that there are now THREE tools to
# choose from. The routing lives entirely in the prompt + the model's Thought.
for step in range(1, MAX_STEPS + 1):
    # stop=["Observation:"] halts the model right after its Action...
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
    text = resp.choices[0].message.content.strip()

    action = ACTION_RE.search(text)
    answer_at = text.find("Answer:")

    # Stop condition: the model reached its final Answer (and no Action comes first).
    if answer_at != -1 and (action is None or answer_at < action.start()):
        print(text)                                 # the final Thought + Answer
        messages.append({"role": "assistant", "content": text})
        break

    if action is None:
        print(text)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": "Observation: no valid Action. Use calculator[...], search_docs[...] or web_search[...]."})
        continue

    # ...and as in Ep2/Ep3 we TRUNCATE at the Action so the model can't hallucinate
    # its own Observation -- the only result it sees is the one WE produced.
    turn = text[: action.end()]
    print(turn)
    messages.append({"role": "assistant", "content": turn})

    name, arg = action.group(1), action.group(2).strip()
    observation = TOOLS[name](arg) if name in TOOLS else f"unknown tool: {name}"
    print(f"Observation:\n{observation}\n")
    messages.append({"role": "user", "content": f"Observation:\n{observation}"})
else:
    # Safety belt -- never loop (and bill) forever if it never reaches an Answer.
    print(f"\n⚠️  gave up after {MAX_STEPS} steps -- no final answer.")
