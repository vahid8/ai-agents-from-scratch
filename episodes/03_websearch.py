"""Episode 3 — A web-search agent: let it research what it doesn't know.

Episodes 1 and 2 built an agent, then made it reason out loud with ReAct. But its
tools were toys: a tiny population dictionary that lived inside our own code. The
agent could only ever "know" what we'd hard-coded.

This episode we give it TWO real tools so it can research a question it has no answer
to — the canonical pattern every research agent uses:

    web_search[query]  -> FIND candidate pages (titles + snippets + URLs)
    fetch_url[url]      -> READ one of them (we GET the page and hand back its text)

Same ReAct loop from Episode 2 — the agent doesn't care that the tools now reach the
live internet. It searches, picks a promising URL, reads it, and answers WITH sources:

    Thought:      what should I look up / read next
    Action:       web_search[...]  or  fetch_url[...]
    Observation:  <what we fetched for it>
    ...repeat until...
    Answer:       a synthesis, on a final line: Sources: <the URLs it used>

Search backend (keyless by default, reliable if you want it):
  • Default: DuckDuckGo via `ddgs` — no key, no bill. It rate-limits, so we RETRY
    with backoff; a flaky tool is a fact of agent life, not a reason to crash.
  • Optional: point TAVILY_API_KEY_FILE at a key file (free tier at tavily.com) and we
    use Tavily instead — an agent-optimized search. Same tool shape, hosted + a key.
    (Keep the key in ~/.secrets/, never inline — same paths-not-secrets rule as Gemini.)

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/03_websearch.py
"""

import os
import re
import time
from html.parser import HTMLParser
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


def get_tavily_key() -> str | None:
    """Optional. Read TAVILY_API_KEY, or the file TAVILY_API_KEY_FILE points to.

    Same paths-not-secrets pattern as the Gemini key: keep the real key in
    ~/.secrets/ and point TAVILY_API_KEY_FILE at it. Returns None if unset --
    the agent just falls back to keyless DuckDuckGo.
    """
    if key := os.environ.get("TAVILY_API_KEY"):
        return key
    if key_file := os.environ.get("TAVILY_API_KEY_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    return None


# Same OpenAI SDK pointed at Gemini's free endpoint -- unchanged since Ep1.
client = OpenAI(
    api_key=get_gemini_key(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"


# --- TOOL 1: web search — FIND pages. ------------------------------------------
MAX_RESULTS = 4


def _format_results(results: list[dict]) -> str:
    """Both backends return rows with title/snippet/url -- render them the same way."""
    if not results:
        return "no results found"
    lines = []
    for i, r in enumerate(results, 1):
        snippet = " ".join((r.get("body") or "").split())[:240]  # trim + collapse whitespace
        lines.append(f"[{i}] {r.get('title', '')}\n    {snippet}\n    {r.get('href', '')}")
    return "\n".join(lines)


def _tavily_search(query: str, key: str) -> list[dict]:
    """Tavily REST (no extra library -- just an HTTP POST). Normalised to our row shape."""
    resp = httpx.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": MAX_RESULTS},
        timeout=20,
    )
    resp.raise_for_status()
    return [{"title": r.get("title"), "body": r.get("content"), "href": r.get("url")}
            for r in resp.json().get("results", [])]


def web_search(query: str) -> str:
    """Search the live web; return the top results as title + snippet + url.

    This is the agent's window to the world -- small and tidy, always with the URL so
    it can cite (and so fetch_url has something to read next). Keyless DuckDuckGo by
    default; Tavily if TAVILY_API_KEY is set. Free search is flaky, so we RETRY.
    """
    query = query.strip()
    if key := get_tavily_key():
        try:
            return _format_results(_tavily_search(query, key))
        except Exception as e:
            return f"search error (tavily): {e}"

    last_err = "no results found"
    for attempt in range(3):  # DuckDuckGo rate-limits; back off and try again
        try:
            results = DDGS().text(query, max_results=MAX_RESULTS)
            if results:
                return _format_results(results)
        except Exception as e:  # network/rate-limit hiccups shouldn't kill the loop
            last_err = f"search error: {e}"
        time.sleep(1.5 * (attempt + 1))
    return last_err


# --- TOOL 2: fetch a URL — READ a page. ----------------------------------------
FETCH_CHARS = 2000  # cap the page text we feed back -- enough to reason over, not a flood


class _TextExtractor(HTMLParser):
    """Turn HTML into readable text with the standard library -- no scraping framework.

    We collect text nodes and skip the noisy containers (script/style/nav chrome). It's
    deliberately simple: real agents reason fine over slightly-messy page text.
    """
    _SKIP = {"script", "style", "head", "noscript", "svg", "template"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip())


def fetch_url(url: str) -> str:
    """GET a page and hand back its readable text (capped). The agent's "read it" tool."""
    url = url.strip()
    try:
        resp = httpx.get(
            url, timeout=20, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ai-agents-from-scratch/1.0)"},
        )
        resp.raise_for_status()
    except Exception as e:
        return f"fetch error: {e}"
    parser = _TextExtractor()
    parser.feed(resp.text[:400_000])  # don't parse multi-MB pages whole
    text = " ".join(" ".join(parser.parts).split())
    if not text:
        return f"fetched {url} but found no readable text"
    return text[:FETCH_CHARS] + (" …[truncated]" if len(text) > FETCH_CHARS else "")


TOOLS = {"web_search": web_search, "fetch_url": fetch_url}


# --- THE ReAct PROMPT: same format as Ep2, now with two real tools. ------------
SYSTEM = """You research questions step by step using this EXACT format:

Thought: reason about what to look up or read next
Action: tool_name[input]

After each Action you are given an Observation with the result. Search to find
promising pages, then READ the best one to get real details. Search/read as many
times as you need. When you have enough to answer, write:

Thought: reason about the final answer
Answer: a concise answer, and on a final line: Sources: <the URLs you used>

Tools you can use:
- web_search[query]    e.g. web_search[ReAct paper authors year]
- fetch_url[url]       e.g. fetch_url[https://arxiv.org/abs/2210.03629]

Rules: ALWAYS begin with a Thought. Output ONE Thought and then ONE Action (or ONE
Answer). Never write an Observation yourself -- it is given to you. Prefer to fetch_url
a real page before answering. Base your Answer only on what the Observations actually
say; do not invent facts or URLs."""

TASK = ("In the last episode we built the ReAct pattern. Research the original ReAct "
        "research paper: its full title, the authors, and the year it was published. "
        "Then name one popular open-source framework that implements ReAct-style agents. "
        "Give a short answer with your sources.")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task: {TASK}"}]

MAX_STEPS = 8  # search + read + refine needs a few more steps than Ep2
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)

print(f"🎯 task: {TASK}\n")

# ============================ THE ReAct LOOP ==================================
# Identical to Episode 2 -- the agent doesn't care that the tools now hit the real
# internet instead of a dictionary. That's the point: tools are just functions.
for step in range(1, MAX_STEPS + 1):
    # stop=["Observation:"] halts the model right after its Action...
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
    text = resp.choices[0].message.content.strip()

    action = ACTION_RE.search(text)
    answer_at = text.find("Answer:")

    # Stop condition: the model reached its final Answer (and no Action comes first).
    if answer_at != -1 and (action is None or answer_at < action.start()):
        print(text)                                 # the final Thought + Answer + Sources
        messages.append({"role": "assistant", "content": text})
        break

    if action is None:
        print(text)
        messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user",
                         "content": "Observation: no valid Action found. Use web_search[query] or fetch_url[url]."})
        continue

    # ...and as in Ep2 we TRUNCATE at the Action so the model can't hallucinate its
    # own results -- the only Observation it sees is the one WE fetched.
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
