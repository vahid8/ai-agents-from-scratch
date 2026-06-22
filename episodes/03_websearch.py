"""Episode 3 — A web-search agent: let it research what it doesn't know.

Episodes 1 and 2 built an agent, then made it reason out loud with ReAct. But its
tools were toys: a tiny population dictionary that lived inside our own code. The
agent could only ever "know" what we'd hard-coded.

This episode we hand it a REAL tool — a web search — so it can go out and research
a question it doesn't have an answer to. Same ReAct loop from Episode 2; we just
swap the toy tools for one that reaches the live internet:

    Thought:      what should I look up next
    Action:       web_search[a search query]
    Observation:  <the top results we fetched for it>
    ...repeat (search again, refine) until...
    Answer:       a synthesis, WITH the source URLs it used

We use DuckDuckGo via the `ddgs` library — no API key, no bill, fits the series.
(In production you'd reach for an agent-optimized search like Tavily; same shape,
just a hosted endpoint with a key.)

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/03_websearch.py
"""

import os
import re
from pathlib import Path

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


# --- THE NEW TOOL: a real web search (DuckDuckGo, no key needed). ---------------
MAX_RESULTS = 4


def web_search(query: str) -> str:
    """Search the live web and return the top results as title + snippet + url.

    This is the agent's window to the world. We keep the output small and tidy --
    just enough for the model to reason over -- and we always include the URL so it
    can cite its sources. (Results are live, so they change run to run.)
    """
    try:
        results = DDGS().text(query.strip(), max_results=MAX_RESULTS)
    except Exception as e:  # network/rate-limit hiccups shouldn't kill the loop
        return f"search error: {e}"
    if not results:
        return "no results found"
    lines = []
    for i, r in enumerate(results, 1):
        snippet = " ".join(r.get("body", "").split())[:240]  # trim + collapse whitespace
        lines.append(f"[{i}] {r.get('title', '')}\n    {snippet}\n    {r.get('href', '')}")
    return "\n".join(lines)


TOOLS = {"web_search": web_search}


# --- THE ReAct PROMPT: same format as Ep2, but now there's just one real tool. --
SYSTEM = """You research questions step by step using this EXACT format:

Thought: reason about what to look up next
Action: web_search[a focused search query]

After each Action you are given an Observation: the top search results (each with a
URL). Search as many times as you need -- refine your query if the first results are
thin. When you have enough to answer, write:

Thought: reason about the final answer
Answer: a concise answer, and on a final line: Sources: <the URLs you used>

Tools you can use:
- web_search[query]    e.g. web_search[ReAct paper authors year]

Rules: ALWAYS begin with a Thought. Output ONE Thought and then ONE Action (or ONE
Answer). Never write an Observation yourself -- it is given to you. Base your Answer
only on what the Observations actually say; do not invent facts or URLs."""

TASK = ("In the last episode we built the ReAct pattern. Research the original ReAct "
        "research paper: its full title, the authors, and the year it was published. "
        "Then name one popular open-source framework that implements ReAct-style agents. "
        "Give a short answer with your sources.")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task: {TASK}"}]

MAX_STEPS = 6
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)

print(f"🎯 task: {TASK}\n")

# ============================ THE ReAct LOOP ==================================
# Identical to Episode 2 -- the agent doesn't care that the tool now hits the real
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
                         "content": "Observation: no valid Action found. Use web_search[query]."})
        continue

    # ...and as in Ep2 we TRUNCATE at the Action so the model can't hallucinate its
    # own search results -- the only Observation it sees is the one WE fetched.
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
