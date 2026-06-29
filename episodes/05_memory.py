"""Episode 5 — Memory: make the agent remember (so it stops forgetting).

Every agent so far had amnesia. Ask it something, it answers, and the moment the
program stops, everything is gone. Real assistants don't work like that -- they
remember what you told them yesterday.

There are TWO kinds of memory, and we build both, by hand:

    SHORT-TERM  = the conversation itself -- the `messages` list we keep appending
                  to. It's why the agent can answer "what did I just say?" within a
                  chat. But it lives in RAM: close the program and it's gone.

    LONG-TERM   = facts written to DISK as vectors (agent_memory.json). It survives
                  restarts, and we search it by MEANING (the same embed + cosine
                  from Ep4's search_docs) -- so "which day do I rotate keys?" finds
                  the fact we saved as "rotates API keys every Monday".

The loop is the SAME ReAct loop from Ep4. The only new idea is the toolbox: instead
of calculator/web/docs, the agent now has two MEMORY tools and decides when to use
them -- remember[fact] to save, recall[query] to look up.

We prove it with two sessions in one run: you tell the agent things, then we wipe
short-term memory (simulating a restart) and watch a brand-new conversation recall
what only the disk remembers.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/05_memory.py
"""

import json
import math
import os
import re
from pathlib import Path

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
EMBED_MODEL = "gemini-embedding-001"  # same family, free key; 768-d vectors
DIMS = 768


# --- the embedding + similarity helpers, unchanged from Ep4's search_docs. ------
def embed(texts: list[str]) -> list[list[float]]:
    """One batched call to Gemini's embeddings endpoint -> one vector per text."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, dimensions=DIMS)
    return [d.embedding for d in resp.data]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b)))


# ============================ LONG-TERM MEMORY ================================
# A tiny vector store that lives in a JSON file. Each fact is saved with its
# embedding, so we can search old facts by MEANING. This file is what survives when
# the program stops -- the whole point of the episode.
MEMORY_FILE = Path(__file__).with_name("agent_memory.json")


def load_memory() -> list[dict]:
    """Read the saved facts back from disk (empty list the first time)."""
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return []


def save_memory(store: list[dict]) -> None:
    """Persist the whole store to disk after every change (small, so just rewrite it)."""
    MEMORY_FILE.write_text(json.dumps(store))


# For a clean, repeatable demo we start each run from an EMPTY long-term memory.
# In real use you would NOT do this -- you'd keep the file; surviving restarts is
# exactly what long-term memory is for.
save_memory([])
memory = load_memory()


def remember(fact: str) -> str:
    """Save a durable fact: embed it once, append to the store, write to disk."""
    fact = fact.strip()
    memory.append({"text": fact, "vec": embed([fact])[0]})
    save_memory(memory)
    return f'saved to long-term memory: "{fact}"'


def recall(query: str, k: int = 3) -> str:
    """Look facts up by MEANING: embed the query, return the closest saved facts."""
    if not memory:
        return "long-term memory is empty"
    qv = embed([query.strip()])[0]
    scored = sorted(((cosine(qv, m["vec"]), m["text"]) for m in memory), reverse=True)
    return "recalled:\n" + "\n".join(f"- {text}" for _, text in scored[:k])


TOOLS = {"remember": remember, "recall": recall}


# --- THE ReAct PROMPT: same format as Ep4, now with two MEMORY tools. ----------
SYSTEM = """You are a helpful assistant with MEMORY. You hold a normal conversation,
and you can also save and look up durable facts using this EXACT format when useful:

Thought: decide whether you need to save a fact or look one up
Action: tool_name[input]

After an Action you are given an Observation. Then take another Action, or reply:

Thought: reason about your reply
Answer: your reply to the user

Your memory tools:
- remember[fact]   save a durable fact the user tells you about THEMSELVES,
                   e.g. remember[Vahid is on the Nimbus Pro plan]
- recall[query]    look up something the user may have told you earlier,
                   e.g. recall[which plan is the user on]

Rules: ALWAYS begin with a Thought. When the user states a durable fact about
themselves, save it with remember[...]. When they ask about something they could
have told you before but that ISN'T in this conversation, use recall[...] first and
answer from what it returns. If the answer is already in this conversation, just
answer. Output ONE Thought then ONE Action, or ONE Thought then ONE Answer."""

MAX_STEPS = 6
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def agent_reply(messages: list[dict]) -> None:
    """Run the ReAct loop for ONE user turn (already appended) until an Answer.

    Identical machinery to Ep4 -- the agent just happens to be choosing between
    memory tools now. Everything it does is appended to `messages`, which IS the
    short-term memory for this conversation.
    """
    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
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
        observation = TOOLS[name](arg) if name in TOOLS else f"unknown tool: {name}"
        print(f"Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})
    else:
        print("⚠️  gave up -- no answer.\n")


def chat(user_messages: list[str]) -> None:
    """Feed a scripted sequence of user turns through one fresh conversation."""
    messages = [{"role": "system", "content": SYSTEM}]   # a NEW short-term scratchpad
    for user in user_messages:
        print(f"🧑 {user}")
        messages.append({"role": "user", "content": user})
        agent_reply(messages)


# ============================ THE DEMO =======================================
# SESSION 1: you tell the agent durable facts -> it saves them to long-term memory.
# The last turn is answered from SHORT-TERM memory (still in this conversation).
SESSION_1 = [
    "Hi! I'm Vahid. For the record, I'm on the Nimbus Pro plan.",
    "One more thing to note: I rotate my API keys every Monday.",
    "Quick check before you forget -- what plan did I just say I'm on?",
]

# SESSION 2: a brand-new conversation. Short-term memory is gone (we never reuse the
# old `messages`), so the ONLY way to answer is LONG-TERM memory loaded from disk.
SESSION_2 = [
    "Hey, I'm back. Remind me -- which day do I rotate my API keys?",
]

print("=== SESSION 1 — you tell the agent things (saved to long-term memory) ===\n")
chat(SESSION_1)

print("\n=== the program 'closes' — short-term memory is wiped, the JSON file stays ===")
memory = load_memory()  # prove it: reload purely from disk into a fresh process state
print(f"💾 long-term memory on disk: {len(memory)} fact(s)\n")

print("=== SESSION 2 — a brand-new conversation; only long-term memory survived ===\n")
chat(SESSION_2)
