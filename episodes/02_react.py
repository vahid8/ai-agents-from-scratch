"""Episode 2 — ReAct: make the agent think OUT LOUD.

In Episode 1 the agent's reasoning was invisible: it decided which tool to call
silently, inside the model. ReAct (Reason + Act) makes that thinking VISIBLE. The
agent works in a loop of:

    Thought:      what should I do next, and why
    Action:       tool_name[input]
    Observation:  <the tool's result>      (WE fill this in)

...repeating until it can write:

    Answer: <the final answer>

We don't use the SDK's function-calling here — we build ReAct FROM SCRATCH with a
plain text prompt, so you can see exactly how the pattern works. The key trick is a
`stop` sequence: we stop the model right before "Observation:" so it can't make up
the result — we run the real tool and feed the observation back ourselves.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/02_react.py
"""

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


# Same OpenAI SDK from Ep1 -- pointed straight at Gemini's free endpoint.
client = OpenAI(
    api_key=get_gemini_key(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"


# --- TOOLS: plain Python functions the agent may call (same idea as Ep1). -------
POPULATIONS = {"France": 68_000_000, "Germany": 84_000_000, "Switzerland": 8_800_000}


def get_population(country: str) -> str:
    """Look up a country's population in our app's data."""
    pop = POPULATIONS.get(country.strip())
    return str(pop) if pop is not None else f"unknown country: {country}"


def calculator(expression: str) -> str:
    """Exact arithmetic. Locked to digits and + - * / ( ) . only -- no names, no calls."""
    expr = expression.strip()
    if "**" in expr or not set(expr) <= set("0123456789+-*/(). "):
        return "error: only numbers and + - * / ( ) are allowed"
    try:
        return str(eval(expr))  # safe: input restricted to arithmetic chars
    except ArithmeticError as e:
        return f"error: {e}"


TOOLS = {"get_population": get_population, "calculator": calculator}


# --- THE ReAct PROMPT: teach the model the Thought / Action / Observation format.
SYSTEM = """You solve tasks step by step using this EXACT format:

Thought: reason about what to do next
Action: tool_name[input]

After each Action you are given an Observation with the result. Repeat
Thought/Action as many times as needed. When you know the final answer, write:

Thought: reason about the final answer
Answer: the final answer

Tools you can use:
- get_population[country]    e.g. get_population[France]
- calculator[expression]    e.g. calculator[68000000-8800000]

Rules: ALWAYS begin with a Thought, even on the very first step. Output ONE
Thought and then ONE Action (or ONE Answer). Never write an Observation
yourself -- it is given to you."""

TASK = ("Is the combined population of France and Switzerland greater than "
        "Germany's? If so, by how much? If not, how much bigger is Germany?")

messages = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Task: {TASK}"}]

MAX_STEPS = 8
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)

print(f"🎯 task: {TASK}\n")

# ============================ THE ReAct LOOP ==================================
# Each turn the model writes a Thought + an Action; we run the tool and hand back
# an Observation. The reasoning is right there on screen the whole time.
for step in range(1, MAX_STEPS + 1):
    # stop=["Observation:"] asks the model to halt right after its Action...
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
                         "content": "Observation: no valid Action found. Use tool_name[input]."})
        continue

    # ...but models sometimes ignore the stop and hallucinate their OWN Observation.
    # So we TRUNCATE the turn at the end of the Action and throw away anything after
    # it -- the only Observation the agent ever sees is the REAL tool result.
    turn = text[: action.end()]
    print(turn)
    messages.append({"role": "assistant", "content": turn})

    name, arg = action.group(1), action.group(2).strip()
    observation = TOOLS[name](arg) if name in TOOLS else f"unknown tool: {name}"
    print(f"Observation: {observation}\n")
    messages.append({"role": "user", "content": f"Observation: {observation}"})
else:
    # Safety belt -- never loop (and bill) forever if it never reaches an Answer.
    print(f"\n⚠️  gave up after {MAX_STEPS} steps -- no final answer.")
