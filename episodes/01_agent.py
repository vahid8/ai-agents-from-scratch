"""Episode 1 — What IS an agent? The minimal agent loop, from scratch.

Season 1 taught one-shot LLM calls and function calling. An *agent* adds the one
missing piece: a LOOP. You give a model some tools and a goal, and it decides --
on its own, step by step -- which tool to call, looks at the result, calls the
next one, and keeps going until it can answer.

Stripped of every buzzword, an agent is just four things:

    1. an LLM            -- the brain that decides what to do next
    2. some tools        -- things it can actually DO (look stuff up, run code)
    3. a loop            -- act, observe the result, decide again
    4. a stop condition  -- quit when the model answers (or we hit a step cap)

That's the whole episode. No framework. About 40 lines of Python.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/01_agent.py
"""

import json
import os
from pathlib import Path

from openai import OpenAI


def get_gemini_key() -> str:
    """Read the key from GEMINI_API_KEY, or from the file GEMINI_API_KEY_FILE points to."""
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    if key_file := os.environ.get("GEMINI_API_KEY_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEY_FILE in your .env")


# The SAME OpenAI SDK from Season 1 -- just pointed at Gemini's free endpoint.
client = OpenAI(
    api_key=get_gemini_key(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"


# --- TOOLS: small Python functions the model is allowed to run. ----------------
# A tiny "database" that lives only inside our app -- the model can't know these.
POPULATIONS = {"France": 68_000_000, "Germany": 84_000_000, "Switzerland": 8_800_000}


def get_population(country: str) -> dict:
    """Look up a country's population in our app's data."""
    return {"country": country, "population": POPULATIONS.get(country, "unknown")}


def calculator(expression: str) -> dict:
    """Evaluate a basic arithmetic expression like '(68000000+84000000)*1.05'.

    LLMs are unreliable at big-number math, so we hand that job to real Python.
    Locked down to digits and + - * / ( ) . only -- no names, no function calls.
    """
    if "**" in expression or not set(expression) <= set("0123456789+-*/(). "):
        return {"error": "only numbers and + - * / ( ) are allowed"}
    try:
        return {"result": eval(expression)}  # safe: input restricted to arithmetic chars
    except ArithmeticError as e:
        return {"error": str(e)}


# Map each tool NAME to the real function, so we can run whatever the model picks.
TOOL_FUNCS = {"get_population": get_population, "calculator": calculator}

# Describe the tools to the model (JSON Schema for each one's arguments).
TOOLS = [
    {"type": "function", "function": {
        "name": "get_population",
        "description": "Look up the population of a country.",
        "parameters": {"type": "object",
                       "properties": {"country": {"type": "string",
                                                  "description": "Country name, e.g. France"}},
                       "required": ["country"]}}},
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Do exact arithmetic. Use this instead of doing math yourself.",
        "parameters": {"type": "object",
                       "properties": {"expression": {"type": "string",
                                                     "description": "e.g. (68000000+84000000)*1.05"}},
                       "required": ["expression"]}}},
]


# A task that NEEDS several steps: look up two numbers, THEN do the math on them.
# One LLM call can't solve this -- it has to use tools, and use them in order.
TASK = ("France and Germany merge into one country. If their combined population "
        "then grows by 5%, how many people is that? "
        "Use the tools for the lookups and the math.")

messages = [{"role": "user", "content": TASK}]
MAX_STEPS = 6  # the stop condition's safety belt: never loop (and bill) forever.

print(f"🎯 task: {TASK}\n")

# ============================ THE AGENT LOOP ===================================
# The entire idea of an "agent" is right here: call the model, run any tools it
# asks for, feed the results back, repeat -- until it stops asking and answers.
for step in range(1, MAX_STEPS + 1):
    resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    msg = resp.choices[0].message

    # Stop condition #1: the model didn't ask for a tool -> it has its answer.
    if not msg.tool_calls:
        print(f"\n✅ final answer (after {step} steps):\n{msg.content}")
        break

    messages.append(msg)  # remember the model's tool request in the history...
    for tc in msg.tool_calls:  # ...run each tool it asked for, feed the result back.
        args = json.loads(tc.function.arguments)
        result = TOOL_FUNCS[tc.function.name](**args)
        print(f"  step {step}: 🔧 {tc.function.name}({args}) -> {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id,
                         "content": json.dumps(result)})
    # loop again: now the model can see the tool results and decide what's next.
else:
    # Stop condition #2: we used every step and it was STILL asking for tools.
    # Without this cap, a confused model could loop (and bill) forever.
    print(f"\n⚠️  gave up after {MAX_STEPS} steps -- no final answer.")
