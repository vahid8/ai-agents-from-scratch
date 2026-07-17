"""Episode 6 — Planning: make the agent lay out the whole route before it walks it.

Every agent so far was a REACTOR. ReAct (Ep2-5) decides ONE step at a time: think,
act, look at the result, think again. It's flexible, but it never steps back to see
the whole task -- it's driving while staring at the road two metres ahead.

This episode builds the other pattern: PLAN-AND-EXECUTE.

    PLAN     -- one LLM call reads the task and writes the whole numbered route up
                front, before any tool runs. You can READ the plan and sanity-check it.
    EXECUTE  -- walk the plan step by step, running the right tool for each, carrying
                the results forward so a later step can use an earlier step's numbers.
    ANSWER   -- one LLM call turns the collected results into a final reply.

ReAct vs plan-and-execute isn't "which is better" -- it's a trade. Planning gives you
a visible, auditable route and is great when the steps are mostly knowable; ReAct
adapts better when each step depends on what the last one turned up. Real agents often
do both (plan, then re-plan when reality disagrees). Here we build the planning half by
hand so you can see the seam.

NEW this episode: a tiny **Tracer** (trace.py). An agent is a multi-step loop that is
invisible from the outside -- you only see the final answer. We wrap the run in a Tracer
that records every LLM call and every tool call, then prints the whole run as a tree
with tokens, cost, and latency, and saves it to a SQLite file. It's the per-RUN analog
of Ep... the gateway's per-CALL bill. Every episode from here runs under it.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/06_planning.py
"""

import os
import re
import time
from pathlib import Path

from openai import OpenAI

from trace import Tracer  # our tiny run tracer (episodes/trace.py) -- new this episode


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


# ============================ THE TOOLS ======================================
# Two tiny, deterministic tools so the demo is stable and offline -- the planning
# machinery is the lesson, not the tools.
POPULATION = {"France": 68_000_000, "Switzerland": 8_700_000, "Germany": 84_000_000}


def get_population(country: str) -> str:
    """Look up a country's population from a fixed table."""
    n = POPULATION.get(country.strip())
    return str(n) if n is not None else f"unknown country: {country}"


# Charset-locked calculator -- same safety trick as Ep1/Ep2: eval only ever sees
# digits and math operators, and has NO builtins, so there's nothing dangerous to run.
ALLOWED = set("0123456789+-*/(). ")


def calculator(expression: str) -> str:
    """Evaluate a pure-arithmetic expression (digits and + - * / ( ) only)."""
    if not set(expression) <= ALLOWED:
        return "error: only numbers and + - * / ( ) are allowed"
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:  # noqa: BLE001 -- surface any math error back to the agent
        return f"error: {e}"


TOOLS = {"get_population": get_population, "calculator": calculator}
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


# ============================ 1) PLAN ========================================
PLANNER_SYSTEM = """You are a PLANNER. Break the user's task into a short ordered list
of concrete steps, where each step is achievable with ONE tool call.

Tools available:
- get_population[country]     look up a country's population
- calculator[expression]      do arithmetic (digits and + - * / ( ) only)

Output ONLY a numbered list, one step per line, and nothing else."""


def plan(task: str, tracer: Tracer) -> list[str]:
    """One LLM call: turn the task into a numbered list of steps (the whole route)."""
    messages = [{"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": task}]
    t = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    tracer.llm(resp, time.perf_counter() - t)
    text = resp.choices[0].message.content.strip()
    # keep only the numbered lines; strip the "1. " / "2) " prefix.
    return [re.sub(r"^\s*\d+[.)]\s*", "", ln).strip()
            for ln in text.splitlines() if re.match(r"^\s*\d+[.)]", ln)]


# ============================ 2) EXECUTE =====================================
STEP_SYSTEM = """You execute ONE step of a plan by choosing a single tool.

Tools:
- get_population[country]
- calculator[expression]   (digits and + - * / ( ) only)

Use the RESULTS SO FAR to fill in real numbers -- e.g. once you know two populations,
a step like "add them" becomes calculator[68000000 + 8700000].

Reply with EXACTLY one line, in this form and nothing else:
Action: tool[input]"""


def execute(task: str, steps: list[str], tracer: Tracer) -> list[str]:
    """Walk the plan: for each step, pick a tool, run it, carry the result forward."""
    notes: list[str] = []
    for step in steps:
        context = "\n".join(notes) if notes else "(nothing yet)"
        messages = [
            {"role": "system", "content": STEP_SYSTEM},
            {"role": "user", "content":
                f"Task: {task}\nResults so far:\n{context}\n\nDo this step: {step}"},
        ]
        t = time.perf_counter()
        resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
        tracer.llm(resp, time.perf_counter() - t)
        action = ACTION_RE.search(resp.choices[0].message.content)
        if action is None:  # step didn't produce a tool call -- record and move on
            notes.append(f"{step} -> (no tool call)")
            continue
        name, arg = action.group(1), action.group(2).strip()
        t = time.perf_counter()
        observation = TOOLS[name](arg) if name in TOOLS else f"unknown tool: {name}"
        tracer.tool(name, arg, observation, time.perf_counter() - t)
        notes.append(f"{step} -> {observation}")
    return notes


# ============================ 3) ANSWER ======================================
ANSWER_SYSTEM = """You are given a task and the result of each step that was run.
Write the final answer for the user in one or two sentences. Be specific with the
numbers; if the task asked "by how much", give the amount."""


def synthesize(task: str, notes: list[str], tracer: Tracer) -> str:
    """One LLM call: fold the collected step results into a final answer."""
    messages = [{"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content":
                    f"Task: {task}\nResults:\n" + "\n".join(notes) + "\n\nFinal answer:"}]
    t = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    tracer.llm(resp, time.perf_counter() - t)
    return resp.choices[0].message.content.strip()


def run(task: str) -> None:
    """Plan -> execute -> answer, all wrapped in one Tracer so the run is visible."""
    print(f"🧑 {task}\n")
    tracer = Tracer(task=task, model=MODEL)

    steps = plan(task, tracer)
    print("📋 PLAN (written up front, before any tool runs):")
    for i, step in enumerate(steps, 1):
        print(f"   {i}. {step}")
    print()

    notes = execute(task, steps, tracer)
    final = synthesize(task, notes, tracer)

    # The Tracer prints the whole run as a tree (tokens/cost/latency) and saves it.
    tracer.finish(final)


# ============================ THE DEMO =======================================
# A task that genuinely needs several dependent steps: two look-ups, an addition that
# needs those results, and a comparison. Watch it write the PLAN first, then execute it,
# then the Tracer shows every step with its cost.
if __name__ == "__main__":
    run("Do France and Switzerland together have more people than Germany? By how much?")
