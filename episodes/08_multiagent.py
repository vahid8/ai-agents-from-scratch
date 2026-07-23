"""Episode 8 — Multi-agent: one agent hands work to another.

Every episode so far had ONE agent doing everything: one system prompt, one pile of
tools, one loop. That works until the job needs different kinds of work -- look
something up, do the arithmetic, write it nicely -- and the single prompt turns into a
list of rules nobody can read, with every tool in reach on every step.

This episode splits the job across SPECIALISTS:

    ORCHESTRATOR  -- owns the goal. It has NO tools of its own. Its "tools" are other
                     agents, and its job is to decide who does what, in what order, and
                     to carry each result to the next one.
    WORKERS       -- small agents with a narrow prompt and only the tools they need:
                     researcher (looks facts up), analyst (does the maths),
                     writer (turns the collected facts into the final text).

The trick is that there is NO new machinery. A hand-off is just a tool call whose
"tool" happens to be another agent: the orchestrator emits `researcher[...]`, we run a
whole ReAct loop for the researcher, and its final Answer comes back as the
orchestrator's Observation. That's why `run_agent()` below is called recursively --
the orchestrator is an agent whose tools are agents.

What you gain: each prompt stays short and testable, and a worker can only reach its
own tools (the analyst can't touch the handbook). What you pay: more LLM calls, so more
latency and more money -- which is exactly why the Tracer now NESTS. The run tree shows
each hand-off as a sub-tree with its own cost, so you can see what delegation actually
cost you.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/08_multiagent.py
"""

import os
import re
import time
from pathlib import Path

from openai import OpenAI

from trace import Tracer  # the run tracer from Ep6 -- it nests as of this episode


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
# Two tiny, deterministic tools so the demo is stable and offline. Note that NEITHER
# belongs to "the agent" any more -- each belongs to ONE worker.
HANDBOOK = {
    "rate limits": "Nimbus Pro allows 60 API requests per minute. Anything above that "
                   "is rejected with HTTP 429 until the next minute starts.",
    "billing": "Nimbus Pro is billed monthly at $49 per seat. Overages are not billed; "
               "excess requests are rejected instead.",
    "support": "Nimbus Pro includes email support with a 24-hour response target.",
}


def lookup(query: str) -> str:
    """Look a topic up in the Nimbus handbook by keyword overlap (the researcher's tool)."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(topic: str) -> int:
        text = set(re.findall(r"[a-z0-9]+", f"{topic} {HANDBOOK[topic]}".lower()))
        return len(words & text)

    best = max(HANDBOOK, key=score)
    if not score(best):
        return "nothing in the handbook matched"
    return f"[{best}] {HANDBOOK[best]}"


_ALLOWED = set("0123456789+-*/(). ")


def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression (the analyst's tool).

    Charset-locked since Ep1: only digits, operators and brackets ever reach eval, so
    the model cannot smuggle code in through a tool argument.
    """
    if not expression or set(expression) - _ALLOWED:
        return "error: numbers and + - * / ( ) only"
    try:
        return str(eval(expression))  # noqa: S307 -- input restricted to the charset above
    except Exception as e:
        return f"error: {e}"


# ========================== THE AGENTS =======================================
# An agent is just a system prompt plus the tools it is allowed to touch. Three narrow
# workers -- each prompt is a few lines, because each one has exactly one job.
REACT_FORMAT = """Work in this EXACT format:

Thought: your reasoning
Action: tool_name[input]

After an Action you get an Observation. Then take another Action, or finish with:

Thought: your reasoning
Answer: your result

Always begin with a Thought. Output ONE Thought then ONE Action, or ONE Thought then
ONE Answer -- never more."""

WORKERS = {
    "researcher": {
        "system": "You are a researcher. You answer ONLY from the Nimbus handbook, using "
                  "your lookup tool -- never from your own knowledge. Your Answer states "
                  "the fact you found, briefly.\n\nYour tool:\n"
                  "- lookup[topic]  search the Nimbus handbook\n\n" + REACT_FORMAT,
        "tools": {"lookup": lookup},
    },
    "analyst": {
        "system": "You are an analyst. You do arithmetic with your calculator tool -- never "
                  "in your head. Your Answer is the number plus a few words of context.\n\n"
                  "Your tool:\n- calculator[expression]  e.g. calculator[150-60]\n\n"
                  + REACT_FORMAT,
        "tools": {"calculator": calculator},
    },
    "writer": {
        "system": "You are a writer. You are given facts and an instruction, and you write "
                  "the final text for a human. You invent NOTHING -- use only the facts you "
                  "were given. You have no tools, so reply immediately with a Thought then "
                  "an Answer containing just the finished text.\n\n" + REACT_FORMAT,
        "tools": {},
    },
}

# The orchestrator: no tools, only colleagues. Notice its "tools" section lists AGENTS,
# in exactly the same bracket syntax -- to the model, delegating and calling a tool are
# the same move.
ORCHESTRATOR = {
    "system": """You are an orchestrator. You never do the work yourself and you never
state facts of your own: you delegate every part of the task to a specialist and pass
what you learn along.

Your specialists:
- researcher[question]     looks a fact up in the Nimbus handbook
- analyst[question]        does arithmetic
- writer[instruction + all the facts]  writes the final text for the user

Rules: delegate one step at a time, and give each specialist everything it needs in the
brackets -- they cannot see this conversation or each other's results. When the task
asks for written output, the writer must produce it, and your final Answer is then that
text, unchanged.

""" + REACT_FORMAT,
    "tools": {},   # the orchestrator's "tools" are the WORKERS, dispatched below
}

MAX_STEPS = 8
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def run_agent(agent: dict, task: str, tracer: Tracer, depth: int = 0) -> str:
    """Run ONE agent's ReAct loop until it answers, and return that answer.

    This is the same loop as Ep5-7, with one addition: if the action names another
    AGENT, we open a child trace and call run_agent again. Recursion IS the hand-off.
    """
    pad = "    " * depth  # indent the transcript so you can see who is talking
    messages = [{"role": "system", "content": agent["system"]},
                {"role": "user", "content": task}]

    for _ in range(MAX_STEPS):
        t = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
        tracer.llm(resp, time.perf_counter() - t)
        text = resp.choices[0].message.content.strip()

        action = ACTION_RE.search(text)
        answer_at = text.find("Answer:")

        # Stop condition: this agent produced its final Answer -- hand it back up.
        if answer_at != -1 and (action is None or answer_at < action.start()):
            print("\n".join(pad + line for line in text.splitlines()))
            return text[answer_at + len("Answer:"):].strip()

        if action is None:  # no tool, no answer -- nudge it back to the format
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Observation: reply with an Action or an Answer."})
            continue

        # Truncate at the Action so the model can't hallucinate its own Observation
        # (the Ep2 honesty rule -- it still applies with agents doing the work).
        turn = text[: action.end()]
        print("\n".join(pad + line for line in turn.splitlines()))
        messages.append({"role": "assistant", "content": turn})

        name, arg = action.group(1), action.group(2).strip()
        if name in WORKERS:
            # ---- THE HAND-OFF: a whole agent runs, and its Answer is our Observation.
            print(f"{pad}↳ handing off to {name}")
            sub = tracer.child(name, arg)                       # nested trace
            observation = run_agent(WORKERS[name], arg, sub, depth + 1)
            sub.finish(observation)
        elif name in agent["tools"]:
            t = time.perf_counter()
            observation = agent["tools"][name](arg)
            tracer.tool(name, arg, observation, time.perf_counter() - t)
        else:
            observation = f"unknown: {name}"

        print(f"{pad}Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "gave up -- no answer."


# ============================ THE DEMO =======================================
# One task that needs three different kinds of work: a fact, a calculation, and some
# writing. A single agent could do it -- the point is to watch it get SPLIT, and to see
# in the trace what each hand-off cost.
TASK = ("We are on the Nimbus Pro plan. Find our per-minute request limit, work out how "
        "many requests would be rejected if we sent 150 in one minute, and have that "
        "written up in two sentences for the team.")

if __name__ == "__main__":
    print(f"🧑 {TASK}\n")
    tracer = Tracer(task="multi-agent: rate limit write-up", model=MODEL)
    answer = run_agent(ORCHESTRATOR, TASK, tracer)
    tracer.finish(answer)
    print("=== FINAL ANSWER ===")
    print(answer)
