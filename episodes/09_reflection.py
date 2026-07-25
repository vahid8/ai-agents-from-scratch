"""Episode 9 — Reflection: make the agent criticise its OWN answer before it ships.

Every agent so far answered in ONE pass. ReAct thinks between tool calls (Ep2), the
planner writes the whole route up front (Ep6), the orchestrator delegates to workers
(Ep8) -- but in every one of them, the first answer the agent writes is the answer the
user gets. Nobody ever checks it.

This episode adds a CRITIC. The agent writes a DRAFT, then looks at its own draft and
asks: is this actually right? If the critic finds a real problem, the draft goes back
for a REVISE, and only an answer the critic approves ever ships. That's the Reflexion
idea, built by hand:

    DRAFT    -- one LLM call answers the task fast, straight from the model's head.
    REFLECT  -- a ReAct loop that VERIFIES the draft against ground truth with tools
                and returns either PASS or a specific, evidence-backed critique.
    REVISE   -- one LLM call rewrites the draft to fix exactly what the critic flagged.

...looping reflect -> revise until the critic passes (or we hit a cap).

There is NO new machinery -- again. REFLECT is the same ReAct loop from Ep2/Ep8, tools
and honesty guard included; DRAFT and REVISE are single LLM calls like Ep6's synthesize.
The whole episode is one extra loop wrapped around an answer.

And the honesty guard matters MORE here, not less. A critic allowed to say "looks good"
without checking is worthless -- it will bless a wrong answer as confidently as a right
one. So the critic doesn't get to hand-wave: it has to look the facts up with the same
tools, under the same Ep2 stop-and-truncate rule that stops the model inventing an
Observation. A model that can hallucinate a tool result can hallucinate a glowing
self-review just as easily.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/09_reflection.py
"""

import os
import re
import time
from pathlib import Path

from openai import OpenAI

from trace import Tracer  # the run tracer from Ep6 -- it nests as of Ep8


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
# Two tiny, deterministic tools so the demo is stable and offline -- reused from Ep8.
# They belong to the CRITIC here: the critic verifies claims with them, it does not
# trust the draft (or its own memory) about a fictional product it cannot possibly know.
HANDBOOK = {
    "rate limits": "Nimbus Pro allows 60 API requests per minute. Anything above that "
                   "is rejected with HTTP 429 until the next minute starts.",
    "billing": "Nimbus Pro is billed monthly at $49 per seat. Overages are not billed; "
               "excess requests are rejected instead.",
    "support": "Nimbus Pro includes email support with a 24-hour response target.",
}


def lookup(query: str) -> str:
    """Look a topic up in the Nimbus handbook by keyword overlap (ground truth)."""
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
    """Evaluate a plain arithmetic expression (the critic re-does the draft's maths).

    Charset-locked since Ep1: only digits, operators and brackets ever reach eval, so
    the model cannot smuggle code in through a tool argument.
    """
    if not expression or set(expression) - _ALLOWED:
        return "error: numbers and + - * / ( ) only"
    try:
        return str(eval(expression))  # noqa: S307 -- input restricted to the charset above
    except Exception as e:
        return f"error: {e}"


CRITIC_TOOLS = {"lookup": lookup, "calculator": calculator}


# ========================== THE PROMPTS ======================================
# The shared ReAct format the critic works in -- same as Ep2/Ep8.
REACT_FORMAT = """Work in this EXACT format:

Thought: your reasoning
Action: tool_name[input]

After an Action you get an Observation. Then take another Action, or finish with:

Thought: your reasoning
Answer: your verdict

Always begin with a Thought. Output ONE Thought then ONE Action, or ONE Thought then
ONE Answer -- never more."""

# DRAFT: the naive first pass. No tools, no hedging -- we want it to commit to an answer
# from its own head so there is something real for the critic to catch. This is the
# agent BEFORE reflection: fast, confident, and about a product it cannot actually know.
DRAFT_SYSTEM = """You are a Nimbus support agent. Answer the customer directly and
confidently, right now, from what you already know about the Nimbus Pro plan. Give a
specific number -- do NOT hedge, do NOT say you need to look anything up. Reply in
exactly two sentences."""

# REFLECT: the critic. Its whole job is to distrust the draft and verify it against the
# handbook and the calculator, then hand back PASS or a specific, evidence-backed fault.
CRITIC_SYSTEM = """You are a reviewer. You are given a TASK and a DRAFT answer written by
another agent. Decide whether the draft is correct and complete -- but you VERIFY every
fact and every number with your tools, NEVER from your own memory. Look the facts up,
redo the arithmetic, and only then judge.

Your tools:
- lookup[topic]            check a fact in the Nimbus handbook
- calculator[expression]   recompute a number, e.g. calculator[150-60]

When you have finished checking, finish with ONE of these as your Answer:

Answer: PASS
    -- if every claim in the draft checks out against the tools.
Answer: PROBLEM: <what is wrong> -- correct value: <the verified fact or number>
    -- if anything is wrong, unverified, or missing.

""" + REACT_FORMAT

# REVISE: fix exactly what the critic flagged, using the verified facts it handed back.
REVISE_SYSTEM = """You are given a TASK, your earlier DRAFT answer, and a critic's report
listing what was wrong together with the verified correct facts. Rewrite the answer so it
fixes EXACTLY what the critic flagged, using the verified facts and inventing nothing.
Keep the format the task asked for. Output only the corrected answer, nothing else."""

MAX_STEPS = 8
MAX_REFLECTIONS = 2
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def _indent(text: str, pad: str = "    ") -> str:
    """Indent a (possibly multi-line) block so the transcript reads cleanly."""
    return "\n".join(pad + line for line in text.splitlines())


# ============================ 1) DRAFT =======================================
def draft(task: str, tracer: Tracer) -> str:
    """One LLM call: the fast, un-checked first answer (no tools)."""
    messages = [{"role": "system", "content": DRAFT_SYSTEM},
                {"role": "user", "content": task}]
    t = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    tracer.llm(resp, time.perf_counter() - t)
    return resp.choices[0].message.content.strip()


# ============================ 2) REFLECT =====================================
def reflect(task: str, draft_text: str, parent: Tracer, round_no: int) -> tuple[bool, str]:
    """Run the critic's ReAct loop over the draft; return (approved, verdict).

    This is the SAME loop as Ep2/Ep8 -- call the model, parse the Action, run the tool,
    feed the Observation back -- with one difference: its tools verify the DRAFT instead
    of solving the task. It opens a child trace (Ep8 nesting) so the critic's cost shows
    up as its own sub-tree in the run.
    """
    sub = parent.child(f"critic#{round_no}", draft_text)  # nested trace for this round
    messages = [{"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": f"TASK: {task}\n\nDRAFT to review:\n{draft_text}"}]

    verdict = "PROBLEM: the critic ran out of steps without a verdict."
    for _ in range(MAX_STEPS):
        t = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
        sub.llm(resp, time.perf_counter() - t)
        text = resp.choices[0].message.content.strip()

        action = ACTION_RE.search(text)
        answer_at = text.find("Answer:")

        # Stop condition: the critic reached its verdict (no Action comes first).
        if answer_at != -1 and (action is None or answer_at < action.start()):
            print(_indent(text))
            verdict = text[answer_at + len("Answer:"):].strip()
            break

        if action is None:  # no tool, no verdict -- nudge it back to the format
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Observation: reply with an Action or an Answer."})
            continue

        # Truncate at the Action so the model can't hallucinate its own Observation
        # (the Ep2 honesty rule -- a critic that invents tool results is worse than none).
        turn = text[: action.end()]
        print(_indent(turn))
        messages.append({"role": "assistant", "content": turn})

        name, arg = action.group(1), action.group(2).strip()
        if name in CRITIC_TOOLS:
            t = time.perf_counter()
            observation = CRITIC_TOOLS[name](arg)
            sub.tool(name, arg, observation, time.perf_counter() - t)
        else:
            observation = f"unknown tool: {name}"

        print(f"    Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    sub.finish(verdict)
    approved = verdict.strip().upper().startswith("PASS")
    return approved, verdict


# ============================ 3) REVISE ======================================
def revise(task: str, draft_text: str, critique: str, tracer: Tracer) -> str:
    """One LLM call: rewrite the draft to fix exactly what the critic flagged."""
    messages = [{"role": "system", "content": REVISE_SYSTEM},
                {"role": "user", "content":
                    f"TASK: {task}\n\nYour draft:\n{draft_text}\n\n"
                    f"The critic found:\n{critique}\n\nWrite the corrected answer:"}]
    t = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    tracer.llm(resp, time.perf_counter() - t)
    return resp.choices[0].message.content.strip()


# ============================ THE DEMO =======================================
# A task whose right answer lives in a handbook the model has never seen. The draft
# answers from its head -- so it GUESSES the rate limit and gets the rejected count
# wrong. The critic looks the real limit up, redoes the subtraction, and catches it;
# the revise fixes it. That's reflection earning its place: without the critic pass,
# the confident-but-wrong first answer would have shipped straight to the customer.
TASK = ("A customer on the Nimbus Pro plan sent 150 API requests in a single minute and "
        "some were rejected. Reply to them in exactly two sentences: how many of the 150 "
        "were rejected, and why.")

if __name__ == "__main__":
    print(f"🧑 {TASK}\n")
    tracer = Tracer(task="reflection: nimbus rejected-requests reply", model=MODEL)

    answer = draft(TASK, tracer)
    print("✍️  DRAFT (first pass, answered straight from the model's head):")
    print(_indent(answer) + "\n")

    for round_no in range(1, MAX_REFLECTIONS + 1):
        print(f"🔎 REFLECT (round {round_no}) — the agent checks its OWN draft with tools:")
        approved, verdict = reflect(TASK, answer, tracer, round_no)
        if approved:
            print("✅ critic: PASS — shipping this answer.\n")
            break
        print(f"❌ critic: {verdict}\n")
        answer = revise(TASK, answer, verdict, tracer)
        print("♻️  REVISED answer (draft sent back and fixed):")
        print(_indent(answer) + "\n")

    tracer.finish(answer)
    print("=== FINAL ANSWER ===")
    print(answer)
