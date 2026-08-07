"""Episode 11 — Guardrails, cost caps and evals: the agent runs on someone else's money.

Ten episodes of making the agent more capable. This one is about taking capability AWAY,
on purpose, because the moment an agent stops running on your laptop three things become
true that were never true before:

    it can loop           -- and a loop that calls a paid API is a bill with no ceiling
    it can spend          -- your budget, in fractions of a cent, forever
    it can reach          -- for any tool you left within reach, including the ones that
                             write, refund, delete, or email a real customer

None of that needs a new subsystem. We already built the thing that knows about all three:
the tracer from Ep6 counts every step, totals every dollar, and sees the name of every tool
before it runs. Recording something and refusing it are the same knowledge asked at
different times -- so the guardrails go IN the tracer, and the agent loop gains exactly two
lines (see `tracer.guard(...)` below).

🔴 THE POINT OF THE EPISODE. Look at SYSTEM: the model is told about `refund`. Look at
POLICY: `refund` is not on the allowlist. The prompt is not a control -- it is a suggestion
to a system that is allowed to ignore it. The allowlist is a control, because it lives on
the path the call actually takes. If you only ever remember one thing from this episode:
never restrict an agent by asking it nicely.

And the second half: a guardrail that quietly breaks your agent is worse than no guardrail,
because you will not notice. So the episode ends with EVALS -- a handful of tasks whose
right answers you know, run under the real policy, printed as a scoreboard. It is the
cheapest test suite you will ever write and the only reason you can change a prompt on a
Friday.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/11_guardrails.py
"""

import os
import re
import time
from pathlib import Path

from openai import OpenAI

from trace import GuardrailTripped, Tracer  # the Ep6 tracer -- it enforces as of this ep


def get_gemini_key() -> str:
    """Read the key from GEMINI_API_KEY, or from the file GEMINI_API_KEY_FILE points to."""
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    if key_file := os.environ.get("GEMINI_API_KEY_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEY_FILE in your .env")


client = OpenAI(
    api_key=get_gemini_key(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-2.5-flash"


# ============================ THE TOOLS ======================================
# Two read-only tools from Ep9... and one that is not read-only at all.
HANDBOOK = {
    "rate limits": "Nimbus Pro allows 60 API requests per minute. Anything above that "
                   "is rejected with HTTP 429 until the next minute starts.",
    "billing": "Nimbus Pro pricing: the plan costs $49 per seat per month. A team pays "
               "for every seat it has; overages are never charged, excess requests are "
               "simply rejected.",
    "refunds": "Refund policy: a confirmed Nimbus service outage lasting a whole "
               "billing month entitles the customer to a full refund of that month's fee.",
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
    """Evaluate a plain arithmetic expression.

    Charset-locked since Ep1 -- and that lock is itself a guardrail, written before we had
    a name for them. Two extra rules since Ep10: `**` is spelled entirely in allowed
    characters, so `9**9**9**9` walks straight through a charset check and hangs the
    process on a number with billions of digits.
    """
    if not expression or set(expression) - _ALLOWED:
        return "error: numbers and + - * / ( ) only"
    if "**" in expression or len(expression) > 100:
        return "error: no exponentiation, max 100 characters"
    try:
        return str(eval(expression))  # noqa: S307 -- input restricted to the charset above
    except Exception as e:
        return f"error: {e}"


def refund(amount: str) -> str:
    """Refund a customer. THIS SPENDS REAL MONEY -- and the agent must never reach it.

    It is here to be reachable. Every real codebase has one: a function that writes, pays,
    deletes or emails, sitting in the same module as the harmless ones because that is
    where it belongs. The question an episode about guardrails has to answer is not "how do
    I avoid writing this function" -- you cannot -- but "what actually stops the agent
    calling it". The answer is below, and it is not the prompt.
    """
    return f"REFUNDED ${amount} to the customer's card"


TOOLS = {"lookup": lookup, "calculator": calculator, "refund": refund}


# ============================ THE POLICY =====================================
# Three numbers and a set -- the entire safety surface of this agent, in one place you can
# read in five seconds, review in a pull request, and change without touching the loop.
#
# max_cost is deliberately generous: a run costs about a third of a cent, so this stops a
# runaway, not a working agent. A cap you trip on the happy path is not a guardrail, it is
# an outage.
POLICY = dict(
    max_steps=10,
    max_cost=0.02,
    allowed_tools={"lookup", "calculator"},   # note who is NOT in here
)


# ============================ THE AGENT ======================================
# Ep2's format string, still not one word different.
REACT_FORMAT = """Work in this EXACT format:

Thought: your reasoning
Action: tool_name[input]

After an Action you get an Observation. Then take another Action, or finish with:

Thought: your reasoning
Answer: your final answer

Always begin with a Thought. Output ONE Thought then ONE Action, or ONE Thought then
ONE Answer -- never more."""

# The model is told about all three tools, refund included. That is the honest situation:
# in a real system the model reads a tool list assembled by somebody else, or discovered
# from a server (Ep10), and you do not get to hand-audit it. The prompt is where you ask;
# the policy is where you decide.
SYSTEM = """You are a Nimbus support agent. Answer the customer's question using your tools.

Your tools:
- lookup[topic]            check a fact in the Nimbus handbook
- calculator[expression]   arithmetic, e.g. calculator[150-60]
- refund[amount]           refund a customer, e.g. refund[49]

Never guess a fact about Nimbus -- look it up. Never do arithmetic in your head: every
calculation goes through the calculator tool, however easy it looks. If the handbook says
the customer is entitled to a refund, issue it with refund[amount].

""" + REACT_FORMAT

ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def _indent(text: str, pad: str = "    ") -> str:
    """Indent a (possibly multi-line) block so the transcript reads cleanly."""
    return "\n".join(pad + line for line in text.splitlines())


def run_agent(task: str, policy: dict, label: str, quiet: bool = False) -> tuple[str, Tracer]:
    """The Ep2 ReAct loop, under a policy. Returns (answer, tracer).

    Two things changed, and only two.

    1. There is no `for _ in range(MAX_STEPS)` any more. The loop cap moved into the
       policy, where it is one of several limits instead of a lonely magic number -- and
       where a sub-agent inherits it instead of getting a fresh one (Ep8).
    2. Two `tracer.guard(...)` calls, at the two places where the run is about to do
       something it might not be allowed to do.

    The interesting part is how differently the same exception is treated in those two
    places. A refused TOOL is caught right there and handed back to the model as an
    Observation: the run continues, better informed. A blown budget or step limit is not
    caught until it has left the loop entirely: the run is over. Same mechanism, two
    policies -- steer, or stop -- and the choice is made by where you put the `try`.
    """
    tracer = Tracer(task=label, model=MODEL, **policy)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task}]
    answer = "(the run was stopped before it answered)"

    try:
        while True:
            tracer.guard("llm")                       # too many steps? too much money?

            t = time.perf_counter()
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
            tracer.llm(resp, time.perf_counter() - t)
            text = resp.choices[0].message.content.strip()

            action = ACTION_RE.search(text)
            answer_at = text.find("Answer:")

            if answer_at != -1 and (action is None or answer_at < action.start()):
                if not quiet:
                    print(_indent(text))
                answer = text[answer_at + len("Answer:"):].strip()
                break

            if action is None:                        # no tool, no answer -- nudge it back
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user",
                                 "content": "Observation: reply with an Action or an Answer."})
                continue

            # Ep2's honesty rule: cut the turn at the Action so the model cannot write its
            # own Observation. Still load-bearing -- an agent that invents tool results
            # will happily invent a successful refund.
            turn = text[: action.end()]
            if not quiet:
                print(_indent(turn))
            messages.append({"role": "assistant", "content": turn})

            name, arg = action.group(1), action.group(2).strip()
            try:
                tracer.guard("tool", name)            # allowed? -- asked BEFORE it runs
                if name in TOOLS:
                    t = time.perf_counter()
                    observation = TOOLS[name](arg)
                    tracer.tool(name, arg, observation, time.perf_counter() - t)
                else:
                    observation = f"unknown tool: {name}"
            except GuardrailTripped as trip:
                # STEER. The tool never ran; the model finds out and can try another way.
                observation = f"refused by policy: {trip}"

            if not quiet:
                print(f"    Observation: {observation}\n")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

    except GuardrailTripped as trip:
        # STOP. Nothing catches this inside the loop, so the run ends here.
        if not quiet:
            print(f"    ✋ run stopped: {trip}\n")
        answer = f"stopped by a guardrail: {trip}"

    tracer.finish(answer, show=not quiet)
    return answer, tracer


# ============================ THE EVALS ======================================
# Tasks whose right answer you already know. That is the whole idea -- there is nothing
# clever here, and its absence is why so many agents are shipped on vibes.
#
# `expect` is a substring, not an exact match: we are testing whether the agent got the
# FACT right, not whether it phrased the sentence the way we would have.
EVALS = [
    ("A customer sent 150 requests in one minute on Nimbus Pro. How many were rejected?",
     "90", "rate limit maths"),
    ("Our team has 12 seats on Nimbus Pro. What does that cost per month?",
     "588", "billing maths"),
    ("How quickly does Nimbus Pro support reply to an email?",
     "24", "support SLA"),
]


def run_evals() -> None:
    """Run every eval under the real policy and print a scoreboard.

    Two columns matter beyond pass/fail. STEPS is how you catch an agent that got the right
    answer by flailing at it -- today's five steps becoming tomorrow's nine is a regression
    even when both pass. COST is the same measure in the unit your finance team uses.
    """
    print("\n" + "=" * 78)
    print("EVALS — the same agent, the same policy, tasks whose answers we know")
    print("=" * 78)
    print(f"\n {'case':<20}{'expects':<10}{'verdict':<10}{'steps':<8}{'cost':<10}")
    print(" " + "-" * 56)

    passed, total_cost = 0, 0.0
    for task, expect, name in EVALS:
        answer, tracer = run_agent(task, POLICY, f"eval: {name}", quiet=True)
        ok = expect in answer.replace(",", "")
        passed += ok
        total_cost += tracer.spent()
        print(f" {name:<20}{expect:<10}{'PASS' if ok else 'FAIL':<10}"
              f"{len(tracer.steps):<8}${tracer.spent():.4f}")

    print(" " + "-" * 56)
    print(f" {passed}/{len(EVALS)} passed{'':<19}"
          f"{'':<10}${total_cost:.4f} total\n")


# ============================ THE DEMO =======================================
if __name__ == "__main__":
    # ---- 1. The allowlist. The prompt offers refund; the policy does not. -------------
    print("=" * 78)
    print("1) THE AGENT REACHES FOR A TOOL IT IS NOT ALLOWED TO USE")
    print("=" * 78)
    # The handbook entitles this customer to the refund, and the prompt tells the agent to
    # issue one. Everything about the situation says yes. The allowlist still says no --
    # which is the only kind of "no" that survives a model deciding otherwise.
    task = ("Your support team confirmed Nimbus was down for my entire billing month. "
            "Please refund my $49.")
    print(f"🧑 {task}\n")
    answer, _ = run_agent(task, POLICY, "guardrails: refund attempt")
    print("=== ANSWER ===")
    print(answer)

    # ---- 2. The budget. Same agent, same task, a cap it cannot finish inside. ---------
    print("\n" + "=" * 78)
    print("2) THE SAME RUN, ON A BUDGET TOO SMALL TO FINISH IT")
    print("=" * 78)
    # A hundredth of a cent -- not a realistic budget, just one small enough to hit inside
    # a demo. In production this number is a month, not a sentence.
    broke = {**POLICY, "max_cost": 0.0001}
    print(f"🧑 {task}\n   policy: max_cost=${broke['max_cost']:.5f}\n")
    answer, _ = run_agent(task, broke, "guardrails: budget too small")
    print("=== ANSWER ===")
    print(answer)

    # ---- 3. Did any of that break the agent? Only one way to know. -------------------
    run_evals()
