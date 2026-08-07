"""Episode 12 (part 2 of 3) — THE FINALE: ship a real one.

Eleven episodes, one idea each. This file has no twelfth idea in it. Every part of it
arrived in an earlier episode, and the only thing that is new is that they are finally
switched on at the same time, in the order a real system needs them:

    Ep2   the ReAct loop, and the stop-and-truncate rule that keeps the model honest
    Ep6   a tracer, so the run is not a black box and the bill is not a surprise
    Ep8   nesting, so a second agent inside the first still shows up in one tree
    Ep9   a critic, so the answer that ships is not simply the first one written
    Ep10  tools discovered over MCP at runtime, from a process we did not write
    Ep11  a policy -- an allowlist, a step cap and a budget -- enforced in the tracer

That is what "production" means here. Not scale, not a queue, not Kubernetes: an agent
whose steps you can see, whose spending has a ceiling, whose answer has been checked, and
whose permissions do not depend on it feeling cooperative today.

🔴 AND THE ONE THING THAT BITES. Ep10's payoff was that no line of the client names a tool
-- the toolbox is whatever the server advertised. Run this and read the first block of
output: the server now advertises a THIRD tool, `refund`, which was not there in Ep10. We
did not upgrade anything. We did not change a line. Somebody else shipped, and our agent's
abilities changed underneath us -- including its ability to move money, described by a
docstring that was written to be persuasive to a model.

Runtime discovery is the feature and it is the hole, and they are the same mechanism. What
closes it is not a smarter prompt. It is the four lines of POLICY below, and the fact that
they are checked on the path the call actually takes.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/12_finale.py
"""

import asyncio
import os
import re
import time
from pathlib import Path

from fastmcp import Client
from openai import OpenAI

from trace import GuardrailTripped, Tracer


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
SERVER = Path(__file__).with_name("12_server.py")


# ============================ THE POLICY =====================================
# The whole safety surface of this agent, in four lines, in one place, reviewable in a pull
# request by somebody who does not know how any of the rest of it works.
#
# allowed_tools is a REVIEWED list, not a discovered one. It would be one line of code to
# write `allowed_tools={t.name for t in tools}` and it would pass every test, and it would
# also mean the server decides what our agent may do -- which is precisely the thing we are
# defending against. A human typed these two names, once, on purpose.
POLICY = dict(
    max_steps=16,
    max_cost=0.05,
    allowed_tools={"lookup", "calculator"},
)


# ============================ THE LOOP =======================================
# Ep2's format string. Twelve episodes later, not one word of it has changed.
REACT_FORMAT = """Work in this EXACT format:

Thought: your reasoning
Action: tool_name[input]

After an Action you get an Observation. Then take another Action, or finish with:

Thought: your reasoning
Answer: your final answer

Always begin with a Thought. Output ONE Thought then ONE Action, or ONE Thought then
ONE Answer -- never more."""

ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def argument_name(tool) -> str:
    """Which argument does this tool want? Read it off the schema FastMCP generated (Ep10)."""
    schema = tool.input_schema or {}
    names = schema.get("required") or list(schema.get("properties") or {})
    return names[0] if names else "input"


def describe(tools) -> str:
    """Render the discovered tools into the prompt the model reads.

    Note what is NOT filtered here. The model is told about every tool the server offered,
    refund included, because that is the honest situation: the catalogue is assembled at
    runtime and nobody hand-edits it before it reaches the prompt. Restricting the prompt
    would only hide the problem -- a model that never sees `refund` can still emit
    `Action: refund[588]`, and on a bad day it will.
    """
    return "\n".join(f"- {t.name}[{argument_name(t)}]   {' '.join((t.description or '').split())}"
                     for t in tools)


async def call_tool(mcp: Client, tool, value: str) -> str:
    """Run one tool over MCP and flatten the reply into an Observation string (Ep10)."""
    result = await mcp.call_tool(tool.name, {argument_name(tool): value},
                                 raise_on_error=False)
    text = "\n".join(block.text for block in result.content
                     if getattr(block, "type", None) == "text")
    return f"tool error: {text}" if result.is_error else text


async def react(mcp: Client, tools, system: str, task: str, tracer: Tracer,
                show: bool = True) -> str:
    """The one loop. Used for the ANSWER and, unchanged, for the CRITIC.

    Ep2 wrote it. Ep10 made the dispatch cross a process boundary. Ep11 added the two
    `tracer.guard(...)` calls. Nothing here is specific to this episode, and the fact that
    the critic runs the identical function with a different system prompt is the whole
    argument of the season: an agent is a loop, and everything else is what you put around
    it.
    """
    by_name = {t.name: t for t in tools}
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]

    while True:
        tracer.guard("llm")                      # step cap + budget, checked at the root

        t = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
        tracer.llm(resp, time.perf_counter() - t)
        text = resp.choices[0].message.content.strip()

        action = ACTION_RE.search(text)
        answer_at = text.find("Answer:")

        if answer_at != -1 and (action is None or answer_at < action.start()):
            if show:
                print(_indent(text))
            return text[answer_at + len("Answer:"):].strip()

        if action is None:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "Observation: reply with an Action or an Answer."})
            continue

        # Ep2's honesty rule. Twelve episodes on it is doing MORE work than it did on day
        # one: the Observation now comes from a stranger's process, so the gap between what
        # the model expects and what actually comes back is at its widest here.
        turn = text[: action.end()]
        if show:
            print(_indent(turn))
        messages.append({"role": "assistant", "content": turn})

        name, arg = action.group(1), action.group(2).strip()
        try:
            tracer.guard("tool", name)           # reviewed? -- asked before it runs
            if name in by_name:
                t = time.perf_counter()
                observation = await call_tool(mcp, by_name[name], arg)
                tracer.tool(name, arg, observation, time.perf_counter() - t)
            else:
                observation = f"unknown tool: {name}"
        except GuardrailTripped as trip:
            observation = f"refused by policy: {trip}"

        if show:
            print(f"    Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})


# ============================ THE PROMPTS ====================================
def answer_system(tools) -> str:
    return f"""You are a Nimbus support agent. Answer the customer using your tools.

Your tools:
{describe(tools)}

Never guess a fact about Nimbus -- look it up. Never do arithmetic in your head: every
calculation goes through the calculator tool, however easy it looks. If a tool is refused
by policy, do not try it again; tell the customer what you can and cannot do.

{REACT_FORMAT}"""


def critic_system(tools) -> str:
    return f"""You are a reviewer. You are given a TASK and a DRAFT reply written by another
agent. Verify every fact and every number in the draft with your tools, NEVER from memory.

Your tools:
{describe(tools)}

Then finish with ONE of these as your Answer:

Answer: PASS
    -- if every claim in the draft checks out.
Answer: PROBLEM: <what is wrong> -- correct value: <the verified fact or number>

{REACT_FORMAT}"""


REVISE_SYSTEM = """You are given a TASK, your earlier DRAFT reply, and a critic's report
listing what was wrong together with the verified correct facts. Rewrite the reply so it
fixes EXACTLY what the critic flagged, inventing nothing. Output only the corrected reply."""


def revise(task: str, draft: str, critique: str, tracer: Tracer) -> str:
    """One LLM call to fix what the critic found (Ep9), under the same budget."""
    tracer.guard("llm")
    messages = [{"role": "system", "content": REVISE_SYSTEM},
                {"role": "user", "content": f"TASK: {task}\n\nYour draft:\n{draft}\n\n"
                                            f"The critic found:\n{critique}\n\nCorrected reply:"}]
    t = time.perf_counter()
    resp = client.chat.completions.create(model=MODEL, messages=messages, temperature=0)
    tracer.llm(resp, time.perf_counter() - t)
    return resp.choices[0].message.content.strip()


# ============================ THE TASK =======================================
# One ticket that needs all of it: a fact from the handbook (the refund policy), a number
# the model must not do in its head (49 x 12), an action it is not allowed to take, and an
# answer worth checking before it reaches a paying customer.
TASK = ("Your support team confirmed Nimbus was down for our entire billing month. "
        "We have 12 seats on Nimbus Pro. Please refund us for the month, and tell us "
        "exactly how much that is.")

MAX_REFLECTIONS = 2


async def main() -> None:
    print("=" * 78)
    print("THE FINALE — every episode of the season, in one run")
    print("=" * 78)

    async with Client(SERVER, mode="auto") as mcp:
        info = mcp.server_info
        print(f"\n🔌 {getattr(info, 'name', '?')} v{getattr(info, 'version', '?')} "
              f"— protocol {mcp.protocol_version}")

        # ---- Ep10: discovery. ---- and the review that Ep10 did not have. -------------
        tools = await mcp.list_tools()
        allowed = POLICY["allowed_tools"]
        print(f"\n🧰 the server advertises {len(tools)} tools. Our policy reviewed {len(allowed)}:")
        for tool in tools:
            ok = tool.name in allowed
            print(f"    {'✓' if ok else '✗'} {tool.name:<12} "
                  f"{'reviewed' if ok else 'NOT ON THE ALLOWLIST — this one is new'}")
        print("\n   Every one of them is in the model's prompt. Two of them can run.\n")

        print(f"🧑 {TASK}\n")
        tracer = Tracer(task="finale: nimbus outage refund", model=MODEL, **POLICY)

        # ---- Ep2 + Ep10 + Ep11: the agent answers. ------------------------------------
        print("🤖 DRAFT — the agent works the ticket:")
        try:
            answer = await react(mcp, tools, answer_system(tools), TASK, tracer)
        except GuardrailTripped as trip:
            answer = f"stopped by a guardrail: {trip}"
            print(f"    ✋ run stopped: {trip}")

        # ---- Ep9 + Ep8: a critic checks it, in its own sub-trace. ---------------------
        for round_no in range(1, MAX_REFLECTIONS + 1):
            print(f"\n🔎 REFLECT (round {round_no}) — a second agent verifies the draft:")
            sub = tracer.child(f"critic#{round_no}", answer)
            verdict = await react(mcp, tools, critic_system(tools),
                                  f"TASK: {TASK}\n\nDRAFT to review:\n{answer}", sub)
            sub.finish(verdict)

            if verdict.strip().upper().startswith("PASS"):
                print("\n✅ critic: PASS — this is the reply that ships.")
                break
            print(f"\n❌ critic: {verdict}")
            answer = revise(TASK, answer, verdict, tracer)
            print("\n♻️  REVISED:")
            print(_indent(answer))

    # ---- Ep6: one tree, one bill, everything above it. --------------------------------
    tracer.finish(answer)
    print("=== THE REPLY THAT SHIPS ===")
    print(answer)
    print("\nThat is the season: a loop, some tools, a memory, a plan, a colleague, a "
          "critic,\na protocol, and a policy. Next, the same agent in a framework.")


if __name__ == "__main__":
    asyncio.run(main())
