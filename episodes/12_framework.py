"""Episode 12 (part 3 of 3) — the same agent, in a framework. The grown-up version.

Twelve episodes of refusing to use one. Here is the one, and the reason we waited: you can
now read every line of it and say exactly which of your own files it replaces.

    Agent(...)          the system prompt + the tool list      -> your `describe()` + prompt
    @function_tool      signature and docstring -> schema      -> Ep10's `@mcp.tool`
    Runner.run(...)     call the model, dispatch, feed back    -> Ep2's while-loop
    max_turns=          stop after N turns                     -> Ep11's max_steps
    tool_filter=        which tools the agent may be offered   -> Ep11's allowed_tools
    result.new_items    what happened, step by step            -> Ep6's Tracer

Six things you no longer write. That is a real gain and you should take it. What you give
up is that the loop is now somebody else's: when the agent answers from its own head
instead of calling a tool, there is no `stop=["Observation:"]` of yours to reach for, and
you are reading a framework's source to find out why. You know what to look for now.
Before Ep2, you would not have.

🔴 A GENUINELY AWKWARD FACT, AUGUST 2026. This file is a PEP 723 script -- its dependencies
are in the header, and `uv run --no-project` builds it a private environment. That is not
tidiness. `openai-agents` pins `mcp>=1.19,<2`; FastMCP 4, the only release that speaks the
stateless 2026-07-28 revision, needs `mcp>=2`. They cannot be installed together. So the
framework episode cannot import the finale's server, and the two files below run in two
different environments on purpose.

That is what a six-week-old protocol revision feels like from inside a real project, and it
is the most useful thing in this episode. Frameworks track a spec; they do not track it
instantly. When they lag, the only person who can tell whether the lag matters is the one
who knows what the protocol actually does -- which, after Ep10, is you.

Run it (its own environment, one command, nothing to install first):
    uv run --env-file .env --no-project episodes/12_framework.py
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai-agents>=0.19.4",
# ]
# ///

import asyncio
import os
import re
from pathlib import Path

from agents import (Agent, OpenAIChatCompletionsModel, Runner, function_tool,
                    set_tracing_disabled)
from openai import AsyncOpenAI


def get_gemini_key() -> str:
    """Read the key from GEMINI_API_KEY, or from the file GEMINI_API_KEY_FILE points to."""
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    if key_file := os.environ.get("GEMINI_API_KEY_FILE"):
        return Path(key_file).expanduser().read_text().strip()
    raise RuntimeError("Set GEMINI_API_KEY or GEMINI_API_KEY_FILE in your .env")


# The SDK is OpenAI's, but the model does not have to be. Point an AsyncOpenAI client at
# Gemini's OpenAI-compatible endpoint -- the same base_url as every episode since Ep1 --
# and hand it over as the agent's model. Tracing off: it would try to upload runs to
# OpenAI's dashboard with an API key we do not have.
set_tracing_disabled(True)
gemini = AsyncOpenAI(
    api_key=get_gemini_key(),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


# ============================ THE TOOLS ======================================
# The same two functions, with a different decorator on top. `@function_tool` reads the
# signature and the docstring and generates the schema -- the identical trade Ep10 made
# with `@mcp.tool`. Your docstring is the prompt, in both worlds.
HANDBOOK = {
    "rate limits": "Nimbus Pro allows 60 API requests per minute. Anything above that "
                   "is rejected with HTTP 429 until the next minute starts.",
    "billing": "Nimbus Pro pricing: the plan costs $49 per seat per month. A team pays "
               "for every seat it has; overages are never charged, excess requests are "
               "simply rejected.",
    "refunds": "Refund policy: a confirmed Nimbus service outage lasting a whole billing "
               "month entitles the customer to a full refund of that month's fee.",
    "support": "Nimbus Pro includes email support with a 24-hour response target.",
}


@function_tool
def lookup(query: str) -> str:
    """Look up a fact about the Nimbus Pro plan (rate limits, billing, refunds, support)."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(topic: str) -> int:
        text = set(re.findall(r"[a-z0-9]+", f"{topic} {HANDBOOK[topic]}".lower()))
        return len(words & text)

    best = max(HANDBOOK, key=score)
    return f"[{best}] {HANDBOOK[best]}" if score(best) else "nothing in the handbook matched"


_ALLOWED = set("0123456789+-*/(). ")


@function_tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. 150-60. Digits and + - * / ( ) only."""
    # Still charset-locked, and note who is enforcing it: us. A framework will happily hand
    # a model's string to your function. What that function then does with it has never
    # been the framework's problem, in any framework, ever.
    if not expression or set(expression) - _ALLOWED:
        return "error: numbers and + - * / ( ) only"
    if "**" in expression or len(expression) > 100:
        return "error: no exponentiation, 100 characters max"
    try:
        return str(eval(expression))  # noqa: S307 -- input restricted to the charset above
    except Exception as e:
        return f"error: {e}"


# ============================ THE AGENT ======================================
# Everything Ep12's finale spends two hundred lines on, minus the parts that are still
# yours. `refund` is simply not in this list -- which is the framework's version of an
# allowlist when the tools are local. (For MCP servers the SDK ships the real thing:
# `MCPServerStdio(..., tool_filter=create_static_tool_filter(allowed_tool_names=[...]))`,
# and `require_approval` for the human-in-the-loop case. The shape you built by hand in
# Ep11 is the shape it ships with.)
agent = Agent(
    name="nimbus-support",
    instructions="You are a Nimbus support agent. Answer the customer using your tools. "
                 "Never guess a fact about Nimbus -- look it up. Never do arithmetic in "
                 "your head: every calculation goes through the calculator tool.",
    model=OpenAIChatCompletionsModel(model="gemini-2.5-flash", openai_client=gemini),
    tools=[lookup, calculator],
)

TASK = ("Your support team confirmed Nimbus was down for our entire billing month. "
        "We have 12 seats on Nimbus Pro. Please refund us for the month, and tell us "
        "exactly how much that is.")


async def main() -> None:
    print("=" * 78)
    print("THE SAME TICKET, IN A FRAMEWORK")
    print("=" * 78)
    print(f"\n🧑 {TASK}\n")

    # max_turns is Ep11's step cap under another name -- and it is the ONLY one of the
    # three limits the framework gives you. There is no max_cost here. The usage totals
    # come back on the result, so a budget is still yours to build; you now know it is
    # about fifteen lines, because you wrote them.
    result = await Runner.run(agent, TASK, max_turns=12)

    print("🤖 what happened (the framework's version of the run tree):")
    labels = {"toolcall": "tool call", "toolcalloutput": "observation",
              "messageoutput": "final message"}
    for item in result.new_items:
        kind = type(item).__name__.replace("Item", "").lower()
        detail = ""
        if kind == "toolcall":
            detail = (f"  {getattr(item.raw_item, 'name', '')}"
                      f"({getattr(item.raw_item, 'arguments', '')})")
        elif kind == "toolcalloutput":
            detail = f"  -> {str(item.output)[:90]}"
        print(f"    {labels.get(kind, kind):<16}{detail}")

    usage = result.context_wrapper.usage
    print(f"\n    {usage.requests} model calls, "
          f"{usage.input_tokens}+{usage.output_tokens} tokens")

    print("\n=== THE REPLY ===")
    print(result.final_output)
    print("\nNo loop. No parser. No dispatch table. And you can read every line of it,\n"
          "because for twelve episodes you were the framework.")


if __name__ == "__main__":
    asyncio.run(main())
