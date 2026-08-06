"""Episode 10 — MCP: let the agent use tools it did NOT write.

Every tool we have built in nine episodes had one thing in common: we wrote it, in the same
file as the agent, and wired it in by hand with a Python dict. The agent could only ever do
what we had personally coded into it.

MCP -- the Model Context Protocol -- breaks that. Tools live in another process, written by
someone else, in a language you don't care about, and the agent finds out what they are at
RUNTIME instead of importing them. It is the USB-C port for tools.

This is the FRAMEWORK episode of the season: the protocol is FastMCP's job, and ours is the
agent. Two things are worth knowing about that division of labour.

🔴 FIRST, MCP CHANGED. The revision dated 2026-07-28 deleted the `initialize` handshake.
There is no session any more -- the protocol is **stateless**: every request carries its own
protocol version and capabilities, and a server may infer nothing from earlier requests on
the same connection. `server/discover` replaces the handshake for clients that want to ask
before they act. FastMCP 4 speaks this revision (3.x is still handshake-era), which is why
the pinned dependency is a beta. Watch the first line the client prints: `mode="auto"` makes
it probe `server/discover`, negotiate the newest revision both ends know, and fall back to a
handshake against an old server -- three code paths we get for one keyword argument.

🔴 SECOND, THE PAYOFF DID NOT MOVE. Look at how SYSTEM is built in run_agent(): the agent's
toolbox is assembled at runtime out of whatever the server advertised. Grep this file for a
tool name and you will only find it in comments -- no line of CODE here knows what the tools
are called. Point it at a different server and the agent can do different things without a
character changing. That is MCP, not FastMCP; the framework just saved us the JSON.

And the loop itself? Untouched since Ep2, honesty guard and all. The only difference is that
dispatching an Action now crosses a process boundary.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/10_mcp.py
"""

import asyncio
import json
import os
import re
import textwrap
import time
from pathlib import Path

from fastmcp import Client
from openai import OpenAI

from trace import Tracer  # the run tracer from Ep6 -- unchanged this episode


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

# The server we launch. Hand FastMCP a path to a .py file and it starts it as a subprocess
# and talks stdio to it -- the same transport every MCP server your editor runs uses.
# Swap this line for a URL and everything below is unchanged.
SERVER = Path(__file__).with_name("10_mcp_server.py")


# ========================== THE AGENT ========================================
# Same format string as Ep2. Not one word of it has changed.
REACT_FORMAT = """Work in this EXACT format:

Thought: your reasoning
Action: tool_name[input]

After an Action you get an Observation. Then take another Action, or finish with:

Thought: your reasoning
Answer: your final answer

Always begin with a Thought. Output ONE Thought then ONE Action, or ONE Thought then
ONE Answer -- never more."""

MAX_STEPS = 10
ACTION_RE = re.compile(r"Action:\s*(\w+)\s*\[(.*?)\]", re.DOTALL)


def _indent(text: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in text.splitlines())


def argument_name(tool) -> str:
    """Which argument does this tool want? Read it off the discovered schema.

    Our ReAct format sends ONE value in brackets -- calculator[150-60] -- but MCP tools take
    a named JSON object. So we read the tool's own input_schema and use its first required
    property as the name. The server never told us in prose that its argument is called
    "expression"; the schema FastMCP generated from the function signature did.

    (A production client skips this entirely: it hands the whole schema to the model as a
    function-calling definition and gets structured arguments back. We keep the Ep2 format
    so you can see the protocol without a second new thing in the way.)
    """
    schema = tool.input_schema or {}
    names = schema.get("required") or list(schema.get("properties") or {})
    return names[0] if names else "input"


def describe(tools) -> str:
    """Render the discovered tools into the prompt the model will read."""
    return "\n".join(f"- {t.name}[{argument_name(t)}]   {(t.description or '').strip()}"
                     for t in tools)


async def call_tool(mcp: Client, tool, value: str) -> str:
    """Run one tool over MCP and flatten the reply into a string for the Observation.

    raise_on_error=False is the interesting flag. MCP has TWO failure modes and FastMCP
    normally turns both into Python exceptions: a *protocol* error (no such tool -- the
    request was malformed, and the model cannot fix that by trying again) and a *tool
    execution* error (the tool ran and blew up, which comes back as an ordinary result with
    isError set). Switching the raise off lets us hand the second kind to the model as an
    Observation, which is exactly what the spec recommends: it is feedback it can act on.
    """
    result = await mcp.call_tool(tool.name, {argument_name(tool): value},
                                 raise_on_error=False)
    # Results are a LIST of content blocks -- text, images, audio, embedded resources.
    # We only asked for text tools, so we join the text ones.
    text = "\n".join(block.text for block in result.content
                     if getattr(block, "type", None) == "text")
    return f"tool error: {text}" if result.is_error else text


async def run_agent(task: str, mcp: Client, tools, tracer: Tracer) -> str:
    """The Ep2 ReAct loop, unchanged -- except tools now live in another process."""
    by_name = {t.name: t for t in tools}

    # The toolbox is DISCOVERED, not hard-coded. describe(tools) is the only thing telling
    # the model what exists -- and it came off the wire a moment ago.
    system = f"""You are a support agent. Answer the user's question using your tools.
Never guess a fact or a number that a tool can give you, and never do arithmetic in your
head -- every calculation goes through the calculator tool, however easy it looks.

Your tools:
{describe(tools)}

{REACT_FORMAT}"""

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]

    for _ in range(MAX_STEPS):
        t = time.perf_counter()
        # A blocking call inside an async function. Fine for one agent talking to one
        # server; the moment you fan out, this is the line you make async first.
        resp = client.chat.completions.create(
            model=MODEL, messages=messages, stop=["Observation:"], temperature=0)
        tracer.llm(resp, time.perf_counter() - t)
        text = resp.choices[0].message.content.strip()

        action = ACTION_RE.search(text)
        answer_at = text.find("Answer:")

        if answer_at != -1 and (action is None or answer_at < action.start()):
            print(_indent(text))
            return text[answer_at + len("Answer:"):].strip()

        if action is None:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user",
                             "content": "Observation: reply with an Action or an Answer."})
            continue

        # The Ep2 honesty rule, still load-bearing: cut the model off at its Action so it
        # cannot write its own Observation. It matters MORE now -- the result is coming from
        # someone else's server, and the model has no idea what it will say.
        turn = text[: action.end()]
        print(_indent(turn))
        messages.append({"role": "assistant", "content": turn})

        name, arg = action.group(1), action.group(2).strip()
        if name in by_name:
            # THIS is the only line in the loop that changed since Ep9: instead of
            # TOOLS[name](arg) we send a tools/call to a process we did not write.
            t = time.perf_counter()
            observation = await call_tool(mcp, by_name[name], arg)
            tracer.tool(name, arg, observation, time.perf_counter() - t)
        else:
            observation = f"unknown tool: {name}"

        print(f"    Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "ran out of steps"


# ============================ THE DEMO =======================================
# Needs both tools, twice each: look the rate limit up, subtract; look the price up,
# multiply. The numbers are stable (60/min, $49/seat) so the answer is always 90 rejected
# and $196 a month -- and not one of those facts is in this file.
TASK = ("A customer on the Nimbus Pro plan sent 150 API requests in a single minute, and "
        "their team of 4 wants to know the monthly bill. Reply in exactly two sentences: "
        "how many of the 150 requests were rejected, and what the team pays per month.")


async def main() -> None:
    print(f"🧑 {TASK}\n")

    # `async with` launches the server, negotiates the protocol era and shuts the child
    # process down on the way out -- the four things our own client would have had to get
    # right. mode="auto" is the default and is what does the negotiating.
    async with Client(SERVER, mode="auto") as mcp:
        info = mcp.server_info
        print(f"🔌 {Path(SERVER).name} — protocol {mcp.protocol_version}, "
              f"{getattr(info, 'name', '?')} v{getattr(info, 'version', '?')}\n")

        print("🧰 asking the server what it can do (tools/list):")
        tools = await mcp.list_tools()
        for tool in tools:
            # Nobody wrote this schema by hand. FastMCP generated it from the function's
            # signature and docstring on the other side of the pipe, and this is the
            # catalogue the MODEL reads to decide what to call.
            print(f"    {tool.name}")
            print(textwrap.fill(" ".join((tool.description or "").split()),
                                width=96, initial_indent=" " * 8,
                                subsequent_indent=" " * 8))
            print(f"        schema: {json.dumps(tool.input_schema)}")
        print()

        print("🤖 same ReAct loop as Ep2 — every Action is now a tools/call:")
        tracer = Tracer(task="mcp: nimbus rejected requests + monthly bill", model=MODEL)
        answer = await run_agent(TASK, mcp, tools, tracer)

    tracer.finish(answer)
    print("=== FINAL ANSWER ===")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
