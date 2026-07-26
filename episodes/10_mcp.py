"""Episode 10 — MCP from scratch: let the agent use tools it did NOT write.

Every tool we have built in nine episodes had one thing in common: we wrote it, in the
same file as the agent, and wired it in by hand with a Python dict. The agent could only
ever do what we had personally coded into it.

MCP -- the Model Context Protocol -- is how an agent uses tools that live somewhere else:
another process, written by someone else, in a language you don't care about, that you
discover at RUNTIME instead of importing. It is the USB-C port for tools. And underneath
the branding it is disarmingly simple:

    JSON-RPC 2.0 messages, one per line, over the server's stdin and stdout.

That is it. So we are not going to install an MCP SDK -- we are going to speak the
protocol by hand, and you will see every byte go past. Three methods do everything:

    initialize   -- agree on a protocol version, exchange capabilities
    tools/list   -- ask the server what it can do; it answers with JSON Schemas
    tools/call   -- run one of those tools and get the result back

The payoff is in tools/list. Look at how SYSTEM is built at the bottom of this file: the
agent's toolbox is assembled at runtime out of whatever the server advertises. Grep this
file for a tool name and you will only find it in comments -- no line of CODE here knows
what the tools are called. Point the client at a different server and the agent can do
different things, without a single line of this file changing.

And the loop itself? Untouched. It is the same ReAct loop from Ep2, honesty guard and
all. The ONLY difference is that dispatching an Action now writes a line to a pipe
instead of calling a Python function.

Run it (free Gemini key -- see README):
    uv run --env-file .env python episodes/10_mcp.py
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

from trace import Tracer  # the run tracer from Ep6 -- unchanged


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

# The newest MCP spec revision at the time of writing. Versions are dates, and the
# client is supposed to ask for the latest one it knows; the server answers with what
# it actually speaks. Flip this to a nonsense date to watch negotiation fail.
PROTOCOL_VERSION = "2025-11-25"

# Print every JSON-RPC line in both directions. Normally you would never do this -- but
# the whole point of this episode is that MCP is just text on a pipe, so let's watch it.
# Raise WIRE_WIDTH if you want to read a whole message; tools/list is the long one.
SHOW_WIRE = True
WIRE_WIDTH = 100


# ========================= THE MCP CLIENT ====================================
class MCPClient:
    """Speaks MCP to one server over its stdin/stdout. About 40 lines of real work.

    The server is a child process. We write a JSON request on its stdin, it writes a
    JSON response on its stdout, and both sides use newlines as the message boundary.
    """

    def __init__(self, command: list[str]):
        # stderr is inherited on purpose: if the server crashes we want its traceback
        # on our terminal, not swallowed. bufsize=1 = line buffered, which is exactly
        # the granularity the protocol works at.
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self._next_id = 0

    # --- the wire ------------------------------------------------------------
    def _send(self, message: dict) -> None:
        """Write ONE message as ONE line, and flush so it actually leaves."""
        line = json.dumps(message)
        if SHOW_WIRE:
            print(f"    \033[36m→ {line[:WIRE_WIDTH]}\033[0m")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _receive(self) -> dict:
        """Read ONE line and parse it as a message."""
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection")
        if SHOW_WIRE:
            print(f"    \033[36m← {line.strip()[:WIRE_WIDTH]}\033[0m")
        return json.loads(line)

    def _request(self, method: str, params: dict | None = None) -> dict:
        """Send a request and wait for the response with the SAME id.

        The id is what makes JSON-RPC work: replies can come back in any order, and a
        server may push notifications at us while we wait, so we match on the id and
        skip anything that isn't ours.
        """
        self._next_id += 1
        request_id = self._next_id
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method,
                    "params": params or {}})
        while True:
            message = self._receive()
            if message.get("id") != request_id:
                continue                      # a notification, or someone else's reply
            if "error" in message:
                raise RuntimeError(f"MCP error: {message['error']}")
            return message.get("result", {})

    def _notify(self, method: str, params: dict | None = None) -> None:
        """Send a notification: no id, no reply, no waiting."""
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # --- the three methods ---------------------------------------------------
    def initialize(self) -> dict:
        """The handshake. Must happen before anything else."""
        result = self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},                       # we're a client with no extras
            "clientInfo": {"name": "agents-from-scratch", "version": "1.0.0"},
        })
        # Then we tell the server we're ready. This one is a NOTIFICATION -- we do not
        # wait for a reply, and the server must not send one.
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        """Ask the server what it can do. This is discovery -- nothing is hard-coded."""
        return self._request("tools/list").get("tools", [])

    def call_tool(self, name: str, arguments: dict) -> str:
        """Run one tool and return its text result."""
        result = self._request("tools/call", {"name": name, "arguments": arguments})
        # Results are a list of content blocks (text, images, audio, embedded
        # resources...). We only asked for text tools, so we join the text ones.
        text = "\n".join(block.get("text", "") for block in result.get("content", [])
                         if block.get("type") == "text")
        if result.get("isError"):
            return f"tool error: {text}"
        return text

    def close(self) -> None:
        """Shut the server down the way the spec asks: close stdin, then wait."""
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


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


def argument_name(tool: dict) -> str:
    """Which argument does this tool want? Read it off the discovered schema.

    Our ReAct format sends ONE value in brackets -- calculator[150-60] -- but MCP tools
    take a named JSON object. So we look at the tool's own inputSchema and use its first
    required property as the name. This is the schema doing real work, not decoration:
    the server tells us its argument is called "expression", so that is what we send.

    (A production client skips this entirely: it hands the whole schema to the model as
    a function-calling definition and gets structured arguments back. We're keeping the
    Ep2 format so you can see the protocol without a second new thing in the way.)
    """
    schema = tool.get("inputSchema") or {}
    names = schema.get("required") or list(schema.get("properties") or {})
    return names[0] if names else "input"


def describe(tools: list[dict]) -> str:
    """Render the discovered tools into the prompt the model will read."""
    return "\n".join(f"- {t['name']}[{argument_name(t)}]   {t.get('description', '')}"
                     for t in tools)


def run_agent(task: str, mcp: MCPClient, tools: list[dict], tracer: Tracer) -> str:
    """The Ep2 ReAct loop, unchanged -- except tools now live on the other end of a pipe."""
    by_name = {t["name"]: t for t in tools}

    # The toolbox is DISCOVERED, not hard-coded. `describe(tools)` is the only thing
    # telling the model what exists -- and it came off the wire a moment ago.
    system = f"""You are a support agent. Answer the user's question using your tools.
Never guess a fact or a number that a tool can give you.

Your tools:
{describe(tools)}

{REACT_FORMAT}"""

    messages = [{"role": "system", "content": system},
                {"role": "user", "content": task}]

    for _ in range(MAX_STEPS):
        t = time.perf_counter()
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

        # The Ep2 honesty rule, still load-bearing: cut the model off at its Action so
        # it cannot write its own Observation. It matters even more now -- the result
        # is coming from someone else's server, and the model has no idea what it says.
        turn = text[: action.end()]
        print(_indent(turn))
        messages.append({"role": "assistant", "content": turn})

        name, arg = action.group(1), action.group(2).strip()
        if name in by_name:
            # THIS is the only line in the whole loop that changed since Ep9: instead of
            # TOOLS[name](arg) we send a tools/call down the pipe.
            t = time.perf_counter()
            observation = mcp.call_tool(name, {argument_name(by_name[name]): arg})
            tracer.tool(name, arg, observation, time.perf_counter() - t)
        else:
            observation = f"unknown tool: {name}"

        print(f"    Observation: {observation}\n")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "ran out of steps"


# ============================ THE DEMO =======================================
# Needs both tools, twice each: look the rate limit up, subtract; look the price up,
# multiply. The numbers are stable (60/min, $49/seat) so the answer is always 90
# rejected and $196 a month -- but not one of those facts is in this file.
TASK = ("A customer on the Nimbus Pro plan sent 150 API requests in a single minute, and "
        "their team of 4 wants to know the monthly bill. Reply in exactly two sentences: "
        "how many of the 150 requests were rejected, and what the team pays per month.")

SERVER_COMMAND = [sys.executable, str(Path(__file__).with_name("10_mcp_server.py"))]

if __name__ == "__main__":
    print(f"🧑 {TASK}\n")

    print("🔌 launching the MCP server and shaking hands:")
    mcp = MCPClient(SERVER_COMMAND)
    info = mcp.initialize()
    server = info.get("serverInfo", {})
    print(f"    connected to {server.get('name')} v{server.get('version')} "
          f"(protocol {info.get('protocolVersion')})\n")

    print("🧰 asking the server what it can do (tools/list):")
    tools = mcp.list_tools()
    for tool in tools:
        print(f"    {tool['name']}({argument_name(tool)}) — {tool.get('description', '')}")
    print()

    print("🤖 same ReAct loop as Ep2 — every Action is now a tools/call:")
    tracer = Tracer(task="mcp: nimbus rejected requests + monthly bill", model=MODEL)
    try:
        answer = run_agent(TASK, mcp, tools, tracer)
    finally:
        mcp.close()   # always shut the child process down, even if the loop blew up

    tracer.finish(answer)
    print("=== FINAL ANSWER ===")
    print(answer)
