"""Episode 10 (part 1 of 2) — an MCP SERVER with FastMCP.

This is the far end of the wire: the "somebody else's tools" half. And it is the shortest
file in the season, which is the whole point of using the framework.

MCP -- the Model Context Protocol -- is how an agent uses tools that live in another
process, written by someone else, discovered at RUNTIME instead of imported. Under the
branding it is JSON-RPC 2.0 messages, one per line, over this program's stdin and stdout.
FastMCP writes every one of those lines for us. What is left is the part that is actually
ours: two Python functions.

🔴 WHY FASTMCP 4, AND WHY THE BETA. MCP changed on 2026-07-28: that revision DELETED the
`initialize` handshake. The protocol is now **stateless** -- there is no session, and a
server may infer nothing from earlier requests on the same pipe. Instead every request
carries its own protocol version and capabilities, and a mandatory `server/discover` call
replaces the handshake for clients that want to look before they leap. FastMCP 4 speaks
that revision; FastMCP 3 is still handshake-era. So this episode pins the 4.0 beta -- see
`[tool.uv] prerelease = "if-necessary-or-explicit"` in pyproject.toml.

Statelessness is not a detail. It is why the same server can sit behind an ordinary load
balancer with a dozen replicas: any replica can answer any request, because the request
brings everything needed to serve it. Nothing about the two functions below changes.

WHAT THE DECORATOR ACTUALLY DOES. `@mcp.tool` reads the function's signature and its
docstring and generates the JSON Schema that goes out over `tools/list`. The type hints
become the schema's types; the docstring becomes the description the MODEL reads when it
decides whether to call this tool. That is the trade with a framework: you stop writing
the catalogue by hand, so the docstring stops being documentation and becomes prompt.
Write it for the model. The client prints the generated schema when it connects -- watch
it and you will see this docstring come back over the wire.

Run it yourself if you like -- it will sit there waiting for JSON on stdin, because that
is all an MCP server is:
    uv run --env-file .env python episodes/10_mcp_server.py
Normally you never do that: 10_mcp.py launches it as a subprocess.
"""

import logging
import re

from fastmcp import FastMCP

# FastMCP logs to stderr at INFO, which the spec explicitly allows (stderr is the server's
# to use; only stdout is sacred). We quiet it so the terminal shows the AGENT and not the
# framework -- if a server of yours ever goes silent, turn this back up first.
logging.getLogger("fastmcp").setLevel(logging.ERROR)

# `instructions` is server-level guidance that a client can read straight off
# server/discover, before it has listed a single tool.
mcp = FastMCP(
    "nimbus-handbook",
    version="2.0.0",
    instructions="Answer support questions about the Nimbus Pro plan. Use lookup for "
                 "facts from the handbook and calculator for arithmetic.",
)


# ============================ THE TOOLS ======================================
# Ground truth about a product the model has never heard of. Unchanged since Ep8 -- on
# purpose. Nothing about the tools is new today; only where they live. In Ep9 the agent
# imported these functions. Today they run in a process the agent knows nothing about
# until it asks.
HANDBOOK = {
    "rate limits": "Nimbus Pro allows 60 API requests per minute. Anything above that "
                   "is rejected with HTTP 429 until the next minute starts.",
    "billing": "Nimbus Pro is billed monthly at $49 per seat. Overages are not billed; "
               "excess requests are rejected instead.",
    "support": "Nimbus Pro includes email support with a 24-hour response target.",
}


@mcp.tool
def lookup(query: str) -> str:
    """Look up a fact about the Nimbus Pro plan (rate limits, billing, support) in the
    official handbook. Use this for any factual claim about the plan."""
    words = set(re.findall(r"[a-z0-9]+", query.lower()))

    def score(topic: str) -> int:
        text = set(re.findall(r"[a-z0-9]+", f"{topic} {HANDBOOK[topic]}".lower()))
        return len(words & text)

    best = max(HANDBOOK, key=score)
    if not score(best):
        return "nothing in the handbook matched"
    return f"[{best}] {HANDBOOK[best]}"


_ALLOWED = set("0123456789+-*/(). ")


@mcp.tool
def calculator(expression: str) -> str:
    """Evaluate a plain arithmetic expression, e.g. 150-60. Digits and + - * / ( ) only."""
    # Charset-locked since Ep1: only digits, operators and brackets ever reach eval. With
    # no letters, quotes, commas or underscores there is no name to look up and no
    # attribute to reach through, so there is nothing to call.
    #
    # Ep10 tightens it by two lines, and the reason is this episode's whole point: the
    # caller is now an agent we did not write, on a protocol with no session, so this
    # function cannot lean on "well, they said hello first". `**` is spelled with two
    # allowed characters, and `9**9**9**9` is a hang, not an answer -- a denial of
    # service that passes a charset check. Length is capped for the same reason.
    if not expression or set(expression) - _ALLOWED:
        return "error: numbers and + - * / ( ) only"
    if "**" in expression or len(expression) > 100:
        return "error: no exponentiation, 100 characters max"
    try:
        return str(eval(expression))  # noqa: S307 -- input restricted to the charset above
    except Exception as e:
        return f"error: {e}"


if __name__ == "__main__":
    # Defaults to the stdio transport: read stdin, write stdout, and NOTHING else may
    # touch stdout or the protocol stream is corrupt. FastMCP's startup banner goes to
    # stderr, which the spec explicitly allows -- we switch it off to keep the terminal
    # readable, not because it would break anything.
    mcp.run(show_banner=False)
