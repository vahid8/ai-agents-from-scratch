"""Episode 10 (part 1 of 2) — an MCP SERVER, by hand.

This is the other end of the wire. It is a normal Python program with one unusual
habit: it reads JSON-RPC messages from **stdin**, one per line, and writes JSON-RPC
messages back to **stdout**, one per line. That is the whole of MCP's stdio
transport -- no HTTP, no sockets, no SDK, no dependencies.

It answers exactly three methods, which is all a tool server needs:

    initialize   -- "hello, here is my protocol version and what I can do"
    tools/list   -- "here are my tools, with a JSON Schema for each one"
    tools/call   -- "run this tool with these arguments, here is the result"

The tools themselves are the SAME boring pair we have used since Ep8: a three-entry
handbook about a fictional product, and the charset-locked calculator. That is on
purpose. Nothing about the tools is new -- only where they live. In Ep9 the agent
imported them from the same file. Here they live in a separate process that the
agent knows nothing about until it asks.

Two rules that will bite you if you break them:
  1. NOTHING but MCP messages may go to stdout. One stray print() and you have
     corrupted the protocol stream. Debug output goes to stderr.
  2. Every message is ONE line -- messages must not contain embedded newlines.
     json.dumps() escapes newlines inside strings for us, so this comes free.

You never run this file yourself -- 10_mcp.py launches it as a subprocess.
"""

import json
import re
import sys

# The version of the MCP spec this server speaks. The client sends the version it
# wants in `initialize`; we answer with the one we actually support, and the two
# sides agree (or the client walks away).
PROTOCOL_VERSION = "2025-11-25"


# ============================ THE TOOLS ======================================
# Ground truth about a product the model has never heard of -- unchanged from Ep9.
HANDBOOK = {
    "rate limits": "Nimbus Pro allows 60 API requests per minute. Anything above that "
                   "is rejected with HTTP 429 until the next minute starts.",
    "billing": "Nimbus Pro is billed monthly at $49 per seat. Overages are not billed; "
               "excess requests are rejected instead.",
    "support": "Nimbus Pro includes email support with a 24-hour response target.",
}


def lookup(query: str) -> str:
    """Look a topic up in the Nimbus handbook by keyword overlap."""
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

    Charset-locked since Ep1: only digits, operators and brackets ever reach eval.
    Worth re-reading now that the caller is a REMOTE agent we did not write -- a
    server is the last thing standing between a tool and whatever sent the request.
    """
    if not expression or set(expression) - _ALLOWED:
        return "error: numbers and + - * / ( ) only"
    try:
        return str(eval(expression))  # noqa: S307 -- input restricted to the charset above
    except Exception as e:
        return f"error: {e}"


# ========================= THE TOOL CATALOGUE ================================
# This is what `tools/list` hands back, and it is the heart of MCP: a machine-readable
# description of each tool -- its name, what it is for, and a JSON Schema for its
# arguments. The client has never seen this file, so this catalogue is the ONLY thing
# telling it what exists. Write the descriptions for the model that will read them.
TOOLS = [
    {
        "name": "lookup",
        "title": "Nimbus handbook lookup",
        "description": "Look up a fact about the Nimbus Pro plan (rate limits, "
                       "billing, support) in the official handbook.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The topic to look up."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "title": "Arithmetic calculator",
        "description": "Evaluate a plain arithmetic expression, e.g. 150-60.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string",
                               "description": "Arithmetic using digits and + - * / ( )."}
            },
            "required": ["expression"],
        },
    },
]

IMPL = {"lookup": lookup, "calculator": calculator}


# ========================== THE THREE METHODS ================================
def handle(msg: dict) -> dict:
    """Turn one JSON-RPC request into one JSON-RPC `result` payload."""
    method = msg.get("method")

    if method == "initialize":
        # The handshake. We announce the version we speak and which capabilities we
        # have -- we only do tools, so that is the only capability we declare.
        return {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "nimbus-handbook", "version": "1.0.0"},
        }

    if method == "tools/list":
        return {"tools": TOOLS}

    if method == "tools/call":
        params = msg.get("params") or {}
        name = params.get("name")
        arguments = params.get("arguments") or {}
        fn = IMPL.get(name)
        if fn is None:
            raise LookupError(f"Unknown tool: {name}")
        try:
            text = fn(**arguments)
            is_error = False
        except Exception as e:
            # A tool that BLEW UP is not a protocol failure -- the request was valid.
            # MCP reports it in the result with isError, so the model can read what
            # went wrong and try again. Protocol errors (below) are for malformed
            # requests, which a model has no hope of fixing.
            text, is_error = f"error: {e}", True
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    raise NotImplementedError(f"Unknown method: {method}")


def main() -> None:
    """Read one JSON message per line forever; answer the ones that want answering."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)

        # No "id" means this is a NOTIFICATION (e.g. notifications/initialized).
        # Notifications are fire-and-forget: the sender is not waiting, so we must
        # not reply -- an unexpected response would desynchronise the stream.
        if "id" not in msg:
            continue

        try:
            response = {"jsonrpc": "2.0", "id": msg["id"], "result": handle(msg)}
        except LookupError as e:
            response = {"jsonrpc": "2.0", "id": msg["id"],
                        "error": {"code": -32602, "message": str(e)}}
        except Exception as e:
            response = {"jsonrpc": "2.0", "id": msg["id"],
                        "error": {"code": -32601, "message": str(e)}}

        # One line, then FLUSH. Without the flush the reply sits in a buffer and the
        # client waits for a message that has technically already been "sent".
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
