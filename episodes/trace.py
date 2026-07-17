"""A tiny agent tracer -- see what your agent actually DID (and what it cost).

An agent is a multi-step loop: think, call a tool, think again, answer. From the
outside you only see the final answer -- the steps in between are invisible. This
makes the loop VISIBLE. Wrap a run in a Tracer, tell it about each LLM call and each
tool call as they happen, and at the end it prints the whole run as a tree and saves
it to a SQLite file you can look at later.

    run: <the task>
    ├─ 1 llm   210+45 tok  $0.0002  480ms
    ├─ 2 tool  recall(which day...) -> every Monday  12ms
    └─ answer: every Monday   total $0.0002  1130ms

No framework, no dependencies -- just the standard library. It grows across the
season (nesting for multi-agent, cost caps for guardrails), but this is the seed:
record steps, print the tree, persist to disk.
"""

import sqlite3
import time
from pathlib import Path

# The run log lives next to the episodes, in one SQLite file (gitignored).
DB = Path(__file__).with_name("traces.db")

# $ per 1,000,000 tokens -- illustrative Gemini 2.5 Flash pricing (input, output).
# The point isn't exact dollars; it's that a multi-step agent has a running BILL you
# can watch add up, step by step -- the per-run analog of the gateway's per-call cost.
PRICES = {"gemini-2.5-flash": (0.30, 2.50)}


class Tracer:
    """Wrap ONE agent run. Call .llm()/.tool() as steps happen, then .finish()."""

    def __init__(self, task: str, model: str):
        self.task, self.model = task, model
        self.steps: list[dict] = []          # flat for now; nesting arrives with multi-agent
        self._t0 = time.perf_counter()

    def llm(self, resp, dt: float) -> None:
        """Record one LLM call: token usage (from the response) + latency + cost."""
        u = resp.usage
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        cost = u.prompt_tokens / 1e6 * pin + u.completion_tokens / 1e6 * pout
        self.steps.append({
            "kind": "llm", "name": self.model, "arg": "",
            "in_tok": u.prompt_tokens, "out_tok": u.completion_tokens,
            "cost": cost, "ms": round(dt * 1000), "result": "",
        })

    def tool(self, name: str, arg: str, result: str, dt: float) -> None:
        """Record one tool call: which tool, its argument, a snippet of the result."""
        self.steps.append({
            "kind": "tool", "name": name, "arg": arg,
            "in_tok": 0, "out_tok": 0, "cost": 0.0,
            "ms": round(dt * 1000), "result": result.replace("\n", " ")[:120],
        })

    def finish(self, answer: str) -> None:
        """End the run: print the tree and save it to disk."""
        total = sum(s["cost"] for s in self.steps)
        ms = round((time.perf_counter() - self._t0) * 1000)
        self._print_tree(answer, total, ms)
        self._save(answer, total, ms)

    # --- output ---------------------------------------------------------------
    def _print_tree(self, answer: str, total: float, ms: int) -> None:
        print(f"\n run: {self.task}")
        for i, s in enumerate(self.steps, 1):
            if s["kind"] == "llm":
                print(f" ├─ {i} llm   {s['in_tok']}+{s['out_tok']} tok  "
                      f"${s['cost']:.4f}  {s['ms']}ms")
            else:
                print(f" ├─ {i} tool  {s['name']}({s['arg']}) -> {s['result']}  {s['ms']}ms")
        print(f" └─ answer: {answer}   total ${total:.4f}  {ms}ms\n")

    def _save(self, answer: str, total: float, ms: int) -> None:
        with sqlite3.connect(DB) as db:
            db.execute("CREATE TABLE IF NOT EXISTS runs("
                       "id INTEGER PRIMARY KEY, task TEXT, answer TEXT, cost REAL, ms INTEGER)")
            db.execute("CREATE TABLE IF NOT EXISTS steps("
                       "run_id INTEGER, seq INTEGER, kind TEXT, name TEXT, arg TEXT, "
                       "in_tok INTEGER, out_tok INTEGER, cost REAL, ms INTEGER, result TEXT)")
            cur = db.execute("INSERT INTO runs(task, answer, cost, ms) VALUES (?,?,?,?)",
                             (self.task, answer, total, ms))
            rid = cur.lastrowid
            db.executemany(
                "INSERT INTO steps VALUES (?,?,?,?,?,?,?,?,?,?)",
                [(rid, i, s["kind"], s["name"], s["arg"], s["in_tok"],
                  s["out_tok"], s["cost"], s["ms"], s["result"])
                 for i, s in enumerate(self.steps, 1)])
