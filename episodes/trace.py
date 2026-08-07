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
season, and Ep11 is the last growth: the tracer already knew every step, every
dollar and every tool name, so it is the natural place to ENFORCE limits on them.
Recording something and refusing it are the same knowledge, asked at different times.
"""

import sqlite3
import time
from pathlib import Path


class GuardrailTripped(RuntimeError):
    """A run tried to do something its own limits forbid (Ep11).

    Raised by Tracer.guard() BEFORE the thing happens, never after. What the caller
    does with it is a policy decision, not a tracing one: catch it around a tool call
    and the run is STEERED (the refusal becomes an Observation and the agent carries
    on); let it propagate out of the loop and the run is STOPPED.
    """


# The run log lives next to the episodes, in one SQLite file (gitignored).
DB = Path(__file__).with_name("traces.db")

# $ per 1,000,000 tokens -- illustrative Gemini 2.5 Flash pricing (input, output).
# The point isn't exact dollars; it's that a multi-step agent has a running BILL you
# can watch add up, step by step -- the per-run analog of the gateway's per-call cost.
PRICES = {"gemini-2.5-flash": (0.30, 2.50)}


def _line(text: str, width: int = 72) -> str:
    """Squash a value onto one short line so the tree stays a tree."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


class Tracer:
    """Wrap ONE agent run. Call .llm()/.tool() as steps happen, then .finish().

    The three limits are Ep11's addition and every one of them is OFF by default, so
    Ep6-Ep10 behave exactly as they always did:

        max_steps      how many steps this run may take at all
        max_cost       how many dollars the whole run tree may spend
        allowed_tools  the only tool names that may ever execute
    """

    def __init__(self, task: str, model: str, parent: "Tracer | None" = None,
                 max_steps: int | None = None, max_cost: float | None = None,
                 allowed_tools: set[str] | None = None):
        self.task, self.model = task, model
        self.steps: list[dict] = []
        self._parent = parent                # set when this run is a sub-agent's run
        self.max_steps, self.max_cost = max_steps, max_cost
        self.allowed_tools = allowed_tools
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

    # --- guardrails (Ep11) ----------------------------------------------------
    def spent(self) -> float:
        """What this run has cost so far, sub-agents included -- even mid-flight.

        A delegation step's own cost is filled in when the child finishes, so read the
        child's live total instead of the parent's placeholder; adding both would count
        every sub-run twice.
        """
        return sum(s["sub"].spent() if "sub" in s else s["cost"] for s in self.steps)

    def _root(self) -> "Tracer":
        """The top of the run tree. Budgets belong to it, not to each agent in it."""
        run = self
        while run._parent is not None:
            run = run._parent
        return run

    def guard(self, kind: str, name: str = "") -> None:
        """Ask -- BEFORE doing the thing -- whether this run is still allowed to.

        Which rules apply depends on what is about to happen, and that split is the whole
        design. Before a TOOL there is one question -- are you allowed to run this? Before
        a MODEL CALL there are two -- have you gone on too long, and have you spent too
        much? Which is why the caller can catch a refused tool and keep going, while a
        blown budget takes the run down: they arrive from different places.

        The budget is checked against the ROOT of the tree. Give each sub-agent its own
        copy of the limit and three workers will happily spend it three times over.
        """
        if kind == "tool":
            if self.allowed_tools is not None and name not in self.allowed_tools:
                self._trip(f"tool {name!r} is not on the allowlist")
            return
        if self.max_steps is not None and len(self.steps) >= self.max_steps:
            self._trip(f"step limit reached ({self.max_steps} steps)")
        if self.max_cost is not None and (spent := self._root().spent()) >= self.max_cost:
            self._trip(f"budget spent (${spent:.5f} of ${self.max_cost:.5f})")

    def _trip(self, why: str) -> None:
        """Record the refusal in the trace, then raise. The tree must show WHY it ended."""
        self.steps.append({
            "kind": "guard", "name": "guard", "arg": why,
            "in_tok": 0, "out_tok": 0, "cost": 0.0,
            "ms": round((time.perf_counter() - self._t0) * 1000), "result": "",
        })
        raise GuardrailTripped(why)

    def child(self, name: str, task: str) -> "Tracer":
        """Hand work to a SUB-AGENT (Ep8): its own run, nested inside this one.

        The delegation shows up as one `agent` step here -- its cost and duration are
        the whole sub-run's -- and the sub-run's own steps print indented underneath.
        Limits are INHERITED: a worker cannot escape its orchestrator's policy by being
        a different Tracer.
        """
        sub = Tracer(task, self.model, parent=self, max_steps=self.max_steps,
                     max_cost=self.max_cost, allowed_tools=self.allowed_tools)
        self.steps.append({
            "kind": "agent", "name": name, "arg": task,
            "in_tok": 0, "out_tok": 0, "cost": 0.0, "ms": 0, "result": "", "sub": sub,
        })
        return sub

    def finish(self, answer: str, show: bool = True) -> None:
        """End the run: print the tree and save it to disk.

        A sub-agent's run doesn't print on its own -- it reports its answer, cost and
        duration up to the delegation step, and the top-level run prints everything.
        `show=False` still records the run; it just doesn't print the tree, which is what
        you want when a suite of evals (Ep11) runs dozens of them in a row.
        """
        total = sum(s["cost"] for s in self.steps)
        ms = round((time.perf_counter() - self._t0) * 1000)
        if self._parent is not None:
            for s in self._parent.steps:                  # fill in OUR step in the parent
                if s.get("sub") is self:
                    s["result"] = answer.replace("\n", " ")[:120]
                    s["cost"], s["ms"] = total, ms
        if self._parent is not None or not show:
            self._save(answer, total, ms)                 # still a run of its own on disk
            return
        print()
        self._print_tree(answer, total, ms)
        print()
        self._save(answer, total, ms)

    # --- output ---------------------------------------------------------------
    def _print_tree(self, answer: str, total: float, ms: int, indent: str = "") -> None:
        print(f"{indent} run: {_line(self.task)}")
        for i, s in enumerate(self.steps, 1):
            if s["kind"] == "llm":
                print(f"{indent} ├─ {i} llm   {s['in_tok']}+{s['out_tok']} tok  "
                      f"${s['cost']:.4f}  {s['ms']}ms")
            elif s["kind"] == "agent":
                print(f"{indent} ├─ {i} agent {s['name']}({_line(s['arg'])})  "
                      f"${s['cost']:.4f}  {s['ms']}ms")
                s["sub"]._print_tree(_line(s["result"]), s["cost"], s["ms"], indent + " │ ")
            elif s["kind"] == "guard":
                print(f"{indent} ├─ {i} guard ✋ {_line(s['arg'])}")
            else:
                print(f"{indent} ├─ {i} tool  {s['name']}({_line(s['arg'])}) -> "
                      f"{_line(s['result'])}  {s['ms']}ms")
        print(f"{indent} └─ answer: {_line(answer)}   total ${total:.4f}  {ms}ms")

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
