# AI Agents from Scratch

Companion code for the **"AI Agents from Scratch"** YouTube series (Season 2) —
faceless, from scratch, no framework, no cloud bill. We build agents by hand in
plain Python with the OpenAI SDK pointed at a **free Gemini key**.

> Sequel to [AI Engineering from Scratch](https://github.com/vahid8/ai-engineering-series).
> Watch on YouTube: [@softwareengineerblog8](https://www.youtube.com/@softwareengineerblog8)

## What's an agent? (the whole series in one line)

> An agent = **an LLM + tools + a loop + a stop condition.**

That's it. Each episode adds one idea on top of that loop: ReAct, web search,
memory, planning, multi-agent, reflection, MCP, guardrails — built by hand so you
can see exactly what's happening, and the finale hands the same job to a real
framework so you can see exactly which of your own files each of its lines replaces.

## Setup (once)

```bash
# 1. install uv  ->  https://docs.astral.sh/uv/
# 2. install deps
uv sync
# 3. add your free Gemini key (https://aistudio.google.com/apikey)
cp .env.template .env      # then edit .env and paste your key
```

## Run an episode

```bash
uv run --env-file .env python episodes/01_agent.py
```

One exception: **`episodes/12_framework.py` runs in its own environment.**
`openai-agents` pins `mcp>=1.19,<2` and FastMCP 4 (Ep10, Ep12) needs `mcp>=2`, so the two
cannot be installed side by side. The file carries its own dependencies in a PEP 723
header, and `--no-project` tells uv to use them instead of this project's:

```bash
uv run --env-file .env --no-project episodes/12_framework.py
```

## Episodes

| Ep | Topic |
|----|-------|
| 01 | What *is* an agent? — the minimal agent loop |
| 02 | ReAct — make the agent think out loud (Thought / Action / Observation) |
| 03 | Web-search agent — give it a real tool to research the live web (DuckDuckGo, no key) |
| 04 | A toolbox — three tools (calculator, web search, RAG retrieval) and the agent *chooses* which to use |
| 05 | Memory — short-term (the conversation) + long-term (a vector store on disk) so the agent stops forgetting |
| 06 | Planning — plan-and-execute vs ReAct, plus a tiny tracer that shows every step, token, and cost |
| 07 | Memory pt.2 — markdown files + a text index (FTS5) instead of vectors; retrieve by keyword, not meaning |
| 08 | Multi-agent — an orchestrator hands work to narrow workers; a hand-off is a recursive tool call, and the tracer nests |
| 09 | Reflection — the agent critiques its OWN draft with tools, then revises until the critic passes (Reflexion, by hand) |
| 10 | MCP with FastMCP — the agent uses tools it didn't write, discovered at runtime, on the new **stateless** `2026-07-28` protocol revision (no more `initialize` handshake) |
| 11 | Guardrails, cost caps & evals — a step cap, a budget and a tool allowlist enforced *in the tracer*, plus the evals that prove you didn't break anything |
| 12 | **The finale** — the whole season in one agent (loop + tracer + critic + MCP + policy), the day the server ships a `refund` tool nobody reviewed, and the same agent in a real framework |

## Security

This repo is **public** and shown on camera. Keys live only in `~/.secrets/`,
`.env` holds a path (never a raw key), and a pre-commit hook scans for leaks.
See [SECURITY.md](SECURITY.md).
