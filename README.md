# AI Agents from Scratch

Companion code for the **"AI Agents from Scratch"** YouTube series (Season 2) —
faceless, from scratch, no framework, no cloud bill. We build agents by hand in
plain Python with the OpenAI SDK pointed at a **free Gemini key**.

> Sequel to [AI Engineering from Scratch](https://github.com/vahid8/ai-engineering-series).
> Watch on YouTube: [@softwareengineerblog8](https://www.youtube.com/@softwareengineerblog8)

## What's an agent? (the whole series in one line)

> An agent = **an LLM + tools + a loop + a stop condition.**

That's it. Each episode adds one idea on top of that loop: ReAct, web search,
memory, planning, multi-agent, reflection, guardrails — built by hand so you can
see exactly what's happening, then one episode shows a framework as "the grown-up
version."

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

## Episodes

| Ep | Topic |
|----|-------|
| 01 | What *is* an agent? — the minimal agent loop |
| 02 | ReAct — make the agent think out loud (Thought / Action / Observation) |
| 03 | Web-search agent — give it a real tool to research the live web (DuckDuckGo, no key) |

## Security

This repo is **public** and shown on camera. Keys live only in `~/.secrets/`,
`.env` holds a path (never a raw key), and a pre-commit hook scans for leaks.
See [SECURITY.md](SECURITY.md).
