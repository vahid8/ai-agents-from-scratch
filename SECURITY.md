# Security & Secret Hygiene

This is a **public** repository. Follow these rules so no key is ever leaked.

## Golden rules
1. **Real keys live ONLY outside the repo** — in `~/.secrets/` (e.g. `~/.secrets/gemini.key`, `chmod 600`).
2. **`.env` is gitignored** and only ever contains a *path* (`GEMINI_API_KEY_FILE=~/.secrets/gemini.key`), never a raw key. Safe to screen-share.
3. **`.env.template`** is the only env file committed — placeholders only, no real values.
4. **Never paste a key into a tracked file** (`.py`, `.md`, notebooks, etc.).
5. **Git history is permanent.** If a key is ever committed, removing it later is NOT enough — you must **rotate (regenerate) the key immediately**.

## Before every commit / before going on camera
```bash
# 1. Confirm .env is ignored
git check-ignore .env            # must print: .env

# 2. Scan staged + working files for key-shaped strings
./scripts/check-secrets.sh
```
A local `pre-commit` hook runs the scan automatically and **blocks the commit** if a key is found.

## If a key leaks anyway
1. **Rotate it now** (Google AI Studio → delete key → create new).
2. Update `~/.secrets/gemini.key` with the new key.
3. The leaked one is now useless — that's why rotation is the fix, not history rewriting.
