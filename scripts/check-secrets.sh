#!/usr/bin/env bash
# Block leaks: fail if .env is tracked or any key-shaped string appears in committed files.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

fail=0

# 1. .env must never be tracked
if git ls-files --error-unmatch .env >/dev/null 2>&1; then
  echo "❌ .env is TRACKED by git — remove it: git rm --cached .env"
  fail=1
fi

# 2. scan tracked + staged + untracked-but-not-ignored files for key patterns
PATTERN='AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}'
if git ls-files -z --cached --others --exclude-standard \
   | xargs -0 grep -InE "$PATTERN" 2>/dev/null; then
  echo "❌ Key-shaped string found above. Move it to ~/.secrets/ and rotate it."
  fail=1
fi

if [ "$fail" -eq 0 ]; then
  echo "✅ No secrets detected."
fi
exit "$fail"
