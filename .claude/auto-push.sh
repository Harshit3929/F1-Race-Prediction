#!/usr/bin/env bash
# Auto-push safety net — runs at session end (SessionEnd hook).
# Commits any leftover changes and pushes to origin so nothing is ever left
# committed-locally-only (the VPS Hermes layer needs the remote current).
# On failure it prints a JSON systemMessage warning. Never blocks the session.

script_dir="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
root="$(cd "$script_dir/.." 2>/dev/null && pwd)"
cd "$root" 2>/dev/null || exit 0

# Emit {"systemMessage":"<text>"} with minimal JSON escaping so it always parses.
warn() {
  local m="$1"
  m="${m//\\/\\\\}"; m="${m//\"/\\\"}"; m="$(printf '%s' "$m" | tr '\n\r\t' '   ')"
  printf '{"systemMessage": "%s"}\n' "$m"
}

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# --- Vault commit lock (atomic, race-free) ---------------------------------
# Several instances — Claude and/or Codex — can reach session-end at once. A bare
# `mkdir` is atomic on every OS: exactly one caller can create the directory, so
# it is a true mutex (no check-then-create gap). The Codex backstop locks the
# very same `.vault.lock` path the same way, so the two tools interlock too.
# `.vault.lock` is gitignored — `git add -A` never stages it.
lock="$root/.vault.lock"
lock_held=0
release_lock() { [ "$lock_held" = 1 ] && rm -rf "$lock" 2>/dev/null; lock_held=0; }
trap release_lock EXIT

acquire_lock() {
  local tries=0 max=40 now age          # ~40 * 0.25s ≈ 10s before giving up
  while [ "$tries" -lt "$max" ]; do
    if mkdir "$lock" 2>/dev/null; then
      printf 'claude-auto-push pid=%s %s\n' "$$" "$(date '+%Y-%m-%d %H:%M:%S')" \
        > "$lock/owner" 2>/dev/null
      lock_held=1
      return 0
    fi
    # Reclaim a stale lock left behind by a crashed run (older than 120s).
    now="$(date +%s)"
    age=$(( now - $(stat -c %Y "$lock" 2>/dev/null || echo "$now") ))
    if [ "$age" -ge 120 ]; then rm -rf "$lock" 2>/dev/null; continue; fi
    tries=$((tries + 1)); sleep 0.25
  done
  return 1
}

# If another instance holds the lock it is committing/pushing the whole shared
# tree (our changes included) — back off quietly rather than racing it.
acquire_lock || exit 0

# 1) Commit leftover changes (respects .gitignore via -A), but NEVER stage
#    .codex/ — Codex owns that folder and edits it concurrently; staging it here
#    would commit its in-progress work and cause push races between the agents.
if [ -n "$(git status --porcelain -- . ':(exclude).codex')" ]; then
  git add -A -- . ':(exclude).codex'
  git commit -q \
    -m "chore: auto-save at session end ($(date '+%Y-%m-%d %H:%M'))" \
    -m "Automatic safety-net commit — see the git rule in CLAUDE.md." >/dev/null 2>&1
fi

# 2) Need an upstream to push to.
if ! git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
  warn "⚠️ Vault: no git upstream configured — commits are LOCAL ONLY. Run: git push -u origin <branch>."
  exit 0
fi

# 3) Anything unpushed?
ahead="$(git rev-list --count @{u}..HEAD 2>/dev/null || echo 0)"
[ "${ahead:-0}" -eq 0 ] && exit 0   # already in sync — stay silent.

# 4) Push; warn explicitly on failure.
if err="$(git push 2>&1)"; then
  exit 0   # pushed — stay silent.
else
  warn "⚠️ Vault: git push FAILED at session end — ${ahead} commit(s) still LOCAL ONLY. Fix and push manually. Error: ${err}"
  exit 0
fi
