#!/usr/bin/env bash
# [ai-generated]
# Stop hook: nudge the agent to keep docs/project-log.md current.
#
# WHY a Stop hook (not SessionEnd): SessionEnd cannot inject feedback or make the
# agent act — it is side-effects only. Stop is the only event that fires at a
# natural end-of-turn AND can block + hand the agent an instruction. So this is
# the closest mechanism to "before the session ends, make sure the log is updated."
#
# Behaviour: if this session edited repo files but touched NONE of the log files
# (docs/project-log.md / .typ, claude-project-log.md), block once and ask the
# agent to either update the log or explicitly acknowledge that no update is
# needed. Persists across turns until the log is updated OR the agent creates the
# ack marker. Never fires on pure read/Q&A sessions (no edits).
#
# Fail-open everywhere: any error / missing tool exits 0 so a broken hook can
# never wedge a session.

set -uo pipefail

input="$(cat)"

# jq is required to parse the transcript; without it, do nothing.
command -v jq >/dev/null 2>&1 || exit 0

# Avoid an infinite block-loop: if we're already continuing because THIS hook
# blocked on the previous turn, don't block again.
[ "$(printf '%s' "$input" | jq -r '.stop_hook_active // false')" = "true" ] && exit 0

transcript="$(printf '%s' "$input" | jq -r '.transcript_path // empty')"
session_id="$(printf '%s' "$input" | jq -r '.session_id // empty')"
repo="$(printf '%s' "$input" | jq -r '.cwd // empty')"
[ -n "$transcript" ] && [ -f "$transcript" ] || exit 0
[ -n "$repo" ] || exit 0

# Per-session acknowledgement marker. The agent creates this (see reason text)
# to say "no project-log update is warranted this session" and silence the nudge.
ack="${TMPDIR:-/tmp}/cc-projlog-ack-${session_id}"
[ -f "$ack" ] && exit 0

# Every file written/edited this session (absolute paths), from the transcript.
edited="$(jq -r '
  select(.type=="assistant")
  | .message.content[]?
  | select(.type=="tool_use" and (.name|test("^(Edit|Write|MultiEdit|NotebookEdit)$")))
  | (.input.file_path // .input.notebook_path // empty)
' "$transcript" 2>/dev/null | sort -u)"

# No edits at all → pure read/analysis/Q&A session. Nothing to log; stay silent.
[ -n "$edited" ] || exit 0

log_touched=0
substantive=0
while IFS= read -r p; do
  [ -n "$p" ] || continue
  case "$p" in
    "$repo"/*) rel="${p#"$repo"/}" ;;
    *) continue ;;                       # edits outside the repo don't count
  esac
  case "$rel" in
    docs/project-log.md|docs/project-log.typ|claude-project-log.md)
      log_touched=1 ;;
    .claude/*)
      : ;;                               # hook/config/settings edits aren't "work"
    *)
      substantive=1 ;;
  esac
done <<< "$edited"

# Repo work happened and a log file was already touched → all good.
[ "$log_touched" -eq 1 ] && exit 0
# Only non-repo or only .claude edits → nothing worth logging.
[ "$substantive" -eq 1 ] || exit 0

# Substantive repo work, no log update. Block once and instruct the agent.
reason="$(printf '%s\n  1. %s\n  2. %s' \
  "This session edited repo files but did NOT update the project log (docs/project-log.md — the canonical experiment ledger + high-level state, read first every session; or claude-project-log.md for the chronological narrative). Per CLAUDE.md, when an experiment lands/retracts or the understanding changes, its ledger row must be updated in the SAME change. Before ending, do ONE of:" \
  "Update docs/project-log.md (and/or claude-project-log.md) to reflect what this session did, or" \
  "If no log update is warranted, run:  touch $ack  — this acknowledges and silences the reminder for the rest of the session (state briefly why no update is needed).")"

jq -n --arg r "$reason" '{decision:"block", reason:$r}'
exit 0
