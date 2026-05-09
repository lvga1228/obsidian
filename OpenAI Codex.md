---
category: "autonomous-ai-agents"
skill_id: "codex"
display_name: "OpenAI Codex"
---

# OpenAI Codex

**Skill ID:** `codex`  
**Category:** [[autonomous-ai-agents]]  
**Description:** Delegate coding to OpenAI Codex CLI (features, PRs).

## Skill Content

# Codex CLI

Delegate coding tasks to [Codex](https://github.com/openai/codex) via the Hermes terminal. Codex is OpenAI's autonomous coding agent CLI.

## When to use

- Building features
- Refactoring
- PR reviews
- Batch issue fixing

Requires the codex CLI and a git repository.

## Prerequisites

- Codex installed: `npm install -g @openai/codex` (may need `sudo`)
- OpenAI auth configured via `codex login --device-auth`
- **Must run inside a git repository** — Codex refuses to run outside one
- Use `pty=true` in terminal calls — Codex is an interactive terminal app
- **China network**: Codex's Rust HTTP client does NOT respect `https_proxy`. Must `unset https_proxy http_proxy` before running. The user's own network/VPN can reach OpenAI directly.

### Authentication

```bash
# Device auth (most reliable approach)
unset https_proxy http_proxy
codex login --device-auth
# → Open https://auth.openai.com/codex/device
# → Enter displayed one-time code
# → Wait up to 15 min for completion
```

For Hermes itself, `model.provider: openai-codex` uses Hermes-managed Codex
OAuth from `~/.hermes/auth.json` after `hermes auth add openai-codex`. For the
standalone Codex CLI, a valid CLI OAuth session may live under
`~/.codex/auth.json`; do not treat a missing `OPENAI_API_KEY` alone as proof
that Codex auth is missing.

### Proxy Pitfall

Codex uses a Rust HTTP client that ignores system proxy env vars. Always run:
```bash
unset https_proxy http_proxy && codex exec "..."
```
The user's machine can reach `api.openai.com` directly (not blocked by GFW like Google services).
```bash
unset https_proxy http_proxy && codex login --device-auth
```

Run in background with long timeout (browser auth takes time):

```terminal
terminal(command="unset https_proxy http_proxy && codex login --device-auth",
         background=true, pty=true, timeout=900)
# Poll for the device code
process(action="poll", session_id="<id>")
# Share URL + code with user
```

For Hermes itself, `model.provider: openai-codex` uses Hermes-managed Codex
OAuth from `~/.hermes/auth.json` after `hermes auth add openai-codex`. For the
standalone Codex CLI, a valid CLI OAuth session may live under
`~/.codex/auth.json`; do not treat a missing `OPENAI_API_KEY` alone as proof
that Codex auth is missing.

## One-Shot Tasks

```
terminal(command="codex exec 'Add dark mode toggle to settings'", workdir="~/project", pty=true)
```

For scratch work (Codex needs a git repo):
```
terminal(command="cd $(mktemp -d) && git init && codex exec 'Build a snake game in Python'", pty=true)
```

## Background Mode (Long Tasks)

```
# Start in background with PTY
terminal(command="codex exec --full-auto 'Refactor the auth module'", workdir="~/project", background=true, pty=true)
# Returns session_id

# Monitor progress
process(action="poll", session_id="<id>")
process(action="log", session_id="<id>")

# Send input if Codex asks a question
process(action="submit", session_id="<id>", data="yes")

# Kill if needed
process(action="kill", session_id="<id>")
```

## Key Flags

| Flag | Effect |
|------|--------|
| `exec "prompt"` | One-shot execution, exits when done |
| `--full-auto` | Sandboxed but auto-approves file changes in workspace |
| `--yolo` | No sandbox, no approvals (fastest, most dangerous) |

## PR Reviews

Clone to a temp directory for safe review:

```
terminal(command="REVIEW=$(mktemp -d) && git clone https://github.com/user/repo.git $REVIEW && cd $REVIEW && gh pr checkout 42 && codex review --base origin/main", pty=true)
```

## Parallel Issue Fixing with Worktrees

```
# Create worktrees
terminal(command="git worktree add -b fix/issue-78 /tmp/issue-78 main", workdir="~/project")
terminal(command="git worktree add -b fix/issue-99 /tmp/issue-99 main", workdir="~/project")

# Launch Codex in each
terminal(command="codex --yolo exec 'Fix issue #78: <description>. Commit when done.'", workdir="/tmp/issue-78", background=true, pty=true)
terminal(command="codex --yolo exec 'Fix issue #99: <description>. Commit when done.'", workdir="/tmp/issue-99", background=true, pty=true)

# Monitor
process(action="list")

# After completion, push and create PRs
terminal(command="cd /tmp/issue-78 && git push -u origin fix/issue-78")
terminal(command="gh pr create --repo user/repo --head fix/issue-78 --title 'fix: ...' --body '...'")

# Cleanup
terminal(command="git worktree remove /tmp/issue-78", workdir="~/project")
```

## Batch PR Reviews

```
# Fetch all PR refs
terminal(command="git fetch origin '+refs/pull/*/head:refs/remotes/origin/pr/*'", workdir="~/project")

# Review multiple PRs in parallel
terminal(command="codex exec 'Review PR #86. git diff origin/main...origin/pr/86'", workdir="~/project", background=true, pty=true)
terminal(command="codex exec 'Review PR #87. git diff origin/main...origin/pr/87'", workdir="~/project", background=true, pty=true)

# Post results
terminal(command="gh pr comment 86 --body '<review>'", workdir="~/project")
```

## Rules

1. **Always use `pty=true`** — Codex is an interactive terminal app and hangs without a PTY
2. **Git repo required** — Codex won't run outside a git directory. Use `mktemp -d && git init` for scratch
3. **Use `exec` for one-shots** — `codex exec "prompt"` runs and exits cleanly
4. **`--sandbox workspace-write` for building** — replaces deprecated `--full-auto`
5. **Background for long tasks** — use `background=true` and monitor with `process` tool
6. **Don't interfere** — monitor with `poll`/`log`, be patient with long-running tasks
7. **Parallel is fine** — run multiple Codex processes at once for batch work
8. **Unset proxy before running** — `unset https_proxy http_proxy` (Rust client ignores env vars)
9. **Troubleshoot token expiry**: `~/.codex/config.toml` may have stale MCP server tokens (e.g. Linear). Remove expired `[mcp_servers.*]` blocks to stop reconnect spam.

## Plugins & Marketplaces

Codex has a plugin marketplace system. View installed: check `[plugins.*]` in `~/.codex/config.toml`.

### Available Marketplaces (add with `codex plugin marketplace add <repo>`)

| Repository | Stars | Description |
|-----------|-------|-------------|
| `numman-ali/n-skills` | 979 | Curated skills marketplace for Codex/Claude |
| `fcakyon/claude-codex-settings` | 673 | Battle-tested skills/plugins/hooks |
| `agent-sh/agentsys` | 787 | 20 plugins, 49 agents, 41 skills |
| `fivetaku/gptaku-plugins-codex` | 21 | Codex-native plugin marketplace |
| `abuiles/codex-whatsapp-relay` | 22 | WhatsApp control for Codex |

### Built-in Plugins (openai-bundled)

| Plugin | Status | Function |
|--------|--------|----------|
| `browser-use` | ✅ Enabled | Browser automation |
| `computer-use` | ✅ Enabled | Desktop control |
| `chrome` | ❌ Available | Native Chrome CDP integration |
| `latex-tectonic` | ❌ Available | LaTeX typesetting |

### Discovery
- `hashgraph-online/awesome-codex-plugins` (★165) — curated plugin index
- `RoggeOhta/awesome-codex-cli` (★174) — 150+ tools and plugins
