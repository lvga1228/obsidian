---
category: "software-development"
skill_id: "deepseek-tui"
display_name: "DeepSeek TUI"
---

# DeepSeek TUI

**Skill ID:** `deepseek-tui`  
**Category:** [[software-development]]  
**Description:** DeepSeek TUI — terminal coding agent for DeepSeek V4 models. Installation, configuration, and usage on macOS in China.

## Skill Content

# DeepSeek TUI

Terminal coding agent for DeepSeek V4. Runs from `deepseek` command, streams reasoning blocks, edits local workspaces with approval gates. Auto mode chooses model + thinking level per turn.

## Triggers

- User asks to install/use/configure DeepSeek TUI
- User mentions `deepseek-tui`, `deepseek` CLI, DeepSeek terminal agent
- User wants to try DeepSeek coding agent in terminal

## Pre-flight

Before any DeepSeek TUI command, ensure proxy is on (required in China):

```bash
zsh -c 'source ~/.zshrc && proxy_on'
```

Verify API connectivity:
```bash
deepseek doctor
```

## Installation (macOS)

### Method 1: npm (recommended, easiest)
```bash
npm install -g deepseek-tui
# China mirror if slow:
npm install -g deepseek-tui --registry=https://registry.npmmirror.com
```

### Method 2: Cargo
```bash
cargo install deepseek-tui-cli --locked   # `deepseek` dispatcher
cargo install deepseek-tui     --locked   # `deepseek-tui` binary
```

### Method 3: Homebrew
```bash
brew tap Hmbown/deepseek-tui
brew install deepseek-tui
```

> ⚠️ Homebrew install may hang on this Mac (known issue with brew install). Prefer npm or Cargo.

## Configuration

```bash
# Set API key (saves to ~/.deepseek/config.toml)
deepseek auth set --provider deepseek

# Or use env var:
export DEEPSEEK_API_KEY="sk-xxx"

# Check auth status
deepseek auth status
```

## Usage Modes

### 1. Interactive TUI (full experience)
```bash
deepseek
```
- Three modes: Plan (read-only) / Agent (with approval) / YOLO (auto-approved)
- `/model auto` — auto-chooses model + reasoning depth
- `Shift+Tab` — cycle reasoning effort (off → high → max)
- Needs real terminal; not suitable for headless/background

### 2. One-shot chat (no tools)
```bash
deepseek-tui exec "你的问题"
deepseek exec "你的问题"          # wrapper equivalent
```

### 3. Agent mode (with tool access)
```bash
deepseek-tui exec --auto "读取 data.csv 并分析"
```
- `--auto` enables agentic mode with tool access and auto-approvals
- `--json` for machine-readable output
- `--model <model>` to override model

> ⚠️ **Pitfall**: `--auto` mode may timeout (>180s) in headless/non-interactive environments. The agent attempts multi-turn tool calls which can hang without a proper terminal session. Use interactive `deepseek` TUI for tool-heavy tasks.

## Command Reference

| Command | Purpose |
|---------|---------|
| `deepseek` | Launch interactive TUI |
| `deepseek doctor` | System diagnostics |
| `deepseek exec "prompt"` | One-shot non-interactive prompt |
| `deepseek-tui exec --auto "prompt"` | Agent mode with tools |
| `deepseek serve` | Start local HTTP server |
| `deepseek auth set --provider deepseek` | Save API key |
| `deepseek auth status` | Check auth |
| `deepseek sessions` | List saved sessions |
| `deepseek resume <id>` | Resume session |
| `deepseek models` | List available models |
| `deepseek --version` | Version check |

## Key Features

- **1M-token context** with streaming reasoning blocks
- **Full tool suite**: file ops, shell, git, web search/browse, apply-patch, sub-agents, MCP
- **Session save/resume** with workspace rollback
- **Auto mode** — picks model + thinking level per turn automatically
- **Skills system** — found 249 skills in `~/.claude/skills` that can be reused
- **Live cost tracking** with cache hit/miss breakdown

## Architecture

Two binaries work together:
- `deepseek` — npm wrapper / dispatcher CLI
- `deepseek-tui` — Rust TUI runtime (ratatui-based)

Both installed to `~/.nvm/versions/node/vXX.XX.X/bin/` when using npm.

## HTTP Serve Mode

```bash
deepseek serve --http    # starts on 127.0.0.1:7878
# Must specify mode: --http, --mcp, or --acp (no default)
```

Health check:
```bash
curl http://127.0.0.1:7878/health
# → {"status":"ok","service":"deepseek-runtime-api","mode":"local"}
```

> ⚠️ The HTTP API is NOT standard OpenAI-compatible. `/v1/chat/completions` returns 404. Use `deepseek-tui exec` for non-interactive use instead.

## Known Issues

1. **`--auto` mode fails in headless**: Agent mode with tools requires a real terminal (TTY). Fails with `tcsetattr: Inappropriate ioctl for device` and/or times out (>180s). Use interactive `deepseek` TUI for tool-heavy tasks.
2. **`--yolo` flag is for `deepseek-tui` top-level, not `exec`**: For `exec`, use `--auto` to enable agent mode. The `--yolo` flag on the top-level binary (`deepseek-tui --yolo -p "prompt"`) enables tools but still needs TTY.
3. **Homebrew hangs**: `brew install` may freeze on this Mac. Use npm or Cargo instead.
4. **China network**: Always run `proxy_on` first. API endpoint is `api.deepseek.com` (needs proxy).
5. **Repo choice**: Use the Hmbown fork (`github.com/Hmbown/DeepSeek-TUI`, 19.5k⭐, 855+ commits) — the original `DeepSeek-TUI/DeepSeek-TUI` (62⭐, 7 commits) is a lightweight stub with only Windows binary.
