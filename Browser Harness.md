---
tags: [skill, hermes]
category: 📋 Productivity
skill_id: browser-harness
---

# Browser Harness

**分类:** 📋 Productivity
**Skill ID:** `browser-harness`

> Direct browser control via CDP. Connects to the user's already-running Chrome for automation, scraping, testing, and web interaction. Bypasses Playwright headless shell download issues in China.

---


# Browser Harness for Hermes

Direct browser control via Chrome DevTools Protocol (CDP). Connects to your existing Chrome browser — no Playwright, no headless shell download.

## Installation

Installed at `~/Developer/browser-harness/` via `uv tool install -e .`. CLI: `browser-harness`.

For full China-specific setup guide: load `references/china-setup.md`.

## Key Advantage for China

- ❌ Playwright: blocked (headless shell 92MB download fails on cdn.playwright.dev)
- ✅ Browser Harness: connects to your REAL Chrome via CDP — no download needed
- ✅ Browser Harness: connects to your REAL Chrome via CDP — no download needed
- Requires `uv` (already on PATH if installed via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `git clone` + `uv tool install` work through Veee proxy (port 15236)

## Installation (China)

```bash
# 1. Clone (through proxy)
export https_proxy=http://127.0.0.1:15236 http_proxy=http://127.0.0.1:15236
cd ~/Developer
git clone https://github.com/browser-use/browser-harness
cd browser-harness

# 2. Install globally
uv tool install -e .

# 3. Verify
command -v browser-harness
browser-harness --doctor
```

**Pitfall:** `uv tool install` downloads dependencies from PyPI. In China, `uv` respects `https_proxy` so the proxy must be set. If it hangs, check proxy is alive on 15236.
**Pitfall:** Do NOT use Homebrew for any part of this — Homebrew's sandbox ignores proxy env vars for pip installs inside formulae.

## Setup (one-time)

1. Install via git clone + uv (no Playwright download needed):
   ```bash
   git clone https://github.com/browser-use/browser-harness ~/Developer/browser-harness
   cd ~/Developer/browser-harness
   uv tool install -e .
   ```
## Setup (one-time)

1. Open Chrome, go to `chrome://inspect/#remote-debugging`
2. Tick "Allow remote debugging for this browser instance"
3. First connection: Chrome will show "Allow remote debugging?" popup → click Allow

## Verified Working (China environment, 2026-05-09)

| Test | Command | Result |
|------|---------|--------|
| Doctor check | `browser-harness --doctor` | Chrome detected, daemon starts on first use |
| Page info | `browser-harness -c 'print(page_info())'` | Returns `about:blank` on new daemon |
| Navigate | `browser-harness -c 'new_tab("https://www.baidu.com"); wait_for_load(); print(page_info())'` | Baidu loaded, title confirmed |
| No proxy needed | Direct connection to Chrome CDP via localhost | Works without proxy — CDP is local |

## Usage

## Usage

```bash
# Basic check
browser-harness -c 'print(page_info())'

# Navigate
browser-harness -c '
new_tab("https://example.com")
wait_for_load()
print(page_info())
'

# Screenshot
browser-harness -c 'capture_screenshot("/tmp/page.png")'
```

## Important Commands

| Command | Purpose |
|---------|---------|
| `browser-harness --doctor` | Check daemon, Chrome, connection status |
| `browser-harness --update -y` | Update to latest release |
| `new_tab(url)` | Open new tab (always use this first, not goto) |
| `wait_for_load()` | Wait for page load |
| `capture_screenshot(path)` | Screenshot current page |
| `click_at_xy(x, y)` | Click at pixel coordinates |
| `page_info()` | Current page URL/title |
| `js("code")` | Execute JavaScript in page |
| `http_get(url)` | HTTP GET without browser |

## Files the Agent Can Edit

- `~/Developer/browser-harness/agent-workspace/agent_helpers.py` — custom helpers
- `~/Developer/browser-harness/agent-workspace/domain-skills/` — per-site playbooks

## Pre-imported Helpers

`new_tab`, `goto_url`, `wait_for_load`, `page_info`, `capture_screenshot`, `click_at_xy`, `js`, `cdp`, `http_get`, `ensure_real_tab`, `start_remote_daemon`, `list_cloud_profiles`, `list_local_profiles`, `sync_local_profile`

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Chrome not detected | Open Chrome first, then retry |
| Daemon not alive | Run `browser-harness -c '1'` to auto-start daemon |
| Popup blocks every attach | Use Way 2: launch Chrome with `--remote-debugging-port=9222` flag |
| Tool not on PATH | `~/.local/bin/browser-harness` — add to PATH or symlink |
| `--doctor` shows "could not reach github" | Proxy issue; ignore if already installed, use `--update -y` over proxy |
| Connection hangs in China | Browser Harness connects to local Chrome CDP via localhost — no proxy needed after daemon starts |

