---
tags: [skill, hermes]
category: 📋 Productivity
skill_id: hammerspoon
---

# Hammerspoon

**分类:** 📋 Productivity
**Skill ID:** `hammerspoon`

> macOS automation via Hammerspoon Lua scripting. Window management, global hotkeys, menubar control, USB events, timers. Hermes can write Lua configs to automate almost anything on macOS.

---


# Hammerspoon for Hermes

Hammerspoon is the ultimate macOS automation framework. Hermes writes `~/.hammerspoon/init.lua` to add new capabilities.

## Current Shortcuts

| Shortcut | Function |
|----------|----------|
| Cmd+Alt+Ctrl + ←/→ | Left/right half screen |
| Cmd+Alt+Ctrl + ↑ | Fullscreen |
| Cmd+Alt+Ctrl + ↓ | Center 80% |
| Cmd+Alt+Ctrl + [ ] | Move to left/right monitor |
| Cmd+Alt+Ctrl + H | Hermes Doctor |
| Cmd+Alt+Ctrl + V | Clipboard history |
| Cmd+Alt+Ctrl + N | Test notification |

## Usage

Hermes can add new automations by editing `~/.hammerspoon/init.lua`. After edit, user clicks 🔨 → Reload Config.

## Pitfalls

### terminal-notifier in hs.task callbacks
**Problem:** Calling `os.execute('terminal-notifier ...')` inside an `hs.task.new()` callback produces `cannot execute binary files` error. The Lua `os.execute()` in the callback context has a restricted PATH.

**Solution:** Use `hs.notify.show()` directly (Hammerspoon's built-in notification), or use `hs.execute()` if synchronous is acceptable. The full path `/opt/homebrew/bin/terminal-notifier` may still fail in certain callback environments.

**Working pattern:**
```lua
hs.task.new("/bin/bash", function(exitCode, stdOut, _)
    hs.notify.show("Title", stdOut:sub(1, 200), "")
end, {"/bin/bash", "-c", "your_command 2>&1"}):start()
## Pitfalls

### Homebrew binary paths
⚠️ Hammerspoon's Lua environment does NOT inherit shell $PATH. Always use full paths for Homebrew binaries:

```lua
-- WRONG
os.execute('terminal-notifier -title "Test" -message "Hi"')

-- CORRECT
os.execute('/opt/homebrew/bin/terminal-notifier -title "Test" -message "Hi"')
```

### Notification permission
Hammerspoon's `hs.notify` requires notification permission in System Settings. If denied, use `terminal-notifier` via `os.execute()` instead — it uses macOS native notifications without extra permissions.

### hs.notify vs terminal-notifier
Prefer `os.execute('/opt/homebrew/bin/terminal-notifier ...')` over `hs.notify.show()` for reliability. It works even without notification permissions and is simpler to debug. (🔨 → Reload Config)
- **Pitfall**: `hs.notify.show()` requires Hammerspoon to have Notification Center permission. If not granted, use `/opt/homebrew/bin/terminal-notifier` via `os.execute()` instead.
- **Pitfall**: `terminal-notifier` path: `/opt/homebrew/bin/terminal-notifier` (Hammerspoon's shell may not have PATH set)
- **Pitfall**: `hs.execute()` is NOT a valid function; use `hs.task.new("/bin/bash", callback, {args})` for async shell execution
- Config file: `~/.hammerspoon/init.lua`
- Enable IPC for CLI control: add `hs.ipc.cliInstall()` to init.lua

