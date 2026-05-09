---
category: "apple"
skill_id: "apple-notes"
display_name: "Apple Notes"
---

# Apple Notes

**Skill ID:** `apple-notes`  
**Category:** [[apple]]  
**Description:** Manage Apple Notes via osascript AppleScript — create, search, edit, pin. No third-party CLI needed.

## Skill Content

# Apple Notes

Use **AppleScript via `osascript`** to manage Apple Notes directly from the terminal. Notes sync across all Apple devices via iCloud.

> ⚠️ `memo` CLI was the original approach but fails to install on this setup (Homebrew pip sandbox timeout). The AppleScript method below is the working alternative and requires no third-party tools.

## Prerequisites

- **macOS** with Notes.app
- ~~Install: `brew tap antoniorodr/memo && brew install antoniorodr/memo/memo`~~ — **memo CLI installation fails (Homebrew pip timeout in China)**. Use osascript directly instead.
- Grant Automation access to Notes.app when prompted (System Settings → Privacy → Automation)
  System Settings → Privacy & Security → Accessibility → Terminal.app ON

## When to Use

- User asks to create, view, or search Apple Notes
- Saving information to Notes.app for cross-device access
- Creating notes with large fonts (via HTML body)
- Pinning important notes to the top

## When NOT to Use

- Obsidian vault management → use the `obsidian` skill
- Quick agent-only notes → use the `memory` tool instead
- Complex rich text editing → AppleScript limited, use Notes.app directly

## Quick Reference

### List Notes

```bash
osascript -e 'tell application "Notes" to return name of every note in default account'
```

### Search Notes

```bash
osascript -e '
tell application "Notes"
    set matches to every note in default account whose name contains "keyword"
    return name of every item of matches
end tell'
```

### Read Note Content

```bash
osascript -e '
tell application "Notes"
    set n to first note in default account whose name contains "标题"
    return body of n
end tell'
```

### Create Note (plain)

```bash
osascript -e '
tell application "Notes"
    make new note at default account with properties {name:"标题", body:"正文"}
end tell'
```

### Create Note (large font via HTML)

```bash
osascript -e '
tell application "Notes"
    set html to "<div style=\"font-size: 48px; font-weight: bold;\">🚀 标题</div><div style=\"font-size: 36px;\">大字正文</div>"
    make new note at default account with properties {name:"标题", body:html}
end tell'
```

### Delete Note

```bash
osascript -e '
tell application "Notes"
    delete every note in default account whose name contains "标题"
end tell'
```

### Pin Note (requires Accessibility permission)

```bash
osascript -e '
tell application "Notes" to activate
delay 1
tell application "System Events"
    tell process "Notes"
        keystroke "f" using command down
        delay 0.5
        keystroke "a" using command down
        delay 0.1
        keystroke "标题关键词"
        delay 0.5
        key code 53
        delay 0.3
        click menu item "Pin Note" of menu "File" of menu bar 1
    end tell
end tell'
```

## Limitations

- `pinned` property is **read-only** in AppleScript — pinning requires UI automation via System Events
- AppleScript body HTML support is limited; test formatting in Notes.app
- Interactive prompts require Accessibility permission for keystrokes
- macOS only — requires Apple Notes.app
- `&` characters in AppleScript strings cause shell issues — use script files (`.applescript`) instead of inline `-e` for complex scripts

## Rules

1. Prefer Apple Notes when user wants cross-device sync (iPhone/iPad/Mac)
2. Use the `memory` tool for agent-internal notes that don't need to sync
3. Use the `obsidian` skill for Markdown-native knowledge management
4. For complex AppleScript, write to `/tmp/*.applescript` file first, then run with `osascript`
