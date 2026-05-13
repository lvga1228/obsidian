---
tags: [skill, hermes]
category: 📋 Productivity
skill_id: hermes-doctor
---

# Hermes Doctor

**分类:** 📋 Productivity
**Skill ID:** `hermes-doctor`

> One-click health check for all Hermes tools and skills — network, Gmail, Chrome, Apple Notes, Obsidian, skill backups. Run with 'python ~/.hermes/scripts/hermes_doctor.py'.

---


# Hermes Doctor

一键体检，检查所有已配置工具的状态。

## Usage

```bash
python ~/.hermes/scripts/hermes_doctor.py
```

Also triggerable via Hammerspoon: `Cmd+Alt+Ctrl+H` (see `hammerspoon` skill).

Script location: `~/.hermes/scripts/hermes_doctor.py`

## Checks

| Category | Items |
|----------|-------|
| Network | Baidu, Google proxy, pip mirror |
| Gmail | OAuth token, API auth |
| Browser | CLI, Chrome connection |
| Apple Notes | Notes.app access |
| Obsidian | Vault, Home.md |
| Skills | Active count, backup, restore script, cron |

## Fix Command

If any check fails, say "帮我修复" and the agent will diagnose and fix.

