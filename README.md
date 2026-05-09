# Hermes Skills Knowledge Graph

> 130+ AI agent skills visualized as an Obsidian knowledge graph — data science, productivity, MLOps, and more.

## What is this?

This is the full skill library of my [Hermes Agent](https://github.com/NousResearch/hermes-agent), organized as an [Obsidian](https://obsidian.md) vault for visual exploration. Every node is a skill, every edge is a relationship.

Open in Obsidian and press `Cmd+G` to see the full graph.

## Structure

```
data_analysis_obsidian/
├── Home.md              ← Entry point: skill map + Mermaid diagram
├── data-science.md      ← 30 skills: EDA → ML → ensemble → SHAP
├── productivity.md      ← 9 skills: freelancing, email, PPT, OCR
├── mlops.md             ← 21 skills: fine-tuning, inference, LLM eval
├── creative.md          ← 19 skills: ASCII art, infographics, design
├── software-development.md ← 11 skills: plan, debug, TDD, review
├── [skill-name].md      ← Individual skill with full documentation
└── .obsidian/           ← Graph settings (color-coded by category)
```

## Key Hubs

| Node | Connections | Role |
|------|------------|------|
| **Master Umbrella** | 30 | Data science pipeline orchestrator |
| **Kaggle提取器** | 17 | Skill generator (extracts from Kaggle notebooks) |
| **中国自由职业平台** | 5 | Freelancing strategy hub |
| **Google Workspace** | 3 | Gmail + Calendar + Drive |

## Highlights

- **Data Science Pipeline**: 3-stage EDA → Modeling → Validation, 30 Kaggle-verified techniques
- **Auto Analysis Pipeline**: Drop data → auto-detect type → recommend skill chain → generate report
- **Browser Harness**: Chrome CDP control (bypasses Playwright GFW block in China)
- **Gmail + Attachments**: OAuth-authenticated email with file attachments
- **Apple Notes**: Read/write/pin via AppleScript
- **Hammerspoon**: macOS window management + global hotkeys

## How to Use

1. Install [Obsidian](https://obsidian.md)
2. Clone this repo: `git clone https://github.com/lvga1228/obsidian.git`
3. Open Obsidian → Open vault → select the cloned folder
4. Press `Cmd+G` for graph view

## Auto-Sync

This repo is automatically synced from my live Hermes skills. When a skill is created or updated:

```
Skill change → skill_sync.py → Obsidian note updated → git push
```

Runs via cronjob every hour.

## Author

**ZANG** — Data analyst & AI automation engineer  
GitHub: [@lvga1228](https://github.com/lvga1228)
