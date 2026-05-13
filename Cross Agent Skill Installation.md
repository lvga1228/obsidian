---
tags: [skill, hermes]
category: 🔧 DevOps
skill_id: cross-agent-skill-installation
---

# Cross Agent Skill Installation

**分类:** 🔧 DevOps
**Skill ID:** `cross-agent-skill-installation`

> Install third-party skill packages (from GitHub repos) into Claude Code and Hermes simultaneously. Covers dependency setup, skill format detection, directory conventions, and multi-agent deployment. Use when installing gstack, superpowers, or any GitHub-based AI agent skill pack.

---


# Cross-Agent Skill Installation

Install third-party skill packages from GitHub into both Claude Code and Hermes in one pass. This skill captures the workflow, pitfalls, and directory conventions discovered across multiple installations.

## When to Use

- User asks to install a skill pack (gstack, superpowers, or similar) for Claude Code, Hermes, or both
- User wants to add skills from a GitHub repository
- Installing skills that require build steps (bun, npm, cargo)

## Prerequisites Check

Before starting, verify:

```bash
# Claude Code installed?
which claude && claude --version

# Hermes skills directory exists?
ls ~/.hermes/skills/

# Claude Code skills directory exists?
ls ~/.claude/skills/

# bun installed? (required by gstack)
~/.bun/bin/bun --version 2>/dev/null || echo "NEED_BUN"
```

## Installation Flow

### Step 1: Identify the Skill Package Format

Check the GitHub repo README for installation instructions. Three common patterns:

| Pattern | Example | How to install |
|---------|---------|---------------|
| Plugin marketplace | superpowers (`/plugin install`) | Add marketplace config + copy skills manually |
| Setup script | gstack (`./setup`) | Clone + run setup, may need build deps |
| Raw SKILL.md files | Most Hermes skills | Clone + copy with prefix to `~/.hermes/skills/` |

### Step 2: Install Dependencies FIRST

Many skill packs have build dependencies. Install before cloning:

| Dependency | Install | Used by |
|-----------|---------|---------|
| bun | `curl -fsSL https://bun.sh/install \| bash` | gstack (browse, design, pdf binaries) |
| Playwright | `npx playwright install chromium` (inside skill dir) | gstack browse/QA |
| Node.js | Already via nvm | Most skills |
| Python packages | `pip install -r requirements.txt` | Some data skills |

**Pitfall: Playwright download can timeout (120s+).** Run `npx playwright install chromium` separately after the main setup, with a 300s timeout.

### Step 3: Clone and Install

#### For Claude Code

Skills go in `~/.claude/skills/` as directories containing `SKILL.md`:

```bash
# Option A: Clone entire repo (for skills with build steps)
git clone --single-branch --depth 1 <repo> ~/.claude/skills/<name>
cd ~/.claude/skills/<name> && ./setup  # if setup script exists

# Option B: Copy individual skills (for raw SKILL.md repos)
for skill_dir in /tmp/repo/skills/*/; do
  skill_name=$(basename "$skill_dir")
  cp -r "$skill_dir" ~/.claude/skills/superpowers-${skill_name}
done
```

**Claude Code sub-skills pattern:** Some packages (like gstack) install as ONE top-level directory with sub-skill directories inside. Others (like superpowers) install as separate top-level directories with a prefix.

#### For Hermes

Skills go in `~/.hermes/skills/` as directories containing `SKILL.md`:

```bash
# Option A: Use the package's Hermes-specific output (if it generates one)
cp -r /path/to/repo/.hermes/skills/* ~/.hermes/skills/

# Option B: Copy raw skills with prefix
for skill_dir in /tmp/repo/skills/*/; do
  skill_name=$(basename "$skill_dir")
  cp -r "$skill_dir" ~/.hermes/skills/${prefix}-${skill_name}
done

# Option C: Symlink for auto-updates
ln -s /path/to/repo/skills/skill-name ~/.hermes/skills/skill-name
```

**Hermes prefix convention:** Skills from external packs should use the pack name as prefix (e.g., `gstack-review`, `superpowers-brainstorming`) to avoid collisions with existing skills.

#### For Both (Simultaneous)

```bash
# Clone once, install to both
git clone --single-branch --depth 1 <repo> /tmp/<name>

# Claude Code: copy skills with prefix
for d in /tmp/<name>/skills/*/; do
  cp -r "$d" ~/.claude/skills/$(basename "$d")
done

# Hermes: copy skills with prefix
for d in /tmp/<name>/skills/*/; do
  cp -r "$d" ~/.hermes/skills/$(basename "$d")
done
```

### Step 4: Verify Installation

```bash
# Check skill directories exist and have SKILL.md
ls ~/.claude/skills/<name>/SKILL.md
ls ~/.hermes/skills/<prefix>-*/SKILL.md

# Check build binaries (if applicable)
ls ~/.claude/skills/<name>/browse/dist/browse  # gstack
ls ~/.claude/skills/<name>/design/dist/design  # gstack

# Count installed skills
ls -d ~/.hermes/skills/<prefix>-* | wc -l
```

## Known Packages Reference

See `references/known-packages.md` for installation details of specific packages (gstack, superpowers, etc.).

## Pitfalls

1. **bun not installed:** gstack setup will fail silently or with cryptic errors. Always check for bun first.
2. **Playwright timeout:** The Chromium download is ~250MB. On slow connections, run it separately with a longer timeout (300s+).
3. **gstack Hermes model:** gstack's `./setup --host hermes` does NOT install skills directly — it just prints instructions and exits. For Hermes, you must copy the generated `.hermes/skills/` from the repo manually.
4. **Skill name collisions:** Always use a prefix (e.g., `gstack-`, `superpowers-`) when copying skills to avoid overwriting existing skills with the same name.
5. **Claude Code plugin marketplace vs manual:** The plugin marketplace system requires interactive Claude Code commands. For non-interactive installation, copy skills directly to `~/.claude/skills/`.
6. **gstack codesign warnings on Apple Silicon:** `codesign failed for browse/dist/find-browse` — the binaries still work, these are non-fatal warnings.
7. **Memory limit pressure:** Installing many skills can push Hermes skill count high. Monitor with `ls -d ~/.hermes/skills/*/ | wc -l`.

## Hermes Update Safety

**User-installed skills (gstack, superpowers, etc.) are SAFE during `hermes update`.**

The update runs `sync_skills()` from `tools/skills_sync.py`, which only touches skills tracked in `~/.hermes/skills/.bundled_manifest`. User-installed skills are NOT in this manifest and are never touched.

**How to verify before updating:**
```bash
# Check if your third-party skills are in the manifest (they shouldn't be)
grep "gstack\|superpowers" ~/.hermes/skills/.bundled_manifest
# Empty result = safe

# Count bundled (tracked) vs total skills
echo "Bundled: $(wc -l < ~/.hermes/skills/.bundled_manifest)"
echo "Total: $(ls -d ~/.hermes/skills/*/ | wc -l)"
```

**What `sync_skills()` does during update:**
- NEW bundled skills (in repo, not in manifest): copied to `~/.hermes/skills/`
- EXISTING bundled skills (in manifest): updated if unchanged, skipped if user modified
- User-deleted bundled skills (in manifest, not on disk): respected, not re-added
- User-installed skills (not in manifest): completely ignored — safe

**What `hermes update` does NOT do:**
- Does NOT touch `~/.hermes/skills/` directly (git pull only affects `~/.hermes/hermes-agent/`)
- Does NOT archive or move user-installed skills
- Does NOT run the curator on user-installed skills

The curator (separate from update) may archive agent-created skills after 90 days of non-use, but this only applies to skills created by the agent itself via `skill_manage`, not externally installed packages.

