---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: kaggle-notebook-extraction
---

# Kaggle Notebook Extraction

**分类:** 📊 Data Science
**Skill ID:** `kaggle-notebook-extraction`

> Extract and analyze Kaggle notebook source code (.ipynb) via Kaggle's public API without authentication or browser. Parse notebook structure to extract data analysis patterns, code templates, and workflows for skill distillation.

---

# Kaggle Notebook Extraction & Pattern Mining — v2.0

Comprehensive update 2026-05-08: 11 new skills distilled in a single session (8 ML foundations gap-fill + 3 domain gap-fill). Added 403 synthesis protocol, gap analysis methodology, and full 22-notebook inventory with skill mappings. Skill persistence mechanism battle-tested (auto-restore from .archive).
# Kaggle Notebook Extraction & Pattern Mining

## Purpose
Systematically extract Kaggle notebook source code via the public API, parse .ipynb content, and distill reusable data analysis patterns into skills. Designed for the "learn from Kaggle → build skills" workflow.

## ⚠️ CRITICAL: API Access Reality (as of April 2026)

```
Notebook Age      →  Pull API Result    Data Download
─────────────────────────────────────────────────────
2017-2020          ✅ Mostly works      ✅ kagglehub works
2021-2024          ⚠️ Mixed (some 403)  ⚠️ Mixed
2025-2026          ❌ Almost all 403    ⚠️ Some work

Dansbecker + Kaggle Learn tutorials → ✅ Always accessible
User-submitted kernels post-2021     → ❌ Mostly 403
```

**When 403 hits**: The notebook slug is valid but Kaggle blocks unauthenticated access. The OAuth path would unlock these.

## Core Endpoint

### Pull Notebook Source (Recommended)
```
GET https://www.kaggle.com/api/v1/kernels/pull/{author_username}/{notebook_slug}
```
No authentication required for public notebooks. Returns JSON with `metadata` and `blob.source` (stringified .ipynb).

## Standard Execution Flow (Dataset + Notebook)

1. Download dataset via kagglehub
2. Pull notebook via `curl` to API endpoint
3. 📋 MANDATORY: Print columns + head(3) + ask user to confirm target/categorical/numerical
4. Create output folder: `分析主题_YYYYMMDD_HHMM`
5. Run full deep analysis (print EVERY code cell)
6. Generate IPYNB with markdown documentation
7. Distill into a new skill via `skill_manage`

**⚠️ LESSON**: ALWAYS do deep analysis BEFORE creating the skill. We once skipped this and missed 5 critical patterns (SMOTETomek vs SMOTE, manual map vs LabelEncoder, XGBoost on original data, plot_tree, train/test accuracy tracking).

## Deep Notebook Analysis

```python
def deep_analyze_notebook(nb_json):
    """Full cell-by-cell extraction — NO SKIPPING."""
    for i, cell in enumerate(nb_json['cells']):
        if cell['cell_type'] == 'markdown':
            src = ''.join(cell.get('source', []))
            for line in src.split('\n'):
                if line.strip().startswith('#'):
                    print(f"[MD {i}] {line.strip()}")
        elif cell['cell_type'] == 'code':
            src = ''.join(cell.get('source', []))
            print(f"\n[CODE {i}] ({len(src)} chars):")
            print(src[:600])
    extract_patterns(nb_json)
```

### Key Patterns Checklist
- [ ] Imbalance handling method (SMOTE vs SMOTETomek vs ADASYN)
- [ ] Encoding method (LabelEncoder vs map() vs OneHot)
- [ ] Which model uses SMOTE data? (Often DT+RF use SMOTE, XGB uses original)
- [ ] Hyperparameters tuned or defaults?
- [ ] Correlation heatmap before modeling?
- [ ] Train vs Test accuracy both reported?
- [ ] Visualizations used (plot_tree, feature_importance, confusion_matrix)?
- [ ] GridSearchCV present? Scoring metric?
- [ ] Library version issues (fbprophet → prophet)

## Confirmed-Working Notebooks (22 total, all distilled as of 2026-05-08)

### Core Collection (distilled pre-2026-05-08)
| Author | Slug | → Skill | Domain | Votes |
|--------|------|---------|--------|:-----:|
| pmarcelino | comprehensive-data-exploration-with-python | enhanced-exploratory-data-analysis | EDA | 33,536 |
| rtatman | data-cleaning-challenge-handling-missing-values | data-cleaning-pipeline | Data Cleaning | 15,162 |
| yassineghouzam | titanic-top-4-with-ensemble-modeling | classification-modeling-workflow | Classification | 5,809 |
| yassineghouzam | introduction-to-cnn-keras-0-997-top-6 | deep-learning-cnn-keras | Deep Learning | 16,499 |
| rounakbanik | movie-recommender-systems | recommender-systems | Recommender | 5,331 |
| abhishek | approaching-almost-any-nlp-problem-on-kaggle | nlp-text-analysis | NLP | 5,049 |
| robikscube | time-series-forecasting-with-prophet | time-series-forecasting | Time Series | 1,698 |
| kushal1996 | customer-segmentation-k-means-analysis | clustering-segmentation | Clustering | 3,356 |
| joparga3 | in-depth-skewed-data-classif-93-recall-acc-now | imbalanced-classification | Imbalanced | 2,577 |
| arthurtok | introduction-to-ensembling-stacking-in-python | ensemble-advanced | Ensemble | 15,461 |
| victorambonati | unsupervised-anomaly-detection | unsupervised-anomaly-detection | Anomaly | 1,443 |
| ldfreeman3 | a-data-science-framework-to-achieve-99-accuracy | ml-pipeline-best-practices | Framework | 13,929 |
| willkoehrsen | introduction-to-feature-selection | feature-selection-methods | Feature Selection | 835 |
| salemali7 | teen-mental-health-prediction-multiple-ml-models | multi-model-smote-workflow | Multi-model | 54 |

### Distilled 2026-05-08 — ML Foundations Gap-Fill
| Author | Slug | → Skill | Domain | Votes |
|--------|------|---------|--------|:-----:|
| dansbecker | shap-values | model-interpretability-shap | Interpretability | 997 |
| dansbecker | cross-validation | cross-validation-strategies | CV | 1,072 |
| dansbecker | pipelines | sklearn-pipeline-automation | Automation | 1,041 |
| dansbecker | random-forests | random-forest-mastery | Ensemble | 4,522 |
| dansbecker | model-validation | model-validation-strategies | Validation | 7,040 |
| ash316 | eda-to-prediction-dietanic | end-to-end-ml-dietanic | Pipeline | 6,314 |

### Distilled 2026-05-08 — Domain Gap-Fill (regression, boosting, stats)
| Author | Slug | → Skill | Domain | Votes |
|--------|------|---------|--------|:-----:|
| serigne | stacked-regressions-top-4-on-leaderboard | regression-modeling-mastery | Regression | 14,337 |
| dansbecker | xgboost | gradient-boosting-mastery | Gradient Boosting | 3,709 |
| dansbecker | basic-data-exploration | statistical-testing-methods | Statistics | 7,020 |
| kanncaa1 | statistical-learning-tutorial-for-beginners | (combined into above) | Statistics | 1,945 |

### 403-Locked — Synthesized from ML Knowledge
| Author | Slug | → Skill | Status |
|--------|------|---------|--------|
| dansbecker | your-first-ML-model | ml-fundamentals | Synthesized (API 403) |
| dansbecker | underfitting-overfitting | overfitting-underfitting-diagnostics | Synthesized (API 403) |

## 403 Synthesis Protocol

When Kaggle API returns 403 for a known-valuable notebook, do NOT skip — synthesize from deep ML knowledge:

1. **Confirm the 403**: Check HTTP status, verify slug is correct
2. **Read notebook title/metadata from Kaggle webpage** (if accessible) to understand scope
3. **Synthesize from first principles**: Use known ML theory + the textbook-standard content these notebooks cover
4. **Mark clearly**: `source: Synthesized from {author}/{slug} (Kaggle, 403 locked)` in frontmatter
5. **Match quality bar**: Same cell-count-equivalent depth as API-fetched skills

## Gap Analysis Methodology

To identify missing skills systematically:
1. List all 22 confirmed-working notebooks → map to existing skills
2. Audit domain coverage: which ML domains have 0 skills?
3. Query Kaggle API for high-vote notebooks in gap domains
4. Prioritize: Regression > Gradient Boosting > Statistical Testing > Dimensionality Reduction
5. Fetch successful candidates, synthesize 403-locked ones

## Current Skill Inventory (30 total)

22 Kaggle-distilled + 2 synthesized + 3 custom + 3 gap-fill (regression, boosting, stats).
Full manifest: `/Users/ordi_de_lvga/Documents/hermes 文件/skills/data-science/MANIFEST.md`

## ⚠️ Skill Persistence — CRITICAL: Backups Do Not Keep Skills Active

**The problem**: Hermes updates automatically move custom skills to `~/.hermes/skills/.archive/`, which makes them **invisible to Hermes**. A file backup alone (e.g., copying to a user path) does NOT make them active again — Hermes only loads from `~/.hermes/skills/`.

**The fix** (deployed 2026-05-08):
1. Auto-restore script: `~/.hermes/scripts/restore_skills.sh`
   - Detects skills archived to `.archive/` and copies them back to `data-science/`
   - Updates the backup at the user's custom path + refreshes `MANIFEST.md`
2. Cron job `Skill Auto-Restore after Hermes Update` — runs every 30 minutes
   - Ensures skills are restored within 30 min of any Hermes update

**Manual restore** (if cron hasn't run yet):
```bash
bash ~/.hermes/scripts/restore_skills.sh
```

**Lesson**: User's requirement was "skills should remain usable after Hermes updates." A file backup to another path is necessary but insufficient — the active load path must be repopulated after every update. The right mental model is **"auto-restore to active path"**, not "backup and hope."

## Pipeline Architecture

All 30 data-science skills form a 3-stage pipeline with branching at the modeling phase:

```
Data Input → [EDA + Cleaning] → [Feature Engineering + Selection] → [Modeling Branch] → Report
                                                   ↓
                  classification / imbalanced / multi-model-smote / ensemble (Core ML)
                  time-series / supply-chain / clustering / recommender / anomaly / nlp / cnn / transfer (Specialized)
```

A visual architecture diagram is available:
`/Users/ordi_de_lvga/Documents/hermes 文件/Upwork接单启动_20260508_1558/04_SkillPipeline架构图.html`

## Relationship to data-science-pipeline
This skill is the GENERATOR: all 17 reference files in `data-science-pipeline` were produced by running this extraction workflow against Kaggle notebooks. This skill was restored from .archive on 2026-05-08 after being moved during a Hermes update.

