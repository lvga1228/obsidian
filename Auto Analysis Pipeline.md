---
category: "data-science"
skill_id: "auto-analysis-pipeline"
display_name: "Auto Analysis Pipeline"
---

# Auto Analysis Pipeline

**Skill ID:** `auto-analysis-pipeline`  
**Category:** [[data-science]]  
**Description:** One-command data analysis pipeline — drop data, get report. Auto-detects problem type (classification/regression/clustering/time-series), creates organized output folder, runs through EDA → cleaning → modeling → report.

## Skill Content

# Auto Analysis Pipeline

一键数据分析流水线。用户提供数据文件路径，自动完成全流程。

## Trigger

User says: "分析这个数据" / "analyze this" / "跑一下分析" + provides data file.

## Workflow (6 steps, fully automated)

### Step 1: Auto-detect problem type

Run detection script:
```bash
Run `python ~/.hermes/scripts/detect_problem.py <data.csv> [target_col]` to auto-detect.

### Execution Script
- `~/.hermes/scripts/detect_problem.py` — 8.5KB, multi-factor problem type detector
- Outputs: detected type, confidence %, reason, recommended skill chain
```

This outputs problem type, confidence, reason, and recommended skill chain.

## Auto-Detection Logic (Multi-Factor)

### Decision Tree

```
Data loaded
  │
  ├─ Target column provided?
  │   ├─ YES → analyze target:
  │   │   ├─ 2 unique values → Binary Classification
  │   │   │   └─ minority ratio < 10% → Imbalanced (SMOTE recommended)
  │   │   ├─ 3-15 unique values → Multi-Class Classification
  │   │   ├─ >50 unique + numeric + continuous → Regression
  │   │   └─ numeric but 15-50 → prompt user (could be ordinal classification)
  │   │
  │   └─ NO → infer from structure:
  │       ├─ Date column + numeric columns → Time Series
  │       ├─ Text columns dominate → NLP Text Analysis
  │       ├─ Image path column → Computer Vision (CNN/Transfer)
  │       ├─ Mostly numeric + few categorical → Clustering / Anomaly
  │       └─ Can't determine → EDA first, ask user
```

### Signals Considered
| Signal | Weight | Indicates |
|--------|--------|-----------|
| Unique value ratio | High | Classification vs Regression |
| Column dtype | High | Numeric vs Categorical |
| Skewness (>2) | Medium | Regression with log-transform need |
| Minority ratio (<10%) | High | Imbalanced → SMOTE |
| Date pattern match (>70%) | High | Time Series |
| Avg text length (>50 chars) | Medium | NLP |
| Image extension pattern | High | Computer Vision |

### Step 2: Create output folder
Format: `{分析主题}_{YYYYMMDD}_{HHMM}`
Path: `/Users/ordi_de_lvga/Documents/hermes 文件/{folder_name}/`

### Step 3: Enhanced EDA
Load `enhanced-exploratory-data-analysis` skill.
Run full EDA: column types, missing values, distributions, correlations, outlier detection.
Generate PPTX summary if applicable.

### Step 4: Data Cleaning
Load `data-cleaning-pipeline` skill.
Handle missing values, outliers, duplicates.
Log all transformations.

### Step 5: Modeling
Load `data-science-pipeline` skill, select sub-skill based on problem type:
- Classification → `classification-modeling-workflow`
- Imbalanced → `imbalanced-classification` + `multi-model-smote-workflow`
- Regression → `regression-modeling-mastery`
- Clustering → `clustering-segmentation`
- Time Series → `time-series-forecasting`
- Anomaly → `unsupervised-anomaly-detection`
Run model training with cross-validation.

### Step 6: Report
- Generate PDF report with findings + charts
- Summarize top 3 insights in Chinese
- Save all outputs to the analysis folder

## References

- `references/detection-logic.md` — Full decision tree, multi-factor signals, test cases for `detect_problem.py`
- Never skip the 3-step data protocol (print columns, show head, confirm target)
- **NEW (2026-05-09): After target confirmation, run `detect_problem.py` and SHOW the user the detected type + confidence + recommended skill chain. Wait for user to confirm or modify BEFORE executing any analysis.**
- All charts use English labels (user preference)
- All report text in Chinese
- File organization follows user's folder convention
- **Never execute analysis or generate files until user explicitly confirms the pipeline**
