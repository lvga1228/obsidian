---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: data-science-pipeline
---

# Data Science Pipeline

**分类:** 📊 Data Science
**Skill ID:** `data-science-pipeline`

> Master umbrella skill for the full data science and machine learning workflow — from EDA and data cleaning through feature engineering, model selection, training, ensemble methods, NLP, computer vision, time series, recommender systems, anomaly detection, regression, gradient boosting, statistical testing, and supply chain analysis. Orchestrates 30 Kaggle-proven technique skills.

---


# Data Science Pipeline — Master Umbrella

Covers the complete data science and ML workflow: exploratory data analysis → data cleaning → feature engineering → feature selection → model training (classification, imbalanced, clustering, NLP, computer vision, time series, recommender systems, anomaly detection) → ensemble methods → best practices (CV, pipeline, overfitting).

Each sub-technique has a self-contained reference file in `references/`. Load those for detailed code recipes and Kaggle-proven workflows.

## When to Use

Load this skill whenever the user asks for any data science or ML modeling task. The umbrella SKILL.md helps you navigate which reference file to load. Common triggers:

- "Build a model", "predict something", "analyze this dataset"
- "Classification", "clustering", "time series", "NLP", "recommender"
- "EDA", "data cleaning", "feature engineering", "feature selection"
- "Imbalanced data", "anomaly detection", "ensemble methods"
- "Supply chain analysis", "customer segmentation"

## Pipeline Decision Framework

```
User has data + goal
  │
  ├─ Step 1: EDA & Data Orientation
  │   └─ Load: enhanced-exploratory-data-analysis skill
  │       Or references/eda.md for quick reference
  │
  ├─ Step 2: Data Cleaning
  │   └─ references/data-cleaning.md
  │
  ├─ Step 3: Feature Engineering
  │   └─ references/feature-engineering.md
  │
  ├─ Step 4: Feature Selection (if >50 features)
  │   └─ references/feature-selection.md
  │
  └─ Step 5: Modeling — PICK YOUR TECHNIQUE
      │
      ├─ Classification (balanced)
      │   └─ references/classification.md
      │
      ├─ Classification (imbalanced, e.g. fraud)
      │   ├─ references/imbalanced-classification.md
      │   └─ references/multi-model-comparison.md (quick SMOTE + DT/XGB/RF shootout)
      │
      ├─ Regression / Time Series
      │   ├─ references/regression-modeling-mastery.md (Lasso, Ridge, ElasticNet, Stacking)
      │   └─ references/time-series-forecasting.md
      │
      ├─ Gradient Boosting
      │   └─ references/gradient-boosting-mastery.md (XGBoost, LightGBM, CatBoost)
      │
      ├─ Statistical Testing
      │   └─ references/statistical-testing-methods.md (t-test, ANOVA, chi-square, effect size)
      │
      ├─ Clustering / Segmentation (unlabeled)
      │   └─ references/clustering.md
      │
      ├─ NLP / Text
      │   └─ references/nlp.md
      │
      ├─ Computer Vision (images)
      │   ├─ references/deep-learning-cnn.md
      │   └─ references/transfer-learning.md
      │
      ├─ Recommender Systems
      │   └─ references/recommender-systems.md
      │
      ├─ Anomaly Detection (unsupervised)
      │   └─ references/anomaly-detection.md
      │
      └─ Supply Chain / Inventory
          └─ references/supply-chain-analysis.md
```

## Cross-Cutting Skills

Loaded AFTER modeling, for improving results:

| Skill | When | Reference |
|-------|------|-----------|
| Model Validation | MAE/MSE/RMSE, overfitting detection, validation curves | `references/model-validation.md` |
| Ensemble Methods | Multiple models trained, want to combine | `references/ensemble-advanced.md` |
| Pipeline Best Practices | CV setup, overfitting diagnosis, sklearn Pipeline | `references/ml-pipeline-best-practices.md` |

## Pipeline Architecture & Skill Management

This umbrella orchestrates 19 data-science skills in a 3-stage pipeline. For the full architecture diagram, skill persistence mechanism (auto-restore from `.archive`), and per-skill source attribution, see:

→ `references/pipeline-architecture.md`

Key: after Hermes updates, skills may be auto-archived. The `restore_skills.sh` cron job (every 30 min) ensures they return to active status.

## Reference Files

| File | Topic | Source |
|------|-------|--------|
| `references/data-cleaning.md` | Missing values, outliers, duplicates, type fixing | Kaggle 15K+ votes |
| `references/feature-engineering.md` | Text extraction, combinations, encoding, scaling, datetime | Kaggle Top 4% |
| `references/feature-selection.md` | Filter/Wrapper/Embedded methods, SHAP | Kaggle classic |
| `references/classification.md` | 8-model CV, GridSearch, learning curves, voting ensemble | Kaggle Titanic Top 4% |
| `references/imbalanced-classification.md` | Undersampling, SMOTE, threshold tuning, PR curves, class weights | Kaggle fraud detection |
| `references/multi-model-comparison.md` | SMOTETomek + DT/XGB/RF parallel comparison | Teen mental health |
| `references/clustering.md` | K-Means, elbow method, silhouette, radar charts, profiling | Kaggle 3.3K votes |
| `references/nlp.md` | TF-IDF, word embeddings, LSTM/GRU, grid search, ensembling | Kaggle 5K+ votes |
| `references/deep-learning-cnn.md` | CNN architecture, data augmentation, LR scheduling, confusion matrix | Kaggle 16.5K votes (99.7%) |
| `references/transfer-learning.md` | VGG16/ResNet/EfficientNet, feature extraction vs fine-tuning | Kaggle 3K votes |
| `references/recommender-systems.md` | Simple/content/collaborative/hybrid recommenders, SVD, cosine similarity | Kaggle 5.3K votes |
| `references/anomaly-detection.md` | Isolation Forest, LOF, One-Class SVM, DBSCAN, profiling | Kaggle 1.4K votes |
| `references/time-series-forecasting.md` | Prophet, seasonality, holidays, changepoint tuning, error metrics | Kaggle 1.7K votes |
| `references/supply-chain-analysis.md` | Stock flow, turnover ratios, inflow calculation, plant benchmarking | Real-world warehouse |
| `references/ensemble-advanced.md` | Stacking, Blending, Voting, weighted ensembles, correlation analysis | Kaggle 15.4K votes |
| `references/model-validation.md` | MAE/MSE/RMSE, overfitting detection, validation curves, train/test gap | Kaggle Model Validation 7K votes |
| `references/ml-pipeline-best-practices.md` | CV strategies, overfitting diagnosis, sklearn Pipeline, model selection tree | Kaggle 12K+ votes |
| `references/random-forest-mastery.md` | RandomForestClassifier/Regressor, n_estimators/max_depth tuning, feature importance, OOB score | Kaggle 4.5K votes |
| `references/regression-modeling-mastery.md` | Lasso, Ridge, ElasticNet, KernelRidge, XGB/LGBM Regressors, Stacking, log transform, RMSLE | Kaggle 14.3K votes |
| `references/gradient-boosting-mastery.md` | XGBoost/XGBClassifier, early_stopping, learning_rate, LightGBM, CatBoost, 3-library comparison | Kaggle 3.7K votes |
| `references/statistical-testing-methods.md` | Descriptive stats, t-tests, ANOVA, chi-square, effect size, normality tests, decision framework | Kaggle 9K votes |

## Standalone Companion Skill

- **`enhanced-exploratory-data-analysis`** — Comprehensive EDA with data orientation protocol, health scoring, business insights, PPTX generation, font configuration. This is the most-developed skill in the collection (7 views, 4 patches). Load it for any EDA task; it has its own `references/` and `templates/`.

## Skill Origin

All 30 skills in this collection were generated by the **`kaggle-notebook-extraction`** skill — a systematic workflow that pulls Kaggle notebook source code via the public API, performs deep cell-by-cell analysis, and distills patterns into reusable skill content. 22 were directly extracted from notebooks, 2 were synthesized from 403-locked notebooks, and 3 are custom/composite skills. Load `kaggle-notebook-extraction` when you need to discover, extract, and distill new data science patterns from Kaggle.

## Quick Start

```bash
# View the umbrella
skill_view data-science-pipeline

# Load a specific reference
skill_view data-science-pipeline --file references/classification.md

# Start with EDA (separate skill)
skill_view enhanced-exploratory-data-analysis

# Generate new skills from Kaggle notebooks
skill_view kaggle-notebook-extraction
```

## Quality Checklist (umbrella-level)

- [ ] EDA completed before modeling? (data orientation confirmed)
- [ ] Data cleaned before feature engineering?
- [ ] Feature selection applied if >50 features?
- [ ] Correct modeling technique chosen for the problem type?
- [ ] Cross-validation used (not single train/test split)?
- [ ] Overfitting checked (learning curves, train/val gap)?
- [ ] Ensemble methods considered if base models are diverse?
- [ ] Results interpretable for the business stakeholder?

