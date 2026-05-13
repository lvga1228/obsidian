---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: random-forest-mastery
---

# Random Forest Mastery

**分类:** 📊 Data Science
**Skill ID:** `random-forest-mastery`

> Random Forest mastery — classifier vs regressor, hyperparameter tuning, feature importance, OOB score, and when to choose RF over single trees. Distilled from Kaggle dansbecker/random-forests (4,522 votes).

---


# Random Forest Mastery

## Purpose
Comprehensive Random Forest guide covering both RandomForestClassifier and RandomForestRegressor. Distilled from Dan Becker's Kaggle tutorial (4,522 votes) with expanded practical patterns.

## Core Concept
Random Forest = ensemble of Decision Trees trained on bootstrap samples with random feature subsets. Key insight: **many mediocre trees voting together** outperform a single excellent tree.

## 1. RandomForestRegressor (from notebook)

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, preds))
```

## 2. RandomForestClassifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,        # No limit by default
    min_samples_split=2,   # Minimum samples to split a node
    min_samples_leaf=1,    # Minimum samples in leaf node
    max_features='sqrt',   # #features per split (sqrt for classification)
    bootstrap=True,        # Bootstrap sampling
    oob_score=True,        # Out-of-bag score (free validation!)
    random_state=0,
    n_jobs=-1              # Parallel training
)
model.fit(X_train, y_train)
print(f"OOB Score: {model.oob_score_:.4f}")
```

## 3. Hyperparameter Tuning Priority

| Priority | Parameter | Effect | Guideline |
|----------|-----------|--------|-----------|
| 1 | n_estimators | More trees = more stable (diminishing returns after ~200) | Start at 100, try 200-500 |
| 2 | max_depth | Controls overfitting | None first, then tune 5-30 |
| 3 | min_samples_split | Higher = less overfitting | Default 2, try 5-20 |
| 4 | min_samples_leaf | Higher = smoother boundaries | Default 1, try 2-10 |
| 5 | max_features | Diversity of trees | 'sqrt' for classif, 0.33 for reg |

## 4. Feature Importance

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values('importance', ascending=False)

# Bar plot
feature_importance_df.head(15).plot.barh(x='feature', y='importance')
plt.title('Random Forest Feature Importance')
plt.show()
```

## 5. OOB Score (Free Built-in Validation)

OOB = Out-of-Bag. Each tree is trained on ~63% of data; the remaining ~37% is "out of bag" and serves as free validation.

```python
model = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=0)
model.fit(X_train, y_train)
print(f"OOB Accuracy: {model.oob_score_:.4f}")

# Compare with test set accuracy
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, model.predict(X_test))
print(f"Test Accuracy: {test_acc:.4f}")
```

Rule of thumb: OOB score close to test score → model generalizes well. OOB >> test → possible data leakage.

## 6. When to Use Random Forest

| Scenario | Random Forest | Single Decision Tree | XGBoost |
|----------|:---:|:---:|:---:|
| Small dataset (<1000 rows) | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| Medium dataset (1K-50K) | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| Highly imbalanced | ⭐ | ⭐ | ⭐⭐⭐ |
| Interpretability needed | ⭐⭐ | ⭐⭐⭐ | ⭐ |
| Mixed categorical/numeric | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| Baseline model | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

## 7. Complete Training Template

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# --- Step 1: Baseline ---
rf = RandomForestClassifier(n_estimators=100, random_state=0, oob_score=True)
rf.fit(X_train, y_train)
print(f"Baseline OOB: {rf.oob_score_:.4f}")
print(f"Baseline Test: {rf.score(X_test, y_test):.4f}")

# --- Step 2: CV Check ---
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='accuracy')
print(f"5-fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# --- Step 3: Hyperparameter Tuning ---
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

grid = GridSearchCV(
    RandomForestClassifier(random_state=0),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")

# --- Step 4: Final Evaluation ---
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# --- Step 5: Feature Importance ---
importances = best_model.feature_importances_
for feat, imp in sorted(zip(X_train.columns, importances), key=lambda x: -x[1])[:10]:
    print(f"  {feat}: {imp:.4f}")
```

## 8. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Too few estimators | High variance between runs | n_estimators >= 100 |
| max_depth too large | Overfitting (train>>test) | Reduce max_depth or increase min_samples_leaf |
| max_features=1.0 | Trees correlated, no ensemble benefit | Use 'sqrt' or 0.3-0.5 |
| Not setting random_state | Results not reproducible | Always set random_state=0 |
| Ignoring OOB | Wasting free validation signal | Set oob_score=True |

## Trigger Keywords
中文: 随机森林, RF模型, 特征重要性, OOB, 集成树
English: random forest, feature importance, ensemble trees, OOB score

