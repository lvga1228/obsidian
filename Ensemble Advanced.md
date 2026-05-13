---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: ensemble-advanced
---

# Ensemble Advanced

**分类:** 📊 Data Science
**Skill ID:** `ensemble-advanced`

> 高级集成学习框架，覆盖 Stacking、Blending、Voting 和多层集成策略，包含特征工程到最终预测的完整竞赛级流程。提炼自 Kaggle 15.4K votes 经典 notebook。

---


# 高级集成学习 (Advanced Ensemble Methods)

## Purpose
超越简单的 VotingClassifier，掌握 Stacking 和 Blending 等竞赛级集成策略，将多个异构模型的优势组合以突破单一模型的性能上限。

## Core Workflow

### Step 1: Base Models
```python
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# 5 diverse base models
base_models = {
    'rf': RandomForestClassifier(n_estimators=500, max_depth=5),
    'et': ExtraTreesClassifier(n_estimators=500, max_depth=5),
    'gb': GradientBoostingClassifier(n_estimators=500, max_depth=3),
    'xgb': XGBClassifier(n_estimators=500, max_depth=3),
    'svc': SVC(kernel='rbf', probability=True),
}
```

### Step 2: Simple Voting
```python
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    voting='soft'  # Probability averaging (usually better than hard)
)
voting.fit(X_train, y_train)
```

### Step 3: Stacking (Meta-Learning)
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier

# Out-of-fold predictions → meta-model
stack = StackingClassifier(
    estimators=[(name, model) for name, model in base_models.items()],
    final_estimator=LogisticRegression(),  # Simple meta-model
    cv=StratifiedKFold(5),                # 5-fold for OOF predictions
    stack_method='predict_proba',          # Use probabilities, not classes
)
stack.fit(X_train, y_train)
print(f"Stacking: {stack.score(X_test, y_test):.4f}")
```

### Step 4: Manual Blending (Multi-Level)
```python
# Level 1: Train base models on fold-1, predict on fold-2
# Level 2: Use predictions as features for meta-model

from sklearn.model_selection import KFold

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

# Container for out-of-fold predictions
oof_preds = np.zeros((X_train.shape[0], len(base_models)))
test_preds = np.zeros((X_test.shape[0], len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    fold_preds = np.zeros(X_train.shape[0])
    fold_test = np.zeros((X_test.shape[0], n_folds))
    
    for j, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr, y_tr)
        fold_preds[val_idx] = model_clone.predict_proba(X_val)[:, 1]
        fold_test[:, j] = model_clone.predict_proba(X_test)[:, 1]
    
    oof_preds[:, i] = fold_preds
    test_preds[:, i] = fold_test.mean(axis=1)

# Meta-model trained on OOF predictions
meta = LogisticRegression()
meta.fit(oof_preds, y_train)
final_preds = meta.predict_proba(test_preds)[:, 1]
```

### Step 5: Correlation Analysis Between Models
```python
# Stack diverse models — high correlation = less benefit
preds_df = pd.DataFrame(oof_preds, columns=base_models.keys())
sns.heatmap(preds_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Model Prediction Correlations (lower = better for stacking)')
plt.show()
```

### Step 6: Weighted Ensemble
```python
# Assign weights based on individual model CV scores
weights = {'rf': 0.15, 'et': 0.15, 'gb': 0.20, 'xgb': 0.35, 'svc': 0.15}
weighted_pred = np.zeros(len(X_test))
for name, model in base_models.items():
    proba = model.predict_proba(X_test)[:, 1]
    weighted_pred += weights[name] * proba
```

## Key Insight: Why Stacking Works
- **Bias-Variance tradeoff**: Base models have different biases; meta-model learns to correct them
- **Diversity**: Combine tree-based (low bias/high variance) with linear (high bias/low variance)
- **OOF predictions**: Critical to avoid data leakage — meta-model must see predictions from unseen data

## Common Pitfalls
1. Stacking highly correlated models → minimal improvement
2. Using same CV splits for all base models → use consistent fold assignments
3. Overly complex meta-model → LogisticRegression is usually sufficient
4. Not normalizing predictions before meta-model → scale mismatch

## Trigger Phrases
- "集成学习"、"模型融合"、"stacking"、"blending"、"ensemble"

