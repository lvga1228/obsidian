---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: overfitting-underfitting-diagnostics
---

# Overfitting Underfitting Diagnostics

**分类:** 📊 Data Science
**Skill ID:** `overfitting-underfitting-diagnostics`

> Systematic diagnosis of overfitting and underfitting — train/test gap analysis, learning curves, bias-variance tradeoff. Synthesized from Kaggle dansbecker/underfitting-overfitting (403 locked) and ML theory.

---


# Overfitting & Underfitting Diagnostics

## Purpose
Systematic diagnosis of model fit quality — detect when a model is too simple (underfitting) or memorizing noise (overfitting), and apply targeted fixes.

## 1. The Three States

```
Underfitting          Good Fit            Overfitting
  (high bias)                              (high variance)

Train: bad           Train: good          Train: excellent
Val:   bad           Val:   good          Val:   bad
       ↓                    ↓                    ↓
  Pattern: both          Pattern: small        Pattern: large
  scores are low         gap between           gap, train >> val
                         train and val
```

## 2. Diagnostic Code

```python
def diagnose_fit(model, X_train, X_val, y_train, y_val):
    """Diagnose whether model is underfitting, good, or overfitting."""
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    gap = train_score - val_score
    
    print(f"Train Score: {train_score:.4f}")
    print(f"Val Score:   {val_score:.4f}")
    print(f"Gap:         {gap:.4f}")
    
    if train_score < 0.6 and val_score < 0.6:
        diagnosis = "UNDERFITTING — try more complexity or better features"
    elif gap > 0.1:
        diagnosis = "OVERFITTING — try regularization or reduce complexity"
    else:
        diagnosis = "GOOD FIT — model generalizes well"
    
    print(f"\nDiagnosis: {diagnosis}")
    return train_score, val_score, gap
```

## 3. Learning Curves (The Gold Standard)

```python
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='neg_mean_absolute_error'
)

train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)
train_std = -train_scores.std(axis=1)
val_std = -val_scores.std(axis=1)

plt.plot(train_sizes, train_mean, 'o-', label='Training error')
plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.2)
plt.plot(train_sizes, val_mean, 'o-', label='Validation error')
plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.2)
plt.xlabel('Training Set Size')
plt.ylabel('MAE')
plt.legend()
plt.title('Learning Curve')
```

**Interpreting Learning Curves:**

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both curves high + converging | Underfitting | Add features, reduce regularization |
| Gap widening with more data | Overfitting | More data / regularization |
| Train high, val low = IMPOSSIBLE | Data leakage! | Check preprocessing pipeline |
| Both low + converged | Good fit | Done! |

## 4. Overfitting: Symptoms & Fixes

**Symptoms:**
- Train accuracy >> validation accuracy (gap > 10%)
- Validation loss increases while training loss decreases
- Model does well on training data but poorly on new data

**Fixes (ordered by impact):**

| Fix | What It Does | When to Use |
|-----|-------------|-------------|
| More training data | Harder to memorize | Always if available |
| Reduce model complexity | Less capacity to memorize | max_depth, n_estimators |
| Regularization (L1/L2) | Penalizes large weights | LogisticRegression, SVM |
| Early stopping | Stop before overfitting | Neural nets, GBM |
| Dropout | Randomly disable neurons | Neural networks |
| Feature selection | Remove noisy features | Many irrelevant features |
| Data augmentation | Create synthetic variations | Image/text data |

## 5. Underfitting: Symptoms & Fixes

**Symptoms:**
- Both train and validation scores are low
- Model cannot capture even training patterns
- High bias — systematic errors

**Fixes:**

| Fix | What It Does |
|-----|-------------|
| More features | Give model more information |
| Increase model complexity | More depth, trees, layers |
| Reduce regularization | Loosen constraints |
| Feature engineering | Create interaction terms |
| Better model architecture | Switch from linear to tree-based |
| Collect more relevant data | Better signal-to-noise |

## 6. max_depth as Overfitting Control

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def plot_max_depth_effect(X_train, X_val, y_train, y_val):
    depths = range(1, 20)
    train_mae, val_mae = [], []
    
    for depth in depths:
        model = DecisionTreeRegressor(max_depth=depth, random_state=0)
        model.fit(X_train, y_train)
        train_mae.append(mean_absolute_error(y_train, model.predict(X_train)))
        val_mae.append(mean_absolute_error(y_val, model.predict(X_val)))
    
    plt.plot(depths, train_mae, label='Train MAE')
    plt.plot(depths, val_mae, label='Val MAE')
    plt.axvline(depths[np.argmin(val_mae)], color='red', linestyle='--')
    plt.xlabel('max_depth')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('Optimal max_depth: {}'.format(depths[np.argmin(val_mae)]))

# Typical result: val MAE drops from depth 1→5, then rises after depth ~8 (overfitting)
```

## 7. Quick Decision Flowchart

```
Is model doing well on training data?
  ├─ NO  → UNDERFITTING
  │        └─ Add features / increase complexity / change model
  │
  └─ YES → Is model doing well on validation data?
           ├─ NO  → OVERFITTING
           │        └─ Regularize / reduce complexity / get more data
           │
           └─ YES → Keep testing on truly unseen data
                    └─ Good fit! Deploy or refine.
```

## Trigger Keywords
中文: 过拟合, 欠拟合, 学习曲线, 偏差方差, 模型诊断
English: overfitting, underfitting, learning curve, bias variance, model diagnostics

