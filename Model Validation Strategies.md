---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: model-validation-strategies
---

# Model Validation Strategies

**分类:** 📊 Data Science
**Skill ID:** `model-validation-strategies`

> Model validation strategies — MAE/MSE/RMSE, in-sample vs out-of-sample, overfitting detection, validation curves. Distilled from Kaggle dansbecker/model-validation (7,040 votes).

---


# Model Validation Strategies

## Purpose
Systematic approach to validating ML models — from basic error metrics to overfitting detection. Distilled from Dan Becker's tutorial (7,040 votes) with expanded validation methods.

## 1. The Fundamental Problem

**In-sample score ≠ real performance.** The notebook's key demonstration:

```python
# In-sample (WRONG approach — deceptively optimistic)
predicted = model.predict(X_train)  # Same data used for training!
mae_in_sample = mean_absolute_error(y_train, predicted)  # ~250
# vs
# Out-of-sample (CORRECT)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
model.fit(X_train, y_train)
preds = model.predict(X_val)
mae_out_sample = mean_absolute_error(y_val, preds)  # ~250,000 (1000x worse!)
```

## 2. Error Metrics Quick Reference

| Metric | Formula | When to Use | sklearn |
|--------|---------|-------------|---------|
| MAE | mean(\|actual - pred\|) | Interpretable, same unit as target | mean_absolute_error |
| MSE | mean((actual - pred)²) | Penalizes large errors heavily | mean_squared_error |
| RMSE | sqrt(MSE) | Same unit, penalizes outliers | sqrt(mean_squared_error) |
| R² | 1 - SS_res/SS_tot | % variance explained | r2_score |
| MAPE | mean(\|(actual-pred)/actual\|) | Percentage error (business-friendly) | Custom |

## 3. Overfitting Detection

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

model.fit(X_train, y_train)

train_mae = mean_absolute_error(y_train, model.predict(X_train))
val_mae = mean_absolute_error(y_val, model.predict(X_val))

print(f"Train MAE: {train_mae}")
print(f"Val MAE:   {val_mae}")
print(f"Gap:       {val_mae - train_mae}")

# Interpretation:
# Gap < 10% of val_mae → model generalizes well
# Gap 10-30%            → mild overfitting (consider regularization)
# Gap > 30%             → severe overfitting (reduce complexity)
```

## 4. Validation Curve

```python
from sklearn.model_selection import validation_curve
import numpy as np

param_range = np.arange(1, 20)
train_scores, val_scores = validation_curve(
    RandomForestRegressor(random_state=0),
    X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5,
    scoring="neg_mean_absolute_error"
)

train_mean = -train_scores.mean(axis=1)
val_mean = -val_scores.mean(axis=1)

# Plot
plt.plot(param_range, train_mean, label='Train MAE', marker='o')
plt.plot(param_range, val_mean, label='Val MAE', marker='o')
plt.axvline(param_range[np.argmin(val_mean)], color='red', linestyle='--', label='Optimal')
plt.xlabel('max_depth')
plt.ylabel('MAE')
plt.legend()
```

## 5. Regression Metrics Template

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_regression(model, X_train, X_val, y_train, y_val):
    """Comprehensive regression evaluation."""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    metrics = {
        'Train MAE': mean_absolute_error(y_train, y_train_pred),
        'Val MAE':   mean_absolute_error(y_val, y_val_pred),
        'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'Val RMSE':   np.sqrt(mean_squared_error(y_val, y_val_pred)),
        'Train R²': r2_score(y_train, y_train_pred),
        'Val R²':   r2_score(y_val, y_val_pred),
    }
    
    overfit_ratio = metrics['Val MAE'] / max(metrics['Train MAE'], 1e-10)
    metrics['Overfit Ratio'] = overfit_ratio
    
    for k, v in metrics.items():
        print(f"{k:15s}: {v:10.4f}")
    
    if overfit_ratio > 1.5:
        print("⚠️  SEVERE OVERFITTING DETECTED!")
    elif overfit_ratio > 1.2:
        print("⚠️  Mild overfitting — consider regularization")
    
    return metrics
```

## 6. Classification Metrics Template

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def evaluate_classification(model, X_train, X_val, y_train, y_val):
    """Comprehensive classification evaluation."""
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    print("="*50)
    print(f"{'Training Set':^50}")
    print("="*50)
    print(classification_report(y_train, y_train_pred))
    
    print("="*50)
    print(f"{'Validation Set':^50}")
    print("="*50)
    print(classification_report(y_val, y_val_pred))
    print(confusion_matrix(y_val, y_val_pred))
    
    # Overfitting check
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    gap = train_acc - val_acc
    print(f"\nAccuracy Gap: {gap:.4f} {'⚠️ OVERFITTING' if gap > 0.05 else '✅ OK'}")
```

## 7. Common Pitfalls

| Pitfall | Consequence | Prevention |
|---------|-------------|------------|
| Evaluating on training data | Hugely optimistic scores | Always use held-out validation set |
| Using accuracy for imbalanced data | 99% accuracy on 99% majority class | Use precision/recall/F1 |
| Data leakage in preprocessing | Validation contamination | Preprocessing AFTER split |
| Single random split | High variance in estimates | Cross-validation |

## Trigger Keywords
中文: 模型验证, 过拟合检测, MAE, 评估指标, 验证曲线
English: model validation, overfitting, MAE, evaluation metrics, validation curve

