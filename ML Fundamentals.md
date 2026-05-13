---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: ml-fundamentals
---

# ML Fundamentals

**分类:** 📊 Data Science
**Skill ID:** `ml-fundamentals`

> ML fundamentals — your first model, train/test split, fit/predict/evaluate lifecycle. Synthesized from Kaggle danbecker/your-first-ML-model (403 locked) and core ML knowledge.

---


# Machine Learning Fundamentals

## Purpose
The absolute entry point to ML — building your first predictive model, understanding the fit/predict/evaluate cycle, and avoiding the most common beginner mistakes.

## 1. The ML Model Lifecycle (5 Steps)

```
1. Load Data    → pd.read_csv()
2. Select Features & Target → X and y
3. Split Data   → train_test_split()
4. Fit Model    → model.fit(X_train, y_train)
5. Predict & Evaluate → model.predict() + metric
```

## 2. Your First Model (Complete Recipe)

```python
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 1: Load
data = pd.read_csv('your_data.csv')

# Step 2: Select
feature_columns = ['Rooms', 'Bathroom', 'Landsize', 'Latitude', 'Longitude']
X = data[feature_columns]
y = data['Price']

# Step 3: Split (THE MOST IMPORTANT LINE!)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

# Step 4: Fit
model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)

# Step 5: Predict & Evaluate
predictions = model.predict(X_val)
mae = mean_absolute_error(y_val, predictions)

print(f"Validation MAE: ${mae:,.0f}")
print(f"Average house price: ${y.mean():,.0f} (for context)")
```

## 3. Core Concepts Explained

### Features (X) vs Target (y)
- **Features (X)**: What you use to make predictions (independent variables)
- **Target (y)**: What you want to predict (dependent variable)
- Shape: X is 2D (DataFrame), y is 1D (Series)

### Train vs Test Split
- **Training data**: Used to fit the model (teach it patterns)
- **Validation data**: Used to evaluate how well it learned (unseen data)
- **Rule**: Model should NEVER see validation data during training

### random_state
- Fixes the random split so results are reproducible
- Any integer works; 0 or 42 are conventions
- Change it to test sensitivity of your results

## 4. Common First-Model Mistakes

| Mistake | Why It's Bad | Fix |
|---------|-------------|-----|
| Forgetting train_test_split | Overfitting — you test on training data | Always split |
| Including ID columns in X | Noise — meaningless features | Drop IDs, names |
| Using categorical strings directly | sklearn needs numbers | Encode first |
| Comparing to y.mean() as baseline | Forgot to check if model beats average | Always check baseline |
| Interpreting MAE without context | $250K error on $1M houses ≠ $250K on $100K houses | Report relative error |

## 5. Decision Tree Intuition

A Decision Tree makes predictions by asking a series of yes/no questions:

```
Is Rooms <= 3?
  ├─ Yes → Is Landsize <= 500?
  │         ├─ Yes → Predict $350,000
  │         └─ No  → Predict $450,000
  └─ No  → Is Latitude <= -37.8?
            ├─ Yes → Predict $600,000
            └─ No  → Predict $800,000
```

## 6. What's Next After Your First Model

1. Try a different model: `RandomForestRegressor()` instead of `DecisionTreeRegressor()`
2. Tune hyperparameters: `max_depth`, `min_samples_leaf`
3. Use more/different features
4. Apply proper validation: cross-validation

## Trigger Keywords
中文: 第一个模型, ML入门, 预测模型, 训练测试分割, 决策树
English: first ML model, beginner ML, predict, fit, train test split

