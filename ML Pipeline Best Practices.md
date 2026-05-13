---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: ml-pipeline-best-practices
---

# ML Pipeline Best Practices

**分类:** 📊 Data Science
**Skill ID:** `ml-pipeline-best-practices`

> 机器学习最佳实践合集，覆盖交叉验证策略、过拟合诊断、Pipeline 自动化、和模型选择决策树。提炼自 Kaggle 官方教程 (Dan Becker, 累计 12K+ votes)。

---


# ML 最佳实践 (Pipeline & Best Practices)

## Purpose
掌握机器学习工程中的关键实践：正确的交叉验证、过拟合诊断与修复、sklearn Pipeline 自动化、和模型选择决策。这是避免"看起来很好、上线就崩"的核心。

## 1. Cross-Validation Done Right

### 为什么不用单次 train_test_split？
```python
# ❌ 错误：单次分割，结果不稳定
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # 高度依赖运气！

# ✅ 正确：5-fold CV
from sklearn.model_selection import cross_val_score, StratifiedKFold
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring='accuracy')
print(f"CV: {scores.mean():.3f} +/- {scores.std()*2:.3f}")  # 95% CI
```

### CV 策略选择
```python
# 分类：StratifiedKFold（保持类别比例）
skf = StratifiedKFold(n_splits=5, shuffle=True)

# 回归/大样本：KFold
kf = KFold(n_splits=5, shuffle=True)

# 时间序列：TimeSeriesSplit（保持时间顺序）
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# 小样本：LeaveOneOut（很慢但无偏）
from sklearn.model_selection import LeaveOneOut
```

## 2. Overfitting & Underfitting

### 诊断三信号
```python
# Signal 1: Train vs Validation gap
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
# Gap > 5% → overfitting
# Both low → underfitting

# Signal 2: Learning Curve
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)
# Converging? → good fit. Diverging? → overfitting.

# Signal 3: Cross-validation variance
# High variance across folds → overfitting
```

### 修复指南
```python
# Overfitting → 降低模型复杂度
model = DecisionTreeClassifier(max_depth=5)        # 限制深度
model = RandomForestClassifier(min_samples_leaf=10) # 限制叶节点
model = LogisticRegression(C=0.1)                  # 强正则化

# Underfitting → 增加模型复杂度
model = DecisionTreeClassifier(max_depth=None)     # 放开深度
model = RandomForestClassifier(n_estimators=300)   # 更多树
# 或加特征：feature engineering!
```

## 3. Pipeline — 防泄漏 + 自动化

### Why Pipeline?
```python
# ❌ 错误：先标准化再 split → 数据泄露！
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # 用了 test 的信息！
X_train, X_test = train_test_split(X_scaled, y)

# ✅ 正确：Pipeline 保证 fit 只在 train 上
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)  # scaler 只在 train 上 fit
pipe.score(X_test, y_test)
```

### ColumnTransformer — 混合类型
```python
from sklearn.compose import ColumnTransformer

num_cols = ['Age', 'Fare']
cat_cols = ['Sex', 'Embarked']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols),
])

pipe = Pipeline([
    ('preprocess', preprocessor),
    ('model', RandomForestClassifier()),
])
pipe.fit(X_train, y_train)
```

### GridSearch + Pipeline
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [3, 5, 7, None],
}
gs = GridSearchCV(pipe, param_grid, cv=5)
gs.fit(X_train, y_train)
print(f"Best: {gs.best_params_}, Score: {gs.best_score_:.3f}")
```

## 4. Model Selection Decision Tree

```
数据量 < 1000?
  ├─ Yes → LogisticRegression 或 DecisionTree（防止过拟合）
  └─ No → 往下
       ├─ 特征 < 20?
       │   ├─ Yes → LogisticRegression, SVC, RandomForest
       │   └─ No → 往下
       └─ 特征 > 100? → XGBoost / RandomForest（内置特征选择）
              └─ 极度高维? → L1 LogisticRegression / XGBoost
```

## Common Mistakes

1. **不 shuffle 分类 CV** → 可能某 fold 没有某类
2. **Pipeline 里放 fit_transform 再后续手动 transform** → 用 Pipeline 包起来
3. **GridSearch 的 scoring 与业务指标不一致** → 用 scoring='recall' 而非 'accuracy'（不平衡时）
4. **学习曲线只看一条线** → 必须同时看 train 和 val
5. **Pipeline 忘记列名** → ColumnTransformer 默认返回 numpy array，用 `set_config(transform_output='pandas')`

## Trigger Phrases
- "交叉验证"、"过拟合"、"Pipeline"
- "数据泄露"、"最佳实践"
- "模型选择"、"learning curve"

