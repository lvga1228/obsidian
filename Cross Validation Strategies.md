---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: cross-validation-strategies
---

# Cross Validation Strategies

**分类:** 📊 Data Science
**Skill ID:** `cross-validation-strategies`

> 交叉验证方法论，覆盖 k-fold、holdout 验证、cross_val_score 使用、CV 与 train_test_split 的权衡。提炼自 Kaggle 经典 Cross-Validation notebook (Dan Becker, 1072 votes)。

---


# 交叉验证策略 (Cross-Validation Strategies)

## Purpose
系统性地评估模型性能，通过多次划分数据获得更可靠的模型质量度量。解决单次 train_test_split 的随机性陷阱。

## When to use
- 数据集较小（< 10,000 行），单次划分不可靠
- 需要对比多个模型或特征组合
- 模型训练快速（< 2 分钟/次）
- 用户说"怎么评估模型"、"交叉验证"、"模型得分不稳定"
- 做超参数调优前的模型筛选

---

## 1. 为什么需要交叉验证 — Train/Test Split 的局限性

### 单次划分的随机性陷阱

```python
# 问题演示：不同 random_state 得到完全不同的结论
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

model_a = RandomForestRegressor(n_estimators=10, random_state=0)
model_b = RandomForestRegressor(n_estimators=50, random_state=0)

# 不同 random_state 可能导致模型排名反转！
for seed in [0, 1, 2]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed)
    score_a = mean_absolute_error(y_te, model_a.fit(X_tr, y_tr).predict(X_te))
    score_b = mean_absolute_error(y_te, model_b.fit(X_tr, y_tr).predict(X_te))
    print(f"seed={seed} | Model A: {score_a:.3f} | Model B: {score_b:.3f} | Winner: {'A' if score_a < score_b else 'B'}")
```

**核心问题：**
- 5000 行数据，20% 测试集 = 1000 行 — 模型可能在这 1000 行上运气好，换 1000 行就糟糕
- 测试集越大 → 度量越可靠，但→ 训练集越小 → 模型越差
- 小数据集上的"最佳"建模决策，往往不是大数据集上的最佳决策

---

## 2. K-Fold 交叉验证 — 核心机制

### 原理
将数据分成 K 等份（folds），每次用 1 份做验证、K-1 份做训练，重复 K 次取平均。

```
Fold 1: [Test] [Train] [Train] [Train] [Train] → Score₁
Fold 2: [Train] [Test] [Train] [Train] [Train] → Score₂
Fold 3: [Train] [Train] [Test] [Train] [Train] → Score₃
Fold 4: [Train] [Train] [Train] [Test] [Train] → Score₄
Fold 5: [Train] [Train] [Train] [Train] [Test] → Score₅

Final Score = mean(Score₁, Score₂, Score₃, Score₄, Score₅)
```

**关键优势：** 100% 的数据都做过验证集（虽然不是同时），5000 行数据最终得到基于 5000 行的模型质量度量。

### 标准用法：cross_val_score

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer  # sklearn ≥ 0.20
# from sklearn.preprocessing import Imputer  # sklearn < 0.20

# ⚠️ 必须使用 Pipeline！否则数据泄露
pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0))

# 默认 5-fold CV
scores = cross_val_score(pipeline, X, y, 
                          scoring='neg_mean_absolute_error',
                          cv=5)

print(f"Scores per fold: {scores}")
print(f"Mean MAE: {-scores.mean():.2f}  (+/- {scores.std():.2f})")
```

### scoring 参数说明

scikit-learn 约定：**所有指标越高越好**。因此回归指标加负号：

| 实际问题 | 原生指标 | scoring 参数 | 解读 |
|---------|---------|-------------|------|
| 回归 MAE | 越小越好 | `'neg_mean_absolute_error'` | 取负数后取反 `-scores.mean()` |
| 回归 MSE | 越小越好 | `'neg_mean_squared_error'` | 取负数后取反 |
| 回归 R² | 越大越好 | `'r2'` | 直接使用 |
| 分类 Accuracy | 越大越好 | `'accuracy'` | 直接使用 |
| 分类 AUC | 越大越好 | `'roc_auc'` | 直接使用 |
| 分类 F1 | 越大越好 | `'f1'` / `'f1_macro'` | 直接使用 |

```python
# 常用 scoring 速查
# 回归
cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
cross_val_score(model, X, y, scoring='r2', cv=5)
cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)

# 分类
cross_val_score(model, X, y, scoring='accuracy', cv=5)
cross_val_score(model, X, y, scoring='roc_auc', cv=5)
cross_val_score(model, X, y, scoring='f1', cv=5)
cross_val_score(model, X, y, scoring='f1_macro', cv=5)  # 多分类
```

---

## 3. Holdout 验证 vs K-Fold — 何时用哪个

### 决策矩阵

| 因素 | 用 Holdout (train_test_split) | 用 K-Fold CV |
|------|-------------------------------|--------------|
| 数据量 | > 100,000 行 | < 10,000 行 |
| 模型训练速度 | > 5 分钟 | < 2 分钟 |
| 决策重要性 | 快速原型 | 模型选型/论文 |
| fold 间得分一致性 | — | 各 fold 相近则 holdout 也 OK |
| 时间序列数据 | ✅ (需时间分割) | ⚠️ 需 TimeSeriesSplit |

### 实用经验法则
- **小数据集 + 快速模型 → CV**（额外计算负担可忽略）
- **大数据集 + 慢模型 → Holdout**（速度快，数据多已足够可靠）
- **不确定？** → 跑一次 CV，看各 fold 标准差。如果 fold 间得分接近 (std/mean < 5%)，holdout 也足够

```python
# 决策辅助代码
scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
cv_mean, cv_std = -scores.mean(), scores.std()
print(f"CV MAE: {cv_mean:.3f} (±{cv_std:.3f})")
print(f"CV%: {cv_std/cv_mean*100:.1f}%")

if cv_std/cv_mean < 0.05:
    print("→ Fold 间一致，单次 holdout 也可接受")
else:
    print("→ Fold 间波动大，建议继续使用 CV")
```

---

## 4. 进阶 CV 策略

### Stratified K-Fold（分类问题标配）
保持每个 fold 中类别比例与整体一致，**分类问题默认使用**。

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
```

### Repeated K-Fold（进一步降噪）
重复 N 次不同随机划分的 K-Fold，取总平均。

```python
from sklearn.model_selection import RepeatedKFold, cross_val_score

rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
scores = cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_absolute_error')
print(f"MAE: {-scores.mean():.2f} ± {scores.std():.2f} (across {len(scores)} folds)")
```

### TimeSeriesSplit（时间序列专用）
保持时间顺序，训练集总是在验证集之前——禁止未来信息泄露。

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
```

### Group K-Fold（防止组泄露）
同一组（如同一用户、同一患者）的数据不会同时出现在训练和验证中。

```python
from sklearn.model_selection import GroupKFold

groups = df['user_id']  # 每组代表一个用户
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, groups=groups, cv=gkf, scoring='accuracy')
```

### Leave-One-Out (LOO) — 极小数据集
每行作为一次验证集，N 行数据跑 N 次。只适用于极小数据集（N < 200）。

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
```

---

## 5. 完整工作流：CV 替代 train_test_split

### 之前（有缺陷）
```python
# ❌ 手动管理 train/test 容易出错
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
score = mean_absolute_error(y_test, model.predict(X_test))
# 还要记住哪个是训练集、哪个是测试集
```

### 之后（CV）
```python
# ✅ 干净、自动、可靠
pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor(random_state=0))
scores = cross_val_score(pipeline, X, y, scoring='neg_mean_absolute_error', cv=5)
print(f"MAE: {-scores.mean():.2f}")
# 无需手动跟踪训练/测试集
```

### 多模型对比工作流
```python
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=0),
    'GradientBoosting': GradientBoostingRegressor(random_state=0),
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=0),
}

results = {}
for name, model in models.items():
    # ⚠️ 每个模型用独立 pipeline 避免数据泄露
    pipe = make_pipeline(SimpleImputer(), model)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_absolute_error')
    results[name] = {'mean': -scores.mean(), 'std': scores.std()}
    print(f"{name:20s}: {-scores.mean():.3f} (±{scores.std():.3f})")

best = min(results, key=lambda k: results[k]['mean'])
print(f"\n→ Best model: {best} (MAE={results[best]['mean']:.3f})")
```

---

## 6. CV 与 Pipeline 必须组合使用

**不正确的做法（数据泄露）：**
```python
# ❌ 先填充全量数据再 CV — 验证集信息泄露到训练集
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X)  # 用了全部数据！
scores = cross_val_score(model, X_imputed, y, cv=5)
```

**正确做法：**
```python
# ✅ Pipeline 确保每个 fold 内独立填充
pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
scores = cross_val_score(pipeline, X, y, cv=5)
# 每个 fold 的 imputer 只 fit 在训练 fold 上，然后 transform 验证 fold
```

**更复杂的前处理：**
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocessor = ColumnTransformer([
    ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numerical_cols),
    ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore')), categorical_cols),
])

pipeline = make_pipeline(preprocessor, RandomForestRegressor())
scores = cross_val_score(pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
```

---

## 7. 常见陷阱与检查清单

| 陷阱 | 症状 | 修复 |
|------|------|------|
| CV 前全量数据预处理 | 验证得分虚高 | 所有预处理放 Pipeline 内 |
| 分类问题用普通 KFold | 某些 fold 缺类别 | 用 StratifiedKFold |
| 时间序列用随机 KFold | 未来泄露 | 用 TimeSeriesSplit |
| 忽略 fold 方差 | 误判模型优劣 | 始终报告 `mean ± std` |
| 组内样本分散到不同 fold | 过拟合特定组 | 用 GroupKFold |
| scoring 选了错误方向 | 负分越"高"反而越差 | 回归用 neg_*，分类用正向指标 |

## Quality Checklist
- [ ] 是否使用了 Pipeline 防止数据泄露？
- [ ] 分类问题是否用了 StratifiedKFold？
- [ ] 是否报告了 mean ± std 而非单一点估计？
- [ ] scoring 参数是否正确（回归用 neg_*）？
- [ ] 是否对比了至少 2-3 种模型？
- [ ] 训练/验证是否完全隔离（无信息泄露）？

## Trigger Phrases
- "交叉验证"、"cross validation"、"k-fold"
- "模型得分不稳定"、"换 random_state 结果就变"
- "怎么评估模型"、"哪个模型更好"
- "train test split 可靠吗"
- "过拟合怎么检测"

