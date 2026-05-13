---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: feature-selection-methods
---

# Feature Selection Methods

**分类:** 📊 Data Science
**Skill ID:** `feature-selection-methods`

> 系统性特征选择方法论，覆盖 Filter/Wrapper/Embedded 三类方法，包含互信息、递归消除、Permutation Importance 和 SHAP 选择。提炼自 Kaggle 经典 feature selection notebook。

---


# 特征选择方法 (Feature Selection Methods)

## Purpose
从高维特征中筛选真正有用的特征，降维、加速训练、防止过拟合。覆盖三大类方法：Filter（统计检验）、Wrapper（递归搜索）、Embedded（模型内置）。

## When to use
- 特征数 > 100，需要筛选
- 模型训练太慢
- 过拟合严重
- 用户说"特征太多了"、"选哪些特征"

## Three Families of Feature Selection

### 1. Filter Methods — 统计检验（快，与模型无关）

```python
# 1a. 方差过滤（去除常数/近常数特征）
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=0.01)
sel.fit(X)

# 1b. 相关性过滤（去除高相关特征对）
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [c for c in upper.columns if any(upper[c] > 0.95)]
X_filtered = X.drop(to_drop, axis=1)

# 1c. 互信息（分类）— 特征与目标的相关性
from sklearn.feature_selection import mutual_info_classif, SelectKBest
mi = mutual_info_classif(X, y)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
selector = SelectKBest(mutual_info_classif, k=20)
X_mi = selector.fit_transform(X, y)

# 1d. F-test / Chi2（分类）
from sklearn.feature_selection import f_classif, chi2
f_scores, p_values = f_classif(X, y)
```

### 2. Wrapper Methods — 搜索最优子集（慢，但精准）

```python
# 2a. Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 自动选最优特征数
rfecv = RFECV(estimator=rf, step=1, cv=5, scoring='accuracy')
rfecv.fit(X, y)
print(f"Optimal features: {rfecv.n_features_}")

plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1),
         rfecv.cv_results_['mean_test_score'])
plt.xlabel('Number of Features'); plt.ylabel('CV Score')
plt.show()

# 2b. SelectFromModel（基于阈值）
from sklearn.feature_selection import SelectFromModel
sfm = SelectFromModel(rf, threshold='median')
X_selected = sfm.fit_transform(X, y)
```

### 3. Embedded Methods — 模型内置（训练时自动选择）

```python
# 3a. L1 正则化（自动稀疏化）
from sklearn.linear_model import LogisticRegression
lr_l1 = LogisticRegression(penalty='l1', solver='saga', C=0.1, max_iter=1000)
lr_l1.fit(X, y)
selected = X.columns[lr_l1.coef_[0] != 0]

# 3b. 树模型特征重要性
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
selected = importances[importances > importances.median()].index

# 3c. Permutation Importance（模型无关，更可靠）
from sklearn.inspection import permutation_importance
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_scores = pd.Series(perm_imp.importances_mean, index=X.columns).sort_values(ascending=False)
```

### 4. SHAP-based Selection（最新，最可靠）
```python
import shap
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)
shap_importance = np.abs(shap_values).mean(axis=0)
selected = X.columns[shap_importance > np.median(shap_importance)]
```

## Decision Framework

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| 特征 > 1000 | Filter (MI/Variance) | 快，缩小范围 |
| 特征 50-500 | Embedded (Tree/SHAP) | 平衡速度与精度 |
| 特征 < 50 且追求极致 | Wrapper (RFECV) | 最优子集，但慢 |
| 事后解释 | Permutation Importance | 模型无关 |
| 线性模型 | L1 正则化 | 自动稀疏 |

## Common Pitfalls
1. 只用 filter → 忽略特征交互
2. 用 full data 做 selection → 必须只在训练集上做
3. 选太少 (k=5) → 可能丢重要特征
4. 不验证 → 选后必须 cross-validate

## Trigger Phrases
- "特征选择"、"特征筛选"、"降维"
- "哪些特征有用"、"特征太多"
- "RFECV"、"互信息"、"Permutation Importance"

