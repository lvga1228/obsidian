---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: multi-model-smote-workflow
---

# Multi Model Smote Workflow

**分类:** 📊 Data Science
**Skill ID:** `multi-model-smote-workflow`

> 多模型分类对比框架，使用 SMOTE 处理不平衡数据，并行训练 Decision Tree/XGBoost/Random Forest，对比特征重要性和评估指标。提炼自 Teen Mental Health Prediction 实战 notebook。

---


# 多模型 SMOTE 分类对比工作流

## Purpose
针对不平衡二分类问题，并行训练多种模型（DT/XGB/RF），使用 SMOTE 平衡训练集，系统对比每个模型的精度、特征重要性，选出最优模型。

## When to use
- 二分类 + 类别不平衡（正样本 < 10%）
- 需要快速对比多个模型的基线性能
- 用户说"对比几个模型"、"谁效果最好"
- 特征数适中 (5-30)，不需要太多特征工程

## Core Workflow

### Step 1: 加载 + 探索 + 相关性热力图
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
import time
import warnings; warnings.filterwarnings("ignore")

df = pd.read_csv('data.csv')
target = 'target_column'

print(f"Shape: {df.shape}")
print(f"Null values: {df.isnull().sum().max()}")
print(f"Target distribution:\n{df[target].value_counts()}")
print(f"Positive rate: {df[target].mean():.1%}")
print(f"Dtypes:\n{df.dtypes}")

# CORRELATION HEATMAP — 必做！看特征间关系
plt.figure(figsize=(10, 7))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), 
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.tight_layout(); plt.show()
```

### Step 2: 手动编码（不用 LabelEncoder！）
```python
# ⚠️ 用 map() 而非 LabelEncoder — 结果可预测、可复现
# 原因：LabelEncoder 按字母序分配，换数据顺序可能变

dfe = df.copy()
dfe['gender'] = dfe['gender'].map({'male': 1, 'female': 0})
dfe['social_interaction_level'] = dfe['social_interaction_level'].map(
    {'low': 0, 'high': 1, 'medium': 2}
)
dfe['platform_usage'] = dfe['platform_usage'].map(
    {'Instagram': 0, 'TikTok': 1, 'Both': 2}
)

print(f"Null after encoding: {dfe.isnull().sum().max()} (should be 0)")

Feature = dfe.drop(target, axis=1)
Label = dfe[target]
```

### Step 3: Stratified Split + SMOTETomek (不是普通 SMOTE！)
```python
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

# Stratified split
xtrain, xtest, ytrain, ytest = train_test_split(
    Feature, Label, test_size=0.2, random_state=42, stratify=Label
)
print(f"Train: {len(xtrain)}, Test: {len(xtest)}")
print(f"Test labels:\n{ytest.value_counts()}")

# SMOTETomek: SMOTE + Tomek Links cleaning
# Tomek Links 清除 SMOTE 产生的噪声样本
print(f"Before SMOTETomek: {dict(ytrain.value_counts())}")
smote = SMOTETomek(random_state=42)
xtrain_sm, ytrain_sm = smote.fit_resample(xtrain, ytrain)
print(f"After SMOTETomek:  {dict(pd.Series(ytrain_sm).value_counts())}")
print(f"Samples: {len(xtrain)} → {len(xtrain_sm)}")
```

### Step 4: Decision Tree（含限制参数 + plot_tree）
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ⚠️ 关键：用 SMOTE 数据训练
model_dt = DecisionTreeClassifier(
    max_depth=4,               # 限制深度防过拟合
    min_samples_split=50,       # 分裂最少样本
    min_samples_leaf=35,        # 叶节点最少样本
    random_state=42,
)
model_dt.fit(xtrain_sm, ytrain_sm)

# 同时看 train 和 test — 诊断过拟合
train_acc = accuracy_score(ytrain_sm, model_dt.predict(xtrain_sm))
test_acc = accuracy_score(ytest, model_dt.predict(xtest))
test_f1 = f1_score(ytest, model_dt.predict(xtest))

print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
if train_acc - test_acc > 0.05:
    print("⚠️ Possible overfitting — increase min_samples_split or decrease max_depth")

# 可视化决策树
plt.figure(figsize=(19, 9))
plot_tree(model_dt, max_depth=5, feature_names=Feature.columns,
          class_names=['Negative', 'Positive'], filled=True, rounded=False, fontsize=10)
plt.title('Decision Tree Structure')
plt.show()

# 特征重要性
pd.Series(model_dt.feature_importances_, index=Feature.columns
         ).sort_values().plot.barh(figsize=(9,6), title='DT Feature Importance')
plt.show()
```

### Step 5: XGBoost（用原始数据训练！不用 SMOTE！）
```python
from xgboost import XGBClassifier

# ⚠️ 关键区别：XGBoost 用 ORIGINAL data（不用 SMOTE）！
# XGBoost 有内置 scale_pos_weight 处理不平衡
# 也可以调 gamma 和 min_child_weight 抑制噪声

Best_Grid_Params = {
    'n_estimators': 500,
    'learning_rate': 0.01,
    'max_depth': 10,
    'gamma': 0.1,              # 分裂最小损失减少（防过拟合）
}

model_xgb = XGBClassifier(**Best_Grid_Params, n_jobs=-1, random_state=42)

# 在 ORIGINAL 训练数据上训练！
model_xgb.fit(xtrain, ytrain)

train_acc = accuracy_score(ytrain, model_xgb.predict(xtrain))
test_acc = accuracy_score(ytest, model_xgb.predict(xtest))
test_f1 = f1_score(ytest, model_xgb.predict(xtest))

print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
```

### Step 6: Random Forest（含限制参数）
```python
from sklearn.ensemble import RandomForestClassifier

# ⚠️ 用 SMOTE 数据训练
model_rf = RandomForestClassifier(
    max_depth=3,               # 浅树防过拟合
    n_estimators=90,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42,
)
model_rf.fit(xtrain_sm, ytrain_sm)

train_acc = accuracy_score(ytrain_sm, model_rf.predict(xtrain_sm))
test_acc = accuracy_score(ytest, model_rf.predict(xtest))
test_f1 = f1_score(ytest, model_rf.predict(xtest))

print(f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | F1: {test_f1:.4f}")

pd.Series(model_rf.feature_importances_, index=Feature.columns
         ).sort_values().plot.barh(figsize=(9,6), title='RF Feature Importance')
plt.show()
```

### Step 7: GridSearchCV for XGBoost（可选，默认注释）
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [5, 8, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.03],
    'n_estimators': [100, 200, 300, 500],
}

search = GridSearchCV(
    XGBClassifier(random_state=42, n_jobs=1),
    param_grid=param_grid,
    scoring='f1_macro',        # 不平衡分类用 F1 而非 accuracy
    cv=5,
    n_jobs=1  # 避免并行冲突
)

# Uncomment to run:
# search.fit(xtrain, ytrain)
# print(f"Best: {search.best_score_:.4f} with {search.best_params_}")
```

### Step 5: 模型对比表格
```python
comparison = pd.DataFrame(results).T
print(f"{'Model':<15} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
print('-'*57)
for name, metrics in results.items():
    print(f"{name:<15} {metrics['Accuracy']:>10.4f} {metrics['F1']:>10.4f} "
          f"{metrics['Precision']:>10.4f} {metrics['Recall']:>10.4f}")

# 可视化
comparison.plot(kind='bar', figsize=(12,6), colormap='Set2')
plt.title('Model Comparison — All Metrics')
plt.ylabel('Score'); plt.ylim(0,1)
plt.legend(loc='lower right'); plt.xticks(rotation=0)
plt.tight_layout(); plt.show()
```

### Step 6: 特征重要性横向对比
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
colors = {'Decision Tree': 'steelblue', 'XGBoost': 'darkorange', 'Random Forest': 'forestgreen'}

for ax, (name, model) in zip(axes, models.items()):
    imp = pd.Series(model.feature_importances_, index=Feature.columns).sort_values()
    imp.plot.barh(ax=ax, color=colors[name], title=name)

plt.suptitle('Feature Importance — Three Models Compared', fontsize=14)
plt.tight_layout(); plt.show()

# 关键洞察：哪些特征三个模型都认为重要？
all_imps = pd.DataFrame({
    'DT': models['Decision Tree'].feature_importances_,
    'XGB': models['XGBoost'].feature_importances_,
    'RF': models['Random Forest'].feature_importances_,
}, index=Feature.columns)
all_imps['avg'] = all_imps.mean(axis=1)
print("Top features by average importance:")
print(all_imps.nlargest(5, 'avg'))
```

### Step 7: 选择模型 + 最终建议
```python
# 选择标准：F1 最高（不平衡分类不看 Accuracy）
best_model_name = comparison['F1'].idxmax()
best_score = comparison.loc[best_model_name, 'F1']
print(f"Best model: {best_model_name} (F1={best_score:.4f})")

# 如果所有模型 F1 < 0.7 → 数据可能不够，需要采集更多样本
# 如果 Accuracy 很高但 F1 很低 → Accuracy paradox，正样本太少
```

## Quality Checklist
- [ ] SMOTE 是否只应用于 training set？
- [ ] Train/test split 是否用了 stratify？
- [ ] 是否报告了 F1 + Precision + Recall（不只是 Accuracy）？
- [ ] 是否对比了三种模型的指标？
- [ ] 是否对比了特征重要性（不同模型可能不同）？
- [ ] 是否检查了 test set 中正样本数量（太少则结果不可靠）？

## Common Pitfalls
1. **SMOTE 用在 test set** → 数据泄露，结果虚高
2. **只看 Accuracy** → 2.6% 正样本时，全预测 0 就有 97.4% 准确率
3. **测试集正样本太少** → 6 个阳性无法可靠评估（至少需要 50+）
4. **Decision Tree 100% 准确率** → 几乎肯定是过拟合
5. **特征重要性不可跨模型直接比较绝对值** → 只看排序

## vs Other Skills
- `imbalanced-classification`: 专注 Recall 优化 + 阈值调优，本 skill 专注多模型对比
- `classification-modeling-workflow`: 完整的 12 步流程，本 skill 是其子流程的快照版本
- `gradient-boosting-advanced`: XGB/LGB/CB 深度调参，本 skill 只用默认参数做快速对比

## Trigger Phrases
- "对比几个分类模型"、"哪个模型最好"
- "SMOTE"、"不平衡分类"、"多模型对比"
- "Decision Tree vs XGBoost vs Random Forest"

