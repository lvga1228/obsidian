---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: imbalanced-classification
---

# Imbalanced Classification

**分类:** 📊 Data Science
**Skill ID:** `imbalanced-classification`

> 不平衡数据分类框架，处理欺诈检测、罕见病诊断等极端不平衡场景。覆盖欠采样、过采样(SMOTE)、阈值调整、Recall 优化和 Precision-Recall 曲线分析。提炼自 Kaggle 2.5K votes 经典 fraud detection notebook。

---


# 不平衡数据分类 (Imbalanced Classification)

## Purpose
处理类别极端不平衡的分类问题（如欺诈检测 99.9% vs 0.1%），目标是在保证合理 Precision 的前提下最大化 Recall（不漏掉正例）。

## When to use

- 正负样本比例 > 1:10（如欺诈交易、罕见病诊断、设备故障）。
- 预测目标中「漏掉正例」的代价远大于「错判负例」。
- 需要同时优化 Recall 和 Precision（而非只看 Accuracy）。
- 用户说"数据很不平衡"、"类别分布不均"、"欺诈检测"。

## Do not use when

- 类别大致平衡（正负比 < 1:3）— 标准分类流程即可。
- 只需要分类不需要概率 — 普通分类即可。
- 数据集极小（<500 条）— 重采样可能失效。

## Core Workflow (核心工作流)

### Step 1: 数据加载与不平衡确认
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('creditcard.csv')
print(f"Shape: {df.shape}")

# 检查目标分布
class_counts = df['Class'].value_counts()
print(f"\nClass distribution:")
print(f"  0 (Normal): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
print(f"  1 (Fraud):  {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")

# 可视化
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
class_counts.plot(kind='bar', ax=ax[0])
ax[0].set_title('Class Distribution')
ax[0].set_ylabel('Count')

class_counts.plot(kind='pie', ax=ax[1], autopct='%1.2f%%', startangle=90)
ax[1].set_title('Class Proportion')
plt.show()

# 不平衡比
imbalance_ratio = class_counts[0] / class_counts[1]
print(f"Imbalance ratio: {imbalance_ratio:.0f}:1 ⚠️" if imbalance_ratio > 10 else f"Imbalance ratio: {imbalance_ratio:.1f}:1")
```

### Step 2: 标准化 + 数据准备
```python
from sklearn.preprocessing import StandardScaler

# 耗时/金额列特殊处理
df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

# 分离特征和目标
X = df.drop('Class', axis=1)
y = df['Class']

print(f"Features: {X.shape[1]}")
print(f"Positive samples: {y.sum()}")
```

### Step 3: 欠采样 (Undersampling)
```python
from sklearn.model_selection import train_test_split

# 随机欠采样 — 让正负样本 1:1
fraud_indices = df[df['Class'] == 1].index
normal_indices = df[df['Class'] == 0].index

# 随机选择与欺诈样本等量的正常样本
random_normal = np.random.choice(normal_indices, len(fraud_indices), replace=False)
undersample_indices = np.concatenate([fraud_indices, random_normal])

df_under = df.iloc[undersample_indices]
X_under = df_under.drop('Class', axis=1)
y_under = df_under['Class']

print(f"After undersampling: {len(df_under)} samples")
print(f"New distribution:\n{y_under.value_counts()}")

# 注意：测试集必须保留原始分布！
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# 训练集用欠采样，测试集用原始
X_train_under, _, y_train_under, _ = train_test_split(
    X_under, y_under, test_size=0.2, random_state=42
)
```

### Step 4: 交叉验证 + Recall 优化
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_auc_score

def print_kfold_scores(X, y, model=None):
    """打印 K-fold 交叉验证的 Recall 分数"""
    if model is None:
        model = LogisticRegression(max_iter=1000)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = cross_val_score(model, X, y, cv=kfold, scoring='recall')
    print(f"Recall: {results.mean():.4f} (+/- {results.std():.4f})")
    
    # 同时报告其他指标
    for metric in ['precision', 'f1', 'roc_auc']:
        scores = cross_val_score(model, X, y, cv=kfold, scoring=metric)
        print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return results

# 欠采样数据上训练
lr = LogisticRegression(C=0.01, max_iter=1000)
print("=== Cross-validation on undersampled data ===")
recall_scores = print_kfold_scores(X_train_under, y_train_under, lr)
```

### Step 5: 混淆矩阵 + 在原始测试集上验证
```python
import itertools

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    """可视化混淆矩阵"""
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# 训练并预测
lr.fit(X_train_under, y_train_under)
y_pred = lr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, classes=['Normal', 'Fraud'])
plt.show()

print("\nClassification Report on ORIGINAL test set:")
print(classification_report(y_test, y_pred))

# 计算 Recall
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(f"\nRecall (fraud detection rate): {recall:.4f}")
print(f"Fraud caught: {cm[1,1]} out of {cm[1,0]+cm[1,1]}")
print(f"False alarms: {cm[0,1]} normal transactions flagged")
```

### Step 6: ROC 曲线 + Precision-Recall 曲线
```python
# ROC Curve
y_pred_proba = lr.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall Curve（不平衡数据中更重要）
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
axes[0].plot(fpr, tpr, label=f'ROC (AUC={roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve')
axes[0].legend()

# Precision-Recall
axes[1].plot(recall, precision, label=f'PR (AUC={pr_auc:.3f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Step 7: 阈值调优 (Threshold Tuning)
```python
# 默认阈值 0.5 未必最优 — 找 Recall-Precision 最佳平衡点
thresholds = np.arange(0.1, 1.0, 0.05)
results = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    cm_thresh = confusion_matrix(y_test, y_pred_thresh)
    
    tn, fp, fn, tp = cm_thresh.ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    results.append({
        'threshold': thresh,
        'recall': rec,
        'precision': prec,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    })

results_df = pd.DataFrame(results)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(results_df['threshold'], results_df['recall'], 'b-', label='Recall', linewidth=2)
ax.plot(results_df['threshold'], results_df['precision'], 'r-', label='Precision', linewidth=2)
ax.plot(results_df['threshold'], results_df['f1'], 'g--', label='F1', linewidth=2)
ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, label='90% Recall target')
ax.set_xlabel('Threshold')
ax.set_ylabel('Score')
ax.set_title('Recall vs Precision vs Threshold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()

# 找到 Recall ≥ 90% 时最高 F1 的阈值
high_recall = results_df[results_df['recall'] >= 0.90]
if len(high_recall) > 0:
    best = high_recall.loc[high_recall['f1'].idxmax()]
    print(f"\nBest threshold for Recall ≥ 90%: {best['threshold']:.2f}")
    print(f"  Recall={best['recall']:.4f}, Precision={best['precision']:.4f}, F1={best['f1']:.4f}")
```

### Step 8: 在完整数据上测试（可选）
```python
# 最终模型在全部数据上的表现
y_pred_all = lr.predict(X)
cm_all = confusion_matrix(y, y_pred_all)
recall_all = cm_all[1,1] / (cm_all[1,0] + cm_all[1,1])
print(f"Recall on entire dataset: {recall_all:.4f}")
```

## 进阶策略

### SMOTE 过采样
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)

print(f"Before SMOTE: {y_train.value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_smote).value_counts().to_dict()}")
```

### 类别权重 (Class Weight)
```python
# 直接在模型中设置权重，无需重采样
lr_weighted = LogisticRegression(
    class_weight='balanced',  # 自动按比例加权
    max_iter=1000,
)
lr_weighted.fit(X_train, y_train)
```

### 多次欠采样集成 (EasyEnsemble)
```python
# 多次欠采样 + 集成
n_models = 10
models = []

for i in range(n_models):
    normal_sample = np.random.choice(normal_indices, len(fraud_indices), replace=False)
    sample_idx = np.concatenate([fraud_indices, normal_sample])
    X_sample = X.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_sample, y_sample)
    models.append(model)

# 概率平均
probas = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
y_ensemble = (probas >= 0.5).astype(int)
```

## Quality Checklist (自检表)

- [ ] 类别分布是否已被量化（比例 + 可视化）？
- [ ] 是否使用了欠采样/过采样/SMOTE/类别加权中至少一种？
- [ ] 测试集是否保留了原始不平衡分布？
- [ ] 是否报告了 Recall + Precision + F1（而非只看 Accuracy）？
- [ ] 是否画了 Precision-Recall 曲线（不平衡场景 PR > ROC）？
- [ ] 是否尝试了阈值调优？
- [ ] 业务上 Recall 的目标值是否明确（95%？99%？）？

## Common Pitfalls

1. **只看 Accuracy**：99.9% 正常→预测全部正常→Accuracy=99.9%但 Recall=0。**Accuracy 是毒药**。
2. **测试集也用欠采样**：应在原始分布上评估，否则指标失真。
3. **SMOTE 不考虑特征分布**：SMOTE 对高维稀疏特征效果差。
4. **阈值永远用 0.5**：不平衡场景应调优阈值。
5. **忽略 False Positive 的业务影响**：100% Recall 但 50% Precision = 一半正常用户被骚扰。

## Trigger Phrases (触发词)

- "数据不平衡"、"类别不均"
- "欺诈检测"、"异常检测"
- "提升 Recall"、"不漏掉正例"
- "代价敏感学习"

## Relationship to Other Skills

- 前置：`data-cleaning-pipeline`
- 前置：`classification-modeling-workflow`
- 配合：`feature-engineering-handbook`

