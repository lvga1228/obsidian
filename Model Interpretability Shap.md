---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: model-interpretability-shap
---

# Model Interpretability Shap

**分类:** 📊 Data Science
**Skill ID:** `model-interpretability-shap`

> 使用SHAP (SHapley Additive exPlanations) 进行模型解释与特征归因分析。覆盖 TreeExplainer/KernelExplainer/DeepExplainer、force_plot/summary_plot/dependence_plot、特征重要性排序与 SHAP 选择。提炼自 Kaggle Dan Becker SHAP Values 教程。

---


# 模型可解释性 — SHAP 值 (Model Interpretability with SHAP)

## Purpose
将黑盒模型预测拆解为每个特征的贡献值，回答"为什么这个样本得到这个预测结果"。SHAP 基于博弈论中的 Shapley 值，保证可加性（所有特征 SHAP 值之和 = 预测值 - 基线值），是目前最可靠的模型解释方法。

## When to use
- 需要解释单个预测结果（贷款拒绝、疾病诊断等合规场景）
- 需要对黑盒模型做全局特征重要性分析
- 需要理解特征如何影响预测方向（正向/负向）
- 需要检测特征交互效应
- 用户问"这个模型是怎么判断的"、"哪些特征最重要"
- 用户提到 SHAP、force plot、summary plot、shapley 值

## Core Concepts (核心概念)

### Shapley 值公式 (分解方程)
```
sum(SHAP values for all features) = prediction - base_value
```
- **base_value**: 模型在整个训练集上的平均预测（基线期望值）
- **prediction**: 当前样本的预测值
- **SHAP value**: 每个特征相对于基线的贡献（正=推高预测，负=拉低预测）

### 三种 Explainers（解释器选择）

| Explainer | 适用模型 | 速度 | 精度 | 场景 |
|-----------|---------|------|------|------|
| `TreeExplainer` | 树模型 (XGBoost/LightGBM/RF/DT/CatBoost) | 极快 | 精确 | 首选，树模型专用 |
| `KernelExplainer` | 任何模型 | 慢 | 近似 | 黑盒模型通用，小数据可用 |
| `DeepExplainer` | 深度学习 (TensorFlow/PyTorch) | 快 | 近似 | CNN/RNN/MLP |
| `LinearExplainer` | 线性模型 | 极快 | 精确 | LogisticRegression/LinearSVC |

---

## 1. SHAP Values 计算 (SHAP Value Calculation)

### 1.1 TreeExplainer — 树模型首选（精确、极快）

```python
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 准备数据
data = pd.read_csv('your_data.csv')
y = (data['target'] == "Yes")
feature_names = [c for c in data.columns if data[c].dtype in [np.int64, np.float64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 训练模型
model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# === SHAP 计算 ===
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(val_X)

# shap_values 结构说明（二分类）:
#   shap_values[0] — 负类（0）的 SHAP 值矩阵
#   shap_values[1] — 正类（1）的 SHAP 值矩阵 ← 通常取这个
# 多分类: shap_values[i] 对应第 i 类
# 回归: shap_values 直接是矩阵，无列表包裹

# 取正类 SHAP 值
shap_pos = shap_values[1]  # shape: (n_samples, n_features)
explainer.expected_value[1]  # 正类的 base_value
```

### 1.2 KernelExplainer — 通用黑盒解释器（慢，近似）

```python
# KernelExplainer 适用于任何模型，但速度慢且是近似值
# 用训练集的一小部分作为背景数据（summary）来加速

# 背景数据采样（用 kmeans 或随机采样 100-200 条）
background = shap.kmeans(train_X, 100)  # 或 train_X.iloc[:100]

k_explainer = shap.KernelExplainer(model.predict_proba, background)

# 解释单条样本
k_shap_values = k_explainer.shap_values(val_X.iloc[5:6])

# 注意: KernelExplainer 对大数据集极慢，优先用 TreeExplainer
```

### 1.3 DeepExplainer — 深度学习模型

```python
# 用于 Keras/TensorFlow/PyTorch 模型
explainer = shap.DeepExplainer(model, background_sample)
shap_values = explainer.shap_values(X_test)
```

### 1.4 关键注意事项 (Critical Details)

```python
# ⚠️ shap_values 结构因模型类型而异：

# 二分类树模型 → 列表 [array_neg, array_pos]
shap_values = explainer.shap_values(X)
print(type(shap_values))  # <class 'list'>
print(shap_values[0].shape, shap_values[1].shape)  # (n, m), (n, m)

# 回归 / 多分类树模型 → 列表（每类一个 array）
# 或使用新版 shap API (v0.40+)
shap_values = explainer(X)  # 返回 Explanation 对象
# shap_values.values → SHAP 值
# shap_values.base_values → base_value
# shap_values.data → 原始特征值

# ⚠️ XGBoost/LightGBM 用原生模型可能需先转 booster
import xgboost as xgb
explainer = shap.TreeExplainer(xgb_model.get_booster())
```

---

## 2. 可视化 (Visualization)

### 2.1 force_plot — 单样本解释（Why This Prediction?）

最经典的 SHAP 可视化。红色=推高预测值，蓝色=拉低预测值。条形宽度=影响幅度。

```python
shap.initjs()  # 在 Notebook 中启用交互式 JS 渲染

# 解释单个样本
shap.force_plot(
    explainer.expected_value[1],  # base_value (正类基线)
    shap_values[1][row_idx],      # 该样本的 SHAP 值向量
    val_X.iloc[row_idx],          # 特征值（用于显示特征名和取值）
    matplotlib=True               # 或 matplotlib=False 用 JS 交互版
)

# 解释多个样本（堆叠 force plot，观察模式）
shap.force_plot(
    explainer.expected_value[1],
    shap_values[1][:50],  # 前 50 个样本
    val_X.iloc[:50]
)
```

**force_plot 解读**:
- 最左侧 `base value` = 模型平均预测
- `f(x)` 最右侧 = 当前样本预测值
- 粉色特征 = 将预测推向更高（正贡献）
- 蓝色特征 = 将预测推向更低（负贡献）
- 粉色总和 - 蓝色总和 = f(x) - base_value

### 2.2 summary_plot — 全局特征重要性（鸟瞰图）

最重要的全局可解释性可视化。一图展示：哪些特征最重要 + 特征值如何影响预测。

```python
# === 基础 summary_plot（蜂群图 / beeswarm）===
shap.summary_plot(shap_values[1], val_X)

# 解读:
#   - Y 轴按特征全局重要性降序排列（|SHAP| 均值）
#   - X 轴为 SHAP 值（对预测的贡献）
#   - 每个点代表一个样本
#   - 颜色 = 特征值（红高蓝低）
#   - 水平偏移 = 该样本该特征的贡献方向和大小
#   - 向右的点 = 推高预测，向左 = 拉低预测

# === 条形图版（更简洁）===
shap.summary_plot(shap_values[1], val_X, plot_type="bar")

# === 分类别对比（多分类时）===
shap.summary_plot(shap_values, val_X, class_names=['Class0', 'Class1', 'Class2'])

# === 绝对值聚合 → 特征重要性排名 ===
shap_importance = np.abs(shap_values[1]).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': val_X.columns,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)
print(importance_df.head(10))
```

**summary_plot 核心洞察**:
1. 排名最高的特征 = 对模型影响最大
2. 红色点偏右 → 高特征值推高预测（正向关系）
3. 蓝色点偏左 → 低特征值推低预测（也是正向关系）
4. 颜色混杂无方向 → 特征影响非线性

### 2.3 dependence_plot — 特征交互与部分依赖

展示单个特征的 SHAP 值如何随其特征值变化，揭示非线性关系和交互效应。

```python
# === 基础 dependence_plot ===
shap.dependence_plot(
    "feature_name",          # 要分析的特征名
    shap_values[1],          # SHAP 值矩阵
    val_X,                   # 特征数据
    interaction_index=None   # 不标注交互特征
)

# 解读:
#   - X 轴 = 特征值
#   - Y 轴 = 该特征值的 SHAP 贡献
#   - 每个点 = 一个样本
#   - 上升趋势 = 高特征值 → 高预测
#   - 下降趋势 = 高特征值 → 拉低预测
#   - 水平束 = 该特征影响饱和（阈值效应）

# === dependence_plot + 交互特征 ===
shap.dependence_plot(
    "feature_name",
    shap_values[1],
    val_X,
    interaction_index="another_feature"  # 颜色显示交互特征的值
)

# 解读:
#   - 颜色按交互特征值着色
#   - 垂直分层（同一X值不同颜色分散）→ 存在交互效应
#   - 例如: 房价数据中，房间数 vs 房价 SHAP，按地区着色
#           同一房间数在不同地区的贡献差异大 → 房间数与地区交互

# === 自动选择最佳交互特征 ===
# interaction_index='auto'（新版 SHAP 支持）
shap.dependence_plot("feature_name", shap_values[1], val_X, 
                     interaction_index=None)  # None 时自动选
```

**dependence_plot 典型模式**:

| 模式 | 含义 | 示例 |
|------|------|------|
| 斜线上升 | 正线性关系 | 进球数 → 获奖概率 |
| 斜线下降 | 负线性关系 | 犯规数 → 获奖概率 |
| 倒U型 | 最优区间 | 温度 → 销量 |
| 水平分层 | 强交互效应 | 房价按地区分层 |
| 水平饱和 | 阈值效应 | 超过某值不再增加影响 |

---

## 3. 特征重要性解释与选择 (Feature Importance Interpretation)

### 3.1 SHAP 重要性 vs 传统重要性

```python
# 传统特征重要性（模型内置，有偏 → 倾向高基数特征）
rf_importance = pd.Series(model.feature_importances_, index=X.columns)

# SHAP 重要性（无偏，基于边际贡献）
shap_importance = np.abs(shap_values[1]).mean(axis=0)
shap_importance = pd.Series(shap_importance, index=X.columns)

# 对比两种重要性
importance_comparison = pd.DataFrame({
    'tree_importance': rf_importance,
    'shap_importance': shap_importance
}).sort_values('shap_importance', ascending=False)
print(importance_comparison.head(15))
# 注意: SHAP 重要性通常更可靠，尤其对有高基数类别特征时
```

### 3.2 SHAP 用于特征选择

```python
# 方法一: 绝对值均值阈值
shap_imp = np.abs(shap_values[1]).mean(axis=0)
selected_features = X.columns[shap_imp > np.median(shap_imp)]
print(f"Selected {len(selected_features)}/{len(X.columns)} features")

# 方法二: Top-K 选择
k = 10
top_k_idx = np.argsort(shap_imp)[-k:]
selected_features = X.columns[top_k_idx]

# 方法三: SHAP RFECV（结合 SHAP 重要性的递归消除）
from sklearn.feature_selection import RFECV
selector = RFECV(model, step=1, cv=5, scoring='accuracy')
selector.fit(X, y)
# 然后对选中的特征运行 SHAP 验证
```

### 3.3 特征方向性分析

```python
# 判断每个特征对正类的贡献方向
shap_mean = shap_values[1].mean(axis=0)
feature_direction = pd.DataFrame({
    'feature': X.columns,
    'mean_shap': shap_mean,
    'direction': ['positive ↑' if v > 0 else 'negative ↓' for v in shap_mean]
}).sort_values('mean_shap', ascending=False)
print(feature_direction)
```

---

## 4. 完整工作流模板

```python
"""
SHAP 模型解释完整工作流
适用: 任何分类/回归模型
"""
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============ Step 1: 训练模型 ============
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============ Step 2: 创建 Explainer ============
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 若为二分类，取正类
if isinstance(shap_values, list):
    shap_pos = shap_values[1]
    base_value = explainer.expected_value[1]
else:
    shap_pos = shap_values
    base_value = explainer.expected_value

# ============ Step 3: 全局特征重要性 (summary_plot) ============
shap.summary_plot(shap_pos, X_test)
plt.title('SHAP Summary Plot — Global Feature Importance')
plt.tight_layout()
plt.savefig('shap_summary.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ Step 4: 特征重要性排序 ============
shap_importance = np.abs(shap_pos).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': X_test.columns,
    'importance': shap_importance
}).sort_values('importance', ascending=False)
print("Top 10 Features by |SHAP| mean:")
print(importance_df.head(10))

# ============ Step 5: Top 特征 dependence_plot ============
top_features = importance_df['feature'].head(5).tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, feat in enumerate(top_features):
    plt.sca(axes[i])
    shap.dependence_plot(feat, shap_pos, X_test, show=False)
    axes[i].set_title(feat, fontsize=11)
# Hide extra subplots
for j in range(len(top_features), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('SHAP Dependence Plots — Top 5 Features', fontsize=14)
plt.tight_layout()
plt.savefig('shap_dependence_top5.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ Step 6: 单样本解释 (force_plot) ============
sample_idx = 0
shap.force_plot(
    base_value,
    shap_pos[sample_idx],
    X_test.iloc[sample_idx],
    matplotlib=True
)
plt.title(f'SHAP Force Plot — Sample {sample_idx}')
plt.tight_layout()
plt.savefig(f'shap_force_sample_{sample_idx}.png', dpi=150, bbox_inches='tight')
plt.show()

# ============ Step 7: 多样本堆叠 (可选) ============
shap.force_plot(base_value, shap_pos[:30], X_test.iloc[:30])

print("\n=== SHAP 分析完成 ===")
print(f"模型基值 (base_value): {base_value:.4f}")
print(f"训练集平均预测: {model.predict_proba(X_train)[:,1].mean():.4f}")
```

---

## 5. 常见陷阱 (Common Pitfalls)

1. **混淆 shap_values 结构**: 二分类树模型的 `shap_values` 是 list（2 个矩阵），回归是直接矩阵。始终 `print(type(shap_values))` 确认。
2. **KernelExplainer 大数据集**: 极慢。必须对背景数据采样（建议 kmeans 采样 100-200 条）。
3. **XGBoost/LightGBM 原生模型**: 用 `shap.TreeExplainer(model.get_booster())` 而非 `shap.TreeExplainer(model)`。
4. **忽略 base_value**: `explainer.expected_value` 可能也是 list，需根据任务取对应类的基线。
5. **只用 feature_importances_: 树模型内置重要性对有高基数类别特征有偏，SHAP 重要性更可靠。
6. **不保存图片**: SHAP 图计算耗时，务必 `plt.savefig()` 保存。
7. **feature_names 不匹配**: `summary_plot` 传入的 X 必须与 `shap_values` 计算的列顺序一致。

## 6. 性能建议

| 场景 | Explainer | 背景数据量 | 预估时间 |
|------|-----------|-----------|---------|
| RF/XGB ~100特征 ~10K样本 | TreeExplainer | 全部 | <1秒 |
| RF/XGB ~500特征 ~100K样本 | TreeExplainer | 全部 | 5-30秒 |
| 任何模型 ~20特征 ~1K样本 | KernelExplainer | 100条 | 1-5分钟 |
| 任何模型 ~100特征 ~10K样本 | KernelExplainer | 200条 | 10-60分钟 |

## Trigger Phrases
- "SHAP"、"shap值"、"shapley"
- "模型解释"、"模型可解释性"、"为什么预测这个结果"
- "force plot"、"summary plot"、"dependence plot"
- "特征归因"、"特征贡献"、"特征重要性排序"
- "哪些特征最重要"、"这个特征如何影响预测"
- "Explainable AI"、"XAI"

