---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: regression-modeling-mastery
---

# Regression Modeling Mastery

**分类:** 📊 Data Science
**Skill ID:** `regression-modeling-mastery`

> 回归建模大师级技能，覆盖 Lasso、Ridge、ElasticNet、KernelRidge、GradientBoosting、XGBoost、LightGBM、Stacking 等回归模型的完整竞赛流程。包含目标变量对数变换、Box-Cox 特征变换、异常值处理、RMSLE 评估以及多层模型堆叠。提炼自 Kaggle 14.3K votes 经典 notebook（Serigne, Top 4% on Leaderboard）。

---


# 回归建模大师级 (Regression Modeling Mastery)

## Purpose
掌握从数据预处理到多层模型堆叠的完整回归建模流程。特别针对房价预测等偏态目标变量的回归任务，融合线性模型与树模型的优势，通过 Stacking 突破单一模型的性能上限。Top 4% on Kaggle Leaderboard 的实战方案。

## 前置知识
本技能假设你已经熟悉以下基础概念：
- 线性回归与正则化（L1/L2）
- 决策树与梯度提升
- 交叉验证的基本思想

## Core Workflow

### Step 1: 目标变量分析 —— 偏态检测与对数变换

回归任务的第一步是检查目标变量的分布。偏态（skewed）的目标变量会严重影响线性模型的性能。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew

# 检查目标变量分布
sns.distplot(train['SalePrice'], fit=norm)

# 获取正态分布拟合参数
(mu, sigma) = norm.fit(train['SalePrice'])
print(f'mu = {mu:.2f}, sigma = {sigma:.2f}')

# QQ图检查偏离正态的程度
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()
```

**为什么对数变换有效**: 房价等经济数据通常呈右偏分布。对数变换压缩高值区间、拉伸低值区间，使分布更接近正态，从而让线性模型（假设误差项正态分布）表现更好。

```python
# 目标变量对数变换: log(1+x) 避免 log(0) 的问题
train["SalePrice"] = np.log1p(train["SalePrice"])

# 验证变换后的分布
sns.distplot(train['SalePrice'], fit=norm)
```

**关键技巧**:
- `np.log1p(x)` = `log(1 + x)`，当 x 可能为 0 时不会出错
- 预测后需要用 `np.expm1(pred)` 还原到原始尺度
- 对数变换后使用 RMSE 等价于原始尺度的 RMSLE（Root Mean Squared Logarithmic Error）

### Step 2: 异常值检测与处理

异常值可以严重扭曲回归模型，特别是对线性模型（Lasso, Ridge, ElasticNet）。

```python
# 散点图检查特征与目标的关系，寻找异常值
fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()

# 删除明显的异常值（右下角：大面积但低价格的离群点）
train = train.drop(
    train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
)
```

**异常值处理原则**:
1. 仅删除"明显不合理"的异常值（如面积极大但价格极低）
2. 使用 `RobustScaler` 替代 `StandardScaler` 可减少剩余异常值的影响
3. 树模型（GBoost, XGBoost, LightGBM）天然对异常值更鲁棒
4. 梯度提升使用 `loss='huber'` 可进一步增强鲁棒性

### Step 3: 缺失值处理策略

回归任务中缺失值的处理需要结合领域知识：

```python
# 策略1: NA 意味着"不存在" → 填 "None"（分类特征）或 0（数值特征）
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")     # NA = 无游泳池
all_data["GarageCars"] = all_data["GarageCars"].fillna(0)   # NA = 无车库 = 0辆车

# 策略2: 用该街区的 LotFrontage 中位数填充（分组填充）
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median())
)

# 策略3: 用众数填充（分类特征最常见值）
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

# 策略4: 删除无信息特征
# 如 Utilities 几乎所有值相同 → 删除
all_data = all_data.drop(['Utilities'], axis=1)
```

### Step 4: 特征工程

#### 4a. 数值→分类转换
```python
# 虽然是数字，但实际是类别编码
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
```

#### 4b. 有序分类变量的 Label Encoding
```python
from sklearn.preprocessing import LabelEncoder

# 这些分类特征的值有优劣顺序（如 ExterQual: Ex > Gd > TA > Fa）
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual',
        'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence',
        'BsmtExposure', 'GarageFinish', 'LandSlope', 'LotShape',
        'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass',
        'OverallCond', 'YrSold', 'MoSold')

for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
```

#### 4c. 创建组合特征
```python
# 总面积 = 地下室 + 一层 + 二层
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
```

#### 4d. Box-Cox 变换（偏态特征处理）

偏态特征会影响线性模型的假设。Box-Cox 变换是一种泛化的幂变换，可以将偏态数据拉向正态分布。

```python
from scipy.special import boxcox1p
from scipy.stats import skew

# 检查所有数值特征的偏度
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew': skewed_feats})

# 对偏度绝对值 > 0.75 的特征进行 Box-Cox 变换
skewness = skewness[abs(skewness) > 0.75]
print(f"There are {skewness.shape[0]} skewed features to transform")

lam = 0.15  # Box-Cox 的 lambda 参数
skewed_features = skewness.index
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
```

**Box-Cox vs Log 变换**:
- `boxcox1p(x, lam=0)` 等价于 `log1p(x)`
- `lam > 0`：介于原始数据和 log 变换之间
- `lam = 0.15`：经验值，常用于房价数据，比纯 log 更温和
- Box-Cox 的重要前提：所有值必须 > 0（因此用 `boxcox1p` 即 `boxcox(1+x)`）

#### 4e. One-Hot Encoding
```python
# 对剩余分类特征做独热编码
all_data = pd.get_dummies(all_data)
```

### Step 5: 基础回归模型

#### 5a. Lasso Regression (L1 正则化)

Lasso 对异常值敏感，必须配合 RobustScaler：

```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

lasso = make_pipeline(
    RobustScaler(),
    Lasso(alpha=0.0005, random_state=1)
)
```

**调参指南**:
- `alpha`: 正则化强度。越小 → 模型越复杂；0.0005 是一个很弱的正则化
- 为什么用 `RobustScaler`：使用中位数和四分位距进行缩放，比 `StandardScaler` 对异常值更鲁棒
- Lasso 倾向于产生稀疏解（部分系数被压缩到 0），有特征选择的作用

#### 5b. ElasticNet (L1 + L2 正则化)

```python
from sklearn.linear_model import ElasticNet

ENet = make_pipeline(
    RobustScaler(),
    ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3)
)
```

**关键参数**:
- `alpha`: 总正则化强度
- `l1_ratio`: L1 与 L2 的比例。0.9 表示 90% L1 + 10% L2
- L1 部分做特征选择，L2 部分处理多重共线性
- 适合特征数量多且存在相关性的场景

#### 5c. Kernel Ridge Regression

```python
from sklearn.kernel_ridge import KernelRidge

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
```

**KernelRidge 详解**:
- Ridge Regression 的核化版本：`LinearRidge → KernelRidge`
- 本质是 Ridge（L2 正则化）在核空间（高维特征空间）中做线性回归
- `kernel='polynomial'` + `degree=2`：考虑特征的二阶多项式交互
- `coef0=2.5`：多项式核的常数项 `(gamma * x^T y + coef0)^degree`
- `alpha=0.6`：Ridge 正则化参数
- **不需要 RobustScaler**：核方法在特征空间中计算，缩放影响较小
- 适合捕捉非线性关系，但计算复杂度 O(n³)，大数据集需谨慎

#### 5d. GradientBoostingRegressor

```python
from sklearn.ensemble import GradientBoostingRegressor

GBoost = GradientBoostingRegressor(
    n_estimators=3000,          # 树的数量
    learning_rate=0.05,         # 学习率（低 → 需要更多树）
    max_depth=4,                # 每棵树的深度
    max_features='sqrt',        # 每次分裂考虑的特征比例
    min_samples_leaf=15,        # 叶节点的最小样本数（防止过拟合）
    min_samples_split=10,       # 分裂的最小样本数
    loss='huber',               # Huber 损失 → 对异常值鲁棒
    random_state=5
)
```

**GradientBoosting 调参优先级**:
1. `n_estimators` × `learning_rate`：大量树 + 低学习率 = 更好的泛化
2. `max_depth`：3-5 适合大多数场景
3. `min_samples_leaf` / `min_samples_split`：越大 → 越保守 → 越不容易过拟合
4. `loss='huber'`：Huber 损失结合了 MSE（小误差时）和 MAE（大误差时），对异常值鲁棒
5. `max_features='sqrt'`：每次分裂随机选择 sqrt(n_features) 个特征，增加树之间的多样性

#### 5e. XGBoost

```python
import xgboost as xgb

model_xgb = xgb.XGBRegressor(
    colsample_bytree=0.4603,    # 每棵树随机采样的特征比例
    gamma=0.0468,               # 分裂所需的最小损失减少量（越大越保守）
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=1.7817,    # 子节点最小权重和（越大越保守）
    n_estimators=2200,
    reg_alpha=0.4640,           # L1 正则化
    reg_lambda=0.8571,          # L2 正则化
    subsample=0.5213,           # 每棵树随机采样的样本比例
    random_state=7,
    nthread=-1
)
```

**XGBoost 关键参数**:
- `colsample_bytree` + `subsample`：双重随机采样，极大减少过拟合
- `gamma`：分裂的最小增益，起到剪枝作用
- `reg_alpha` / `reg_lambda`：内置 L1/L2 正则化
- `min_child_weight`：类似 min_samples_leaf，阻值学噪声

#### 5f. LightGBM

```python
import lightgbm as lgb

model_lgb = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=5,               # 叶节点数（小 → 简单树 → 防过拟合）
    learning_rate=0.05,
    n_estimators=720,
    max_bin=55,                 # 特征分桶数（小 = 更快但可能欠拟合）
    bagging_fraction=0.8,       # Bagging 采样比例
    bagging_freq=5,             # 每 5 轮迭代做一次 Bagging
    feature_fraction=0.2319,    # 特征采样比例
    feature_fraction_seed=9,
    bagging_seed=9,
    min_data_in_leaf=6,         # 叶节点最少数据量
    min_sum_hessian_in_leaf=11  # 叶节点最小 Hessian 和（类似 min_child_weight）
)
```

**LightGBM vs XGBoost**:
- LightGBM 使用 leaf-wise 生长策略（XGBoost 是 level-wise）→ 更快、更准确但易过拟合
- `num_leaves` 必须 ≤ 2^max_depth（LightGBM 的核心控制参数）
- 配合 `min_data_in_leaf` 和 `min_sum_hessian_in_leaf` 防止过拟合
- `bagging_fraction` + `bagging_freq`：内置 Bagging 减少方差

### Step 6: 交叉验证策略

针对回归任务，使用 K-Fold CV 配合 RMSE 评分：

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
import numpy as np

n_folds = 5

def rmsle_cv(model):
    """计算交叉验证的 RMSE（通过对数变换后等价于 RMSLE）"""
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse = np.sqrt(-cross_val_score(
        model, train.values, y_train,
        scoring="neg_mean_squared_error",
        cv=kf
    ))
    return rmse
```

**重要理解**:
- `scoring="neg_mean_squared_error"`：sklearn 返回负的 MSE，取反再开根号得到 RMSE
- 因为目标变量已经做了 `log1p`，所以这里的 RMSE 实际上就是 RMSLE
- 公式关系：`RMSE(log(y)) = RMSLE(y)` 当 y_pred 和 y_true 都在 log 空间中

**RMSE vs RMSLE**:
| 指标 | 公式 | 特点 |
|------|------|------|
| RMSE | √(平均((ŷ - y)²)) | 对大误差敏感，受异常值影响大 |
| RMSLE | √(平均((log(ŷ+1) - log(y+1))²)) | 关注相对误差，对异常值鲁棒 |
| 实现技巧 | 对 y 做 log1p → 用 RMSE = RMSLE | 无需单独实现 RMSLE |

### Step 7: 基础模型评分

```python
score = rmsle_cv(lasso)
print(f"Lasso score: {score.mean():.4f} ({score.std():.4f})")

score = rmsle_cv(ENet)
print(f"ElasticNet score: {score.mean():.4f} ({score.std():.4f})")

score = rmsle_cv(KRR)
print(f"Kernel Ridge score: {score.mean():.4f} ({score.std():.4f})")

score = rmsle_cv(GBoost)
print(f"Gradient Boosting score: {score.mean():.4f} ({score.std():.4f})")

score = rmsle_cv(model_xgb)
print(f"XGBoost score: {score.mean():.4f} ({score.std():.4f})")

score = rmsle_cv(model_lgb)
print(f"LGBM score: {score.mean():.4f} ({score.std():.4f})")
```

### Step 8: Stacking Level 1 —— 简单平均

最简单的 Stacking：直接取多个模型的预测平均值。

```python
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        # 克隆原始模型，避免在原模型上重复训练
        self.models_ = [clone(x) for x in self.models]

        # 训练所有克隆模型
        for model in self.models_:
            model.fit(X, y)

        return self

    def predict(self, X):
        # 收集所有模型的预测并取平均值
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)

# 使用
averaged_models = AveragingModels(models=(ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
```

**为什么简单平均已经有效**:
- 不同模型犯不同类型的错误 → 平均后误差互相抵消
- 线性模型（Lasso, ENet）和树模型（GBoost）的偏差来源互补
- 核方法（KRR）捕获非线性，树模型捕获交互效应

### Step 9: Stacking Level 2 —— 元模型堆叠

更高级的方案：用基础模型的 Out-of-Fold（OOF）预测作为特征，训练一个元模型。

```python
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for _ in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        # 为每个 base model 生成 OOF 预测
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 用 OOF 预测训练元模型
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        # 每个 base model 在所有 fold 上的预测取平均 → 作为元特征
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])
        return self.meta_model_.predict(meta_features)

# 使用: base models 取平均 + Lasso 作为 meta-model
stacked_averaged_models = StackingAveragedModels(
    base_models=(ENet, GBoost, KRR),
    meta_model=lasso
)
score = rmsle_cv(stacked_averaged_models)
```

**Stacking 的核心原则**:
1. **OOF 预测是关键**：元模型必须看到训练期间未见过的数据，否则会导致数据泄露和过拟合
2. **简单的 meta-model 通常最好**：Lasso 或 Ridge 作为 meta-model，避免过拟合
3. **Base models 需要多样化**：相关性太高的模型堆叠意义不大

**完整 Stacking 流程**:
```
Training Set (A+B, y known)
  ↓ KFold split (n_folds=5)
  ├─ Fold 1-4 (train) → train base model → predict on Fold 5 (holdout)
  ├─ Fold 2-5 (train) → train base model → predict on Fold 1 (holdout)
  ├─ ... repeat for all folds
  ↓
OOF predictions (size = n_train × n_base_models)
  ↓ train meta-model on (OOF_preds, y)
Test Set (C)
  ↓ predict with all fold-models → average → meta_features
  ↓ predict with meta-model
Final Prediction
```

**OOF (Out-of-Fold) 学习流水线图解**:
```
Round 1: train on [Fold2+Fold3+Fold4+Fold5], predict Fold1 → OOF₁
Round 2: train on [Fold1+Fold3+Fold4+Fold5], predict Fold2 → OOF₂
Round 3: train on [Fold1+Fold2+Fold4+Fold5], predict Fold3 → OOF₃
Round 4: train on [Fold1+Fold2+Fold3+Fold5], predict Fold4 → OOF₄
Round 5: train on [Fold1+Fold2+Fold3+Fold4], predict Fold5 → OOF₅

Meta-features = [OOF₁, OOF₂, OOF₃, OOF₄, OOF₅]  → 训练元模型
```

### Step 10: 多层级集成 —— 加权平均最终预测

将 Stacking 模型的结果与 XGBoost、LightGBM 的独立预测做加权融合：

```python
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

# 训练 StackedRegressor
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
# 注意：预测后需要 expm1 还原到原始价格空间
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

# 训练 XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))

# 训练 LightGBM
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))

# 加权平均（权重基于 CV 表现和经验调优）
print('RMSLE score on train data:')
print(rmsle(y_train, stacked_train_pred * 0.70 +
                     xgb_train_pred * 0.15 +
                     lgb_train_pred * 0.15))

# 最终集成预测
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15
```

**权重分配原则**:
- Stacking 模型（0.70）：表现最好的模型，权重最高
- XGBoost（0.15）：与 Stacking 互补的梯度提升方法
- LightGBM（0.15）：不同的提升策略，进一步增加多样性
- 权重和为 1.0，可根据各模型 CV 表现调整

## 完整 Pipeline 总结

```
1. 数据分析
   ├─ 目标变量分布检查（偏态检测）
   ├─ 异常值散点图
   └─ 缺失值分析

2. 数据预处理
   ├─ 删除极端异常值
   ├─ 缺失值填充（领域知识驱动）
   ├─ 特征类型转换（数值→分类）
   └─ 删除无用特征

3. 特征工程
   ├─ Label Encoding（有序分类变量）
   ├─ Box-Cox 变换（偏态数值特征，lam=0.15）
   ├─ 组合特征（如 TotalSF）
   └─ One-Hot Encoding

4. 目标变换
   └─ log1p 变换（使目标变量接近正态分布）

5. 模型训练（5-Fold CV，RMSLE 评估）
   ├─ Lasso + RobustScaler
   ├─ ElasticNet + RobustScaler
   ├─ KernelRidge (polynomial kernel)
   ├─ GradientBoostingRegressor (huber loss)
   ├─ XGBoost
   └─ LightGBM

6. Stacking
   ├─ Level 1: AveragingModels (简单平均)
   ├─ Level 2: StackingAveragedModels (OOF + meta-model)
   └─ 元模型：Lasso

7. 最终集成
   └─ 加权平均：Stacked × 0.70 + XGB × 0.15 + LGB × 0.15

8. 预测还原
   └─ np.expm1(pred) 还原到原始价格尺度
```

## Key Insights

1. **log1p + RMSE = RMSLE**：对目标做对数变换后使用 RMSE，等价于在原始尺度用 RMSLE。这既简化了实现，又让线性模型受益于正态分布假设。

2. **RobustScaler 是回归的关键组件**：Lasso 和 ElasticNet 对异常值极其敏感。`RobustScaler`（基于中位数和四分位距）比 `StandardScaler`（基于均值和标准差）更能抵抗异常值的干扰。

3. **Box-Cox λ=0.15 的经验价值**：对于房价类数据，λ=0.15 的 Box-Cox 变换往往优于纯 log 变换（λ=0），因为它更温和，能保留更多原始数据的结构信息。

4. **Huber Loss 让梯度提升更鲁棒**：`loss='huber'` 在处理有异常值的回归问题时是默认选择。

5. **Stacking 的核心是 OOF**：永远不要在训练数据上生成 meta-features。必须使用交叉验证的 holdout 预测，确保元模型评估的是"模型在未见数据上的表现"。

6. **多样性 > 数量**：堆叠 3 个差异大的模型（线性 + 核方法 + 树模型）比堆叠 10 个相似的树模型更有效。

## Common Pitfalls

1. **忘记 expm1 还原**：在 log 空间训练，预测时没有 `expm1` 会导致预测值完全错误
2. **未对测试集做相同的预处理**：Box-Cox、编码器、缺失值填充必须对测试集使用训练集的参数
3. **OOF 预测泄露**：元模型的训练数据必须来自未参与训练的 fold，否则严重过拟合
4. **异常值删除过度**：只删除"明显不可能"的点，保留数据多样性
5. **元模型过于复杂**：Lasso/Ridge 作为 meta-model 通常已足够，Deep NN 会过拟合 meta-features
6. **忽略特征偏度**：偏态特征不处理会严重损害线性模型的表现

## 模型选择速查表

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 特征多/有共线性 | ElasticNet | L1+L2 正则化，特征选择 + 处理共线性 |
| 需要特征选择 | Lasso | L1 正则化产生稀疏解 |
| 非线性关系 | KernelRidge | 核方法在特征空间中建模非线性 |
| 大数据集 | LightGBM | Leaf-wise 生长，极快 |
| 需要可解释性 | Lasso / Ridge | 线性系数可直接解释 |
| 最高精度 | GradientBoosting + Stacking | 集成多个强模型 |
| 有异常值 | GBoost(loss='huber') + RobustScaler | Huber 损失 + 鲁棒缩放 |

## Trigger Phrases
- "回归模型"、"房价预测"、"stacked regressions"、"Lasso"、"Ridge"、"KernelRidge"、"ElasticNet"
- "对数变换"、"log transformation"、"target encoding"、"Box-Cox"
- "RMSLE"、"RMSE"、"偏态"、"异常值"、"stacking"、"blending"、"模型堆叠"
- "GradientBoostingRegressor"、"XGBRegressor"、"LGBMRegressor"
- "log1p"、"expm1"、"RobustScaler"

