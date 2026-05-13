---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: feature-engineering-handbook
---

# Feature Engineering Handbook

**分类:** 📊 Data Science
**Skill ID:** `feature-engineering-handbook`

> 特征工程全手册，覆盖文本提取、组合特征、缺失标记、编码转换、分箱、标准化、日期分解等核心技巧，提炼自多个 Kaggle 经典 notebook。

---


# 特征工程手册 (Feature Engineering Handbook)

## Purpose
从原始数据中创建能提升模型性能的新特征，是建模流程中 ROI 最高的环节。覆盖文本抽取、数值变换、组合特征、缺失标记、编码转换等技术。

## When to use

- 基础 EDA 完成后，建模前。
- 模型准确率停滞，需要新特征突破瓶颈。
- 原始列直接入模效果差。
- 用户说"帮我做特征工程"、"创建新特征"。

## Do not use when

- 数据已经是经过精心设计的特征表。
- 模型已经是预训练模型（如 BERT、GPT），有自己的 tokenizer。
- 数据集极小（特征过多容易过拟合）。

## Core Techniques (核心技巧)

### 1. 文本特征提取 (Text Feature Extraction)

```python
# 1a. 正则提取 — 从字符串中抽结构化信息
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# 1b. 首字母/前缀
df['First_letter'] = df['Name'].str[0]
df['Cabin_deck'] = df['Cabin'].str[0]  # 甲板层

# 1c. 字符串长度
df['Name_length'] = df['Name'].str.len()
df['Word_count'] = df['Description'].str.split().str.len()

# 1d. 包含关键词
df['Has_premium'] = df['Description'].str.contains('premium', case=False).astype(int)

# 1e. 稀有类别合并
title_counts = df['Title'].value_counts()
rare_titles = title_counts[title_counts < 5].index
df['Title'] = df['Title'].replace(rare_titles, 'Rare')

# 映射同类项
df['Title'] = df['Title'].replace({
    'Mlle': 'Miss', 'Ms': 'Miss',
    'Mme': 'Mrs', 'Lady': 'Rare',
    'Countess': 'Rare', 'Sir': 'Rare',
    'Jonkheer': 'Rare', 'Don': 'Rare',
    'Dona': 'Rare',
})
```

### 2. 组合特征 (Combination Features)

```python
# 2a. 加减乘除组合
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Total_rooms'] = df['BedroomAbvGr'] + df['KitchenAbvGr']
df['Ratio'] = df['Column_A'] / (df['Column_B'] + 1e-6)

# 2b. 布尔组合
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Has_multiple_conditions'] = (
    (df['Condition_1'] > 0) & (df['Condition_2'] > 0)
).astype(int)

# 2c. 多条件分箱组合
def classify_size(row):
    if row['FamilySize'] == 1: return 'Alone'
    elif row['FamilySize'] <= 4: return 'Small'
    else: return 'Large'
df['FamilySizeGroup'] = df.apply(classify_size, axis=1)
```

### 3. 缺失值作为特征 (Missing as Feature)

```python
# 缺失本身可能是重要信号！
df['Age_missing'] = df['Age'].isnull().astype(int)
df['Cabin_missing'] = df['Cabin'].isnull().astype(int)
df['Fare_zero'] = (df['Fare'] == 0).astype(int)

# 缺失占比（多列）
df['missing_count'] = df.isnull().sum(axis=1)
df['missing_pct'] = df.isnull().sum(axis=1) / df.shape[1]
```

### 4. 数值变换 (Numeric Transformations)

```python
# 4a. Log 变换（处理长尾分布）
df['Log_price'] = np.log1p(df['Price'])  # log(1+x)，避免 log(0)

# 4b. Box-Cox 变换（正态化）
from scipy import stats
df['BoxCox_price'], lambda_param = stats.boxcox(df['Price'] + 1)

# 4c. 平方/平方根
df['Sqrt_area'] = np.sqrt(df['Area'])
df['Square_size'] = df['Size'] ** 2

# 4d. 交互项
df['Age_Income'] = df['Age'] * df['Income']
```

### 5. 分箱 (Binning / Discretization)

```python
# 5a. 等距分箱
df['Age_bin'] = pd.cut(df['Age'], bins=5, labels=['0-20', '20-40', '40-60', '60-80', '80+'])

# 5b. 等频分箱
df['Income_quantile'] = pd.qcut(df['Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

# 5c. 业务逻辑分箱
bins = [0, 12, 18, 35, 60, 100]
labels = ['Child', 'Teenager', 'YoungAdult', 'Adult', 'Senior']
df['Age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
```

### 6. 编码转换 (Encoding)

```python
# 6a. One-Hot Encoding（低基数分类变量）
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# 6b. Label Encoding（树模型适用）
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['City_encoded'] = le.fit_transform(df['City'])

# 6c. Ordinal Encoding（有序分类）
from sklearn.preprocessing import OrdinalEncoder
ordinal_map = [['Low', 'Medium', 'High']]  # 指定顺序
oe = OrdinalEncoder(categories=ordinal_map)
df['Quality_encoded'] = oe.fit_transform(df[['Quality']])

# 6d. Frequency Encoding（高基数分类变量）
freq = df['Product_ID'].value_counts(normalize=True)
df['Product_freq'] = df['Product_ID'].map(freq)

# 6e. Target Encoding（有监督）
# 按分类计算目标变量的均值
target_mean = df.groupby('Category')['Target'].mean()
df['Category_target_mean'] = df['Category'].map(target_mean)
# 注意：必须用交叉验证避免数据泄露！
```

### 7. 日期时间分解 (Date/Time Decomposition)

```python
df['date'] = pd.to_datetime(df['date_column'])

# 提取时间组件
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek          # 0=Monday
df['quarter'] = df['date'].dt.quarter
df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
df['hour'] = df['date'].dt.hour

# 布尔时间特征
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)

# 距离某日期的天数
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
df['days_to_event'] = (pd.Timestamp('2024-12-25') - df['date']).dt.days

# 循环编码（保留周期性，如小时、月份）
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
```

### 8. 聚合统计特征 (Aggregation Features)

```python
# 按分组计算统计量
df['avg_price_by_category'] = df.groupby('Category')['Price'].transform('mean')
df['max_price_by_category'] = df.groupby('Category')['Price'].transform('max')
df['count_by_category'] = df.groupby('Category')['Price'].transform('count')
df['std_price_by_category'] = df.groupby('Category')['Price'].transform('std')

# 排名特征
df['price_rank_in_category'] = df.groupby('Category')['Price'].rank(pct=True)
```

### 9. 标准化 (Scaling)

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Z-score 标准化（均值0，标准差1）— 适用于线性模型
scaler = StandardScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

# MinMax 缩放（0-1范围）— 适用于神经网络
scaler = MinMaxScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])

# Robust 缩放（对异常值不敏感）— 用中位数和 IQR
scaler = RobustScaler()
df[['Age', 'Income']] = scaler.fit_transform(df[['Age', 'Income']])
```

### 10. 特征选择 (Feature Selection)

```python
# 10a. 基于重要性选择（树模型）
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# 保留 Top N 特征
top_features = importances.head(20)['feature'].tolist()
X_train_selected = X_train[top_features]

# 10b. 去除高相关特征（>0.95）
corr_matrix = X_train.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
X_train_uncorr = X_train.drop(to_drop, axis=1)

# 10c. 去除低方差特征
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)  # 方差<1%的特征
X_train_var = selector.fit_transform(X_train)
```

## Feature Engineering Pipeline Template

```python
def create_features(df, is_train=True):
    """一站式特征工程函数"""
    df = df.copy()
    
    # 1. 文本特征
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss').replace('Mme', 'Mrs')
    df['Title'] = df['Title'].replace(df['Title'].value_counts()[lambda x: x<5].index, 'Rare')
    
    # 2. 组合特征
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # 3. 缺失标记
    df['Age_missing'] = df['Age'].isnull().astype(int)
    
    # 4. 编码
    df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    # 5. 数值变换
    df['LogFare'] = np.log1p(df['Fare'])
    
    # 6. 清理无用列
    drop_cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)
    
    return df
```

## Quality Checklist (自检表)

- [ ] 文本中是否提取了结构化信息？
- [ ] 是否有组合特征（加减乘除、布尔组合）？
- [ ] 缺失值是否被标记为特征？
- [ ] 是否需要 Log/Box-Cox 变换处理偏态？
- [ ] 分类变量是否选择了合适的编码方式？
- [ ] 日期是否被完全分解（年/月/日/星期/小时/循环编码）？
- [ ] 是否有按分组计算的聚合统计特征？
- [ ] 是否有冗余/高相关特征需要剔除？
- [ ] 标准化方式是否与模型匹配（线性→StandardScaler，树→不需要）？

## Common Pitfalls

1. **数据泄露**：Target Encoding 不能用全量数据算，必须在每个 CV fold 内单独计算。
2. **编码后爆炸**：高基数特征用 One-Hot 会产生成千上万个新列，改用 Frequency/Label/Target Encoding。
3. **标准化不分 train/test**：先 fit train，再 transform test，不要用全量数据 fit。
4. **Log 变换忘记 +1**：log(0) = -inf，必须 log1p 或 +1。
5. **特征过多导致过拟合**：特征选择要走在建模前面。

## Trigger Phrases (触发词)

- "做特征工程"
- "创建新特征"
- "特征提取/特征构造"
- "准备建模数据"
- "数据预处理和编码"

## Relationship to Other Skills

- 前置：`data-cleaning-pipeline`（先清洗再工程）
- 前置：`enhanced-exploratory-data-analysis`（从 EDA 中获取特征灵感）
- 后置：`classification-modeling-workflow`（特征工程后建模）

