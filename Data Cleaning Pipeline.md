---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: data-cleaning-pipeline
---

# Data Cleaning Pipeline

**分类:** 📊 Data Science
**Skill ID:** `data-cleaning-pipeline`

> 系统性的数据清洗管道，处理缺失值、异常值、重复值，确保数据质量达到建模标准。提炼自 Kaggle 15K+ votes 经典 notebook。

---


# 数据清洗管道

## Purpose
在建模或分析之前，系统性地检查和处理数据质量问题：缺失值、异常值、重复值、类型错误。

## Prerequisite: Data Role Confirmation ⚠️
Before cleaning, the user must confirm which columns are Target, Features, and Categorical/Numerical.
If the user has not provided this, stop and ask:
```
Before cleaning, please confirm:
- Target column: ______
- Feature columns: ______  
- Categorical: ______
- Numerical: ______
```
Do NOT infer these roles from statistics — same column can be target in one analysis and feature in another.

## Core Workflow

### Step 1: 初览
```python
import pandas as pd
import numpy as np
df = pd.read_csv('data.csv')
print(f"Shape: {df.shape}\n{df.dtypes}")
```

### Step 2: 缺失值诊断
```python
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
print(pd.DataFrame({'count': missing, 'pct': missing_pct})[missing > 0].sort_values('pct', ascending=False))
```
分类：MCAR→删除 / MAR→预测填充 / MNAR→标记为特征

### Step 3: 处理
- 列缺失>50%：整列删除
- 行缺失<5%：`df.dropna(subset=['col'])`
- 数值：median填充 / 分类：mode填充 / 时序：ffill

### Step 4: 异常值
- IQR：Q1-1.5IQR ~ Q3+1.5IQR
- Z-score：|z| > 3
- 处理：clip(Winsorize) 或标记为特征或删除

### Step 5: 重复值
- `df.duplicated().sum()` / `df.duplicated(subset=['id']).sum()`

### Step 6: 类型修正
- `pd.to_numeric(errors='coerce')` / `pd.to_datetime(errors='coerce')` / downcast

## Common Pitfalls
1. 偏态数据用均值→用中位数 2. 忽略缺失业务含义→标记 3. 不保留清洗日志

## Trigger Phrases
"清理数据" / "检查数据质量" / "处理缺失值"

