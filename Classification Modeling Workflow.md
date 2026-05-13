---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: classification-modeling-workflow
---

# Classification Modeling Workflow

**分类:** 📊 Data Science
**Skill ID:** `classification-modeling-workflow`

> 端到端分类建模工作流，从数据探索、特征工程、多模型交叉验证到超参数调优和集成学习。提炼自 Kaggle Titanic Top 4% 经典 notebook。

---


# 分类建模工作流 (Classification Modeling Workflow)

## Purpose
为分类问题提供端到端建模管道：数据准备→特征工程→模型筛选→超参调优→集成学习→预测输出。

## Core Workflow

### Step 1: 加载与整合
```python
import pandas as pd; import numpy as np
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
ntrain = train.shape[0]; target = train['Target']
full = pd.concat([train.drop('Target',axis=1), test], axis=0)
```

### Step 2: 缺失值 + 异常值
```python
missing = full.isnull().sum()
print(pd.DataFrame({'pct': missing/len(full)*100})[missing>0].sort_values('pct', ascending=False))
# 填充：数值用中位数，分类用众数
```

### Step 3: 特征工程
```python
# 文本提取
full['Title'] = full['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# 合并稀有类别
rare = full['Title'].value_counts()[lambda x: x<5].index
full['Title'] = full['Title'].replace(rare, 'Rare')
# 组合特征
full['FamilySize'] = full['SibSp'] + full['Parch'] + 1
full['IsAlone'] = (full['FamilySize']==1).astype(int)
# 缺失标记
full['Age_missing'] = full['Age'].isnull().astype(int)
```

### Step 4: 编码 + 标准化
```python
full = pd.get_dummies(full, columns=['Sex','Embarked','Title'], drop_first=True)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = full.select_dtypes(include=[np.number]).columns
full[num_cols] = scaler.fit_transform(full[num_cols])
```

### Step 5: 多模型交叉验证
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

X = full[:ntrain]; y = target
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'LR': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'DT': DecisionTreeClassifier(),
    'RF': RandomForestClassifier(n_estimators=100),
    'ET': ExtraTreesClassifier(n_estimators=100),
    'Ada': AdaBoostClassifier(),
    'GB': GradientBoostingClassifier(),
    'SVC': SVC(),
}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Step 6: 超参数调优
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':[100,200,300], 'max_depth':[3,5,7,None], 'min_samples_split':[2,5,10]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X, y)
print(f"Best: {gs.best_score_:.4f} | {gs.best_params_}")
```

### Step 7: 学习曲线 + 特征重要性
```python
from sklearn.model_selection import learning_curve
train_sizes, train_scores, val_scores = learning_curve(gs.best_estimator_, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,10))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(train_sizes, val_scores.mean(axis=1), 'o-', label='Val')
plt.legend(); plt.show()

importances = pd.DataFrame({'feature':X.columns, 'importance':gs.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
plt.barh(importances['feature'][:15][::-1], importances['importance'][:15][::-1])
plt.show()
```

### Step 8: 集成学习
```python
from sklearn.ensemble import VotingClassifier
voting = VotingClassifier(estimators=[('rf',RandomForestClassifier(**rf_params)),('gb',GradientBoostingClassifier(**gb_params))], voting='soft')
voting.fit(X, y)
predictions = voting.predict(full[ntrain:].drop('PassengerId',axis=1))
```

## Quality Checklist
- [ ] 训练/测试是否先合并再处理？
- [ ] 是否测试了至少5种模型？
- [ ] 是否用了交叉验证？
- [ ] 超参数是否调优？
- [ ] 是否检查了学习曲线？
- [ ] 是否尝试了集成？

## Trigger Phrases
- "帮我建个分类模型"、"预测二分类结果"、"对比模型效果"

