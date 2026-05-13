---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: end-to-end-ml-dietanic
---

# END TO END ML Dietanic

**分类:** 📊 Data Science
**Skill ID:** `end-to-end-ml-dietanic`

> Complete ML project workflow from EDA through feature engineering, multi-model comparison, GridSearchCV, and ensembling (Voting, Bagging, Boosting, XGBoost). Distilled from Kaggle ash316/eda-to-prediction-dietanic (6,314 votes, 149 cells).

---


# End-to-End ML Workflow (DieTanic)

## Purpose
Complete machine learning project workflow distilled from a 6,314-vote Kaggle notebook. Covers the full pipeline: EDA → data cleaning → feature engineering → multi-model comparison → cross-validation → hyperparameter tuning → ensembling.

## Phase 1: Exploratory Data Analysis (EDA)

### 1.1 Quick Data Inspection
```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

data = pd.read_csv('train.csv')
data.head()
data.isnull().sum()  # Identify missing columns
data.describe()
```

### 1.2 Target Distribution
```python
f, ax = plt.subplots(1, 2, figsize=(18, 8))
data['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0])
sns.countplot(x='Survived', data=data, ax=ax[1])
plt.show()
```

### 1.3 Categorical Features vs Target
```python
# Sex vs Survived
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar()
sns.countplot(x='Sex', hue='Survived', data=data)

# Pclass vs Survived
pd.crosstab(data.Pclass, data.Survived, margins=True)
sns.countplot(x='Pclass', hue='Survived', data=data)

# Multi-way analysis
sns.factorplot('Pclass', 'Survived', hue='Sex', data=data)
sns.factorplot('Pclass', 'Survived', hue='Sex', col='Embarked', data=data)
```

### 1.4 Continuous Features Analysis
```python
# Age distribution
f, ax = plt.subplots(1, 2, figsize=(18, 8))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=data, split=True, ax=ax[0])
sns.violinplot(x='Sex', y='Age', hue='Survived', data=data, split=True, ax=ax[1])

# Fare distribution by Pclass
f, ax = plt.subplots(1, 3, figsize=(20, 8))
for i, pclass in enumerate([1, 2, 3]):
    sns.distplot(data[data['Pclass']==pclass].Fare, ax=ax[i])
    ax[i].set_title(f'Fares in Pclass {pclass}')
```

### 1.5 Correlation Heatmap
```python
sns.heatmap(data.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
```

## Phase 2: Data Cleaning & Feature Engineering

### 2.1 Smart Missing Value Imputation (Title-Based)
```python
# Extract title from Name
data['Initial'] = data.Name.str.extract('([A-Za-z]+)\.')
data['Initial'].replace(
    ['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
    ['Miss','Miss','Miss','Other','Other','Other','Other','Other','Other','Other','Other','Other','Other']
)

# Fill Age by title group means
avg_ages = data.groupby('Initial')['Age'].mean()
data.loc[(data.Age.isnull()) & (data.Initial=='Mr'), 'Age'] = 33
data.loc[(data.Age.isnull()) & (data.Initial=='Mrs'), 'Age'] = 36
data.loc[(data.Age.isnull()) & (data.Initial=='Master'), 'Age'] = 5
data.loc[(data.Age.isnull()) & (data.Initial=='Miss'), 'Age'] = 22
```

### 2.2 Feature Creation
```python
# Age bands
data['Age_band'] = 0
data.loc[data['Age'] <= 16, 'Age_band'] = 0
data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age_band'] = 1
data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age_band'] = 2
data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age_band'] = 3
data.loc[data['Age'] > 64, 'Age_band'] = 4

# Family size and Alone flag
data['Family_Size'] = data['Parch'] + data['SibSp']
data['Alone'] = (data['Family_Size'] == 0).astype(int)

# Fare categorization
data['Fare_cat'] = 0
data.loc[data['Fare'] <= 7.91, 'Fare_cat'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_cat'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_cat'] = 2
data.loc[data['Fare'] > 31, 'Fare_cat'] = 3
```

### 2.3 Encoding
```python
# Manual map (preferred over LabelEncoder for small categories)
data['Sex'].replace(['male', 'female'], [0, 1], inplace=True)
data['Embarked'].replace(['S', 'C', 'Q'], [0, 1, 2], inplace=True)
data['Initial'].replace(['Mr','Mrs','Miss','Master','Other'], [0,1,2,3,4], inplace=True)

# Drop unused columns
data.drop(['Name', 'Age', 'Ticket', 'Fare', 'Cabin', 'Fare_Range', 'PassengerId'], axis=1, inplace=True)
```

## Phase 3: Multi-Model Comparison

```python
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

# Train/test split with stratification
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['Survived'])
train_X = train[train.columns[1:]]
train_Y = train[train.columns[:1]]
test_X = test[test.columns[1:]]
test_Y = test[test.columns[:1]]
X = data[data.columns[1:]]
Y = data['Survived']

# Models dictionary
models = {
    'rbf-SVM': svm.SVC(kernel='rbf', C=1, gamma=0.1),
    'linear-SVM': svm.SVC(kernel='linear', C=0.1, gamma=0.1),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100)
}

# Train and evaluate
for name, model in models.items():
    model.fit(train_X, train_Y.values.ravel())
    pred = model.predict(test_X)
    acc = metrics.accuracy_score(test_Y, pred)
    print(f'{name:25s}: {acc:.4f}')
```

## Phase 4: KNN n_neighbors Tuning

```python
a_index = list(range(1, 11))
accuracies = []

for i in a_index:
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(train_X, train_Y.values.ravel())
    pred = model.predict(test_X)
    accuracies.append(metrics.accuracy_score(test_Y, pred))

plt.plot(a_index, accuracies, marker='o')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
print(f'Best n: {a_index[np.argmax(accuracies)]}, Accuracy: {max(accuracies):.4f}')
```

## Phase 5: K-Fold Cross-Validation

```python
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict

kfold = KFold(n_splits=10, random_state=22, shuffle=True)
classifiers = ['rbf-SVM', 'linear-SVM', 'LR', 'DT', 'KNN', 'NB', 'RF']
accuracy = []

for clf_name in classifiers:
    model = models[clf_name]
    cv_result = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
    accuracy.append(cv_result)
    print(f'{clf_name:15s}: {cv_result.mean():.4f} (+/- {cv_result.std():.4f})')

# Box plot comparison
plt.subplots(figsize=(12, 6))
pd.DataFrame(accuracy, index=classifiers).T.boxplot()
plt.title('Cross-Validation Accuracy Distribution')
```

## Phase 6: Confusion Matrices

```python
f, ax = plt.subplots(3, 3, figsize=(12, 10))
y_pred = cross_val_predict(svm.SVC(kernel='rbf'), X, Y, cv=10)
sns.heatmap(confusion_matrix(Y, y_pred), ax=ax[0,0], annot=True, fmt='2.0f')
ax[0,0].set_title('rbf-SVM')
# ... repeat for all models
```

## Phase 7: GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# SVM Grid Search
C = [0.05, 0.1, 0.2, 0.3, 0.5, 1]
gamma = [0.1, 0.2, 0.3, 0.4, 0.5]
kernel = ['rbf', 'linear']
hyper = {'kernel': kernel, 'C': C, 'gamma': gamma}

gd = GridSearchCV(estimator=svm.SVC(), param_grid=hyper, verbose=True)
gd.fit(X, Y)
print(gd.best_score_)
print(gd.best_estimator_)

# RF Grid Search
n_estimators = range(100, 1000, 100)
hyper = {'n_estimators': list(n_estimators)}
gd = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=hyper, verbose=True)
gd.fit(X, Y)
print(gd.best_score_)
```

## Phase 8: Ensembling

### 8.1 Voting Classifier
```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[
    ('KNN', KNeighborsClassifier(n_neighbors=10)),
    ('RBF', svm.SVC(probability=True, kernel='rbf', C=0.5, gamma=0.1)),
    ('RF', RandomForestClassifier(n_estimators=500, random_state=0)),
    ('LR', LogisticRegression(C=0.05)),
    ('DT', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('linear-SVM', svm.SVC(probability=True, kernel='linear', C=0.1))
], voting='soft')

cv_score = cross_val_score(ensemble, X, Y, cv=10, scoring='accuracy')
print(f'Voting Ensemble CV: {cv_score.mean():.4f}')
```

### 8.2 Bagging
```python
from sklearn.ensemble import BaggingClassifier

# Bagged KNN
model = BaggingClassifier(
    base_estimator=KNeighborsClassifier(n_neighbors=3),
    random_state=0, n_estimators=100
)
cv = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print(f'Bagged KNN CV: {cv.mean():.4f}')

# Bagged Decision Tree
model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    random_state=0, n_estimators=100
)
cv = cross_val_score(model, X, Y, cv=10, scoring='accuracy')
print(f'Bagged DT CV: {cv.mean():.4f}')
```

### 8.3 AdaBoost
```python
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier(n_estimators=200, random_state=0, learning_rate=0.1)
cv = cross_val_score(ada, X, Y, cv=10, scoring='accuracy')
print(f'AdaBoost CV: {cv.mean():.4f}')

# Tuning AdaBoost
n_estimators = list(range(100, 1100, 100))
learn_rate = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1]
hyper = {'n_estimators': n_estimators, 'learning_rate': learn_rate}
gd = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=hyper, verbose=True)
gd.fit(X, Y)
```

### 8.4 Gradient Boosting + XGBoost
```python
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xg

# GBM
grad = GradientBoostingClassifier(n_estimators=500, random_state=0, learning_rate=0.1)
cv = cross_val_score(grad, X, Y, cv=10, scoring='accuracy')

# XGBoost
xgboost = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
cv = cross_val_score(xgboost, X, Y, cv=10, scoring='accuracy')
```

## Phase 9: Feature Importance Comparison

```python
f, ax = plt.subplots(2, 2, figsize=(15, 12))

# Random Forest
model = RandomForestClassifier(n_estimators=500, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(ax=ax[0, 0])
ax[0, 0].set_title('Random Forest')

# AdaBoost
model = AdaBoostClassifier(n_estimators=200, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(ax=ax[0, 1])
ax[0, 1].set_title('AdaBoost')

# Gradient Boosting
model = GradientBoostingClassifier(n_estimators=500, random_state=0)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(ax=ax[1, 0])
ax[1, 0].set_title('Gradient Boosting')

# XGBoost
model = xg.XGBClassifier(n_estimators=900, learning_rate=0.1)
model.fit(X, Y)
pd.Series(model.feature_importances_, X.columns).sort_values(ascending=True).plot.barh(ax=ax[1, 1])
ax[1, 1].set_title('XGBoost')

plt.show()
```

## Complete Workflow Summary

```
1. EDA: Null check → pie chart → bar/violin plots → corr heatmap
2. Feature Engineering: Title extraction → Age imputation → bands + family + fare_cat
3. Encoding: Manual map() for small cardinality categories
4. Split: train_test_split with stratify
5. Multi-Model: Run 6-8 classifiers with default params
6. CV: K-Fold (10 splits) with cross_val_score
7. Confusion Matrices: cross_val_predict for each model
8. GridSearchCV: Tune SVM + RF + AdaBoost
9. Ensemble: Voting → Bagging → AdaBoost → GBM → XGBoost
10. Feature Importance: Compare across 4 models
```

## Key Insights from the Notebook
1. **Title-based imputation** beats mean/median for missing Age
2. **Family_Size=0 (Alone)** is a strong survival predictor
3. **Voting Classifier** with diverse base models outperforms individual models
4. **AdaBoost with tuned lr=0.05** achieved best single-model performance (83.16%)
5. **Feature importance varies across models** — compare don't trust one

## Trigger Keywords
中文: 完整建模, 端到端, 模型对比, 集成学习, GridSearch, 混淆矩阵
English: end-to-end ML, model comparison, ensemble, grid search, confusion matrix

