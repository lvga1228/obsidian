---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: recommender-systems
---

# Recommender Systems

**分类:** 📊 Data Science
**Skill ID:** `recommender-systems`

> 推荐系统全栈框架，从简单评分排序、基于内容的推荐、协同过滤到混合推荐。覆盖 TF-IDF + 余弦相似度、Surprise SVD 矩阵分解。提炼自 Kaggle 5.3K votes 经典 notebook。

---


# 推荐系统 (Recommender Systems)

## Purpose
为用户/物品构建推荐引擎，涵盖三种主流方法：简单推荐（普适热门）、基于内容推荐（物品相似度）、协同过滤（用户行为矩阵分解），最终组合成混合推荐。

## When to use

- 需要为用户推荐内容（电影、商品、文章）。
- 有用户-物品交互数据（评分、点击、购买）。
- 物品有文本描述/元数据可用于内容匹配。
- 用户说"做推荐系统"、"物品相似度"、"猜你喜欢"。

## Do not use when

- 数据完全是新品/新用户（冷启动）— 先用内容推荐。
- 没有用户行为数据 — 只能用基于内容的推荐。
- 实时性要求极高（毫秒级）— 需预计算 + 缓存。

## Core Workflow (核心工作流)

### 1. 简单推荐 (Simple Recommender)
基于全局统计，不考虑用户偏好。

```python
import pandas as pd
import numpy as np

df = pd.read_csv('movies_metadata.csv')

# 加权评分：IMDB 风格
# WR = (v/(v+m) * R) + (m/(v+m) * C)
# R = 物品平均评分
# v = 评分数量
# m = 最少评分阈值（如 95 分位数）
# C = 全局平均评分

m = df['vote_count'].quantile(0.95)  # 最少需要排多少名
C = df['vote_average'].mean()

qualified = df[df['vote_count'] >= m].copy()

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(v+m) * C)

qualified['score'] = qualified.apply(weighted_rating, axis=1)
top_movies = qualified.sort_values('score', ascending=False)

print("Top 10 movies:")
print(top_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

# 按类别筛选
def build_chart(genre, percentile=0.85):
    genre_movies = df[df['genres'].str.contains(genre, na=False)]
    m = genre_movies['vote_count'].quantile(percentile)
    qualified = genre_movies[genre_movies['vote_count'] >= m]
    qualified['score'] = qualified.apply(weighted_rating, axis=1)
    return qualified.sort_values('score', ascending=False)
```

### 2. 基于内容推荐 (Content-Based)
用物品描述/元数据计算相似度。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

# 2a. 基于文本描述
df['description'] = df['overview'].fillna('')

tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
)
tfidf_matrix = tfidf.fit_transform(df['description'])

# 计算余弦相似度矩阵
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 构建索引映射
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def get_recommendations(title, cosine_sim=cosine_sim, n=10):
    """基于内容的推荐"""
    idx = indices[title]
    
    # 获取该电影与其他所有电影的相似度
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # 跳过自己
    
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

print(get_recommendations('The Dark Knight'))
```

### 2b. 基于元数据（演职员、关键词）
```python
from ast import literal_eval

# 解析 JSON 字段
df['cast'] = df['cast'].apply(literal_eval)
df['crew'] = df['crew'].apply(literal_eval)
df['keywords'] = df['keywords'].apply(literal_eval)

# 提取导演
def get_director(x):
    for member in x:
        if member['job'] == 'Director':
            return member['name']
    return ''
df['director'] = df['crew'].apply(get_director)

# 提取主要演员（Top 3）
df['cast_names'] = df['cast'].apply(
    lambda x: [i['name'] for i in x[:3]] if isinstance(x, list) else []
)

# 提取关键词
df['keywords_list'] = df['keywords'].apply(
    lambda x: [i['name'] for i in x] if isinstance(x, list) else []
)

# 标准化名称（去掉空格，小写）
def clean_names(x):
    if isinstance(x, list):
        return [str.lower(name.replace(' ', '')) for name in x]
    return []

df['cast_clean'] = df['cast_names'].apply(clean_names)
df['director_clean'] = df['director'].apply(
    lambda x: str.lower(x.replace(' ', '')) if x else ''
)
df['keywords_clean'] = df['keywords_list'].apply(clean_names)

# 拼接特征字符串
def create_soup(x):
    return ' '.join(x['keywords_clean']) + ' ' + ' '.join(x['cast_clean']) + ' ' + x['director_clean']

df['soup'] = df.apply(create_soup, axis=1)

# 用 CountVectorizer + 余弦相似度
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english', max_features=5000)
count_matrix = count.fit_transform(df['soup'])
cosine_sim_meta = cosine_similarity(count_matrix, count_matrix)
```

### 3. 协同过滤 (Collaborative Filtering)
基于用户-物品交互矩阵。

```python
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Surprise 库需要: user_id, item_id, rating
ratings = pd.read_csv('ratings.csv')
ratings = ratings[['userId', 'movieId', 'rating']]

# 加载数据
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# SVD 矩阵分解
svd = SVD(n_factors=100, n_epochs=20, random_state=42)
cross_validate(svd, data, cv=5, measures=['RMSE', 'MAE'], verbose=True)

# 训练全量数据
trainset = data.build_full_trainset()
svd.fit(trainset)

# 预测用户对物品的评分
user_id = 1
movie_id = 302  # e.g., Inception
pred = svd.predict(user_id, movie_id)
print(f"Predicted rating: {pred.est:.2f}")

# 为用户推荐 Top N
def get_top_n_for_user(user_id, n=10):
    all_movie_ids = ratings['movieId'].unique()
    predictions = [
        (mid, svd.predict(user_id, mid).est)
        for mid in all_movie_ids
    ]
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    top_movies = []
    for mid, score in predictions[:n]:
        title = df[df['id'] == mid]['title'].values
        if len(title) > 0:
            top_movies.append((title[0], score))
    return top_movies
```

### 4. 混合推荐 (Hybrid)
结合内容推荐和协同过滤的结果。

```python
def hybrid_recommendation(user_id, title, n=10):
    """
    混合推荐：
    1. 获取基于内容的推荐（取前30）
    2. 用协同过滤为这些候选项打分
    3. 返回协同过滤分最高的 Top N
    """
    # 基于内容的候选项
    content_recs = get_recommendations(title, n=30)
    content_indices = indices[content_recs].values
    
    # 协同过滤打分
    scored = []
    for idx in content_indices:
        movie_id = df.iloc[idx]['id']
        pred = svd.predict(user_id, movie_id)
        scored.append((idx, pred.est))
    
    # 排序取 Top N
    scored.sort(key=lambda x: x[1], reverse=True)
    top_indices = [s[0] for s in scored[:n]]
    return df['title'].iloc[top_indices]
```

## Quality Checklist (自检表)

- [ ] 是否实现了至少两种推荐方法（内容 + 协同/混合）？
- [ ] 加权评分是否考虑了评分数量（避免小众高分霸榜）？
- [ ] TF-IDF/CountVectorizer 是否设置了 max_features？
- [ ] 元数据字段（cast, crew, keywords）是否正确解析？
- [ ] 协同过滤的 RMSE/MAE 是否合理？
- [ ] 冷启动（新用户/新物品）是否有处理策略？

## Common Pitfalls

1. **只用平均评分**：一部 5 分的电影如果只有 2 人评分，加权评分更合理。
2. **余弦相似度矩阵过大**：N 个物品需要 N×N 矩阵，超 10 万时需用近似算法。
3. **不处理长尾**：只推荐热门→信息茧房。加入内容多样性约束。
4. **忽略数据稀疏性**：用户-物品矩阵通常 > 99% 空，SVD/矩阵分解是核心解法。
5. **冷启动无策略**：新用户=全局热门，新物品=内容推荐。

## Trigger Phrases (触发词)

- "做推荐系统"、"推荐引擎"
- "猜你喜欢"、"物品相似"
- "协同过滤"、"内容推荐"
- "构建推荐模型"

