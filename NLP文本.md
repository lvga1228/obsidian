---
category: "data-science"
skill_id: "nlp-text-analysis"
display_name: "NLP文本"
---

## Skill Content

# NLP 文本分析工作流 (NLP Text Analysis)

## Purpose
对文本数据进行分类、情感分析、主题建模等任务。覆盖从传统机器学习（TF-IDF + 分类器）到深度学习（LSTM/GRU）再到模型集成的完整技术栈。

## When to use

- 文本分类（垃圾邮件检测、情感分析、主题分类）。
- 文本相似度计算。
- 需要从自然语言中提取特征。
- 用户说"分析文本数据"、"做情感分析"、"文本分类"。

## Core Workflow (核心工作流)

### Step 1: 数据加载与探索
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('text_data.csv')
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# 目标分布
df['target'].value_counts().plot(kind='bar')
plt.title('Target Distribution')
plt.show()

# 文本长度分布
df['text_len'] = df['text'].str.len()
df['text_len'].hist(bins=50)
plt.title('Text Length Distribution')
plt.show()
```

### Step 2: TF-IDF + 基础分类器 (Baseline)
```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# TF-IDF 特征提取
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),      # unigrams + bigrams
    stop_words='english',     # 或传自定义停用词列表
    sublinear_tf=True,        # 1 + log(tf)
    min_df=3,                 # 最少出现在3个文档中
)
X = tfidf.fit_transform(df['text'])
y = df['target']

# 快速 baseline
model = LogisticRegression(max_iter=1000)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"TF-IDF + LR baseline: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
```

### Step 3: CountVectorizer + 多种分类器对比
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import xgboost as xgb

# 词频特征
ctv = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 3),
    stop_words='english',
    analyzer='word',
)
X_ctv = ctv.fit_transform(df['text'])

# 多模型对比
models = {
    'NaiveBayes': MultinomialNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'SVM': SVC(kernel='linear', probability=True),
    'XGBoost': xgb.XGBClassifier(n_estimators=100),
}

for name, model in models.items():
    scores = cross_val_score(model, X_ctv, y, cv=5, scoring='accuracy')
    print(f"{name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Step 4: 网格搜索最佳参数
```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000)),
])

params = {
    'tfidf__max_features': [5000, 10000],
    'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'tfidf__sublinear_tf': [True, False],
    'clf__C': [0.1, 1, 10],
}

gs = GridSearchCV(pipeline, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
gs.fit(df['text'], y)
print(f"Best score: {gs.best_score_:.4f}")
print(f"Best params: {gs.best_params_}")
```

### Step 5: 词向量 (Word Embeddings)
```python
# GloVe / FastText 预训练词向量加载
def load_glove(path, dim=300):
    embeddings = {}
    with open(path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# 或使用 gensim 训练 Word2Vec
from gensim.models import Word2Vec
sentences = [text.split() for text in df['text']]
w2v_model = Word2Vec(sentences, vector_size=300, window=5, min_count=2, workers=4)

# 文档向量 = 词向量的平均
def document_vector(doc, model, dim=300):
    words = doc.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    if len(vectors) == 0:
        return np.zeros(dim)
    return np.mean(vectors, axis=0)

X_w2v = np.array([document_vector(text, w2v_model) for text in df['text']])
```

### Step 6: 深度学习 — LSTM/GRU
```python
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU, Dropout, SpatialDropout1D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# Tokenization
MAX_WORDS = 50000
MAX_LEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
X_seq = pad_sequences(sequences, maxlen=MAX_LEN)

# One-hot 目标
y_cat = to_categorical(df['target'])

# LSTM 模型
model = Sequential()
model.add(Embedding(MAX_WORDS, 300, input_length=MAX_LEN))
model.add(SpatialDropout1D(0.3))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(
    X_seq, y_cat,
    batch_size=64,
    epochs=10,
    validation_split=0.1,
    verbose=1
)
```

### Step 7: 深度学习 — 使用预训练词向量
```python
# 用 GloVe 初始化 Embedding 层
embedding_matrix = np.zeros((MAX_WORDS, 300))
for word, i in tokenizer.word_index.items():
    if i >= MAX_WORDS:
        continue
    if word in glove_embeddings:
        embedding_matrix[i] = glove_embeddings[word]

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False  # 冻结 Embedding 层
```

### Step 8: 模型集成 (Ensembling)
```python
# 多个模型的概率平均
from sklearn.ensemble import VotingClassifier

# 方法1: VotingClassifier
ensemble = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('svm', SVC(kernel='linear', probability=True)),
        ('xgb', xgb.XGBClassifier()),
    ],
    voting='soft',
)

# 方法2: Stacking
from sklearn.ensemble import StackingClassifier
estimators = [
    ('nb', MultinomialNB()),
    ('svm', SVC(probability=True)),
]
stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    cv=5,
)
```

## Quality Checklist (自检表)

- [ ] 文本是否做了预处理（小写、去标点、去停用词）？
- [ ] 是否对比了 TF-IDF 和 CountVectorizer？
- [ ] ngram_range 是否尝试了 (1,1), (1,2), (1,3)？
- [ ] 是否对比了传统 ML（NB, LR, SVM, XGB）和深度学习？
- [ ] GridSearch 是否覆盖了关键超参数？
- [ ] 深度学习是否使用了预训练词向量？
- [ ] 是否尝试了集成学习？

## Common Pitfalls

1. **不控制 max_features**：TF-IDF 默认生成全部词，内存爆炸。始终设置上限。
2. **过拟合小数据集**：深度学习在 < 1000 条文本时效果差于 TF-IDF+LR。
3. **忽略文本长度差异**：序列 padding 过长浪费计算，过短丢失信息（取 95 分位）。
4. **忘记 sublinear_tf**：TF-IDF 中 `sublinear_tf=True` 可抑制高频词权重。
5. **不处理类别不平衡**：使用 `class_weight='balanced'` 或采样策略。

## Trigger Phrases (触发词)

- "分析文本数据"、"做 NLP"
- "文本分类"、"情感分析"
- "提取文本特征"
- "训练词向量"