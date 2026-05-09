---
category: "data-science"
skill_id: "deep-learning-cnn-keras"
display_name: "深度学习CNN"
---

## Skill Content

# 深度学习 CNN (Keras/TensorFlow)

## Purpose
使用卷积神经网络 (CNN) 进行图像分类。覆盖数据准备（归一化、reshape、one-hot）、CNN 架构设计（Conv2D、MaxPool、Dropout、BatchNorm）、数据增强、学习率调度和模型评估。目标：在 MNIST 级别数据集上达到 99%+ 准确率。

## When to use

- 图像分类任务（手写数字、物体识别、医学影像）。
- 需要比传统 ML 更高的准确率。
- 有足够的 GPU/TPU 资源。
- 用户说"用 CNN 做图像分类"、"深度学习建模"。

## Do not use when

- 数据集极小（< 1000 张）— 考虑迁移学习或传统 ML。
- 没有 GPU — 训练会很慢。
- 问题可以用简单模型解决（如线性可分的图像）。

## Core Workflow (核心工作流)

### Step 1: 加载与探索数据
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# 加载数据
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# 分离标签和像素
y_train = train['label']
X_train = train.drop('label', axis=1)
X_test = test  # 测试集无标签

# 类别分布
plt.figure(figsize=(10, 4))
y_train.value_counts().sort_index().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# 可视化一些样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    img = X_train.iloc[i].values.reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {y_train.iloc[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### Step 2: 数据预处理
```python
# 2a. 归一化 — 将 [0, 255] 缩放到 [0, 1]
X_train = X_train / 255.0
X_test = X_test / 255.0

# 2b. Reshape — 展平 → 图像格式 (samples, height, width, channels)
# MNIST: 28x28 灰度图 → (N, 28, 28, 1)
X_train = X_train.values.reshape(-1, 28, 28, 1)
X_test = X_test.values.reshape(-1, 28, 28, 1)

print(f"X_train shape: {X_train.shape}")  # (42000, 28, 28, 1)
print(f"X_test shape: {X_test.shape}")    # (28000, 28, 28, 1)

# 2c. One-Hot 编码标签
from keras.utils import to_categorical
y_train_cat = to_categorical(y_train, num_classes=10)
print(f"y_train shape: {y_train_cat.shape}")  # (42000, 10)
```

### Step 3: 训练/验证集分割
```python
# 留出 10% 做验证集
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_cat,
    test_size=0.1,
    random_state=42,
    stratify=y_train  # 保持类别分布
)

print(f"Train: {X_train_split.shape[0]}")
print(f"Val:   {X_val.shape[0]}")
```

### Step 4: 构建 CNN 模型
```python
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
)
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ReduceLROnPlateau

model = Sequential()

# Block 1 — 特征提取
model.add(Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(32, (5, 5), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Block 2 — 更深的特征
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Block 3 — 进一步加深
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 分类头
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# 编译
model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### Step 5: 学习率调度
```python
# 当验证精度不再提升时，自动降低学习率
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=3,       # 3 个 epoch 不提升就降
    factor=0.5,       # 降为原来的一半
    min_lr=1e-6,
    verbose=1
)
```

### Step 6: 数据增强 (Data Augmentation)
```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=10,      # 随机旋转 ±10°
    zoom_range=0.1,         # 随机缩放 ±10%
    width_shift_range=0.1,  # 水平平移 ±10%
    height_shift_range=0.1, # 垂直平移 ±10%
    shear_range=0.1,        # 剪切变换
)

# 不要对验证集做增强！只 fit 训练集
datagen.fit(X_train_split)

# 可视化增强效果
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    batch = next(datagen.flow(X_train_split[:1], batch_size=1))
    ax.imshow(batch[0].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.suptitle('Augmented Samples')
plt.show()
```

### Step 7: 训练模型
```python
epochs = 30
batch_size = 86

history = model.fit(
    datagen.flow(X_train_split, y_train_split, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, y_val),
    steps_per_epoch=len(X_train_split) // batch_size,
    callbacks=[learning_rate_reduction],
    verbose=1
)
```

### Step 8: 评估 — 训练曲线
```python
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Loss
axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# Accuracy
axes[1].plot(history.history['accuracy'], label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.show()

# 最终指标
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final train accuracy: {final_train_acc:.4f}")
print(f"Final val accuracy:   {final_val_acc:.4f}")
print(f"Gap (overfitting?):   {final_train_acc - final_val_acc:.4f}")
```

### Step 9: 混淆矩阵
```python
import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix'):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

# 验证集预测
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plot_confusion_matrix(cm, classes=range(10))
plt.show()

# 找出错误分类的样本
errors = np.where(y_pred_classes != y_true)[0]
print(f"Total errors: {len(errors)} out of {len(y_true)}")
print("Error examples:")
for err_idx in errors[:5]:
    print(f"  True: {y_true[err_idx]}, Predicted: {y_pred_classes[err_idx]}")
```

### Step 10: 可视化错误样本
```python
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    if i < len(errors):
        err_idx = errors[i]
        img = X_val[err_idx].reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'True:{y_true[err_idx]} Pred:{y_pred_classes[err_idx]}',
                     color='red')
    ax.axis('off')
plt.suptitle('Misclassified Examples')
plt.show()
```

### Step 11: 预测测试集
```python
predictions = model.predict(X_test)
pred_classes = np.argmax(predictions, axis=1)

# 保存提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(pred_classes) + 1),
    'Label': pred_classes
})
submission.to_csv('submission.csv', index=False)
print("Submission saved!")
```

## CNN 架构设计原则

| 组件 | 作用 | 常用参数 |
|------|------|---------|
| Conv2D | 提取空间特征 | filters=32→64→128, kernel=(3,3) |
| MaxPool2D | 降采样，减少参数 | pool_size=(2,2) |
| Dropout | 防过拟合 | 0.25-0.5 |
| BatchNormalization | 加速训练，稳定梯度 | 在激活函数前 |
| Flatten | 展平 → 全连接层 | — |
| Dense | 分类头 | 最后一层 units=类别数 |

## 迁移学习 (Transfer Learning)
```python
from keras.applications import VGG16, ResNet50, EfficientNetB0

# 加载预训练模型（不含分类头）
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # 冻结权重

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

## Quality Checklist (自检表)

- [ ] 数据是否归一化到 [0, 1]？
- [ ] 输入 shape 是否正确 (H, W, Channels)？
- [ ] 标签是否做了 One-Hot 编码？
- [ ] 是否使用了数据增强（训练集）？
- [ ] 是否有 Dropout + BatchNorm 防止过拟合？
- [ ] 是否使用了学习率调度（ReduceLROnPlateau）？
- [ ] 训练/验证曲线是否有明显过拟合？
- [ ] 是否分析了错误分类的样本？

## Common Pitfalls

1. **忘记归一化**：像素值 0-255 直接入模 → 梯度爆炸。
2. **数据增强用于验证集**：只对训练集增强！验证集保持原样。
3. **过大的 batch size**：默认 32-128，过大导致泛化差。
4. **没有学习率调度**：固定 LR 后期震荡，ReduceLROnPlateau 是标配。
5. **卷积核尺寸过大**：3×3 是主流，5×5 用于第一层，7×7 以上很少用。

## Trigger Phrases (触发词)

- "图像分类"、"用 CNN"
- "深度学习模型"、"Keras/TensorFlow"
- "卷积神经网络"
- "训练神经网络"
- "Data Augmentation"