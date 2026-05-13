---
tags: [skill, hermes]
category: 📊 Data Science
skill_id: time-series-forecasting
---

# Time Series Forecasting

**分类:** 📊 Data Science
**Skill ID:** `time-series-forecasting`

> 时间序列预测框架，使用 Facebook Prophet 进行趋势、季节性和节假日效应建模。包含 EDA、模型训练、误差评估和多模型对比。提炼自 Kaggle 经典 Prophet notebook。

---


# 时间序列预测 (Time Series Forecasting)

## Purpose
对时间序列数据进行系统性分析：趋势识别、季节分解、节假日效应建模，使用 Facebook Prophet 进行预测，并提供误差评估和模型优化建议。

## When to use

- 数据有明显的时间戳，需要预测未来值。
- 业务场景：销售预测、库存预测、流量预测。
- 数据包含趋势、周期性（周/月/年）和节假日效应。
- 用户说"预测下个月销量"、"分析时间趋势"、"做时序预测"。

## Do not use when

- 数据无时间维度（用回归/分类）。
- 预测变量主要依赖外部因素而非时间规律（考虑外部回归模型）。
- 数据点极少（<30 个时间点），Prophet 效果不佳。

## Core Workflow (核心工作流)

### Step 1: 加载与 EDA
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('time_series.csv')
df['ds'] = pd.to_datetime(df['datetime_column'])
df['y'] = df['value_column']

print(f"Range: {df['ds'].min()} → {df['ds'].max()}")
print(f"Records: {len(df)}")
print(f"Granularity: hourly/daily/weekly?")
```

### Step 2: 可视化趋势
```python
# 整体时间序列
plt.figure(figsize=(16, 6))
plt.plot(df['ds'], df['y'], alpha=0.7, linewidth=0.5)
plt.title('Time Series Overview')
plt.xlabel('Date')
plt.ylabel('Value')
plt.grid(True, alpha=0.3)
plt.show()

# 多维度探查
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# 按小时聚合
df.groupby(df['ds'].dt.hour)['y'].mean().plot(ax=axes[0], marker='o')
axes[0].set_title('Average by Hour of Day')
axes[0].set_xlabel('Hour')

# 按周几聚合
df.groupby(df['ds'].dt.dayofweek)['y'].mean().plot(ax=axes[1], marker='o')
axes[1].set_title('Average by Day of Week')
axes[1].set_xlabel('Day (0=Monday)')

# 按月份聚合
df.groupby(df['ds'].dt.month)['y'].mean().plot(ax=axes[2], marker='o')
axes[2].set_title('Average by Month')
axes[2].set_xlabel('Month')

plt.tight_layout()
plt.show()
```

### Step 3: 训练/测试分割 (Time-Series Safe Split)
```python
# 时间序列不能随机分割！必须按时间序分割
split_date = df['ds'].quantile(0.8)  # 前80%训练

train = df[df['ds'] <= split_date]
test = df[df['ds'] > split_date]

print(f"Train: {len(train)} ({train['ds'].min()} to {train['ds'].max()})")
print(f"Test:  {len(test)} ({test['ds'].min()} to {test['ds'].max()})")

plt.figure(figsize=(16, 6))
plt.plot(train['ds'], train['y'], label='Train', alpha=0.7)
plt.plot(test['ds'], test['y'], label='Test', alpha=0.7)
plt.axvline(split_date, color='black', linestyle='--', label='Split')
plt.legend()
plt.show()
```

### Step 4: Prophet 基础模型
```python
# Prophet 要求列名为 ds (日期) 和 y (数值)
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,  # 趋势灵活度（默认0.05）
)
model.fit(train[['ds', 'y']])

# 创建未来时间范围
future = model.make_future_dataframe(periods=len(test), freq='H')  # 或 'D'
forecast = model.predict(future)

# 可视化
model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

# 季节性分解
model.plot_components(forecast)
plt.show()
```

### Step 5: 对比预测 vs 实际
```python
# 合并实际值与预测值
comparison = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].merge(
    test[['ds', 'y']], on='ds', how='inner'
)

# 可视化
plt.figure(figsize=(16, 8))

# 整体
plt.subplot(2, 1, 1)
plt.plot(train['ds'], train['y'], label='Train', alpha=0.5)
plt.plot(test['ds'], test['y'], label='Actual', alpha=0.7)
plt.plot(comparison['ds'], comparison['yhat'], label='Predicted', alpha=0.7)
plt.fill_between(comparison['ds'], comparison['yhat_lower'], comparison['yhat_upper'],
                 alpha=0.2, label='Uncertainty')
plt.legend()
plt.title('Forecast vs Actual — Full Range')

# 放大首月
plt.subplot(2, 1, 2)
first_month = comparison.head(30 * 24)  # 月*小时
plt.plot(first_month['ds'], first_month['y'], label='Actual', marker='o', markersize=2)
plt.plot(first_month['ds'], first_month['yhat'], label='Predicted', marker='o', markersize=2)
plt.fill_between(first_month['ds'], first_month['yhat_lower'], first_month['yhat_upper'],
                 alpha=0.2)
plt.legend()
plt.title('Forecast vs Actual — First Month Detail')

plt.tight_layout()
plt.show()
```

### Step 6: 误差评估
```python
# 多种误差指标
rmse = np.sqrt(mean_squared_error(comparison['y'], comparison['yhat']))
mae = mean_absolute_error(comparison['y'], comparison['yhat'])
mape = np.mean(np.abs((comparison['y'] - comparison['yhat']) / comparison['y'])) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"MAPE: {mape:.1f}%")

# 误差分布
comparison['error'] = comparison['y'] - comparison['yhat']
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
comparison['error'].hist(bins=50)
plt.title('Error Distribution')
plt.xlabel('Error (Actual - Predicted)')

plt.subplot(1, 2, 2)
plt.scatter(comparison['yhat'], comparison['error'], alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Predicted')
plt.ylabel('Error')

plt.tight_layout()
plt.show()
```

### Step 7: 添加节假日效应
```python
# Prophet 原生支持节假日
from pandas.tseries.holiday import USFederalHolidayCalendar

calendar = USFederalHolidayCalendar()
holidays = calendar.holidays(
    start=train['ds'].min(),
    end=test['ds'].max()
)

holidays_df = pd.DataFrame({
    'holiday': 'us_federal',
    'ds': holidays,
    'lower_window': 0,
    'upper_window': 1,  # 节后1天有影响
})

model_holiday = Prophet(
    holidays=holidays_df,
    yearly_seasonality=True,
    weekly_seasonality=True,
)
model_holiday.fit(train[['ds', 'y']])

forecast_holiday = model_holiday.predict(future)

# 可视化节假日效应
model_holiday.plot_components(forecast_holiday)
plt.show()
```

### Step 8: 模型对比
```python
# 计算节假日模型的误差
comparison_holiday = forecast_holiday[['ds', 'yhat']].merge(
    test[['ds', 'y']], on='ds', how='inner'
)
rmse_holiday = np.sqrt(mean_squared_error(
    comparison_holiday['y'], comparison_holiday['yhat']
))

print(f"Base model RMSE:    {rmse:.2f}")
print(f"Holiday model RMSE: {rmse_holiday:.2f}")
print(f"Improvement:        {(rmse - rmse_holiday) / rmse * 100:.1f}%")

# 仅评估节假日日的误差
holiday_dates = set(holidays)
comparison['is_holiday'] = comparison['ds'].dt.date.isin(
    [d.date() for d in holiday_dates]
)
print("\nError on holidays:")
print(comparison.groupby('is_holiday')['error'].describe())
```

## Advanced Tuning

```python
# Prophet 关键参数调整
model_tuned = Prophet(
    # 趋势灵活性
    changepoint_prior_scale=0.5,        # 更高 = 更灵活趋势（可能过拟合）
    changepoint_range=0.9,              # 推断变点的数据比例
    
    # 季节性
    seasonality_prior_scale=10.0,       # 更高 = 更强季节性拟合
    seasonality_mode='multiplicative',  # 乘法季节性（当振幅随时间变化时）
    
    # 节假日
    holidays_prior_scale=10.0,          # 更高 = 更强调节假日效应
    
    # 不确定性
    interval_width=0.95,                # 置信区间宽度
)
```

## Quality Checklist (自检表)

- [ ] 数据是否按时间排序并检查了连续性（无缺失时间点）？
- [ ] 训练/测试分割是否按时间顺序（非随机）？
- [ ] Prophet 的 ds/y 列名是否正确？
- [ ] 是否检查了季节性分解（trend + weekly + yearly）？
- [ ] 是否评估了多种误差指标（RMSE, MAE, MAPE）？
- [ ] 残差是否随机分布（无异方差性）？
- [ ] 节假日是否被正确识别和建模？

## Common Pitfalls

1. **随机分割数据**：时间序列必须按时间序分训练/测试。
2. **忽略缺失时间点**：Prophet 对不规则间隔敏感，需先填充缺失时间戳。
3. **默认参数不调**：至少调整 changepoint_prior_scale 和 seasonality_mode。
4. **只用 RMSE**：MAE 和 MAPE 在业务上更直观，应同时报告。
5. **Prophet 不适用所有序列**：对高频金融数据、多季节复杂模式效果有限。

## Environment & Installation Notes

### Prophet Installation (China Network)
Chinese pip mirrors (tsinghua, etc.) may fail with SSL errors when installing Prophet.
Use the official PyPI directly:
```bash
pip install prophet -i https://pypi.org/simple/ --trusted-host pypi.org
```
Import: use `from prophet import Prophet` (not `fbprophet` — the package was renamed).

### Matplotlib Headless Execution
When running scripts outside Jupyter (e.g., `nbconvert --execute`, subprocess, or cron):
```python
import matplotlib
matplotlib.use('Agg')
```
Or set environment variable: `MPLBACKEND=Agg python script.py`
Without this, matplotlib may crash on macOS with `NSWindow` errors in headless mode.

### Jupyter nbconvert Issues
`jupyter nbconvert --execute --inplace` may fail due to `jupyter_contrib_nbextensions` dependency conflicts.
**Alternative**: Extract code cells from IPYNB JSON, join into a single script, execute with subprocess:
```python
import json, subprocess
with open('notebook.ipynb') as f:
    nb = json.load(f)
code = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
subprocess.run(['python', '-c', code], cwd=output_dir, env={**os.environ, 'MPLBACKEND': 'Agg'})
```

### IPYNB Generation
To programmatically create IPYNB files with embedded markdown and code:
See `templates/ipynb_generation_template.py` for a complete template.

## Trigger Phrases (触发词)

- "做时间序列预测"
- "预测未来趋势"
- "分析季节性规律"
- "节假日效应分析"
- "销量/流量/库存预测"

