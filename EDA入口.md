---
category: "data-science"
skill_id: "enhanced-exploratory-data-analysis"
display_name: "EDA入口"
---

## Skill Content

# 增强型探索性数据分析 (Enhanced EDA)

## Purpose
对数据集进行系统性的首轮分析，识别架构问题、数据质量隐患、统计分布规律，并提炼可落地（Actionable）的业务洞察，评估其是否具备建模条件。

## When to use

- 用户提供数据集、表格、CSV/Excel 或 DataFrame 要求分析时。
- 在构建机器学习模型之前的预处理阶段。
- 需要评估第三方数据质量或检查数据漂移时。
- 需要从原始数据中快速提取商业发现时。

## Do not use when

- 仅需进行单一的数据转换（如：只要求将 A 列改为日期格式）。
- 纯粹的模型训练任务，无需理解数据背景。
- 用户仅询问特定的单一指标（如：计算平均值）。

## Core Workflow (核心工作流)

### 0. 数据定向确认 (Data Orientation) — ⚠️ 必须先执行

**原则：Target vs Feature 是业务判断，不是统计判断。禁止自行推断。**

**V2 Enhanced Protocol (2026-05-09):** After user confirms target/feature/cat/num, run automatic problem type detection BEFORE starting analysis:

```bash
python ~/.hermes/scripts/detect_problem.py <data.csv> [target_column]
```

The detector analyzes: unique value ratios, column dtypes, skewness, class imbalance, date patterns, text length, image extensions. It outputs:
- Detected problem type (classification / regression / time series / clustering / NLP / CV)
- Confidence score
- Recommended skill chain
- Reasoning

**Show the detection results to the user and wait for confirmation or modification before executing.** Never skip this step or run analysis without user approval.

在开始任何分析前，必须向用户确认以下三项信息。如果用户未提供，**停止分析并询问**：

1. **🎯 Target（目标列）**：你要预测/分析的核心变量是什么？
2. **📊 Features（特征列）**：哪些列是用来预测 Target 的？
3. **🏷️ 定性/定量**：哪些是分类变量（Categorical），哪些是数值变量（Numerical）？

**完整协议流程**：
```
① 打印列名
② 展示前 3 行
③ 确认 target / feature / categorical / numerical
④ 运行 detect_problem.py → 展示检测结果 + 推荐技能链
⑤ 用户审查 → 确认或修改管道
⑥ 确认后才开始生成文件
```

**用户标准协议**：用户会主动提供以上信息。如果遗漏任何一项，按以下格式提醒：

```
在开始分析前，请确认：
- 目标列 (Target)：______
- 特征列 (Features)：______
- 定性变量 (Categorical)：______
- 定量变量 (Numerical)：______
```

**常见陷阱**：
- `gender`（男/女）只有 2 个值，但通常是 Feature 不是 Target
- `amount` 在欺诈检测中是 Feature，在销量预测中是 Target
- 列名 `flag_*` 可能是 Target 也可能是 Feature，必须确认
- 同一列在不同分析场景中角色可能完全不同

### 1. 确定业务颗粒度 (Identify Grain)
- 明确每一行代表什么（用户、交易、设备、时间点？）。
- 识别主键/唯一标识符。
- 校验：实际数据行数与预期的颗粒度是否匹配。

### 2. Schema 架构审计
- 列出所有字段及其原始类型。
- 类型纠偏：标记应为数值但被识别为字符串的字段，或应为分类（Category）的 ID 类字段。

### 3. 数据健康度评估 (Data Health Check)
- 缺失率：计算各列缺失比例，区分"随机缺失"与"业务闭环缺失"。
- 唯一性：测试重复行及业务主键的唯一性。
- 异常值：检测逻辑错误（如年龄为负、未来日期、极端的 9999 填充值）。

### 4. 单变量分布分析 (Univariate Analysis)
- 数值型：统计 Count, Mean, Median, Min, Max, Skewness (偏度)。识别长尾分布或双峰分布。
- 分类型：统计唯一值数量，识别 Top 类别及"稀有类别"（Rare levels）。

### 5. 变量关联与特征挖掘 (Bivariate/Multivariate)
- 相关性分析：数值列之间的相关系数矩阵。
- 交叉分析：目标变量在不同分类维度下的表现差异。

### 6. 时间序列模式 (如有)
- 覆盖范围、数据断档检测、季节性或结构性断点识别。

### 7. 建模风险评估
- 目标泄露 (Leakage)：识别包含"未来信息"的字段。
- 样本偏差：评估训练样本是否能代表真实业务场景。

### 8. 撰写结论与建议
- 给出数据健康分。
- 提取 3-5 条关键业务发现。
- 给出明确的清洗、工程化及建模建议。

## Output Format (输出格式)

### 1. 数据概览与健康评价
- 基本统计： [行数] 行 x [列数] 列 | 业务颗粒度：[如：单笔订单]
- 健康评分： [0-10 分]
- 就绪状态： [可以直接建模 / 需清洗后使用 / 存在严重缺陷不可用]

### 2. 数据质量详细报告
- 缺失值： (列出高缺失率字段及可能原因)
- 重复项： (是否存在重复记录或键值冲突)
- 异常偏离： (逻辑错误值、极值说明)

### 3. 核心分布与洞察
- 统计特征： (描述核心数值列的分布形态，如正态、左偏、幂律)
- 分类特征： (Top 类别占比，类别不平衡状况)
- 关联模式： (发现变量 A 与 B 之间的显著正/负相关性)

### 4. 业务洞察 (Insight)
- (基于数据的发现 1：例如"80% 的收入来自于 5% 的高价值客户")
- (基于数据的发现 2)
- (基于数据的发现 3)

### 5. 风险预警
- (如：存在目标泄露风险、样本时间分布不均、某些特征粒度过粗等)

### 6. 下一步操作建议
- 即时清洗： (如：剔除缺失值、处理异常极值)
- 特征工程： (如：建议对 A 列进行 Log 转换，或从日期中提取星期特征)
- 分析/建模建议： (如：建议使用对异常值不敏感的随机森林模型)

## Quality Checklist (自检表)

- [ ] 是否明确了数据行代表的业务含义？
- [ ] 是否检查了数据类型转换的必要性？
- [ ] 是否计算了各维度的缺失率和重复率？
- [ ] 是否识别了对业务有重大影响的偏离值？
- [ ] 结论是否直接回答了用户的业务目标？
- [ ] 是否给出了具备操作性的后续步骤？

## Trigger Phrases (触发词)

- "分析一下这个数据集"
- "帮我做个 EDA"
- "检查这份数据的质量并给建议"
- "这份数据能用来建模吗？"
- "总结这个表的规律和潜在问题"

## Implementation Notes

### 技术工具建议
- 使用 pandas 进行数据加载和基础分析
- 使用 matplotlib/seaborn 进行可视化
- 使用 scipy/statsmodels 进行统计检验

### 环境配置
- 如果用户指定了特定 conda 环境（如 `code_mbp`），确保在该环境中执行 Python 代码
- 检查必要的 Python 包是否已安装（pandas, numpy, matplotlib, seaborn, scipy）
- 在用户环境中执行命令示例：`/opt/anaconda3/envs/code_mbp/bin/python script.py`
- **用户特定环境**：已知用户使用 `code_mbp` conda 环境（路径：`/opt/anaconda3/envs/code_mbp`）

### 可视化语言偏好配置
**重要**：在开始分析前，询问用户对可视化图表语言的偏好：
- **英文图表**（推荐）：所有标题、轴标签、图例使用英文，避免字体兼容性问题
- **中文图表**：需要配置系统字体支持

**用户特定偏好**：
- 当前用户要求所有可视化图表中的文字说明（标题、轴标签、图例等）只使用英文
- 在生成图表时确保所有文本为英文，无需配置中文字体

**最佳实践**：
1. 默认使用英文图表，确保跨平台兼容性
2. 如果用户需要本地化图表，在分析开始时确认语言偏好
3. 记录用户偏好到记忆（使用 `memory` 工具）

### 跨平台字体配置
**问题**：在不同操作系统上使用 matplotlib 生成图片时，特殊字符（如中文）可能显示异常。

#### macOS 环境配置
**中文字体解决方案**：
1. **检查系统字体**：macOS 已内置多种中文字体，包括：
   - Hiragino Sans GB（冬青黑体简体中文）
   - STHeiti（华文黑体）
   - Apple SD Gothic Neo
   - Arial Unicode MS
   - Kai（楷体）
   - LiSong Pro（隶书）

2. **配置 matplotlib**：
   ```python
   import matplotlib.pyplot as plt
   plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 使用中文字体
   plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
   ```

3. **验证字体可用性**：
   ```python
   def check_and_configure_chinese_fonts():
       \"\"\"检查并配置中文字体\"\"\"
       import matplotlib.font_manager as fm
       
       # 字体候选列表（按优先级排序）
       font_candidates = [
           'PingFang SC',           # macOS 苹方简体
           'Hiragino Sans GB',      # macOS 冬青黑体简体中文
           'STHeiti',               # macOS 华文黑体
           'Apple SD Gothic Neo',   # macOS 苹果SD Gothic Neo
           'Arial Unicode MS',      # 通用
           'Microsoft YaHei',       # Windows 微软雅黑
           'SimHei',                # Windows 黑体
       ]
       
       # 检查可用字体
       available_fonts = set([f.name for f in fm.fontManager.ttflist])
       
       # 选择第一个可用的中文字体
       selected_font = None
       for font in font_candidates:
           if font in available_fonts:
               selected_font = font
               break
       
       if selected_font:
           import matplotlib.pyplot as plt
           plt.rcParams['font.sans-serif'] = [selected_font, 'DejaVu Sans', 'Arial']
           plt.rcParams['axes.unicode_minus'] = False
           return selected_font
       return None
   
   # 使用示例
   font_used = check_and_configure_chinese_fonts()
   if font_used:
       print(f\"已配置字体: {font_used}\")
   ```

4. **修复现有脚本**：
   ```python
   def add_font_config_to_script(script_path):
       \"\"\"在现有脚本中添加字体配置\"\"\"
       with open(script_path, 'r') as f:
           lines = f.read().split('\\n')
       
       new_lines = []
       font_config_added = False
       
       for line in lines:
           new_lines.append(line)
           if 'import matplotlib.pyplot as plt' in line and not font_config_added:
               new_lines.append('')
               new_lines.append('# Font configuration')
               new_lines.append(\"plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']\")
               new_lines.append(\"plt.rcParams['axes.unicode_minus'] = False\")
               font_config_added = True
       
       if not font_config_added and any('import matplotlib' in line for line in lines):
           # 如果导入了matplotlib但没找到plt导入，在文件开头添加
           lines = [\"import matplotlib.pyplot as plt\",
                    \"plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']\",
                    \"plt.rcParams['axes.unicode_minus'] = False\",
                    \"\"] + lines
           new_lines = lines
       
       with open(script_path, 'w') as f:
           f.write('\\n'.join(new_lines))
   ```

**注意**：无需额外 pip 安装字体包，主流操作系统都已内置必要字体。优先使用英文图表可避免字体兼容性问题。

### 文件组织规范
- 在用户指定的根目录下创建独立分析文件夹（如 `/Users/ordi_de_lvga/Documents/hermes 文件/`）
- 文件夹命名格式：`分析主题_YYYYMMDD_HHMM`（如 `库存周转分析_20260415_1416`）
- 将所有分析输出文件（报告、可视化、清洗数据、统计摘要）放在该文件夹内
- 在分析开始时确定分析主题名称，结束时整理文件

### 关键指标计算
- 缺失率阈值：>30% 为高风险，10-30% 为中风险，<10% 为低风险
- 异常值检测：使用 IQR 方法或 Z-score 方法
- 相关性阈值：|r| > 0.7 为强相关，0.3-0.7 为中等相关

### 报告生成
- 优先使用 Markdown 格式输出，便于阅读和分享
- 重要发现使用 **加粗** 或列表形式突出
- 对于复杂分析，生成可视化图表并保存到文件

### PPTX 演示文稿生成（可交付物）
- 当用户需要将分析结果转为演示文稿（PPT）时，使用 python-pptx 直接生成专业 .pptx 文件
- 模板参考：`references/pptx_from_eda.md` — 包含 12-slide 结构模板、配色方案、布局规范
- **安装**：`pip install python-pptx`
- **幻灯片标准结构**：封面 → 目录 → 背景/问题 → 数据概览 → 方法论 → 分析结果(4-5页) → 关键发现 → 建议
- 每页：标题(32pt bold) + 3-4个要点(16pt/15pt) + 1张图表(png)；深蓝/海军蓝主题
- **关键代码模式**：`prs = Presentation()` → `prs.slide_width/height = Inches(13.333, 7.5)` (16:9) → `add_bullet_slide()` → `prs.save()`

## Example Commands

### 基础示例
```python
# 加载数据
import pandas as pd
df = pd.read_csv('data.csv')

# 基础信息
print(f"数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"数据类型:\\n{df.dtypes}")
```

### 完整模板
技能包含完整的EDA模板脚本：`templates/full_eda_template.py`
- 该脚本实现了本技能的所有核心工作流程
- 包含数据健康度评估、可视化生成、报告输出等功能
- 可根据具体数据集修改配置参数使用

### 环境执行示例
```bash
# 在指定conda环境中执行
/opt/anaconda3/envs/code_mbp/bin/python full_eda_template.py

# 或使用相对路径
cd "/Users/ordi_de_lvga/Documents/hermes 文件" && /opt/anaconda3/envs/code_mbp/bin/python /tmp/analysis_script.py
```

## 常见问题与解决方案

### 问题1：内存不足处理大数据集
- 解决方案：使用 chunksize 参数分块读取，或使用 dask 库

### 问题2：日期时间格式不一致
- 解决方案：使用 pd.to_datetime() 统一格式，设置 errors='coerce'

### 问题3：分类变量类别过多
- 解决方案：将低频类别合并为"其他"类别

### 问题4：数值型字段包含非数值字符
- 解决方案：使用 pd.to_numeric(errors='coerce') 转换，然后分析缺失情况

### 问题5：Categorical列字符串拼接报错
- **问题描述**：使用 `.astype('category')` 转换后，尝试做字符串拼接如 `df['col1'] + ' / ' + df['col2']` 报错：`TypeError: unsupported operand type(s) for +: 'Categorical' and 'str'`
- **解决方案**：在拼接前先将分类列转为字符串：`df['col1'].astype(str) + ' / ' + df['col2'].astype(str)`
- **常见场景**：crosstab 后用 reset_index() 创建标签列、生成复合分类标签等
- **预防**：在涉及字符串操作的代码中，对已设为 category 的列使用 `.astype(str)` 安全转换

### 问题6：ModuleNotFoundError（Python包未找到）
- **解决方案**：检查用户是否指定了特定conda环境，使用该环境执行Python代码
- **验证步骤**：
  1. 确认环境存在：`conda env list | grep code_mbp`
  2. 测试包导入：`/opt/anaconda3/envs/code_mbp/bin/python -c \\\"import pandas, numpy, matplotlib, seaborn\\\"`
  3. 如果失败，安装缺失包：`/opt/anaconda3/envs/code_mbp/bin/pip install pandas numpy matplotlib seaborn scipy`
- **执行示例**：
  - 直接使用环境Python：`/opt/anaconda3/envs/code_mbp/bin/python script.py`
  - 或在指定目录执行：`cd \\\"/Users/ordi_de_lvga/Documents/hermes 文件\\\" && /opt/anaconda3/envs/code_mbp/bin/python /tmp/analysis_script.py`
- **预防措施**：在编写分析脚本前，先用小段测试代码验证环境配置

### 问题6：特殊字符无法显示（PNG图片中显示为方框）
- **问题描述**：使用 matplotlib 生成的可视化图表中，特殊字符（如中文、特定符号）显示为方框
- **根本原因**：matplotlib 默认字体不支持特定字符集
- **最佳实践**：
  1. **优先使用英文图表**：建议所有可视化使用英文标签，确保跨平台兼容性
  2. **询问用户偏好**：在分析开始时确认用户对可视化语言的要求
  3. **记录用户偏好**：将语言偏好保存到记忆（使用 `memory` 工具）

- **中文字体解决方案**（macOS）：
  1. **检查可用字体**：macOS 已内置多种中文字体，无需额外安装
  2. **配置 matplotlib**：
     ```python
     import matplotlib.pyplot as plt
     plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 冬青黑体简体中文
     plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
     ```
  3. **备选字体**：如果 Hiragino Sans GB 不可用，尝试：
     - 'STHeiti' (华文黑体)
     - 'Apple SD Gothic Neo'
     - 'Arial Unicode MS'
     - 'PingFang SC' (macOS 苹方)
  4. **系统字体检查**：使用 `check_and_configure_chinese_fonts()` 函数（见上方"跨平台字体配置"部分）
- **英文图表解决方案**：
  - 确保所有图表标题、轴标签、图例使用英文
  - 无需特殊字体配置，使用 matplotlib 默认字体即可
  - 示例：`plt.title('Weekly Stock Out Trend', fontsize=16)` 而不是中文标题
- **验证步骤**：
  1. 生成测试图表检查字符是否正常显示
  2. 重新生成所有可视化图表
- **预防措施**：
  1. 默认使用英文图表
  2. 如需本地化图表，在脚本开头添加相应字体配置
  3. 保存用户语言偏好到长期记忆

*本技能适用于需要对数据集进行全面初步分析的场景，特别关注从数据中提取可落地的业务洞察。*

## Related Skills

- **`data-science-pipeline`** (umbrella): Master umbrella covering the FULL data science workflow. After EDA, load this skill — it contains 16 technique reference files (classification, clustering, NLP, CNN, time series, recommender systems, anomaly detection, ensemble methods, supply chain analysis, and more) under one discoverable entry point. Use `skill_view('data-science-pipeline')` and then load the specific reference file needed.
- **`jupyter-live-kernel`**: Interactive Python exploration for iterative EDA.