---
category: "data-science"
skill_id: "spreadsheet-formula-assistant"
display_name: "Excel公式助手"
---

## Skill Content

# Spreadsheet Formula Assistant

写公式、调 Pivot Table、跨中英文语境转换。数据分析师必备。

## When to Use

- 需要写 Excel/Google Sheets 公式
- Pivot Table 不会配
- 公式报错需要调试
- 中文→英文函数名转换
- 批量生成报告模板

## Common Scenarios

### Excel Formula Writing
```
user: "写一个 VLOOKUP 公式，根据 A 列在 Sheet2 的 B:C 查找"
→ Output: =VLOOKUP(A2, Sheet2!B:C, 2, FALSE)
```

### Pivot Table Setup
```
user: "按月份汇总销售额，按地区分行"
→ Guide: Row=日期(按月分组), Column=地区, Value=SUM(销售额)
```

### Chinese-English Function Translation
```
user: "SUMIFS 中文版怎么写"
→ Output: =SUMIFS(求和区域, 条件区域1, 条件1, 条件区域2, 条件2)
  CN: =SUMIFS
```

### Formula Debugging
```
user: "这个公式为什么返回 #N/A？=VLOOKUP(A2,Sheet2!A:B,3,FALSE)"
→ Analysis: 第3参数超出范围（Sheet2!A:B 只有2列，你却要返回第3列）
   Fix: =VLOOKUP(A2, Sheet2!A:C, 3, FALSE)
```

## Key Excel Functions (Chinese ↔ English)

| CN | EN |
|----|-----|
| VLOOKUP | VLOOKUP |
| SUMIFS | SUMIFS |
| COUNTIFS | COUNTIFS |
| IFERROR | IFERROR |
| INDEX+MATCH | INDEX+MATCH |
| XLOOKUP | XLOOKUP |
| TEXTJOIN | TEXTJOIN |
| UNIQUE | UNIQUE |
| FILTER | FILTER |

## Instructions for Agent

1. 理解用户的数据结构和需求
2. 给出公式 + 解释每个参数
3. 标注是否支持 Excel / Google Sheets / 两者
4. 提示边界情况（空值、重复、大数据量）