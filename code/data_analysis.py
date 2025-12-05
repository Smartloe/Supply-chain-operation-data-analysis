# -*- coding: utf-8 -*-
"""
供应链经营数据分析 - 380平台数据分析
主要任务：
- 任务3: 数据预处理
- 任务4: 平台行业分析
- 任务5: 母婴行业分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GRAPH_DIR = os.path.join(BASE_DIR, 'graph')

# 确保输出目录存在
os.makedirs(GRAPH_DIR, exist_ok=True)

# =============================================================================
# 任务3: 数据预处理
# =============================================================================
print("=" * 60)
print("任务3: 数据预处理")
print("=" * 60)

# 读取数据
file_path = os.path.join(DATA_DIR, '380平台－数据源.xlsx')
print(f"\n读取数据文件: {file_path}")

# 读取Excel文件
df = pd.read_excel(file_path)
print(f"原始数据形状: {df.shape}")
print(f"\n数据列名:\n{df.columns.tolist()}")

# 查看数据前几行
print(f"\n数据预览:\n{df.head()}")

# 查看数据基本信息
print(f"\n数据类型信息:")
print(df.dtypes)

# 检查缺失值
print(f"\n缺失值统计:")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
print(missing_df[missing_df['缺失数量'] > 0])

# 检查重复值
duplicates = df.duplicated().sum()
print(f"\n重复值数量: {duplicates}")

# 数据清洗
print("\n开始数据清洗...")

# 处理缺失值（根据实际情况处理）
# 如果是数值列，填充0或均值；如果是分类列，填充众数或'未知'
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64', 'int64']:
            # 数值型用0填充
            df[col].fillna(0, inplace=True)
        else:
            # 非数值型用'未知'填充
            df[col].fillna('未知', inplace=True)

# 删除重复值
df.drop_duplicates(inplace=True)
print(f"清洗后数据形状: {df.shape}")

# 保存处理后的数据
processed_file = os.path.join(DATA_DIR, '380平台(已处理).csv')
df.to_csv(processed_file, index=False, encoding='utf-8-sig')
print(f"已保存处理后的数据到: {processed_file}")

# 查看数据基本统计信息
print(f"\n数据基本统计信息:")
print(df.describe())

# =============================================================================
# 任务4: 平台行业分析
# =============================================================================
print("\n" + "=" * 60)
print("任务4: 平台行业分析")
print("=" * 60)

# 识别数据列名（根据实际数据调整）
print(f"\n数据列名: {df.columns.tolist()}")

# 假设数据包含以下字段（需要根据实际数据调整）
# 日期、行业、销售额、成本、利润等

# 尝试识别日期列和相关列
date_cols = [col for col in df.columns if '日期' in col or '时间' in col or 'date' in col.lower()]
if date_cols:
    date_col = date_cols[0]
    print(f"识别到日期列: {date_col}")
    # 转换日期格式
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    # 提取年月
    df['年月'] = df[date_col].dt.to_period('M')
    df['年'] = df[date_col].dt.year
    df['月'] = df[date_col].dt.month

# 识别金额相关列
print(f"\n正在识别金额相关列...")
for col in df.columns:
    print(f"  - {col}: {df[col].dtype}")

print("\n" + "-" * 40)
print("数据结构探索完成，请根据实际列名调整分析代码")
print("-" * 40)

# 保存数据列名信息，方便后续分析
columns_info = pd.DataFrame({
    '列名': df.columns,
    '数据类型': df.dtypes.values,
    '非空数量': df.count().values,
    '唯一值数量': [df[col].nunique() for col in df.columns]
})
columns_info.to_csv(os.path.join(DATA_DIR, '数据列信息.csv'), index=False, encoding='utf-8-sig')
print(f"\n列信息已保存到: {os.path.join(DATA_DIR, '数据列信息.csv')}")

print("\n" + "=" * 60)
print("数据预处理完成！")
print("=" * 60)
