# -*- coding: utf-8 -*-
"""
供应链经营数据分析 - 380平台完整分析
包含：任务3-数据预处理、任务4-平台行业分析、任务5-母婴行业分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from datetime import datetime
import warnings
import os

# 忽略警告
warnings.filterwarnings('ignore')

# =============================================================================
# 中文字体设置 - 解决中文显示问题
# =============================================================================
# 方法1: 直接设置系统中存在的中文字体
import matplotlib.font_manager as fm

# 查找系统中可用的中文字体
def get_chinese_font():
    """获取可用的中文字体"""
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong', 
                     'STHeiti', 'STKaiti', 'STSong', 'STFangsong']
    
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    
    for font in chinese_fonts:
        if font in available_fonts:
            return font
    
    # 如果没有找到常用中文字体，尝试查找包含中文的字体
    for f in fm.fontManager.ttflist:
        if 'Hei' in f.name or 'Song' in f.name or 'Kai' in f.name or 'Ming' in f.name:
            return f.name
    
    return 'SimHei'  # 默认返回黑体

# 获取中文字体
CHINESE_FONT = get_chinese_font()
print(f"使用中文字体: {CHINESE_FONT}")

# 设置matplotlib参数
plt.rcParams['font.sans-serif'] = [CHINESE_FONT]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 100

# 重要：在设置seaborn样式后重新设置字体，因为seaborn会覆盖字体设置
sns.set_style("whitegrid")
sns.set_palette("husl")
# 再次确保字体设置正确
plt.rcParams['font.sans-serif'] = [CHINESE_FONT]
plt.rcParams['axes.unicode_minus'] = False

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GRAPH_DIR = os.path.join(BASE_DIR, 'graph')

# 确保输出目录存在
os.makedirs(GRAPH_DIR, exist_ok=True)

# =============================================================================
# 读取处理后的数据
# =============================================================================
print("=" * 70)
print("供应链经营数据分析 - 380平台完整分析")
print("=" * 70)

# 读取处理后的数据
processed_file = os.path.join(DATA_DIR, '380平台(已处理).csv')
if os.path.exists(processed_file):
    df = pd.read_csv(processed_file)
    df['日期'] = pd.to_datetime(df['日期'])
else:
    # 如果没有处理过的数据，重新处理
    file_path = os.path.join(DATA_DIR, '380平台－数据源.xlsx')
    df = pd.read_excel(file_path)
    df['日期'] = pd.to_datetime(df['日期'], errors='coerce')
    
    # 处理缺失值
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna('未知', inplace=True)
    
    # 删除重复值
    df.drop_duplicates(inplace=True)
    df.to_csv(processed_file, index=False, encoding='utf-8-sig')

# 添加年月列
df['年月'] = df['日期'].dt.to_period('M')
df['年'] = df['日期'].dt.year
df['月'] = df['日期'].dt.month
df['年月字符串'] = df['日期'].dt.strftime('%Y-%m')

print(f"数据加载完成，数据形状: {df.shape}")
print(f"时间范围: {df['日期'].min()} 至 {df['日期'].max()}")
print(f"行业类别: {df['行业'].unique()}")

# =============================================================================
# 任务4.1: 统计分析380平台每月的盈利情况
# =============================================================================
print("\n" + "=" * 70)
print("任务4.1: 统计分析380平台每月的盈利情况")
print("=" * 70)

# 使用"经营利润1"作为盈利指标
monthly_profit = df.groupby('年月字符串').agg({
    '销售收入': 'sum',
    '销售成本': 'sum',
    '综合毛利': 'sum',
    '经营利润1': 'sum'
}).reset_index()

monthly_profit.columns = ['年月', '销售收入', '销售成本', '综合毛利', '经营利润']
monthly_profit = monthly_profit.sort_values('年月')

print("\n380平台每月盈利情况统计:")
print(monthly_profit.to_string(index=False))

# 保存结果
monthly_profit.to_csv(os.path.join(DATA_DIR, '任务4.1(380平台月盈利).csv'), 
                      index=False, encoding='utf-8-sig')

# 绘制图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 每月销售收入和成本对比
ax1 = axes[0, 0]
x = range(len(monthly_profit))
width = 0.35
ax1.bar([i - width/2 for i in x], monthly_profit['销售收入']/10000, width, label='销售收入', color='#3498db', alpha=0.8)
ax1.bar([i + width/2 for i in x], monthly_profit['销售成本']/10000, width, label='销售成本', color='#e74c3c', alpha=0.8)
ax1.set_xlabel('月份')
ax1.set_ylabel('金额（万元）')
ax1.set_title('380平台每月销售收入与成本对比', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(monthly_profit['年月'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 每月经营利润折线图
ax2 = axes[0, 1]
colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in monthly_profit['经营利润']]
ax2.bar(monthly_profit['年月'], monthly_profit['经营利润']/10000, color=colors, alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('月份')
ax2.set_ylabel('经营利润（万元）')
ax2.set_title('380平台每月经营利润', fontsize=14, fontweight='bold')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3)

# 图3: 综合毛利趋势
ax3 = axes[1, 0]
ax3.plot(monthly_profit['年月'], monthly_profit['综合毛利']/10000, 
         marker='o', linewidth=2, markersize=6, color='#9b59b6')
ax3.fill_between(range(len(monthly_profit)), monthly_profit['综合毛利']/10000, 
                  alpha=0.3, color='#9b59b6')
ax3.set_xlabel('月份')
ax3.set_ylabel('综合毛利（万元）')
ax3.set_title('380平台每月综合毛利趋势', fontsize=14, fontweight='bold')
ax3.set_xticks(range(len(monthly_profit)))
ax3.set_xticklabels(monthly_profit['年月'], rotation=45, ha='right')
ax3.grid(True, alpha=0.3)

# 图4: 盈利情况汇总统计
ax4 = axes[1, 1]
summary_data = {
    '指标': ['总销售收入', '总销售成本', '总综合毛利', '总经营利润'],
    '金额(万元)': [
        monthly_profit['销售收入'].sum()/10000,
        monthly_profit['销售成本'].sum()/10000,
        monthly_profit['综合毛利'].sum()/10000,
        monthly_profit['经营利润'].sum()/10000
    ]
}
colors_summary = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
bars = ax4.bar(summary_data['指标'], summary_data['金额(万元)'], color=colors_summary, alpha=0.8)
ax4.set_ylabel('金额（万元）')
ax4.set_title('380平台盈利情况汇总', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, summary_data['金额(万元)']):
    height = bar.get_height()
    ax4.annotate(f'{val:,.0f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '4.1_平台月盈利分析.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '4.1_平台月盈利分析.png')}")

# =============================================================================
# 任务4.2: 统计分析380平台各行业的盈利分布情况
# =============================================================================
print("\n" + "=" * 70)
print("任务4.2: 统计分析380平台各行业的盈利分布情况")
print("=" * 70)

# 按行业统计盈利
industry_profit = df.groupby('行业').agg({
    '销售收入': 'sum',
    '销售成本': 'sum',
    '综合毛利': 'sum',
    '经营利润1': 'sum',
    '结算编码': 'count'
}).reset_index()

industry_profit.columns = ['行业', '销售收入', '销售成本', '综合毛利', '经营利润', '订单数']
industry_profit['利润率(%)'] = (industry_profit['经营利润'] / industry_profit['销售收入'] * 100).round(2)
industry_profit = industry_profit.sort_values('经营利润', ascending=False)

print("\n380平台各行业盈利分布情况:")
print(industry_profit.to_string(index=False))

# 保存结果
industry_profit.to_csv(os.path.join(DATA_DIR, '任务4.2(各行业盈利分布).csv'), 
                       index=False, encoding='utf-8-sig')

# 绘制图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 各行业经营利润柱状图
ax1 = axes[0, 0]
colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in industry_profit['经营利润']]
bars = ax1.barh(industry_profit['行业'], industry_profit['经营利润']/10000, color=colors, alpha=0.8)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('经营利润（万元）')
ax1.set_title('各行业经营利润分布', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')

# 图2: 各行业销售收入饼图
ax2 = axes[0, 1]
positive_profit = industry_profit[industry_profit['销售收入'] > 0]
ax2.pie(positive_profit['销售收入'], labels=positive_profit['行业'], autopct='%1.1f%%',
        colors=sns.color_palette("husl", len(positive_profit)), startangle=90)
ax2.set_title('各行业销售收入占比', fontsize=14, fontweight='bold')

# 图3: 各行业利润率对比
ax3 = axes[1, 0]
colors_rate = ['#27ae60' if v >= 0 else '#e74c3c' for v in industry_profit['利润率(%)']]
bars = ax3.bar(industry_profit['行业'], industry_profit['利润率(%)'], color=colors_rate, alpha=0.8)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_xlabel('行业')
ax3.set_ylabel('利润率（%）')
ax3.set_title('各行业利润率对比', fontsize=14, fontweight='bold')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax3.grid(True, alpha=0.3, axis='y')

# 图4: 各行业订单数
ax4 = axes[1, 1]
ax4.bar(industry_profit['行业'], industry_profit['订单数'], color='#3498db', alpha=0.8)
ax4.set_xlabel('行业')
ax4.set_ylabel('订单数')
ax4.set_title('各行业订单数量分布', fontsize=14, fontweight='bold')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '4.2_各行业盈利分布.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '4.2_各行业盈利分布.png')}")

# =============================================================================
# 任务4.3: 统计分析380平台各行业每月的盈利情况，并预测下个月的盈利情况
# =============================================================================
print("\n" + "=" * 70)
print("任务4.3: 统计分析380平台各行业每月的盈利情况及预测")
print("=" * 70)

# 按行业和月份统计盈利
industry_monthly = df.groupby(['行业', '年月字符串']).agg({
    '经营利润1': 'sum'
}).reset_index()

industry_monthly.columns = ['行业', '年月', '经营利润']
industry_monthly = industry_monthly.sort_values(['行业', '年月'])

# 创建透视表
pivot_table = industry_monthly.pivot(index='年月', columns='行业', values='经营利润')
pivot_table = pivot_table.fillna(0)

print("\n各行业月度经营利润（万元）:")
print((pivot_table/10000).round(2).to_string())

# 保存结果
pivot_table.to_csv(os.path.join(DATA_DIR, '任务4.3(各行业月盈利).csv'), encoding='utf-8-sig')

# 简单线性预测下个月的盈利
# 使用最近6个月的数据进行简单移动平均预测
predictions = {}
for industry in pivot_table.columns:
    recent_data = pivot_table[industry].tail(6)
    # 简单移动平均预测
    prediction = recent_data.mean()
    predictions[industry] = prediction

predictions_df = pd.DataFrame({
    '行业': list(predictions.keys()),
    '预测盈利（2016-01）': list(predictions.values())
})
predictions_df['预测盈利（万元）'] = (predictions_df['预测盈利（2016-01）'] / 10000).round(2)
print("\n2016年1月盈利预测（基于最近6个月移动平均）:")
print(predictions_df[['行业', '预测盈利（万元）']].to_string(index=False))

# 保存预测结果
predictions_df.to_csv(os.path.join(DATA_DIR, '任务4.3(盈利预测).csv'), index=False, encoding='utf-8-sig')

# 绘制图表
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# 图1: 各行业月度盈利趋势
ax1 = axes[0]
for industry in pivot_table.columns:
    ax1.plot(pivot_table.index, pivot_table[industry]/10000, marker='o', label=industry, linewidth=2, markersize=4)
ax1.set_xlabel('月份')
ax1.set_ylabel('经营利润（万元）')
ax1.set_title('各行业月度经营利润趋势', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
ax1.grid(True, alpha=0.3)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# 图2: 预测结果
ax2 = axes[1]
colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in predictions_df['预测盈利（2016-01）']]
bars = ax2.bar(predictions_df['行业'], predictions_df['预测盈利（万元）'], color=colors, alpha=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel('行业')
ax2.set_ylabel('预测经营利润（万元）')
ax2.set_title('2016年1月各行业经营利润预测', fontsize=14, fontweight='bold')
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, val in zip(bars, predictions_df['预测盈利（万元）']):
    height = bar.get_height()
    ax2.annotate(f'{val:,.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -15),
                textcoords="offset points",
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '4.3_各行业月盈利及预测.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '4.3_各行业月盈利及预测.png')}")

# =============================================================================
# 任务4.4: 为380平台各行业投入比重分配提供建议
# =============================================================================
print("\n" + "=" * 70)
print("任务4.4: 为380平台各行业投入比重分配提供建议")
print("=" * 70)

# 综合分析：盈利能力 + 增长趋势 + 稳定性
industry_analysis = industry_profit[['行业', '经营利润', '利润率(%)']].copy()

# 计算各行业的利润增长趋势（最近3个月 vs 前3个月）
growth_rates = {}
for industry in pivot_table.columns:
    recent = pivot_table[industry].tail(3).mean()
    earlier = pivot_table[industry].iloc[-6:-3].mean() if len(pivot_table) >= 6 else pivot_table[industry].head(3).mean()
    if earlier != 0:
        growth = (recent - earlier) / abs(earlier) * 100
    else:
        growth = 100 if recent > 0 else -100
    growth_rates[industry] = growth

industry_analysis['增长率(%)'] = industry_analysis['行业'].map(growth_rates)

# 计算波动性（标准差/均值）
volatility = {}
for industry in pivot_table.columns:
    if pivot_table[industry].mean() != 0:
        vol = pivot_table[industry].std() / abs(pivot_table[industry].mean()) * 100
    else:
        vol = 100
    volatility[industry] = vol

industry_analysis['波动性(%)'] = industry_analysis['行业'].map(volatility)

# 综合评分（利润权重40%，利润率权重25%，增长率权重25%，稳定性权重10%）
def normalize(series):
    if series.max() == series.min():
        return pd.Series([0.5] * len(series))
    return (series - series.min()) / (series.max() - series.min())

industry_analysis['利润得分'] = normalize(industry_analysis['经营利润'])
industry_analysis['利润率得分'] = normalize(industry_analysis['利润率(%)'])
industry_analysis['增长得分'] = normalize(industry_analysis['增长率(%)'])
industry_analysis['稳定性得分'] = 1 - normalize(industry_analysis['波动性(%)'])

industry_analysis['综合得分'] = (
    industry_analysis['利润得分'] * 0.4 +
    industry_analysis['利润率得分'] * 0.25 +
    industry_analysis['增长得分'] * 0.25 +
    industry_analysis['稳定性得分'] * 0.1
)

# 计算建议投入比重
total_score = industry_analysis['综合得分'].sum()
industry_analysis['建议投入比重(%)'] = (industry_analysis['综合得分'] / total_score * 100).round(1)

# 排序
industry_analysis = industry_analysis.sort_values('综合得分', ascending=False)

# 添加投入建议
def get_recommendation(row):
    if row['综合得分'] >= 0.7:
        return '重点投入'
    elif row['综合得分'] >= 0.4:
        return '适度投入'
    elif row['综合得分'] >= 0.2:
        return '维持现状'
    else:
        return '减少投入'

industry_analysis['投入建议'] = industry_analysis.apply(get_recommendation, axis=1)

print("\n各行业投入比重分配建议:")
result_cols = ['行业', '经营利润', '利润率(%)', '增长率(%)', '波动性(%)', '综合得分', '建议投入比重(%)', '投入建议']
print(industry_analysis[result_cols].round(2).to_string(index=False))

# 保存结果
industry_analysis[result_cols].to_csv(os.path.join(DATA_DIR, '任务4.4(投入比重建议).csv'), 
                                       index=False, encoding='utf-8-sig')

# 绘制图表
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图1: 建议投入比重饼图
ax1 = axes[0]
ax1.pie(industry_analysis['建议投入比重(%)'], labels=industry_analysis['行业'], 
        autopct='%1.1f%%', colors=sns.color_palette("husl", len(industry_analysis)), startangle=90)
ax1.set_title('各行业建议投入比重分配', fontsize=14, fontweight='bold')

# 图2: 综合得分雷达图（简化为柱状图）
ax2 = axes[1]
colors = {'重点投入': '#27ae60', '适度投入': '#3498db', '维持现状': '#f39c12', '减少投入': '#e74c3c'}
bar_colors = [colors.get(r, '#95a5a6') for r in industry_analysis['投入建议']]
bars = ax2.barh(industry_analysis['行业'], industry_analysis['综合得分'], color=bar_colors, alpha=0.8)
ax2.set_xlabel('综合得分')
ax2.set_title('各行业综合评分及投入建议', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in colors.items()]
ax2.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '4.4_投入比重建议.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '4.4_投入比重建议.png')}")

# =============================================================================
# 任务5: 母婴行业分析
# =============================================================================
print("\n" + "=" * 70)
print("任务5: 母婴行业分析")
print("=" * 70)

# 筛选母婴行业数据
baby_df = df[df['行业'] == '母婴'].copy()
print(f"\n母婴行业数据量: {len(baby_df)}")

if len(baby_df) == 0:
    # 尝试查找包含"母婴"关键字的行业
    baby_industries = [ind for ind in df['行业'].unique() if '母婴' in str(ind) or '婴' in str(ind)]
    if baby_industries:
        baby_df = df[df['行业'].isin(baby_industries)].copy()
        print(f"识别到母婴相关行业: {baby_industries}")
    else:
        print("未找到母婴行业数据，请检查数据")
        # 为了演示，使用第一个行业的数据
        first_industry = df['行业'].iloc[0]
        baby_df = df[df['行业'] == first_industry].copy()
        print(f"使用 '{first_industry}' 行业数据进行演示")

# 识别品牌列（项目简称可能是品牌）
brand_col = '项目简称'
print(f"品牌识别列: {brand_col}")
print(f"品牌数量: {baby_df[brand_col].nunique()}")

# =============================================================================
# 任务5.1: 统计分析母婴行业各品牌销售分布情况
# =============================================================================
print("\n" + "-" * 60)
print("任务5.1: 统计分析母婴行业各品牌销售分布情况")
print("-" * 60)

brand_sales = baby_df.groupby(brand_col).agg({
    '销售收入': 'sum',
    '销售成本': 'sum',
    '综合毛利': 'sum',
    '经营利润1': 'sum',
    '结算编码': 'count'
}).reset_index()

brand_sales.columns = ['品牌', '销售收入', '销售成本', '综合毛利', '经营利润', '订单数']
brand_sales['毛利率(%)'] = (brand_sales['综合毛利'] / brand_sales['销售收入'] * 100).round(2)
brand_sales['销售占比(%)'] = (brand_sales['销售收入'] / brand_sales['销售收入'].sum() * 100).round(2)
brand_sales = brand_sales.sort_values('销售收入', ascending=False)

print("\n母婴行业各品牌销售分布情况:")
print(brand_sales.head(15).to_string(index=False))

# 保存结果
brand_sales.to_csv(os.path.join(DATA_DIR, '任务5.1(品牌销售分布).csv'), 
                   index=False, encoding='utf-8-sig')

# 调整建议
def get_sales_suggestion(row, total_sales, avg_margin):
    if row['销售占比(%)'] < 1:
        if row['毛利率(%)'] > avg_margin:
            return '建议加大推广'
        else:
            return '建议淘汰'
    elif row['销售占比(%)'] < 5:
        return '需关注'
    else:
        return '保持现状'

avg_margin = brand_sales['毛利率(%)'].mean()
brand_sales['调整建议'] = brand_sales.apply(
    lambda x: get_sales_suggestion(x, brand_sales['销售收入'].sum(), avg_margin), axis=1
)

print("\n品牌调整建议:")
print(brand_sales[['品牌', '销售占比(%)', '毛利率(%)', '调整建议']].head(20).to_string(index=False))

# 绘制图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 取销售额Top 10品牌
top_brands = brand_sales.head(10)

# 图1: Top10品牌销售收入
ax1 = axes[0, 0]
bars = ax1.barh(top_brands['品牌'], top_brands['销售收入']/10000, color='#3498db', alpha=0.8)
ax1.set_xlabel('销售收入（万元）')
ax1.set_title('母婴行业Top10品牌销售收入', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# 图2: Top10品牌销售占比
ax2 = axes[0, 1]
ax2.pie(top_brands['销售收入'], labels=top_brands['品牌'], autopct='%1.1f%%',
        colors=sns.color_palette("husl", len(top_brands)), startangle=90)
ax2.set_title('Top10品牌销售收入占比', fontsize=14, fontweight='bold')

# 图3: 品牌毛利率分布
ax3 = axes[1, 0]
ax3.bar(top_brands['品牌'], top_brands['毛利率(%)'], color='#27ae60', alpha=0.8)
ax3.axhline(y=avg_margin, color='red', linestyle='--', label=f'平均毛利率: {avg_margin:.1f}%')
ax3.set_xlabel('品牌')
ax3.set_ylabel('毛利率（%）')
ax3.set_title('Top10品牌毛利率对比', fontsize=14, fontweight='bold')
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 图4: 调整建议分布
ax4 = axes[1, 1]
suggestion_counts = brand_sales['调整建议'].value_counts()
colors_sugg = {'保持现状': '#27ae60', '需关注': '#f39c12', '建议加大推广': '#3498db', '建议淘汰': '#e74c3c'}
ax4.pie(suggestion_counts, labels=suggestion_counts.index, autopct='%1.1f%%',
        colors=[colors_sugg.get(s, '#95a5a6') for s in suggestion_counts.index], startangle=90)
ax4.set_title('品牌调整建议分布', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '5.1_品牌销售分布.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '5.1_品牌销售分布.png')}")

# =============================================================================
# 任务5.2: 统计分析母婴行业各品牌在各地区的盈利分布情况
# =============================================================================
print("\n" + "-" * 60)
print("任务5.2: 统计分析母婴行业各品牌在各地区的盈利分布情况")
print("-" * 60)

# 按品牌和地区统计
brand_region = baby_df.groupby([brand_col, '省区平台']).agg({
    '销售收入': 'sum',
    '经营利润1': 'sum'
}).reset_index()

brand_region.columns = ['品牌', '地区', '销售收入', '经营利润']

# 创建品牌-地区透视表
region_pivot = brand_region.pivot_table(
    index='品牌', 
    columns='地区', 
    values='经营利润', 
    aggfunc='sum',
    fill_value=0
)

# 取Top 10品牌和Top 10地区
top10_brands = brand_sales.head(10)['品牌'].tolist()
top_regions = brand_region.groupby('地区')['经营利润'].sum().sort_values(ascending=False).head(10).index.tolist()

region_pivot_filtered = region_pivot.loc[
    region_pivot.index.isin(top10_brands),
    region_pivot.columns.isin(top_regions)
]

print("\nTop10品牌在各主要地区的经营利润（万元）:")
print((region_pivot_filtered/10000).round(2).to_string())

# 保存结果
region_pivot.to_csv(os.path.join(DATA_DIR, '任务5.2(品牌地区盈利).csv'), encoding='utf-8-sig')

# 绘制热力图
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(region_pivot_filtered/10000, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
            ax=ax, cbar_kws={'label': '经营利润（万元）'})
ax.set_title('母婴行业Top10品牌在各地区的盈利分布(万元)', fontsize=14, fontweight='bold')
ax.set_xlabel('地区')
ax.set_ylabel('品牌')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '5.2_品牌地区盈利热力图.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '5.2_品牌地区盈利热力图.png')}")

# =============================================================================
# 任务5.3: 统计分析母婴行业各品牌库存和周转情况
# =============================================================================
print("\n" + "-" * 60)
print("任务5.3: 统计分析母婴行业各品牌库存和周转情况")
print("-" * 60)

# 计算库存周转相关指标
brand_inventory = baby_df.groupby(brand_col).agg({
    '期末存货余额': 'mean',
    '销售成本': 'sum',
    '销售收入': 'sum',
    '结算编码': 'count'
}).reset_index()

brand_inventory.columns = ['品牌', '平均库存', '总销售成本', '总销售收入', '月份数']

# 计算库存周转率 = 销售成本 / 平均库存
brand_inventory['库存周转率'] = np.where(
    brand_inventory['平均库存'] != 0,
    brand_inventory['总销售成本'] / brand_inventory['平均库存'],
    0
)

# 计算周转天数 = 365 / 库存周转率
brand_inventory['周转天数'] = np.where(
    brand_inventory['库存周转率'] != 0,
    365 / brand_inventory['库存周转率'],
    999
)

# 销售速度指标（月均销售额）
brand_inventory['月均销售额'] = brand_inventory['总销售收入'] / brand_inventory['月份数']

# 给品牌打标签
def label_brand(row):
    # 基于周转天数和销售额综合判断
    if row['库存周转率'] == 0 or row['周转天数'] > 180:
        return '滞销'
    elif row['周转天数'] < 60:
        return '热销'
    else:
        return '正常'

brand_inventory['标签'] = brand_inventory.apply(label_brand, axis=1)

# 排序
brand_inventory = brand_inventory.sort_values('库存周转率', ascending=False)

print("\n母婴行业各品牌库存和周转情况:")
result_cols = ['品牌', '平均库存', '总销售成本', '库存周转率', '周转天数', '标签']
print(brand_inventory[result_cols].round(2).head(20).to_string(index=False))

# 保存结果
brand_inventory[result_cols].to_csv(os.path.join(DATA_DIR, '任务5.3(品牌库存周转).csv'), 
                                     index=False, encoding='utf-8-sig')

# 生成标签表格
label_table = brand_inventory[['品牌', '标签']].copy()
label_table.insert(0, '序号', range(1, len(label_table) + 1))
label_table.to_csv(os.path.join(DATA_DIR, '任务5.3(品牌标签表).csv'), index=False, encoding='utf-8-sig')

print("\n品牌标签统计:")
print(label_table['标签'].value_counts())

# 绘制图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 取Top 15品牌用于展示
top15 = brand_inventory.head(15)

# 图1: 各品牌库存周转率
ax1 = axes[0, 0]
colors_label = {'热销': '#27ae60', '正常': '#3498db', '滞销': '#e74c3c'}
bar_colors = [colors_label.get(l, '#95a5a6') for l in top15['标签']]
ax1.barh(top15['品牌'], top15['库存周转率'], color=bar_colors, alpha=0.8)
ax1.set_xlabel('库存周转率')
ax1.set_title('各品牌库存周转率（Top15）', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

# 添加图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=l) for l, c in colors_label.items()]
ax1.legend(handles=legend_elements, loc='lower right')

# 图2: 品牌标签分布
ax2 = axes[0, 1]
label_counts = brand_inventory['标签'].value_counts()
ax2.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%',
        colors=[colors_label.get(l, '#95a5a6') for l in label_counts.index], startangle=90,
        explode=[0.05 if l == '热销' else 0 for l in label_counts.index])
ax2.set_title('品牌销售标签分布', fontsize=14, fontweight='bold')

# 图3: 周转天数分布
ax3 = axes[1, 0]
# 过滤极端值
filtered = brand_inventory[brand_inventory['周转天数'] < 500]
ax3.hist(filtered['周转天数'], bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
ax3.axvline(x=60, color='green', linestyle='--', label='热销阈值(60天)')
ax3.axvline(x=180, color='red', linestyle='--', label='滞销阈值(180天)')
ax3.set_xlabel('周转天数')
ax3.set_ylabel('品牌数量')
ax3.set_title('品牌周转天数分布', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 图4: 平均库存 vs 周转率散点图
ax4 = axes[1, 1]
for label in colors_label:
    subset = brand_inventory[brand_inventory['标签'] == label]
    ax4.scatter(subset['平均库存']/10000, subset['库存周转率'], 
                c=colors_label[label], label=label, alpha=0.6, s=50)
ax4.set_xlabel('平均库存（万元）')
ax4.set_ylabel('库存周转率')
ax4.set_title('平均库存 vs 库存周转率', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '5.3_库存周转分析.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '5.3_库存周转分析.png')}")

# =============================================================================
# 任务5.4: 综合分析母婴行业运营情况
# =============================================================================
print("\n" + "-" * 60)
print("任务5.4: 综合分析母婴行业运营情况")
print("-" * 60)

# 合并销售和库存数据
comprehensive = brand_sales[['品牌', '销售收入', '综合毛利', '经营利润', '毛利率(%)', '调整建议']].merge(
    brand_inventory[['品牌', '平均库存', '库存周转率', '周转天数', '标签']],
    on='品牌',
    how='inner'
)

# 综合评价
def get_health_status(row):
    score = 0
    # 盈利状况评分
    if row['经营利润'] > 0:
        score += 3
    elif row['经营利润'] > -50000:
        score += 1
    
    # 毛利率评分
    if row['毛利率(%)'] > 10:
        score += 3
    elif row['毛利率(%)'] > 5:
        score += 2
    elif row['毛利率(%)'] > 0:
        score += 1
    
    # 库存周转评分
    if row['标签'] == '热销':
        score += 3
    elif row['标签'] == '正常':
        score += 2
    else:
        score += 0
    
    if score >= 7:
        return '优秀'
    elif score >= 5:
        return '良好'
    elif score >= 3:
        return '一般'
    else:
        return '较差'

comprehensive['健康状况'] = comprehensive.apply(get_health_status, axis=1)

# 发展趋势评估
def get_trend(row):
    if row['健康状况'] in ['优秀', '良好'] and row['标签'] == '热销':
        return '上升'
    elif row['健康状况'] == '较差' or row['标签'] == '滞销':
        return '下降'
    else:
        return '稳定'

comprehensive['发展趋势'] = comprehensive.apply(get_trend, axis=1)

# 优化建议
def get_optimization(row):
    suggestions = []
    if row['标签'] == '滞销':
        suggestions.append('清理库存')
    if row['毛利率(%)'] < 5:
        suggestions.append('优化成本结构')
    if row['经营利润'] < 0:
        suggestions.append('减少投入或淘汰')
    if row['标签'] == '热销' and row['经营利润'] > 0:
        suggestions.append('增加备货')
    if not suggestions:
        suggestions.append('保持现状')
    return '、'.join(suggestions)

comprehensive['优化建议'] = comprehensive.apply(get_optimization, axis=1)

print("\n母婴行业综合分析结果:")
result_cols = ['品牌', '销售收入', '毛利率(%)', '标签', '健康状况', '发展趋势', '优化建议']
comprehensive_sorted = comprehensive.sort_values('销售收入', ascending=False)
print(comprehensive_sorted[result_cols].head(20).to_string(index=False))

# 保存结果
comprehensive_sorted.to_csv(os.path.join(DATA_DIR, '任务5.4(综合分析).csv'), 
                             index=False, encoding='utf-8-sig')

# 生成需加大推广的品牌列表
promotion_brands = comprehensive[
    (comprehensive['调整建议'] == '建议加大推广') | 
    ((comprehensive['标签'] == '热销') & (comprehensive['经营利润'] > 0))
][['品牌', '销售收入', '毛利率(%)', '标签', '优化建议']]
promotion_brands.to_csv(os.path.join(DATA_DIR, '任务5.4(需加大推广的品牌).csv'), 
                         index=False, encoding='utf-8-sig')

# 总结报告
print("\n" + "=" * 70)
print("母婴部门健康状况评估总结")
print("=" * 70)

total_revenue = comprehensive['销售收入'].sum()
total_profit = comprehensive['经营利润'].sum()
avg_margin = comprehensive['毛利率(%)'].mean()
health_distribution = comprehensive['健康状况'].value_counts()
label_distribution = comprehensive['标签'].value_counts()

print(f"\n【运营情况】")
print(f"  - 总销售收入: {total_revenue/10000:,.2f} 万元")
print(f"  - 总经营利润: {total_profit/10000:,.2f} 万元")
print(f"  - 平均毛利率: {avg_margin:.2f}%")
print(f"  - 品牌数量: {len(comprehensive)}")

print(f"\n【财务状况】")
profit_brands = len(comprehensive[comprehensive['经营利润'] > 0])
loss_brands = len(comprehensive[comprehensive['经营利润'] <= 0])
print(f"  - 盈利品牌: {profit_brands} 个")
print(f"  - 亏损品牌: {loss_brands} 个")

print(f"\n【物流管理】")
print(f"  - 热销品牌: {label_distribution.get('热销', 0)} 个")
print(f"  - 正常品牌: {label_distribution.get('正常', 0)} 个")
print(f"  - 滞销品牌: {label_distribution.get('滞销', 0)} 个")

print(f"\n【健康状况评估】")
print(f"  - 优秀: {health_distribution.get('优秀', 0)} 个品牌")
print(f"  - 良好: {health_distribution.get('良好', 0)} 个品牌")
print(f"  - 一般: {health_distribution.get('一般', 0)} 个品牌")
print(f"  - 较差: {health_distribution.get('较差', 0)} 个品牌")

print(f"\n【发展趋势】")
trend_dist = comprehensive['发展趋势'].value_counts()
for trend, count in trend_dist.items():
    print(f"  - {trend}: {count} 个品牌 ({count/len(comprehensive)*100:.1f}%)")

print(f"\n【问题发现与优化建议】")
print("  1. 对于滞销品牌，建议及时清理库存，避免资金占用")
print("  2. 对于亏损品牌，需分析原因，优化成本结构或考虑淘汰")
print("  3. 对于热销且盈利的品牌，建议增加备货，避免缺货损失")
print("  4. 需要重点关注毛利率较低的品牌，优化议价能力")

# 绘制综合分析图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1: 健康状况分布
ax1 = axes[0, 0]
health_colors = {'优秀': '#27ae60', '良好': '#3498db', '一般': '#f39c12', '较差': '#e74c3c'}
health_counts = comprehensive['健康状况'].value_counts()
ax1.pie(health_counts, labels=health_counts.index, autopct='%1.1f%%',
        colors=[health_colors.get(h, '#95a5a6') for h in health_counts.index], startangle=90)
ax1.set_title('品牌健康状况分布', fontsize=14, fontweight='bold')

# 图2: 发展趋势分布
ax2 = axes[0, 1]
trend_colors = {'上升': '#27ae60', '稳定': '#3498db', '下降': '#e74c3c'}
trend_counts = comprehensive['发展趋势'].value_counts()
ax2.pie(trend_counts, labels=trend_counts.index, autopct='%1.1f%%',
        colors=[trend_colors.get(t, '#95a5a6') for t in trend_counts.index], startangle=90)
ax2.set_title('品牌发展趋势分布', fontsize=14, fontweight='bold')

# 图3: Top15品牌销售收入vs利润
ax3 = axes[1, 0]
top15_comp = comprehensive_sorted.head(15)
x = range(len(top15_comp))
width = 0.35
ax3.bar([i - width/2 for i in x], top15_comp['销售收入']/10000, width, label='销售收入', color='#3498db', alpha=0.8)
ax3.bar([i + width/2 for i in x], top15_comp['经营利润']/10000, width, label='经营利润', color='#27ae60', alpha=0.8)
ax3.set_xlabel('品牌')
ax3.set_ylabel('金额（万元）')
ax3.set_title('Top15品牌销售收入与经营利润对比', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(top15_comp['品牌'], rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 图4: 健康状况与销售标签交叉分析
ax4 = axes[1, 1]
cross_tab = pd.crosstab(comprehensive['健康状况'], comprehensive['标签'])
cross_tab.plot(kind='bar', ax=ax4, color=['#e74c3c', '#3498db', '#27ae60'], alpha=0.8)
ax4.set_xlabel('健康状况')
ax4.set_ylabel('品牌数量')
ax4.set_title('健康状况与销售标签交叉分析', fontsize=14, fontweight='bold')
ax4.legend(title='销售标签')
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=0)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, '5.4_综合分析.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n图表已保存到: {os.path.join(GRAPH_DIR, '5.4_综合分析.png')}")

print("\n" + "=" * 70)
print("所有分析任务已完成！")
print("=" * 70)
print(f"\n输出文件目录:")
print(f"  - 数据文件: {DATA_DIR}")
print(f"  - 图表文件: {GRAPH_DIR}")
