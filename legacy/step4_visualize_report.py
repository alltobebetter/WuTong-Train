# -*- coding: utf-8 -*-
# Source: 04_可视化大屏与报告生成.ipynb


# ===== code cell 1 =====
# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# 设置Seaborn样式
sns.set_style('whitegrid')
sns.set_palette('husl')

print('库导入成功！')

# ===== code cell 2 =====
# 加载分析结果数据
import os

possible_paths = [
    os.path.join(os.path.dirname(__file__), 'final_results.pkl'),
    os.path.join(os.path.dirname(__file__), '..', '01_????', 'final_results.pkl'),
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f'成功加载数据: {path}')
        break

if df is None:
    raise FileNotFoundError('请先运行前置模块生成分析结果')

print(f'数据形状: {df.shape}')
print(f'数据列: {list(df.columns)}')

# ===== code cell 3 =====
# 创建安全态势大屏
fig = plt.figure(figsize=(20, 14))
fig.suptitle('AI安全告警智能研判系统 - 安全态势大屏', fontsize=20, fontweight='bold', y=0.98)

# 定义颜色方案
colors_risk = {'严重': '#d62728', '高危': '#ff7f0e', '中危': '#ffbb78', '低危': '#98df8a', '信息': '#aec7e8'}
colors_attack = plt.cm.Set3(np.linspace(0, 1, 9))

# 1. 核心指标卡片区域 (顶部)
ax_metrics = fig.add_axes([0.02, 0.85, 0.96, 0.10])
ax_metrics.axis('off')

# 计算核心指标
total_alerts = len(df)
critical_alerts = len(df[df['risk_level'] == '严重'])
high_alerts = len(df[df['risk_level'] == '高危'])
attack_types = df['attack_type'].nunique()
avg_risk = df['risk_score'].mean()
attack_rate = (df['attack_type'] != '正常访问').sum() / len(df) * 100

metrics = [
    ('总告警数', f'{total_alerts:,}', '#3498db'),
    ('严重告警', f'{critical_alerts:,}', '#e74c3c'),
    ('高危告警', f'{high_alerts:,}', '#f39c12'),
    ('攻击类型', f'{attack_types}种', '#9b59b6'),
    ('平均风险分', f'{avg_risk:.1f}', '#1abc9c'),
    ('攻击占比', f'{attack_rate:.1f}%', '#e67e22')
]

for i, (label, value, color) in enumerate(metrics):
    x = 0.08 + i * 0.15
    ax_metrics.text(x, 0.7, value, fontsize=24, fontweight='bold', color=color, ha='center')
    ax_metrics.text(x, 0.2, label, fontsize=12, color='gray', ha='center')

# 2. 攻击类型分布 (左上)
ax1 = fig.add_axes([0.05, 0.45, 0.28, 0.35])
attack_counts = df['attack_type'].value_counts()
wedges, texts, autotexts = ax1.pie(attack_counts.values, labels=None, autopct='%1.1f%%',
                                    colors=colors_attack, pctdistance=0.75)
ax1.set_title('攻击类型分布', fontsize=14, fontweight='bold', pad=10)
ax1.legend(wedges, attack_counts.index, loc='center left', bbox_to_anchor=(-0.3, 0.5), fontsize=9)

# 3. 风险等级分布 (中上)
ax2 = fig.add_axes([0.38, 0.45, 0.25, 0.35])
risk_counts = df['risk_level'].value_counts()
risk_order = ['严重', '高危', '中危', '低危', '信息']
risk_counts = risk_counts.reindex([r for r in risk_order if r in risk_counts.index])
bars = ax2.bar(risk_counts.index, risk_counts.values, 
               color=[colors_risk[r] for r in risk_counts.index], edgecolor='white', linewidth=2)
ax2.set_ylabel('数量')
ax2.set_title('风险等级分布', fontsize=14, fontweight='bold')
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 50, f'{int(height):,}',
             ha='center', va='bottom', fontsize=10)

# 4. 风险评分分布 (右上)
ax3 = fig.add_axes([0.70, 0.45, 0.27, 0.35])
ax3.hist(df['risk_score'], bins=40, color='steelblue', edgecolor='white', alpha=0.8)
ax3.axvline(x=60, color='orange', linestyle='--', linewidth=2, label='高危阈值')
ax3.axvline(x=80, color='red', linestyle='--', linewidth=2, label='严重阈值')
ax3.set_xlabel('风险评分')
ax3.set_ylabel('数量')
ax3.set_title('风险评分分布', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')

# 5. 24小时告警趋势 (左下)
ax4 = fig.add_axes([0.05, 0.08, 0.28, 0.30])
if 'timestamp' in df.columns:
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    hourly = df.groupby('hour').size()
    ax4.fill_between(hourly.index, hourly.values, alpha=0.3, color='steelblue')
    ax4.plot(hourly.index, hourly.values, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax4.set_xlabel('小时')
    ax4.set_ylabel('告警数量')
    ax4.set_xticks(range(0, 24, 3))
ax4.set_title('24小时告警趋势', fontsize=14, fontweight='bold')

# 6. HTTP方法分布 (中下)
ax5 = fig.add_axes([0.38, 0.08, 0.25, 0.30])
method_counts = df['method'].value_counts()
colors_method = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
ax5.pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
        colors=colors_method[:len(method_counts)], startangle=90)
ax5.set_title('HTTP请求方法分布', fontsize=14, fontweight='bold')

# 7. Top 10 源IP (右下)
ax6 = fig.add_axes([0.70, 0.08, 0.27, 0.30])
top_ips = df[df['attack_type'] != '正常访问']['src_ip'].value_counts().head(10)
ax6.barh(range(len(top_ips)), top_ips.values, color='coral')
ax6.set_yticks(range(len(top_ips)))
ax6.set_yticklabels([ip[:15] + '...' if len(ip) > 15 else ip for ip in top_ips.index], fontsize=9)
ax6.set_xlabel('攻击次数')
ax6.set_title('Top 10 攻击源IP', fontsize=14, fontweight='bold')
ax6.invert_yaxis()

plt.savefig('security_dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print('安全态势大屏已保存: security_dashboard.png')

# ===== code cell 4 =====
# 攻击类型详细分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('攻击类型详细分析', fontsize=16, fontweight='bold')

# 1. 各攻击类型的风险评分箱线图
attack_types = df['attack_type'].unique()
data_for_box = [df[df['attack_type'] == at]['risk_score'].values for at in attack_types]
bp = axes[0, 0].boxplot(data_for_box, labels=attack_types, vert=True, patch_artist=True)
colors = plt.cm.Set3(np.linspace(0, 1, len(attack_types)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 0].set_xticklabels(attack_types, rotation=45, ha='right')
axes[0, 0].set_ylabel('风险评分')
axes[0, 0].set_title('各攻击类型风险评分分布')

# 2. 攻击类型与HTTP方法的关系
attack_method = pd.crosstab(df['attack_type'], df['method'])
attack_method_pct = attack_method.div(attack_method.sum(axis=1), axis=0) * 100
attack_method_pct.plot(kind='barh', stacked=True, ax=axes[0, 1], colormap='Set2')
axes[0, 1].set_xlabel('百分比 (%)')
axes[0, 1].set_title('各攻击类型的HTTP方法分布')
axes[0, 1].legend(title='HTTP方法', bbox_to_anchor=(1.02, 1))

# 3. 攻击时间热力图
if 'timestamp' in df.columns:
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    heatmap_data = df.pivot_table(index='weekday', columns='hour', values='risk_score', aggfunc='count', fill_value=0)
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=axes[1, 0], cbar_kws={'label': '告警数量'})
    axes[1, 0].set_xlabel('小时')
    axes[1, 0].set_ylabel('星期')
    axes[1, 0].set_yticklabels(['周一', '周二', '周三', '周四', '周五', '周六', '周日'])
    axes[1, 0].set_title('告警时间热力图')

# 4. 攻击特征匹配统计
pattern_cols = [col for col in df.columns if col.startswith('pattern_')]
if pattern_cols:
    pattern_sums = df[pattern_cols].sum().sort_values(ascending=True)
    pattern_sums.index = [col.replace('pattern_', '') for col in pattern_sums.index]
    axes[1, 1].barh(pattern_sums.index, pattern_sums.values, color='teal')
    axes[1, 1].set_xlabel('匹配次数')
    axes[1, 1].set_title('攻击特征模式匹配统计')

plt.tight_layout()
plt.savefig('attack_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 5 =====
def generate_security_report(df):
    """
    生成结构化安全态势报告
    """
    report = []
    report.append('='*70)
    report.append('AI安全告警智能研判系统 - 安全态势分析报告')
    report.append('='*70)
    report.append(f'报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    report.append('')
    
    # 1. 概述
    report.append('一、总体概述')
    report.append('-'*50)
    report.append(f'  分析告警总数: {len(df):,} 条')
    report.append(f'  时间范围: {df["timestamp"].min()} 至 {df["timestamp"].max()}')
    report.append(f'  涉及源IP数: {df["src_ip"].nunique()} 个')
    report.append(f'  涉及目标IP数: {df["dst_ip"].nunique()} 个')
    report.append('')
    
    # 2. 风险统计
    report.append('二、风险统计')
    report.append('-'*50)
    risk_counts = df['risk_level'].value_counts()
    for level in ['严重', '高危', '中危', '低危', '信息']:
        count = risk_counts.get(level, 0)
        pct = count / len(df) * 100
        report.append(f'  {level}风险: {count:,} 条 ({pct:.1f}%)')
    report.append(f'  平均风险评分: {df["risk_score"].mean():.1f}')
    report.append(f'  最高风险评分: {df["risk_score"].max():.1f}')
    report.append('')
    
    # 3. 攻击类型分析
    report.append('三、攻击类型分析')
    report.append('-'*50)
    attack_counts = df['attack_type'].value_counts()
    for attack_type, count in attack_counts.items():
        pct = count / len(df) * 100
        report.append(f'  {attack_type}: {count:,} 条 ({pct:.1f}%)')
    report.append('')
    
    # 4. 高风险告警详情
    report.append('四、高风险告警详情 (Top 10)')
    report.append('-'*50)
    high_risk = df[df['risk_level'].isin(['严重', '高危'])].nlargest(10, 'risk_score')
    for i, (_, row) in enumerate(high_risk.iterrows(), 1):
        report.append(f'  {i}. 风险评分: {row["risk_score"]:.1f}')
        report.append(f'     攻击类型: {row["attack_type"]}')
        report.append(f'     源IP: {row["src_ip"]}')
        report.append(f'     目标IP: {row["dst_ip"]}')
        report.append(f'     时间: {row["timestamp"]}')
        report.append('')
    
    # 5. 攻击源分析
    report.append('五、主要攻击源 (Top 10)')
    report.append('-'*50)
    attack_df = df[df['attack_type'] != '正常访问']
    top_sources = attack_df['src_ip'].value_counts().head(10)
    for ip, count in top_sources.items():
        avg_risk = attack_df[attack_df['src_ip'] == ip]['risk_score'].mean()
        report.append(f'  {ip}: {count} 次攻击, 平均风险分 {avg_risk:.1f}')
    report.append('')
    
    # 6. 处置建议
    report.append('六、处置建议')
    report.append('-'*50)
    report.append('  1. 立即处置:')
    report.append(f'     - 阻断高风险源IP ({len(top_sources)} 个)')
    report.append('     - 检查受影响系统的完整性')
    report.append('     - 审计数据库和文件系统访问日志')
    report.append('')
    report.append('  2. 短期措施:')
    report.append('     - 更新WAF规则，增强攻击检测能力')
    report.append('     - 修复已识别的安全漏洞')
    report.append('     - 加强访问控制和身份认证')
    report.append('')
    report.append('  3. 长期建议:')
    report.append('     - 建立持续的安全监控机制')
    report.append('     - 定期进行安全评估和渗透测试')
    report.append('     - 加强安全意识培训')
    report.append('')
    
    report.append('='*70)
    report.append('报告结束')
    report.append('='*70)
    
    return '\n'.join(report)

# 生成报告
report_text = generate_security_report(df)
print(report_text)

# 保存报告
with open('security_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)
print('\n报告已保存: security_report.txt')

# ===== code cell 6 =====
# 生成CSV格式的详细报告
report_df = df[['timestamp', 'src_ip', 'dst_ip', 'method', 'url_path', 
                'attack_type', 'predicted_attack_type', 'risk_score', 'risk_level']].copy()
report_df = report_df.sort_values('risk_score', ascending=False)
report_df.to_csv('detailed_alert_report.csv', index=False, encoding='utf-8-sig')

print('详细告警报告已保存: detailed_alert_report.csv')
print(f'包含 {len(report_df)} 条告警记录')

# ===== code cell 7 =====
# 汇总统计
print('='*60)
print('可视化与报告生成完成')
print('='*60)
print('\n生成的文件:')
print('  - security_dashboard.png (安全态势大屏)')
print('  - attack_analysis.png (攻击分析详情)')
print('  - security_report.txt (文本格式报告)')
print('  - detailed_alert_report.csv (详细告警报告)')
