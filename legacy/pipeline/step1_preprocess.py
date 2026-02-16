# -*- coding: utf-8 -*-
# Source: 01_数据预处理与特征工程.ipynb


# ===== code cell 1 =====
# 导入必要的库
import pandas as pd
import numpy as np
import re
import warnings
from urllib.parse import unquote, urlparse, parse_qs
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 100)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print('库导入成功！')
print(f'Pandas版本: {pd.__version__}')
print(f'NumPy版本: {np.__version__}')

# ===== code cell 2 =====
# 加载官方提供的原始告警信息数据集
# 注意：请确保数据文件在正确的路径下
import os

# 尝试多个可能的路径
possible_paths = [
    os.path.join(os.path.dirname(__file__), 'data', '???????????V1.2.xlsx'),
    os.path.join(os.path.dirname(__file__), '..', '04_???', '???????????V1.2.xlsx'),
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_excel(path)
        print(f'成功加载数据集: {path}')
        break

if df is None:
    raise FileNotFoundError('请将数据文件放置在正确的路径下')

# 重命名攻击类型列，简化后续操作
df = df.rename(columns={'attack_type(结果列，需参赛选手判断输出)': 'attack_type'})

print(f'\n数据集形状: {df.shape}')
print(f'总记录数: {len(df):,} 条')
print(f'\n列名: {list(df.columns)}')

# ===== code cell 3 =====
# 查看数据基本信息
print('='*60)
print('数据集基本信息')
print('='*60)
df.info()

print('\n' + '='*60)
print('数据集前5行')
print('='*60)
df.head()

# ===== code cell 4 =====
# 攻击类型分布统计
print('='*60)
print('攻击类型分布统计')
print('='*60)
attack_dist = df['attack_type'].value_counts()
print(attack_dist)

print(f'\n共 {len(attack_dist)} 种类型（含正常访问）')
print(f'正常访问占比: {attack_dist["正常访问"]/len(df)*100:.1f}%')
print(f'攻击记录占比: {(len(df)-attack_dist["正常访问"])/len(df)*100:.1f}%')

# ===== code cell 5 =====
# 检查缺失值
print('='*60)
print('缺失值统计')
print('='*60)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'缺失数量': missing, '缺失比例(%)': missing_pct})
print(missing_df)

# 填充request_body的缺失值
df['request_body'] = df['request_body'].fillna('')
print(f'\n已将request_body的缺失值填充为空字符串')

# ===== code cell 6 =====
# 检查重复记录
duplicates = df.duplicated().sum()
print(f'重复记录数: {duplicates}')

if duplicates > 0:
    df = df.drop_duplicates()
    print(f'已删除重复记录，剩余: {len(df)} 条')

# ===== code cell 7 =====
# 定义攻击特征检测模式
ATTACK_PATTERNS = {
    # SQL注入特征
    'sql_injection': [
        r'(?i)(union\s+select|select\s+.*\s+from|insert\s+into|delete\s+from|drop\s+table)',
        r'(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1|\'\s*or\s*\'|\'\s*=\s*\')',
        r'(?i)(exec\s*\(|execute\s*\(|sp_executesql)',
        r'(?i)(\%27|\%22|\-\-|\%23|#)',
        r'(?i)(benchmark\s*\(|sleep\s*\(|waitfor\s+delay)',
    ],
    # XSS跨站脚本特征
    'xss': [
        r'(?i)(<script|</script|javascript:|onerror\s*=|onload\s*=)',
        r'(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()',
        r'(?i)(document\.cookie|document\.location|document\.write)',
        r'(?i)(<img[^>]+onerror|<svg[^>]+onload)',
        r'(?i)(\%3cscript|\%3c/script)',
    ],
    # 目录遍历特征
    'path_traversal': [
        r'(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.\.%2f)',
        r'(?i)(/etc/passwd|/etc/shadow|/windows/system32)',
        r'(?i)(\.ssh/|id_rsa|authorized_keys)',
        r'(?i)(boot\.ini|win\.ini|system\.ini)',
    ],
    # 命令执行特征
    'command_injection': [
        r'(;|\||\$\(|`|&&|\|\|)',
        r'(?i)(ping\s+-|wget\s+|curl\s+|nc\s+-|bash\s+-)',
        r'(?i)(/bin/sh|/bin/bash|cmd\.exe|powershell)',
        r'(?i)(whoami|id\s|uname|cat\s+/etc)',
    ],
    # 文件包含特征
    'file_inclusion': [
        r'(?i)(include\s*=|require\s*=|file\s*=)',
        r'(?i)(php://|data://|expect://|zip://)',
        r'(?i)(\?file=|\?page=|\?path=|\?include=)',
    ],
    # 文件上传特征
    'file_upload': [
        r'(?i)(\.php|\.jsp|\.asp|\.aspx|\.exe|\.sh)$',
        r'(?i)(multipart/form-data)',
        r'(?i)(filename=.*\.(php|jsp|asp|exe|sh))',
    ],
    # Java反序列化特征
    'java_deserialization': [
        r'(rO0AB|aced0005)',
        r'(?i)(java\.lang\.Runtime|java\.lang\.ProcessBuilder)',
        r'(?i)(ObjectInputStream|readObject)',
    ],
    # CSRF特征
    'csrf': [
        r'(?i)(transfer|withdraw|delete|update).*amount',
        r'(?i)(action=|method=post).*token',
    ]
}

print('攻击特征模式定义完成')
print(f'共定义 {len(ATTACK_PATTERNS)} 类攻击特征')

# ===== code cell 8 =====
def decode_url(text):
    """URL解码，处理多层编码"""
    if pd.isna(text) or text == '':
        return ''
    try:
        decoded = str(text)
        for _ in range(3):  # 最多解码3层
            new_decoded = unquote(decoded)
            if new_decoded == decoded:
                break
            decoded = new_decoded
        return decoded.lower()
    except:
        return str(text).lower()

def count_pattern_matches(text, patterns):
    """统计文本中匹配攻击模式的数量"""
    if not text:
        return 0
    count = 0
    for pattern in patterns:
        try:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        except:
            pass
    return count

def extract_features(row):
    """从单条记录中提取特征"""
    features = {}
    
    # 解码URL和请求体
    url_decoded = decode_url(row['url_path'])
    body_decoded = decode_url(row['request_body'])
    combined_text = url_decoded + ' ' + body_decoded
    
    # 1. 基础特征
    features['url_length'] = len(str(row['url_path']))
    features['body_length'] = len(str(row['request_body'])) if row['request_body'] else 0
    features['has_body'] = 1 if row['request_body'] else 0
    
    # 2. HTTP方法特征
    features['method_get'] = 1 if row['method'] == 'GET' else 0
    features['method_post'] = 1 if row['method'] == 'POST' else 0
    features['method_put'] = 1 if row['method'] == 'PUT' else 0
    features['method_delete'] = 1 if row['method'] == 'DELETE' else 0
    
    # 3. URL特征
    features['url_param_count'] = url_decoded.count('=')  # 参数数量
    features['url_depth'] = url_decoded.count('/')  # URL深度
    features['url_special_chars'] = len(re.findall(r'[<>"\';(){}\[\]]', url_decoded))
    
    # 4. 攻击模式匹配特征
    for attack_type, patterns in ATTACK_PATTERNS.items():
        features[f'pattern_{attack_type}'] = count_pattern_matches(combined_text, patterns)
    
    # 5. 编码特征
    features['url_encoding_count'] = str(row['url_path']).count('%')
    features['double_encoding'] = 1 if '%25' in str(row['url_path']) else 0
    
    # 6. 敏感关键词特征
    sensitive_keywords = ['admin', 'root', 'password', 'passwd', 'login', 'shell', 'cmd', 'exec']
    features['sensitive_keyword_count'] = sum(1 for kw in sensitive_keywords if kw in combined_text)
    
    # 7. User-Agent特征
    ua = str(row['user_agent']).lower()
    features['ua_is_bot'] = 1 if any(bot in ua for bot in ['bot', 'spider', 'crawler', 'curl', 'wget']) else 0
    features['ua_is_mobile'] = 1 if any(m in ua for m in ['mobile', 'android', 'iphone']) else 0
    
    # 8. 时间特征
    if pd.notna(row['timestamp']):
        features['hour'] = row['timestamp'].hour
        features['is_night'] = 1 if row['timestamp'].hour < 6 or row['timestamp'].hour > 22 else 0
        features['is_weekend'] = 1 if row['timestamp'].weekday() >= 5 else 0
    else:
        features['hour'] = 0
        features['is_night'] = 0
        features['is_weekend'] = 0
    
    return features

print('特征提取函数定义完成')

# ===== code cell 9 =====
import sys

# ===== code cell 10 =====
# 批量提取特征
print('开始提取特征...')
print(f'处理 {len(df):,} 条记录')

from tqdm.notebook import tqdm
try:
    features_list = [extract_features(row) for _, row in tqdm(df.iterrows(), total=len(df), desc='特征提取')]
except:
    # 如果tqdm不可用，使用普通循环
    features_list = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 1000 == 0:
            print(f'处理进度: {i}/{len(df)}')
        features_list.append(extract_features(row))

# 转换为DataFrame
features_df = pd.DataFrame(features_list)
print(f'\n特征提取完成！')
print(f'提取特征数量: {len(features_df.columns)} 个')
print(f'特征列表: {list(features_df.columns)}')

# ===== code cell 11 =====
# 合并原始数据和特征
df_processed = pd.concat([df.reset_index(drop=True), features_df], axis=1)
print(f'合并后数据形状: {df_processed.shape}')
df_processed.head()

# ===== code cell 12 =====
# 攻击类型分布可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 饼图
attack_counts = df['attack_type'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(attack_counts)))
axes[0].pie(attack_counts.values, labels=attack_counts.index, autopct='%1.1f%%', colors=colors)
axes[0].set_title('攻击类型分布', fontsize=14)

# 柱状图
bars = axes[1].barh(attack_counts.index, attack_counts.values, color=colors)
axes[1].set_xlabel('数量')
axes[1].set_title('攻击类型数量统计', fontsize=14)
for i, v in enumerate(attack_counts.values):
    axes[1].text(v + 50, i, str(v), va='center')

plt.tight_layout()
plt.savefig('attack_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('图表已保存: attack_distribution.png')

# ===== code cell 13 =====
# 时间分布分析
df_processed['hour'] = df_processed['timestamp'].dt.hour
df_processed['date'] = df_processed['timestamp'].dt.date

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# 24小时分布
hourly = df_processed.groupby('hour').size()
axes[0].bar(hourly.index, hourly.values, color='steelblue', edgecolor='white')
axes[0].set_xlabel('小时')
axes[0].set_ylabel('告警数量')
axes[0].set_title('24小时告警分布', fontsize=14)
axes[0].set_xticks(range(0, 24, 2))

# 每日趋势
daily = df_processed.groupby('date').size()
axes[1].plot(range(len(daily)), daily.values, marker='o', linewidth=2, markersize=4)
axes[1].fill_between(range(len(daily)), daily.values, alpha=0.3)
axes[1].set_xlabel('日期序号')
axes[1].set_ylabel('告警数量')
axes[1].set_title('每日告警趋势', fontsize=14)

plt.tight_layout()
plt.savefig('attack_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 14 =====
# 特征相关性分析
pattern_cols = [col for col in features_df.columns if col.startswith('pattern_')]

# 按攻击类型统计各模式匹配情况
pattern_by_attack = df_processed.groupby('attack_type')[pattern_cols].mean()

plt.figure(figsize=(12, 8))
sns.heatmap(pattern_by_attack, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
plt.title('攻击类型与特征模式匹配热力图', fontsize=14)
plt.xlabel('特征模式')
plt.ylabel('攻击类型')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('attack_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 15 =====
# 保存预处理后的数据
df_processed.to_pickle('processed_data.pkl')
df_processed.to_csv('processed_data.csv', index=False, encoding='utf-8-sig')

print('='*60)
print('数据预处理完成！')
print('='*60)
print(f'原始数据: {len(df):,} 条')
print(f'处理后数据: {len(df_processed):,} 条')
print(f'特征数量: {len(features_df.columns)} 个')
print(f'\n保存文件:')
print('  - processed_data.pkl (用于模型训练)')
print('  - processed_data.csv (可查看)')
print('  - attack_distribution.png')
print('  - time_distribution.png')
print('  - feature_heatmap.png')
