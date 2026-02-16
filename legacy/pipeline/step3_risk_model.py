# -*- coding: utf-8 -*-
# Source: 03_风险精准研判模型.ipynb


# ===== code cell 1 =====
# 导入必要的库
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print('库导入成功！')

# ===== code cell 2 =====
# 加载数据
import os

possible_paths = [
    os.path.join(os.path.dirname(__file__), 'data_with_clusters.pkl'),
    os.path.join(os.path.dirname(__file__), '..', '01_????', 'data_with_clusters.pkl'),
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f'成功加载数据: {path}')
        break

if df is None:
    raise FileNotFoundError('请先运行前置模块')

print(f'数据形状: {df.shape}')

# ===== code cell 3 =====
# 准备多分类数据
feature_cols = [
    'url_length', 'body_length', 'has_body',
    'method_get', 'method_post', 'method_put', 'method_delete',
    'url_param_count', 'url_depth', 'url_special_chars',
    'pattern_sql_injection', 'pattern_xss', 'pattern_path_traversal',
    'pattern_command_injection', 'pattern_file_inclusion',
    'pattern_file_upload', 'pattern_java_deserialization', 'pattern_csrf',
    'url_encoding_count', 'double_encoding', 'sensitive_keyword_count',
    'ua_is_bot', 'ua_is_mobile', 'is_night', 'is_weekend'
]

available_features = [col for col in feature_cols if col in df.columns]
X = df[available_features].fillna(0)

# 标签编码
le = LabelEncoder()
y = le.fit_transform(df['attack_type'])

print(f'特征数量: {len(available_features)}')
print(f'类别数量: {len(le.classes_)}')
print(f'类别列表: {list(le.classes_)}')

# ===== code cell 4 =====
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'训练集: {len(X_train)} 条')
print(f'测试集: {len(X_test)} 条')

# ===== code cell 5 =====
# 训练随机森林多分类模型
print('='*60)
print('训练随机森林多分类模型')
print('='*60)

rf_multi = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_multi.fit(X_train, y_train)
rf_pred = rf_multi.predict(X_test)

print('\n分类报告:')
print(classification_report(y_test, rf_pred, target_names=le.classes_))

rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'\n总体准确率: {rf_accuracy*100:.2f}%')

# ===== code cell 6 =====
# 训练梯度提升多分类模型
print('='*60)
print('训练梯度提升多分类模型')
print('='*60)

gb_multi = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)

gb_multi.fit(X_train, y_train)
gb_pred = gb_multi.predict(X_test)

print('\n分类报告:')
print(classification_report(y_test, gb_pred, target_names=le.classes_))

gb_accuracy = accuracy_score(y_test, gb_pred)
print(f'\n总体准确率: {gb_accuracy*100:.2f}%')

# ===== code cell 7 =====
# 集成模型 - 投票分类器
print('='*60)
print('训练集成模型 (Voting Classifier)')
print('='*60)

ensemble_model = VotingClassifier(
    estimators=[
        ('rf', rf_multi),
        ('gb', gb_multi)
    ],
    voting='soft'
)

ensemble_model.fit(X_train, y_train)
ensemble_pred = ensemble_model.predict(X_test)

print('\n分类报告:')
print(classification_report(y_test, ensemble_pred, target_names=le.classes_))

ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f'\n总体准确率: {ensemble_accuracy*100:.2f}%')

# ===== code cell 8 =====
# 混淆矩阵可视化
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, ensemble_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('攻击类型分类 - 混淆矩阵', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('multiclass_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 9 =====
# 定义攻击类型的基础风险等级
ATTACK_SEVERITY = {
    '正常访问': 0,
    'CSRF攻击': 40,
    'XSS跨站脚本攻击': 50,
    '目录遍历攻击': 55,
    '文件包含攻击': 65,
    '文件上传攻击': 70,
    'SQL注入攻击': 75,
    '远程命令执行攻击': 90,
    'Java反序列化漏洞利用攻击': 95
}

def calculate_risk_score(row, predicted_attack_type):
    """
    计算综合风险评分 (0-100)
    
    评分维度:
    1. 攻击类型基础分 (40%)
    2. 攻击特征匹配度 (25%)
    3. 时间因素 (10%)
    4. 目标敏感度 (15%)
    5. 攻击复杂度 (10%)
    """
    score = 0
    
    # 1. 攻击类型基础分 (40%)
    base_score = ATTACK_SEVERITY.get(predicted_attack_type, 50)
    score += base_score * 0.4
    
    # 2. 攻击特征匹配度 (25%)
    pattern_cols = [col for col in row.index if col.startswith('pattern_')]
    if pattern_cols:
        pattern_score = min(sum(row[col] for col in pattern_cols) * 10, 100)
        score += pattern_score * 0.25
    
    # 3. 时间因素 (10%) - 夜间和周末风险更高
    time_score = 0
    if 'is_night' in row and row['is_night']:
        time_score += 50
    if 'is_weekend' in row and row['is_weekend']:
        time_score += 30
    score += time_score * 0.1
    
    # 4. 目标敏感度 (15%)
    sensitivity_score = 0
    if 'sensitive_keyword_count' in row:
        sensitivity_score = min(row['sensitive_keyword_count'] * 20, 100)
    score += sensitivity_score * 0.15
    
    # 5. 攻击复杂度 (10%)
    complexity_score = 0
    if 'url_encoding_count' in row:
        complexity_score += min(row['url_encoding_count'] * 5, 50)
    if 'double_encoding' in row and row['double_encoding']:
        complexity_score += 30
    if 'url_length' in row and row['url_length'] > 500:
        complexity_score += 20
    score += min(complexity_score, 100) * 0.1
    
    return min(round(score, 1), 100)

print('风险评分函数定义完成')

# ===== code cell 10 =====
# 对所有数据计算风险评分
print('计算风险评分...')

# 使用模型预测攻击类型
X_all = df[available_features].fillna(0)
predicted_labels = ensemble_model.predict(X_all)
predicted_types = le.inverse_transform(predicted_labels)

# 计算风险评分
risk_scores = []
for i, (_, row) in enumerate(df.iterrows()):
    score = calculate_risk_score(row, predicted_types[i])
    risk_scores.append(score)

df['predicted_attack_type'] = predicted_types
df['risk_score'] = risk_scores

# 定义风险等级
def get_risk_level(score):
    if score >= 80:
        return '严重'
    elif score >= 60:
        return '高危'
    elif score >= 40:
        return '中危'
    elif score >= 20:
        return '低危'
    else:
        return '信息'

df['risk_level'] = df['risk_score'].apply(get_risk_level)

print('\n风险等级分布:')
print(df['risk_level'].value_counts())

# ===== code cell 11 =====
print(f'df类型: {type(df)}')
print(f'df列名: {list(df.columns)}')
print(f'是否有risk_score列: {"risk_score" in df.columns}')

# ===== code cell 12 =====
import matplotlib.pyplot as plt
import numpy as np

# 风险评分可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 确保axes是二维数组
if axes is None:
    print("创建子图失败，请重试")
else:
    # 1. 风险评分分布
    axes[0, 0].hist(df['risk_score'].dropna(), bins=50, edgecolor='white', color='steelblue')
    axes[0, 0].set_xlabel('风险评分')
    axes[0, 0].set_ylabel('数量')
    axes[0, 0].set_title('风险评分分布', fontsize=12)
    axes[0, 0].axvline(x=60, color='orange', linestyle='--', label='高危阈值')
    axes[0, 0].axvline(x=80, color='red', linestyle='--', label='严重阈值')
    axes[0, 0].legend()

    # 2. 风险等级分布
    risk_counts = df['risk_level'].value_counts()
    colors = {'严重': '#d62728', '高危': '#ff7f0e', '中危': '#ffbb78', '低危': '#98df8a', '信息': '#aec7e8'}
    risk_order = ['严重', '高危', '中危', '低危', '信息']
    risk_counts = risk_counts.reindex(risk_order).dropna()
    axes[0, 1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                   colors=[colors.get(r, 'gray') for r in risk_counts.index])
    axes[0, 1].set_title('风险等级分布', fontsize=12)

    # 3. 各攻击类型的平均风险评分
    attack_risk = df.groupby('attack_type')['risk_score'].mean().sort_values(ascending=True)
    axes[1, 0].barh(attack_risk.index, attack_risk.values, color='coral')
    axes[1, 0].set_xlabel('平均风险评分')
    axes[1, 0].set_title('各攻击类型平均风险评分', fontsize=12)

    # 4. 风险评分箱线图
    attack_types = df['attack_type'].unique()
    data_for_box = [df[df['attack_type'] == at]['risk_score'].dropna().values for at in attack_types]
    bp = axes[1, 1].boxplot(data_for_box, labels=attack_types, vert=True, patch_artist=True)
    axes[1, 1].set_xticklabels(attack_types, rotation=45, ha='right')
    axes[1, 1].set_ylabel('风险评分')
    axes[1, 1].set_title('各攻击类型风险评分分布', fontsize=12)

    plt.tight_layout()
    plt.savefig('risk_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# ===== code cell 13 =====
# 定义处置建议模板
RESPONSE_SUGGESTIONS = {
    'SQL注入攻击': {
        'immediate': ['立即阻断源IP访问', '检查数据库是否被非法访问', '审计数据库操作日志'],
        'short_term': ['修复SQL注入漏洞', '实施参数化查询', '部署WAF规则'],
        'long_term': ['代码安全审计', '安全开发培训', '定期渗透测试']
    },
    'XSS跨站脚本攻击': {
        'immediate': ['阻断恶意请求', '检查是否有用户数据泄露'],
        'short_term': ['实施输入输出过滤', '部署CSP策略', '更新WAF规则'],
        'long_term': ['前端安全加固', '安全编码规范培训']
    },
    '远程命令执行攻击': {
        'immediate': ['立即隔离受影响主机', '阻断源IP', '检查系统完整性'],
        'short_term': ['修复命令执行漏洞', '加固系统权限', '部署入侵检测'],
        'long_term': ['系统安全加固', '最小权限原则实施', '定期漏洞扫描']
    },
    '目录遍历攻击': {
        'immediate': ['阻断恶意请求', '检查敏感文件是否被访问'],
        'short_term': ['修复路径遍历漏洞', '实施路径白名单'],
        'long_term': ['文件访问权限审计', '安全配置加固']
    },
    '文件包含攻击': {
        'immediate': ['阻断恶意请求', '检查是否有恶意代码执行'],
        'short_term': ['禁用危险函数', '实施文件白名单'],
        'long_term': ['代码安全审计', 'PHP配置加固']
    },
    '文件上传攻击': {
        'immediate': ['删除可疑上传文件', '阻断源IP'],
        'short_term': ['加强文件类型验证', '限制上传目录权限'],
        'long_term': ['实施文件沙箱', '部署文件安全扫描']
    },
    'Java反序列化漏洞利用攻击': {
        'immediate': ['立即隔离受影响系统', '检查是否有后门植入'],
        'short_term': ['更新Java组件', '禁用危险类'],
        'long_term': ['组件安全管理', '定期安全更新']
    },
    'CSRF攻击': {
        'immediate': ['检查是否有未授权操作'],
        'short_term': ['实施CSRF Token', '验证Referer'],
        'long_term': ['安全开发规范', 'SameSite Cookie策略']
    },
    '正常访问': {
        'immediate': ['无需处置'],
        'short_term': ['持续监控'],
        'long_term': ['优化告警规则']
    }
}

def generate_response_suggestion(attack_type, risk_level):
    """生成处置建议"""
    suggestions = RESPONSE_SUGGESTIONS.get(attack_type, RESPONSE_SUGGESTIONS['正常访问'])
    
    result = {
        'attack_type': attack_type,
        'risk_level': risk_level,
        'immediate_actions': suggestions['immediate'],
        'short_term_actions': suggestions['short_term'],
        'long_term_actions': suggestions['long_term']
    }
    
    return result

print('处置建议生成函数定义完成')

# ===== code cell 14 =====
# 生成高风险告警的处置建议
high_risk_alerts = df[df['risk_level'].isin(['严重', '高危'])].head(10)

print('='*60)
print('高风险告警处置建议示例')
print('='*60)

for i, (_, row) in enumerate(high_risk_alerts.iterrows()):
    suggestion = generate_response_suggestion(row['predicted_attack_type'], row['risk_level'])
    
    print(f'\n【告警 {i+1}】')
    print(f'  攻击类型: {suggestion["attack_type"]}')
    print(f'  风险等级: {suggestion["risk_level"]}')
    print(f'  风险评分: {row["risk_score"]}')
    print(f'  源IP: {row["src_ip"]}')
    print(f'  立即处置: {", ".join(suggestion["immediate_actions"])}')
    print(f'  短期措施: {", ".join(suggestion["short_term_actions"])}')

# ===== code cell 15 =====
# 保存模型和结果
import pickle

# 保存集成模型
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble_model, f)

# 保存标签编码器
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# 保存完整结果
df.to_pickle('final_results.pkl')
df.to_csv('final_results.csv', index=False, encoding='utf-8-sig')

# 生成风险报告摘要
report_summary = {
    '总告警数': len(df),
    '严重风险': len(df[df['risk_level'] == '严重']),
    '高危风险': len(df[df['risk_level'] == '高危']),
    '中危风险': len(df[df['risk_level'] == '中危']),
    '低危风险': len(df[df['risk_level'] == '低危']),
    '信息级别': len(df[df['risk_level'] == '信息']),
    '平均风险评分': round(df['risk_score'].mean(), 2),
    '最高风险评分': df['risk_score'].max(),
    '模型准确率': f'{ensemble_accuracy*100:.2f}%'
}

print('\n' + '='*60)
print('风险研判报告摘要')
print('='*60)
for key, value in report_summary.items():
    print(f'{key}: {value}')

print('\n模型和结果已保存')
