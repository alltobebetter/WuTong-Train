# -*- coding: utf-8 -*-
# Source: 02_告警智能降噪模型.ipynb


# ===== code cell 1 =====
# 导入必要的库
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

print('库导入成功！')

# ===== code cell 2 =====
# 加载预处理后的数据
import os

possible_paths = [
    os.path.join(os.path.dirname(__file__), 'processed_data.pkl'),
    os.path.join(os.path.dirname(__file__), '..', '01_????', 'processed_data.pkl'),
]

df = None
for path in possible_paths:
    if os.path.exists(path):
        df = pd.read_pickle(path)
        print(f'成功加载数据: {path}')
        break

if df is None:
    # 如果没有预处理数据，从原始数据加载
    print('未找到预处理数据，请先运行 01_数据预处理与特征工程.ipynb')
    raise FileNotFoundError('请先运行数据预处理模块')

print(f'数据形状: {df.shape}')
print(f'攻击类型: {df["attack_type"].unique()}')

# ===== code cell 3 =====
class AlertAggregator:
    """
    告警聚合器
    基于TF-IDF和DBSCAN实现相似告警的自动聚合
    """
    
    def __init__(self, eps=0.3, min_samples=2):
        self.eps = eps  # DBSCAN邻域半径
        self.min_samples = min_samples  # 最小样本数
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.cluster_labels = None
        
    def _prepare_text(self, df):
        """准备用于聚类的文本特征"""
        texts = []
        for _, row in df.iterrows():
            text = f"{row['url_path']} {row['request_body']} {row['attack_type']}"
            texts.append(str(text))
        return texts
    
    def fit_transform(self, df):
        """执行告警聚合"""
        print('开始告警聚合...')
        
        # 准备文本
        texts = self._prepare_text(df)
        print(f'  - 处理 {len(texts)} 条告警')
        
        # TF-IDF向量化
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f'  - TF-IDF特征维度: {tfidf_matrix.shape}')
        
        # DBSCAN聚类
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='cosine')
        self.cluster_labels = clustering.fit_predict(tfidf_matrix)
        
        # 统计聚类结果
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        print(f'  - 聚类数量: {n_clusters}')
        print(f'  - 噪声点(独立告警): {n_noise}')
        
        return self.cluster_labels
    
    def get_aggregation_stats(self, df):
        """获取聚合统计信息"""
        if self.cluster_labels is None:
            raise ValueError('请先执行fit_transform')
        
        df_with_clusters = df.copy()
        df_with_clusters['cluster_id'] = self.cluster_labels
        
        # 统计每个聚类的信息
        cluster_stats = []
        for cluster_id in set(self.cluster_labels):
            if cluster_id == -1:
                continue
            cluster_data = df_with_clusters[df_with_clusters['cluster_id'] == cluster_id]
            cluster_stats.append({
                'cluster_id': cluster_id,
                'count': len(cluster_data),
                'attack_types': cluster_data['attack_type'].unique().tolist(),
                'src_ips': cluster_data['src_ip'].nunique(),
                'time_span_minutes': (cluster_data['timestamp'].max() - cluster_data['timestamp'].min()).total_seconds() / 60
            })
        
        return pd.DataFrame(cluster_stats)

print('AlertAggregator 类定义完成')

# ===== code cell 4 =====
# 执行告警聚合
aggregator = AlertAggregator(eps=0.4, min_samples=3)
cluster_labels = aggregator.fit_transform(df)

# 添加聚类标签到数据
df['cluster_id'] = cluster_labels

# 获取聚合统计
cluster_stats = aggregator.get_aggregation_stats(df)
print(f'\n聚合后告警组数量: {len(cluster_stats)}')
print(f'原始告警数量: {len(df)}')
print(f'降噪率: {(1 - len(cluster_stats)/len(df))*100:.1f}%')

# ===== code cell 5 =====
# 可视化聚合效果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 聚类大小分布
cluster_sizes = df['cluster_id'].value_counts()
cluster_sizes_filtered = cluster_sizes[cluster_sizes.index != -1]
axes[0].hist(cluster_sizes_filtered.values, bins=30, edgecolor='white', color='steelblue')
axes[0].set_xlabel('聚类大小（告警数量）')
axes[0].set_ylabel('聚类数量')
axes[0].set_title('告警聚类大小分布', fontsize=12)

# 降噪效果对比
labels = ['原始告警', '聚合后告警组']
values = [len(df), len(cluster_stats) + list(cluster_labels).count(-1)]
colors = ['#ff6b6b', '#4ecdc4']
bars = axes[1].bar(labels, values, color=colors, edgecolor='white', linewidth=2)
axes[1].set_ylabel('数量')
axes[1].set_title('告警降噪效果对比', fontsize=12)
for bar, val in zip(bars, values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                 f'{val:,}', ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('aggregation_effect.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 6 =====
# 准备训练数据
# 特征列
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

# 检查特征列是否存在
available_features = [col for col in feature_cols if col in df.columns]
print(f'可用特征数量: {len(available_features)}')

X = df[available_features].fillna(0)

# 创建二分类标签：正常访问 vs 攻击
y_binary = (df['attack_type'] != '正常访问').astype(int)
print(f'\n二分类标签分布:')
print(f'  - 正常访问: {(y_binary == 0).sum()}')
print(f'  - 攻击行为: {(y_binary == 1).sum()}')

# ===== code cell 7 =====
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f'训练集: {len(X_train)} 条')
print(f'测试集: {len(X_test)} 条')

# ===== code cell 8 =====
# 训练随机森林分类器
print('='*60)
print('训练随机森林分类器')
print('='*60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print('\n分类报告:')
print(classification_report(y_test, rf_pred, target_names=['正常访问', '攻击行为']))

rf_accuracy = accuracy_score(y_test, rf_pred)
print(f'准确率: {rf_accuracy*100:.2f}%')

# ===== code cell 9 =====
# 训练梯度提升分类器
print('='*60)
print('训练梯度提升分类器 (Gradient Boosting)')
print('='*60)

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=8,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

print('\n分类报告:')
print(classification_report(y_test, gb_pred, target_names=['正常访问', '攻击行为']))

gb_accuracy = accuracy_score(y_test, gb_pred)
print(f'准确率: {gb_accuracy*100:.2f}%')

# ===== code cell 10 =====
# 特征重要性分析
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance_rf': rf_model.feature_importances_,
    'importance_gb': gb_model.feature_importances_
}).sort_values('importance_rf', ascending=False)

# 可视化特征重要性
fig, axes = plt.subplots(1, 2, figsize=(14, 8))

top_n = 15
top_features = feature_importance.head(top_n)

axes[0].barh(range(top_n), top_features['importance_rf'].values, color='steelblue')
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_features['feature'].values)
axes[0].set_xlabel('重要性')
axes[0].set_title('随机森林 - 特征重要性 Top 15', fontsize=12)
axes[0].invert_yaxis()

top_features_gb = feature_importance.sort_values('importance_gb', ascending=False).head(top_n)
axes[1].barh(range(top_n), top_features_gb['importance_gb'].values, color='coral')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_features_gb['feature'].values)
axes[1].set_xlabel('重要性')
axes[1].set_title('梯度提升 - 特征重要性 Top 15', fontsize=12)
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 11 =====
# 混淆矩阵可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, pred, title in [(axes[0], rf_pred, '随机森林'), (axes[1], gb_pred, '梯度提升')]:
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['正常访问', '攻击行为'],
                yticklabels=['正常访问', '攻击行为'])
    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')
    ax.set_title(f'{title} - 混淆矩阵', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# ===== code cell 12 =====
# 综合降噪效果评估
print('='*60)
print('告警智能降噪效果评估报告')
print('='*60)

# 1. 聚合降噪
original_count = len(df)
unique_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
noise_points = list(cluster_labels).count(-1)
aggregated_count = unique_clusters + noise_points
aggregation_reduction = (1 - aggregated_count / original_count) * 100

print(f'\n1. 告警聚合降噪')
print(f'   原始告警数量: {original_count:,}')
print(f'   聚合后数量: {aggregated_count:,}')
print(f'   降噪率: {aggregation_reduction:.1f}%')

# 2. 误报过滤
normal_count = (df['attack_type'] == '正常访问').sum()
attack_count = (df['attack_type'] != '正常访问').sum()
false_positive_rate = normal_count / original_count * 100

print(f'\n2. 误报过滤效果')
print(f'   正常访问（潜在误报）: {normal_count:,} ({false_positive_rate:.1f}%)')
print(f'   真实攻击: {attack_count:,} ({100-false_positive_rate:.1f}%)')
print(f'   误报识别准确率: {rf_accuracy*100:.2f}%')

# 3. 综合效果
final_alerts = attack_count  # 过滤误报后的告警
total_reduction = (1 - final_alerts / original_count) * 100

print(f'\n3. 综合降噪效果')
print(f'   最终需处理告警: {final_alerts:,}')
print(f'   总体降噪率: {total_reduction:.1f}%')
print(f'   人工工作量减少: {total_reduction:.1f}%')

# ===== code cell 13 =====
# 保存模型和结果
import pickle

# 保存模型
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
with open('gb_model.pkl', 'wb') as f:
    pickle.dump(gb_model, f)

# 保存特征重要性
feature_importance.to_csv('feature_importance.csv', index=False)

# 保存处理后的数据
df.to_pickle('data_with_clusters.pkl')

print('\n模型和结果已保存:')
print('  - rf_model.pkl (随机森林模型)')
print('  - gb_model.pkl (梯度提升模型)')
print('  - feature_importance.csv')
print('  - data_with_clusters.pkl')
