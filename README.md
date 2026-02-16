# WuTong 梧桐 — AI 安全告警智能研判系统

基于集成学习的网络安全告警分类系统，支持 9 类攻击类型识别。

## 性能

| 版本 | 准确率 | F1 | 方法 |
|------|--------|-----|------|
| V6 | 98.55% | 0.9856 | XGB+Cat+LGB Stacking |
| V7 | **99.35%** | 0.9935 | Optuna + 交互特征 + 4 模型 Stacking |
| V8 | 99.23% | 0.9923 | V7 + 特征选择 + 多种子集成 |

当前最优：**V7 — 99.35%**

## 攻击类型（9 类）

SQL注入、XSS跨站脚本、CSRF、目录遍历、文件包含、文件上传、远程命令执行、Java反序列化、正常访问

## 项目结构

```
wutong/          核心库（特征提取、降噪聚合、配置）
src_train/       训练模块（V6/V7/V8）
scripts/         工具脚本（数据处理、训练入口、Notebook 生成）
jupyter/         Kaggle 一键训练 Notebook
docs/            文档与训练日志
data_raw/        原始数据集
legacy/          历史版本归档
```

## 快速开始

### Kaggle 一键训练（推荐）

1. 上传 `jupyter/kaggle_train_v7.ipynb` 到 Kaggle
2. 开启 GPU（Tesla T4）
3. Run All，约 35 分钟完成

### 本地训练

```bash
pip install -r requirements.txt
python scripts/ingest.py                # 数据预处理 → data/staging/
python scripts/augment_data.py          # 数据增强
python scripts/train_v7.py              # 训练 V7（最优版本）
```

训练产物保存在 `models/` 目录下。

## 技术栈

- XGBoost / CatBoost / LightGBM / ExtraTrees — 4 模型 Stacking 集成
- Optuna — 贝叶斯超参搜索（30 trials/模型）
- SMOTE-ENN — 过采样 + 清洗
- scikit-learn — StackingClassifier + GradientBoosting meta-learner
- 49 维特征（25 基础 + 13 高级 + 11 交互特征）

## 许可

MIT
