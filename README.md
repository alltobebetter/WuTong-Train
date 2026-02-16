# WuTong 梧桐 — AI 安全告警智能研判系统

基于集成学习的网络安全告警分类系统，可对原始告警日志自动识别 9 类攻击类型，最高准确率 **99.35%**。

## 性能

| 版本 | 准确率 | F1 | 方法 |
|------|--------|-----|------|
| V6 | 98.55% | 0.9856 | XGBoost + CatBoost + LightGBM Stacking |
| **V7** | **99.35%** | **0.9935** | Optuna 调参 + 交互特征 + 4 模型 Stacking |
| V8 | 99.23% | 0.9923 | V7 基础上增加特征选择 + 多种子集成 |

## 支持的攻击类型

| # | 类型 |
|---|------|
| 1 | SQL 注入 |
| 2 | XSS 跨站脚本 |
| 3 | CSRF |
| 4 | 目录遍历 |
| 5 | 文件包含 |
| 6 | 文件上传 |
| 7 | 远程命令执行 |
| 8 | Java 反序列化 |
| 9 | 正常访问 |

## 项目结构

```
wutong/          核心库（特征提取、降噪聚合、全局配置）
src_train/       训练模块（V6 / V7 / V8）
scripts/         工具脚本（数据预处理、训练入口、Notebook 生成）
jupyter/         Kaggle 一键训练 Notebook
data_raw/        原始告警数据集
legacy/          历史版本归档（V1–V5）
```

## 快速开始

### Kaggle 一键训练（推荐）

1. 上传 `jupyter/kaggle_train_v7.ipynb` 到 Kaggle
2. 开启 GPU（Tesla T4）
3. 点击 Run All，约 35 分钟完成

### 本地训练

```bash
pip install -r requirements.txt
python scripts/ingest.py           # 原始数据 → data/staging/
python scripts/augment_data.py     # 数据增强（可选）
python scripts/train_v7.py         # 训练最优模型 V7
```

模型产物输出到 `models/` 目录。

## 技术方案

- **特征工程**：49 维特征（25 基础 + 13 高级 + 11 交互），针对 CSRF / 正常访问混淆做了专项特征
- **集成学习**：XGBoost + CatBoost + LightGBM + ExtraTrees → StackingClassifier（GradientBoosting 作为 meta-learner）
- **超参搜索**：Optuna 贝叶斯优化，每模型 30 trials
- **数据均衡**：SMOTE-ENN 过采样 + 清洗
- **外部数据**：CSIC 2010 受控混入，仅补充少数类

## 依赖

```
pandas / numpy / scikit-learn / xgboost / catboost / lightgbm
imbalanced-learn / optuna / openpyxl / pyarrow
```

详见 [requirements.txt](requirements.txt)

## License

MIT
