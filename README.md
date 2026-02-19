# WuTong 梧桐 — AI 安全告警智能研判系统

基于集成学习的网络安全告警分类系统，可对原始告警日志自动识别 9 类攻击类型。

## 性能

| 版本 | 准确率 | F1 | 方法 |
|------|--------|-----|------|
| V6 | 98.55% | 0.9856 | XGBoost + CatBoost + LightGBM Stacking |
| V7 | 99.35% | 0.9935 | Optuna 调参 + 交互特征 + 4 模型 Stacking |
| V8 | 99.23% | 0.9923 | V7 + 特征选择 + 多种子集成（过度工程，性能下降） |
| V9 | 99.37% | 0.9937 | V7 基础 + 混淆对专家 + 置信度路由 + 对抗验证 |
| **V10** | **99.37%** | **0.9937** | V9 bug 修复 + LGB meta-learner + 自适应路由 |

> V10 最终选择 LightGBM 单模型作为最优模型，证明在当前数据集上单模型已达到集成的天花板。

### V10 各模型详细表现

| 模型 | 测试准确率 | CV 准确率 |
|------|-----------|----------|
| LightGBM | 99.37% | 99.93% |
| XGBoost | 99.28% | 99.91% |
| CatBoost | 99.22% | 99.88% |
| ExtraTrees | 99.03% | 99.82% |
| Stacking 集成 | 99.32% | — |
| Voting 集成 | 99.34% | — |
| 校准 Voting | 99.35% | — |

### 各类别表现（V10）

| 攻击类型 | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| CSRF 攻击 | 98.78% | 100.00% | 99.38% |
| Java 反序列化漏洞利用 | 100.00% | 100.00% | 100.00% |
| SQL 注入攻击 | 99.26% | 99.02% | 99.14% |
| XSS 跨站脚本攻击 | 100.00% | 98.97% | 99.48% |
| 文件上传攻击 | 100.00% | 100.00% | 100.00% |
| 文件包含攻击 | 98.95% | 97.93% | 98.44% |
| 正常访问 | 98.90% | 98.90% | 98.90% |
| 目录遍历攻击 | 99.01% | 100.00% | 99.50% |
| 远程命令执行攻击 | 99.52% | 100.00% | 99.76% |

## 支持的攻击类型

| # | 类型 | 说明 |
|---|------|------|
| 1 | SQL 注入 | UNION 注入、盲注、报错注入等 |
| 2 | XSS 跨站脚本 | 反射型、存储型、DOM 型 XSS |
| 3 | CSRF | 跨站请求伪造 |
| 4 | 目录遍历 | `../` 路径穿越攻击 |
| 5 | 文件包含 | 本地/远程文件包含（LFI/RFI） |
| 6 | 文件上传 | 恶意文件上传攻击 |
| 7 | 远程命令执行 | OS 命令注入、RCE |
| 8 | Java 反序列化 | Java 反序列化漏洞利用 |
| 9 | 正常访问 | 非攻击的正常 HTTP 请求 |

## 项目结构

```
WuTong-Train/
├── wutong/                  核心库
│   ├── config.py            全局配置（特征列、目录、训练参数）
│   ├── features.py          基础特征提取（25 维）
│   ├── denoise.py           告警降噪聚合器
│   └── __init__.py
├── src_train/               训练模块
│   ├── train_models_v6.py   V6: 基础 3 模型 Stacking
│   ├── train_models_v7.py   V7: +Optuna +交互特征 +4 模型
│   ├── train_models_v8.py   V8: +特征选择 +多种子（失败实验）
│   ├── train_models_v9.py   V9: +混淆对专家 +置信度路由 +对抗验证
│   ├── train_models_v10.py  V10: bug 修复 +LGB meta +自适应路由
│   └── __init__.py
├── scripts/                 工具脚本
│   ├── ingest.py            原始数据预处理 → parquet
│   ├── augment_data.py      数据增强（同义替换、变异）
│   ├── integrate_csic2010.py  CSIC 2010 外部数据集集成
│   ├── train_v6~v10.py      各版本训练入口脚本
│   └── generate_kaggle_notebook_v7~v10.py  Kaggle notebook 生成器
├── jupyter/                 Kaggle 一键训练 Notebook
│   ├── kaggle_train_v7.ipynb
│   ├── kaggle_train_v8.ipynb
│   ├── kaggle_train_v9.ipynb
│   └── kaggle_train_v10.ipynb
├── data_raw/                原始告警数据集（随仓库分发）
│   └── 原始告警信息样例数据集V1.2.xlsx
├── legacy/                  历史版本归档（V1–V5）
├── requirements.txt
└── README.md
```

## 快速开始

### Kaggle 一键训练（推荐）

1. 上传 `jupyter/kaggle_train_v10.ipynb` 到 Kaggle Notebook
2. 开启 GPU 加速器（Tesla T4）
3. 点击 **Run All**，约 2.5 小时完成全部训练（含 Optuna 超参搜索）
4. 训练完成后在右侧 Output 面板下载模型 zip 包

### 本地训练

```bash
# 安装依赖
pip install -r requirements.txt
pip install imbalanced-learn optuna

# 数据预处理
python scripts/ingest.py "data_raw/原始告警信息样例数据集V1.2.xlsx"

# 数据增强（可选，推荐）
python scripts/augment_data.py --target-size 30000 --ratio 2.5

# 外部数据集成（可选，需联网下载 CSIC 2010）
python scripts/integrate_csic2010.py

# 训练 V10 模型
python scripts/train_v10.py --version v10.0.0 --cv-splits 10 --optuna-trials 30
```

模型产物输出到 `models/v10.0.0/` 目录，包含：
- `ensemble.pkl` — 最终集成模型
- `manifest.json` — 训练配置与指标
- `classification_report.txt` — 分类报告
- `label_encoder.pkl` — 标签编码器
- 各基模型 pkl 文件

### 训练参数说明

```bash
python scripts/train_v10.py \
  --version v10.0.0        # 模型版本号
  --cv-splits 10           # 交叉验证折数
  --optuna-trials 30       # Optuna 搜索次数（越大越慢越准）
  --confidence-threshold 0.45  # 置信度路由阈值
  --margin 0.15            # 路由 margin
  --no-optuna              # 跳过超参搜索（快速训练）
  --no-experts             # 跳过混淆对专家
  --no-adversarial         # 跳过对抗验证
  --no-smote               # 跳过 SMOTE-ENN
```

## 技术方案

### 特征工程（54 维）

| 类别 | 数量 | 说明 |
|------|------|------|
| 基础特征 | 25 | URL 长度、参数数量、HTTP 方法、攻击模式匹配、编码检测等 |
| V6 高级特征 | 13 | URL/Body 熵、特殊字符比例、SQL 关键词密度、XSS 标签计数等 |
| V7 交互特征 | 11 | CSRF 复合分数、良性请求分数、攻击模式聚合、URL 复杂度等 |
| V9 精准特征 | 5 | 文件包含协议检测、SQL 函数计数、XSS 事件处理器、干净请求分数 |

### 集成学习架构

```
                    ┌─────────────┐
                    │  XGBoost    │──┐
                    ├─────────────┤  │
                    │  CatBoost   │──┤  Stacking / Voting
                    ├─────────────┤  │  (LightGBM meta-learner)
                    │  LightGBM   │──┤        │
                    ├─────────────┤  │        ▼
                    │  ExtraTrees │──┘   最终预测
                    └─────────────┘
```

### 训练流程

1. **数据加载** — 读取 parquet，可选混入外部数据集（CSIC 2010）
2. **特征提取** — 25 基础特征 + 29 高级/交互/精准特征 = 54 维
3. **数据均衡** — SMOTE-ENN 过采样 + 清洗，目标比例 0.9
4. **对抗验证** — 检测训练/测试分布偏移，自动调整样本权重
5. **Optuna 超参搜索** — XGBoost/LightGBM 各 30 trials，CatBoost 15 trials
6. **10 折交叉验证** — 训练 4 个基模型
7. **集成** — Stacking（LightGBM meta）+ Voting（平方加权）
8. **混淆对专家** — 6 个二分类专家模型，自适应路由
9. **概率校准** — CalibratedClassifierCV isotonic 校准
10. **模型选择** — 从集成、路由、校准、最强单模型中自动选最优

### 版本演进

| 版本 | 核心变化 | 结果 |
|------|---------|------|
| V1–V5 | 基础探索（单模型 → 简单集成） | ~97% |
| V6 | 3 模型 Stacking + 高级特征 | 98.55% |
| V7 | +Optuna +ExtraTrees +交互特征 | 99.35% ✅ |
| V8 | +特征选择 +多种子集成 +passthrough | 99.23% ❌ 过度工程 |
| V9 | 回归 V7 +混淆对专家 +对抗验证 | 99.37% ✅ |
| V10 | V9 bug 修复 +LGB meta +自适应路由 | 99.37% ✅ |

> V8 的教训：特征选择误删关键特征、多种子集成稀释强模型置信度、passthrough 导致 meta-learner 过拟合。V9/V10 回归"少即是多"策略。

## 依赖

```
pandas>=2.0        numpy>=1.24       scikit-learn>=1.3
xgboost>=2.0       catboost>=1.2     lightgbm>=4.0
imbalanced-learn>=0.11               optuna>=3.4
openpyxl>=3.1      pyarrow>=14.0
```

Python >= 3.10，详见 [requirements.txt](requirements.txt)

## License

MIT
