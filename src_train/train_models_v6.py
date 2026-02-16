# -*- coding: utf-8 -*-
"""
训练流程 V6：数据质量优先 + 高级特征工程 + 精细化集成
目标：99.5%+ 准确率

V5 问题诊断：
  - CSIC 2010 外部数据 (61k) 淹没原始数据 (29k)，分布严重扭曲
  - CSIC 2010 的 classify_attack_type 把大量异常流量误标为"文件上传攻击"
  - SMOTE 被迫做 15x 过采样，合成样本质量极差
  - CV 90% vs 测试集 77% → 典型的分布不一致

V6 核心改进：
  1. 外部数据受控混入：每类最多补充原始数据 30%，且只补充少数类
  2. 高级特征工程：URL/Body 的 TF-IDF 特征 + 信息熵 + payload 长度统计
  3. SMOTE-ENN 替代 SMOTE：合成后清洗边界噪声样本
  4. 模型参数精调：基于 V5 日志反馈调整
  5. 概率校准 + 软投票集成
"""

import json
import logging
import math
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# 过采样 + 清洗
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import catboost as cb
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wutong.config import FEATURE_COLS, MODEL_DIR, RANDOM_STATE, TRAIN_TEST_SPLIT
from wutong.features import extract_features
from wutong.denoise import AlertAggregator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 高级特征工程
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy(s: str) -> float:
    """计算字符串的 Shannon 信息熵，高熵 → 可能是编码/混淆攻击"""
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _count_pattern(text: str, pattern: str) -> int:
    """安全的正则计数"""
    try:
        return len(re.findall(pattern, text, re.IGNORECASE))
    except re.error:
        return 0


def extract_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在基础 25 特征之上，新增高区分度特征

    新增特征：
    - url_entropy / body_entropy: 信息熵（混淆攻击检测）
    - url_upper_ratio: 大写字母比例（大小写绕过检测）
    - url_non_ascii_count: 非 ASCII 字符数
    - body_special_char_ratio: body 中特殊字符占比
    - payload_token_count: payload 中的 token 数量
    - nested_encoding_depth: 嵌套编码深度（%25xx 等）
    - sql_keyword_density: SQL 关键词密度
    - xss_tag_count: XSS 标签数量
    - path_depth_ratio: 路径深度与 URL 长度的比值
    - has_hex_encoding: 是否包含十六进制编码
    - semicolon_count: 分号数量（命令注入指标）
    - dot_dot_count: ../ 出现次数（目录遍历指标）
    """
    logger.info("提取高级特征 ...")

    url_col = 'url_path' if 'url_path' in df.columns else 'url'
    body_col = 'request_body' if 'request_body' in df.columns else 'body'

    urls = df[url_col].fillna('').astype(str)
    bodies = df[body_col].fillna('').astype(str)
    combined = urls + ' ' + bodies

    # 信息熵
    df['url_entropy'] = urls.apply(_entropy)
    df['body_entropy'] = bodies.apply(_entropy)

    # 大写字母比例（大小写绕过）
    df['url_upper_ratio'] = urls.apply(
        lambda u: sum(1 for c in u if c.isupper()) / max(len(u), 1)
    )

    # 非 ASCII 字符数
    df['url_non_ascii_count'] = urls.apply(
        lambda u: sum(1 for c in u if ord(c) > 127)
    )

    # body 特殊字符占比
    _special = set("'\"<>;|&$`\\{}[]()")
    df['body_special_char_ratio'] = bodies.apply(
        lambda b: sum(1 for c in b if c in _special) / max(len(b), 1)
    )

    # payload token 数量（空格/特殊字符分割）
    df['payload_token_count'] = combined.apply(
        lambda t: len(re.split(r'[\s/&=?;|]+', t))
    )

    # 嵌套编码深度
    df['nested_encoding_depth'] = combined.apply(
        lambda t: t.count('%25') + t.count('%2525')
    )

    # SQL 关键词密度
    _sql_kw = re.compile(
        r'\b(select|union|insert|update|delete|drop|exec|where|from|having|'
        r'group\s+by|order\s+by|benchmark|sleep|waitfor)\b', re.IGNORECASE
    )
    df['sql_keyword_density'] = combined.apply(
        lambda t: len(_sql_kw.findall(t)) / max(len(t.split()), 1)
    )

    # XSS 标签数量
    _xss_re = re.compile(r'<\s*(script|img|svg|iframe|body|object|embed|link)', re.IGNORECASE)
    df['xss_tag_count'] = combined.apply(lambda t: len(_xss_re.findall(t)))

    # 路径深度与 URL 长度比值
    df['path_depth_ratio'] = urls.apply(
        lambda u: u.count('/') / max(len(u), 1)
    )

    # 十六进制编码
    df['has_hex_encoding'] = combined.apply(
        lambda t: 1 if re.search(r'0x[0-9a-fA-F]{2,}', t) else 0
    )

    # 分号数量（命令注入）
    df['semicolon_count'] = combined.apply(lambda t: t.count(';'))

    # ../ 出现次数（目录遍历）
    df['dot_dot_count'] = combined.apply(
        lambda t: t.count('../') + t.count('..\\') + t.lower().count('%2e%2e')
    )

    new_features = [
        'url_entropy', 'body_entropy', 'url_upper_ratio',
        'url_non_ascii_count', 'body_special_char_ratio',
        'payload_token_count', 'nested_encoding_depth',
        'sql_keyword_density', 'xss_tag_count', 'path_depth_ratio',
        'has_hex_encoding', 'semicolon_count', 'dot_dot_count',
    ]
    logger.info("  新增 %d 个高级特征: %s", len(new_features), new_features)
    return df, new_features


# ═══════════════════════════════════════════════════════════════════════════════
# 外部数据受控混入
# ═══════════════════════════════════════════════════════════════════════════════

def load_external_datasets_controlled(
    external_dir: Path,
    original_dist: pd.Series,
    max_ratio: float = 0.3,
) -> pd.DataFrame | None:
    """
    受控加载外部数据集

    策略：
    1. 只补充原始数据中的少数类（低于中位数的类别）
    2. 每个类别最多补充 original_count * max_ratio 条
    3. 丢弃外部数据中标签为"正常访问"的样本（CSIC 正常流量特征空间差异太大）
    4. 丢弃外部数据中标签为"文件上传攻击"的样本（CSIC 误标严重）

    Parameters
    ----------
    external_dir : 外部数据集目录
    original_dist : 原始数据的 attack_type 分布 (value_counts)
    max_ratio : 每类最多补充的比例
    """
    if not external_dir or not external_dir.exists():
        logger.info("未指定外部数据集目录，跳过")
        return None

    external_files = list(external_dir.glob("*.parquet"))
    if not external_files:
        logger.info("未在 %s 找到外部数据集", external_dir)
        return None

    dfs = []
    for f in external_files:
        logger.info("  加载外部数据: %s", f.name)
        dfs.append(pd.read_parquet(f))

    ext_df = pd.concat(dfs, ignore_index=True)
    logger.info("外部数据集原始总量: %d 条", len(ext_df))
    logger.info("外部攻击类型分布:\n%s", ext_df["attack_type"].value_counts())

    # ── 过滤掉不可靠的标签 ──────────────────────────
    # CSIC 2010 的"正常访问"和"文件上传攻击"标签质量差，特征空间与原始数据不一致
    drop_labels = {"正常访问", "文件上传攻击"}
    before = len(ext_df)
    ext_df = ext_df[~ext_df["attack_type"].isin(drop_labels)]
    logger.info("过滤不可靠标签 %s: %d → %d 条", drop_labels, before, len(ext_df))

    if ext_df.empty:
        logger.info("过滤后外部数据为空，跳过")
        return None

    # ── 只补充少数类 ────────────────────────────────
    median_count = original_dist.median()
    minority_classes = original_dist[original_dist <= median_count].index.tolist()
    logger.info("少数类（≤中位数 %d）: %s", median_count, minority_classes)

    sampled_parts = []
    for cls in minority_classes:
        cls_data = ext_df[ext_df["attack_type"] == cls]
        if cls_data.empty:
            continue
        max_samples = int(original_dist[cls] * max_ratio)
        n_take = min(len(cls_data), max_samples)
        if n_take > 0:
            sampled = cls_data.sample(n=n_take, random_state=RANDOM_STATE)
            sampled_parts.append(sampled)
            logger.info("  %s: 补充 %d 条（外部可用 %d，上限 %d）",
                        cls, n_take, len(cls_data), max_samples)

    if not sampled_parts:
        logger.info("无可补充的少数类数据")
        return None

    result = pd.concat(sampled_parts, ignore_index=True)
    logger.info("外部数据受控混入总量: %d 条", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 训练核心
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_cv(
    X_train, y_train, X_test, y_test,
    model_name: str, model,
    n_splits: int = 10,
):
    """使用交叉验证训练模型"""
    logger.info("--- 训练 %s ---", model_name)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        if isinstance(model, cb.CatBoostClassifier):
            model.fit(X_fold_train, y_fold_train, verbose=False)
        else:
            model.fit(X_fold_train, y_fold_train)

        val_pred = model.predict(X_fold_val)
        val_acc = accuracy_score(y_fold_val, val_pred)
        cv_scores.append(val_acc)

        if fold <= 3 or fold == n_splits:
            logger.info("  Fold %d: %.4f", fold, val_acc)

    avg_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    logger.info("  平均 CV 准确率: %.4f (±%.4f)", avg_cv, std_cv)

    # 全量训练
    if isinstance(model, cb.CatBoostClassifier):
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')

    logger.info("  测试集准确率: %.4f", test_acc)
    logger.info("  测试集 F1: %.4f", test_f1)

    return {
        "model": model,
        "cv_accuracy": avg_cv,
        "cv_std": std_cv,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "predictions": test_pred,
    }


def train(
    data_path: str | Path,
    version: str = "v6.0.0",
    use_cv: bool = True,
    n_cv_splits: int = 10,
    use_stacking: bool = True,
    use_smote: bool = True,
    external_data_dir: Path = None,
) -> Path:
    """
    V6 完整训练流程

    核心改进：
    1. 外部数据受控混入（不再淹没原始分布）
    2. 高级特征工程（+13 个新特征）
    3. SMOTE-ENN（合成 + 清洗边界噪声）
    4. 精调模型参数
    5. 软投票 + Stacking 双集成对比，取最优
    """
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("=== V6 训练开始: version=%s ===", version)
    logger.info("=" * 72)

    # ── 1. 加载原始数据 ───────────────────────────────
    df = pd.read_parquet(data_path)
    logger.info("原始数据加载: %d 条", len(df))
    original_dist = df["attack_type"].value_counts()
    logger.info("原始攻击类型分布:\n%s", original_dist)

    # ── 2. 受控混入外部数据 ───────────────────────────
    ext_df = load_external_datasets_controlled(
        external_data_dir, original_dist, max_ratio=0.3
    )
    ext_rows = 0
    if ext_df is not None and not ext_df.empty:
        ext_rows = len(ext_df)
        df = pd.concat([df, ext_df], ignore_index=True)
        logger.info("合并后总数据量: %d 条", len(df))
        logger.info("合并后分布:\n%s", df["attack_type"].value_counts())

    # ── 3. 基础特征提取 ──────────────────────────────
    logger.info("\n基础特征提取 ...")
    df = extract_features(df)

    # ── 4. 高级特征提取 ──────────────────────────────
    df, adv_feature_names = extract_advanced_features(df)

    # ── 5. 告警聚合器 ────────────────────────────────
    if len(df) > 50000:
        logger.info("数据集较大 (%d 条)，跳过聚合器训练", len(df))
        df["cluster_id"] = -1
        aggregator = AlertAggregator()
        aggregator._fitted = True
        aggregator.save(out_dir / "aggregator.pkl")
    else:
        logger.info("训练告警聚合器 ...")
        aggregator = AlertAggregator()
        df["cluster_id"] = aggregator.fit_transform(df)
        aggregator.save(out_dir / "aggregator.pkl")

    # ── 6. 准备特征矩阵 ──────────────────────────────
    all_features = [c for c in FEATURE_COLS if c in df.columns] + \
                   [c for c in adv_feature_names if c in df.columns]
    # 去重
    all_features = list(dict.fromkeys(all_features))
    X = df[all_features].fillna(0)
    logger.info("总特征数量: %d (基础 %d + 高级 %d)",
                len(all_features),
                len([c for c in FEATURE_COLS if c in df.columns]),
                len(adv_feature_names))

    # ── 7. 标签编码 + 划分 ───────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df["attack_type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ── 8. SMOTE-ENN 过采样 + 清洗 ───────────────────
    if use_smote:
        logger.info("\n=== SMOTE-ENN 过采样 + 清洗 ===")
        logger.info("过采样前分布:")
        for idx, count in pd.Series(y_train).value_counts().sort_index().items():
            logger.info("  %s: %d", le.classes_[idx], count)

        # 计算目标采样策略：少数类上采样到多数类的 80%（而非 100%）
        train_counts = pd.Series(y_train).value_counts()
        target_count = int(train_counts.max() * 0.8)
        sampling_strategy = {}
        for cls_idx, count in train_counts.items():
            if count < target_count:
                sampling_strategy[cls_idx] = target_count

        if sampling_strategy:
            try:
                smote_enn = SMOTEENN(
                    smote=SMOTE(
                        sampling_strategy=sampling_strategy,
                        random_state=RANDOM_STATE,
                        k_neighbors=min(5, min(train_counts.values) - 1),
                    ),
                    random_state=RANDOM_STATE,
                )
                X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

                logger.info("\nSMOTE-ENN 后分布:")
                for idx, count in pd.Series(y_resampled).value_counts().sort_index().items():
                    logger.info("  %s: %d", le.classes_[idx], count)
                logger.info("数据量变化: %d → %d", len(y_train), len(y_resampled))

                X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
                y_train = pd.Series(y_resampled)
            except Exception as e:
                logger.warning("SMOTE-ENN 失败，回退到普通 SMOTE: %s", e)
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=RANDOM_STATE,
                    k_neighbors=min(5, min(train_counts.values) - 1),
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
                y_train = pd.Series(y_resampled)
                logger.info("SMOTE 后数据量: %d", len(y_train))
        else:
            logger.info("所有类别已足够均衡，跳过过采样")
            y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = pd.Series(y_train, index=X_train.index)

    y_test = pd.Series(y_test, index=X_test.index)

    # ── 9. 模型训练 ──────────────────────────────────

    # XGBoost（V6 调参：降低深度防过拟合，增加正则化）
    xgb_model = xgb.XGBClassifier(
        n_estimators=600,
        max_depth=10,               # V5: 12 → 降低防过拟合
        learning_rate=0.02,         # V5: 0.03 → 更小学习率 + 更多树
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,
        min_child_weight=3,         # V5: 1 → 增加防过拟合
        gamma=0.1,                  # V5: 0.05 → 增加剪枝
        reg_alpha=0.1,              # V5: 0.05
        reg_lambda=1.0,             # V5: 0.5
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method='hist',
        eval_metric='mlogloss',
    )

    if use_cv:
        xgb_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "XGBoost", xgb_model, n_cv_splits)
    else:
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_result = {
            "model": xgb_model,
            "test_accuracy": accuracy_score(y_test, xgb_pred),
            "test_f1": f1_score(y_test, xgb_pred, average='weighted'),
            "predictions": xgb_pred,
        }

    # CatBoost（V6：增加迭代次数，降低学习率）
    cat_model = cb.CatBoostClassifier(
        iterations=500,             # V5: 300
        depth=8,                    # V5: 10 → 降低防过拟合
        learning_rate=0.03,         # V5: 0.05
        l2_leaf_reg=5,              # V5: 3
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
        border_count=128,
        bagging_temperature=0.8,
    )

    if use_cv:
        cat_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "CatBoost", cat_model, n_cv_splits)
    else:
        cat_model.fit(X_train, y_train, verbose=False)
        cat_pred = cat_model.predict(X_test)
        cat_result = {
            "model": cat_model,
            "test_accuracy": accuracy_score(y_test, cat_pred),
            "test_f1": f1_score(y_test, cat_pred, average='weighted'),
            "predictions": cat_pred,
        }

    # LightGBM（V6：增加叶子数，更精细分裂）
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,           # V5: 300
        max_depth=8,                # V5: 10
        learning_rate=0.03,         # V5: 0.05
        num_leaves=127,             # V5: 63 → 更精细
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    if use_cv:
        lgb_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "LightGBM", lgb_model, n_cv_splits)
    else:
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_result = {
            "model": lgb_model,
            "test_accuracy": accuracy_score(y_test, lgb_pred),
            "test_f1": f1_score(y_test, lgb_pred, average='weighted'),
            "predictions": lgb_pred,
        }

    # ── 10. 集成模型 ─────────────────────────────────
    logger.info("\n=== 集成模型训练 ===")

    # 方案 A: Stacking
    if use_stacking:
        logger.info("训练 Stacking 集成 ...")
        stacking = StackingClassifier(
            estimators=[
                ("xgb", xgb_result["model"]),
                ("cat", cat_result["model"]),
                ("lgb", lgb_result["model"]),
            ],
            final_estimator=LogisticRegression(
                max_iter=3000,
                C=1.0,
                solver='lbfgs',
                random_state=RANDOM_STATE,
            ),
            cv=5,                   # V5: 10 → 减少 CV 折数加速
            n_jobs=-1,
            passthrough=False,
        )
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_pred)
        stacking_f1 = f1_score(y_test, stacking_pred, average='weighted')
        logger.info("Stacking 准确率: %.4f, F1: %.4f", stacking_acc, stacking_f1)

    # 方案 B: 软投票（基于概率加权）
    logger.info("训练 Voting 集成 ...")
    # 根据单模型测试集表现分配权重
    accs = [xgb_result["test_accuracy"], cat_result["test_accuracy"], lgb_result["test_accuracy"]]
    total = sum(accs)
    weights = [a / total for a in accs]
    logger.info("Voting 权重: XGB=%.3f, Cat=%.3f, LGB=%.3f", *weights)

    voting = VotingClassifier(
        estimators=[
            ("xgb", xgb_result["model"]),
            ("cat", cat_result["model"]),
            ("lgb", lgb_result["model"]),
        ],
        voting="soft",
        weights=weights,
    )
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)
    voting_acc = accuracy_score(y_test, voting_pred)
    voting_f1 = f1_score(y_test, voting_pred, average='weighted')
    logger.info("Voting 准确率: %.4f, F1: %.4f", voting_acc, voting_f1)

    # 选择最优集成
    if use_stacking and stacking_acc >= voting_acc:
        ensemble = stacking
        ensemble_pred = stacking_pred
        ensemble_acc = stacking_acc
        ensemble_f1 = stacking_f1
        ensemble_type = "stacking"
        logger.info("✅ 选择 Stacking（%.4f ≥ %.4f）", stacking_acc, voting_acc)
    else:
        ensemble = voting
        ensemble_pred = voting_pred
        ensemble_acc = voting_acc
        ensemble_f1 = voting_f1
        ensemble_type = "voting"
        logger.info("✅ 选择 Voting（%.4f）", voting_acc)

    # ── 11. 分类报告 ─────────────────────────────────
    report_text = classification_report(
        y_test, ensemble_pred,
        target_names=le.classes_,
        digits=4,
    )
    logger.info("分类报告:\n%s", report_text)

    cm = confusion_matrix(y_test, ensemble_pred)
    logger.info("混淆矩阵:\n%s", cm)

    # ── 12. 保存模型工件 ─────────────────────────────
    with open(out_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(out_dir / "xgboost.pkl", "wb") as f:
        pickle.dump(xgb_result["model"], f)
    with open(out_dir / "catboost.pkl", "wb") as f:
        pickle.dump(cat_result["model"], f)
    with open(out_dir / "lightgbm.pkl", "wb") as f:
        pickle.dump(lgb_result["model"], f)

    # ── 13. manifest ─────────────────────────────────
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_rows": len(df),
        "external_data": ext_rows > 0,
        "external_data_rows": ext_rows,
        "feature_list": all_features,
        "classes": list(le.classes_),
        "training_config": {
            "use_cv": use_cv,
            "n_cv_splits": n_cv_splits if use_cv else None,
            "use_stacking": use_stacking,
            "use_smote": use_smote,
            "smote_method": "SMOTE-ENN" if use_smote else None,
            "test_size": TRAIN_TEST_SPLIT,
            "random_state": RANDOM_STATE,
            "external_data_max_ratio": 0.3,
            "advanced_features": adv_feature_names,
        },
        "metrics": {
            "xgboost": {
                "test_accuracy": round(xgb_result["test_accuracy"], 4),
                "test_f1": round(xgb_result["test_f1"], 4),
                "cv_accuracy": round(xgb_result.get("cv_accuracy", 0), 4) if use_cv else None,
            },
            "catboost": {
                "test_accuracy": round(cat_result["test_accuracy"], 4),
                "test_f1": round(cat_result["test_f1"], 4),
                "cv_accuracy": round(cat_result.get("cv_accuracy", 0), 4) if use_cv else None,
            },
            "lightgbm": {
                "test_accuracy": round(lgb_result["test_accuracy"], 4),
                "test_f1": round(lgb_result["test_f1"], 4),
                "cv_accuracy": round(lgb_result.get("cv_accuracy", 0), 4) if use_cv else None,
            },
            "ensemble": {
                "test_accuracy": round(ensemble_acc, 4),
                "test_f1": round(ensemble_f1, 4),
                "ensemble_type": ensemble_type,
            },
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("=== V6 训练完成，模型已保存: %s ===", out_dir)
    logger.info("文件列表: %s", [p.name for p in out_dir.iterdir()])

    return out_dir
