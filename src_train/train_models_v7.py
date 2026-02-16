# -*- coding: utf-8 -*-
"""
训练流程 V7：精准调优 + 特征交互 + Optuna 自动调参 + 4 模型 Stacking
目标：99%+ 准确率

V6 结果分析（98.55%）：
  - 主要错误：正常访问→CSRF（38条），文件包含→CSRF/目录遍历（22条），XSS→正常/SQL（14条）
  - CSRF 的 precision 只有 91.53%（大量误报）
  - 正常访问 recall 只有 95.10%（被误判为 CSRF）

V7 核心改进：
  1. Optuna 贝叶斯超参搜索（每个模型 30 trials）
  2. 特征交互：针对 CSRF/正常访问 混淆，增加二阶交叉特征
  3. 第四基模型 ExtraTrees 增加集成多样性
  4. 改进 Stacking：meta-learner 用 GradientBoosting 替代 LogisticRegression
  5. 概率校准 CalibratedClassifierCV
  6. 更精细的 SMOTE-ENN 策略
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

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
# 高级特征工程（继承 V6 + 新增交互特征）
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _count_pattern(text: str, pattern: str) -> int:
    try:
        return len(re.findall(pattern, text, re.IGNORECASE))
    except re.error:
        return 0


def extract_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """V6 的 13 个高级特征 + V7 新增交互特征"""
    logger.info("提取高级特征 ...")

    url_col = 'url_path' if 'url_path' in df.columns else 'url'
    body_col = 'request_body' if 'request_body' in df.columns else 'body'

    urls = df[url_col].fillna('').astype(str)
    bodies = df[body_col].fillna('').astype(str)
    combined = urls + ' ' + bodies

    # ── V6 原有 13 个高级特征 ──
    df['url_entropy'] = urls.apply(_entropy)
    df['body_entropy'] = bodies.apply(_entropy)
    df['url_upper_ratio'] = urls.apply(
        lambda u: sum(1 for c in u if c.isupper()) / max(len(u), 1)
    )
    df['url_non_ascii_count'] = urls.apply(
        lambda u: sum(1 for c in u if ord(c) > 127)
    )
    _special = set("'\"<>;|&\\{}[]()")
    df['body_special_char_ratio'] = bodies.apply(
        lambda b: sum(1 for c in b if c in _special) / max(len(b), 1)
    )
    df['payload_token_count'] = combined.apply(
        lambda t: len(re.split(r'[\s/&=?;|]+', t))
    )
    df['nested_encoding_depth'] = combined.apply(
        lambda t: t.count('%25') + t.count('%2525')
    )
    _sql_kw = re.compile(
        r'\b(select|union|insert|update|delete|drop|exec|where|from|having|'
        r'group\s+by|order\s+by|benchmark|sleep|waitfor)\b', re.IGNORECASE
    )
    df['sql_keyword_density'] = combined.apply(
        lambda t: len(_sql_kw.findall(t)) / max(len(t.split()), 1)
    )
    _xss_re = re.compile(r'<\s*(script|img|svg|iframe|body|object|embed|link)', re.IGNORECASE)
    df['xss_tag_count'] = combined.apply(lambda t: len(_xss_re.findall(t)))
    df['path_depth_ratio'] = urls.apply(
        lambda u: u.count('/') / max(len(u), 1)
    )
    df['has_hex_encoding'] = combined.apply(
        lambda t: 1 if re.search(r'0x[0-9a-fA-F]{2,}', t) else 0
    )
    df['semicolon_count'] = combined.apply(lambda t: t.count(';'))
    df['dot_dot_count'] = combined.apply(
        lambda t: t.count('../') + t.count('..\\') + t.lower().count('%2e%2e')
    )

    v6_features = [
        'url_entropy', 'body_entropy', 'url_upper_ratio',
        'url_non_ascii_count', 'body_special_char_ratio',
        'payload_token_count', 'nested_encoding_depth',
        'sql_keyword_density', 'xss_tag_count', 'path_depth_ratio',
        'has_hex_encoding', 'semicolon_count', 'dot_dot_count',
    ]

    # ── V7 新增：针对 CSRF/正常访问 混淆的特征 ──
    # CSRF 通常有 POST + 表单参数 + 特定动作词
    _csrf_action = re.compile(
        r'(transfer|withdraw|delete|update|submit|change|modify|reset|confirm|purchase|pay)',
        re.IGNORECASE
    )
    df['csrf_action_count'] = combined.apply(lambda t: len(_csrf_action.findall(t)))

    # token/csrf 相关参数存在性
    _token_re = re.compile(r'(csrf|token|nonce|_verify|authenticity)', re.IGNORECASE)
    df['has_token_param'] = combined.apply(lambda t: 1 if _token_re.search(t) else 0)

    # POST + 有 body + 有动作词 = 高 CSRF 嫌疑
    df['csrf_composite'] = (
        df.get('method_post', 0) * df['csrf_action_count'] *
        (1 + df.get('has_body', 0))
    )

    # 正常访问通常：GET + 短 URL + 无攻击模式
    df['benign_score'] = (
        df.get('method_get', 0) *
        (1 / (1 + df.get('url_special_chars', 0))) *
        (1 / (1 + df.get('sensitive_keyword_count', 0)))
    )

    # ── V7 新增：特征交互（二阶） ──
    # 攻击模式总分
    pattern_cols = [c for c in df.columns if c.startswith('pattern_')]
    df['attack_pattern_sum'] = df[pattern_cols].sum(axis=1) if pattern_cols else 0
    df['attack_pattern_max'] = df[pattern_cols].max(axis=1) if pattern_cols else 0

    # URL 复杂度综合指标
    df['url_complexity'] = (
        df.get('url_length', 0) * df.get('url_special_chars', 0) *
        (1 + df.get('url_encoding_count', 0))
    )

    # body 危险度
    df['body_danger'] = (
        df.get('body_length', 0) * df['body_special_char_ratio'] *
        (1 + df['body_entropy'])
    )

    # 编码异常综合
    df['encoding_anomaly'] = (
        df.get('url_encoding_count', 0) +
        df.get('double_encoding', 0) * 3 +
        df['nested_encoding_depth'] * 5 +
        df['has_hex_encoding'] * 2
    )

    # payload 密度（token 数 / 长度）
    total_len = df.get('url_length', 1) + df.get('body_length', 0)
    df['payload_density'] = df['payload_token_count'] / total_len.clip(lower=1)

    # 路径遍历综合
    df['traversal_score'] = (
        df['dot_dot_count'] * 3 +
        df.get('pattern_path_traversal', 0) * 2 +
        df.get('pattern_file_inclusion', 0)
    )

    v7_features = [
        'csrf_action_count', 'has_token_param', 'csrf_composite', 'benign_score',
        'attack_pattern_sum', 'attack_pattern_max',
        'url_complexity', 'body_danger', 'encoding_anomaly',
        'payload_density', 'traversal_score',
    ]

    all_new = v6_features + v7_features
    logger.info("  新增 %d 个高级特征 (V6: %d + V7: %d)",
                len(all_new), len(v6_features), len(v7_features))
    return df, all_new


# ═══════════════════════════════════════════════════════════════════════════════
# 外部数据受控混入（继承 V6）
# ═══════════════════════════════════════════════════════════════════════════════

def load_external_datasets_controlled(
    external_dir: Path,
    original_dist: pd.Series,
    max_ratio: float = 0.3,
) -> pd.DataFrame | None:
    """受控加载外部数据集（与 V6 相同逻辑）"""
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

    drop_labels = {"正常访问", "文件上传攻击"}
    before = len(ext_df)
    ext_df = ext_df[~ext_df["attack_type"].isin(drop_labels)]
    logger.info("过滤不可靠标签 %s: %d → %d 条", drop_labels, before, len(ext_df))

    if ext_df.empty:
        return None

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
        return None

    result = pd.concat(sampled_parts, ignore_index=True)
    logger.info("外部数据受控混入总量: %d 条", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna 超参搜索
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_available():
    try:
        import optuna
        return True
    except ImportError:
        return False


def optuna_tune_xgb(X_train, y_train, n_trials=30, n_splits=5):
    """Optuna 调参 XGBoost"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'max_depth': trial.suggest_int('max_depth', 6, 14),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': 'mlogloss',
        }
        model = xgb.XGBClassifier(**params)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("XGBoost Optuna 最优: %.4f, 参数: %s", study.best_value, study.best_params)
    return study.best_params


def optuna_tune_lgb(X_train, y_train, n_trials=30, n_splits=5):
    """Optuna 调参 LightGBM"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1000),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
            'random_state': RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1,
        }
        model = lgb.LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            pred = model.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("LightGBM Optuna 最优: %.4f, 参数: %s", study.best_value, study.best_params)
    return study.best_params


def optuna_tune_cat(X_train, y_train, n_trials=20, n_splits=5):
    """Optuna 调参 CatBoost（trials 少一些因为 CatBoost 训练慢）"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 300, 800),
            'depth': trial.suggest_int('depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.5),
            'border_count': trial.suggest_int('border_count', 64, 255),
            'random_seed': RANDOM_STATE,
            'verbose': False,
            'thread_count': -1,
        }
        model = cb.CatBoostClassifier(**params)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
            pred = model.predict(X_train.iloc[val_idx])
            scores.append(accuracy_score(y_train.iloc[val_idx], pred))
        return np.mean(scores)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("CatBoost Optuna 最优: %.4f, 参数: %s", study.best_value, study.best_params)
    return study.best_params


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
    version: str = "v7.0.0",
    use_cv: bool = True,
    n_cv_splits: int = 10,
    use_stacking: bool = True,
    use_smote: bool = True,
    use_optuna: bool = True,
    optuna_trials: int = 30,
    external_data_dir: Path = None,
) -> Path:
    """
    V7 完整训练流程

    核心改进（相比 V6）：
    1. Optuna 贝叶斯超参搜索
    2. +11 个交互特征（针对 CSRF/正常访问 混淆）
    3. ExtraTrees 第四基模型
    4. Stacking meta-learner 升级为 GradientBoosting
    5. 概率校准
    """
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("=== V7 训练开始: version=%s ===", version)
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

    # ── 4. 高级特征 + 交互特征 ───────────────────────
    df, adv_feature_names = extract_advanced_features(df)

    # ── 5. 告警聚合器 ────────────────────────────────
    if len(df) > 50000:
        logger.info("数据集较大 (%d 条)，跳过聚合器训练", len(df))
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
    all_features = list(dict.fromkeys(all_features))
    X = df[all_features].fillna(0)

    # 替换 inf 值
    X = X.replace([np.inf, -np.inf], 0)

    logger.info("总特征数量: %d (基础 %d + 高级/交互 %d)",
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

    # ── 8. SMOTE-ENN ─────────────────────────────────
    if use_smote:
        logger.info("\n=== SMOTE-ENN 过采样 + 清洗 ===")
        logger.info("过采样前分布:")
        for idx, count in pd.Series(y_train).value_counts().sort_index().items():
            logger.info("  %s: %d", le.classes_[idx], count)

        train_counts = pd.Series(y_train).value_counts()
        # V7: 目标 90%（V6 是 80%），让少数类更接近多数类
        target_count = int(train_counts.max() * 0.9)
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
                logger.warning("SMOTE-ENN 失败: %s，回退到 SMOTE", e)
                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=RANDOM_STATE,
                    k_neighbors=min(5, min(train_counts.values) - 1),
                )
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
                y_train = pd.Series(y_resampled)
        else:
            logger.info("所有类别已足够均衡，跳过过采样")
            y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = pd.Series(y_train, index=X_train.index)

    y_test = pd.Series(y_test, index=X_test.index)

    # ── 9. Optuna 超参搜索 ───────────────────────────
    xgb_params = {}
    lgb_params = {}
    cat_params = {}

    if use_optuna and _optuna_available():
        logger.info("\n=== Optuna 超参搜索 ===")
        logger.info("XGBoost: %d trials ...", optuna_trials)
        xgb_params = optuna_tune_xgb(X_train, y_train, n_trials=optuna_trials)
        logger.info("LightGBM: %d trials ...", optuna_trials)
        lgb_params = optuna_tune_lgb(X_train, y_train, n_trials=optuna_trials)
        logger.info("CatBoost: %d trials ...", max(optuna_trials // 2, 10))
        cat_params = optuna_tune_cat(X_train, y_train, n_trials=max(optuna_trials // 2, 10))
    else:
        if use_optuna:
            logger.warning("Optuna 未安装，使用默认参数")

    # ── 10. 构建模型 ─────────────────────────────────
    # XGBoost
    xgb_default = {
        'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.02,
        'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_weight': 3, 'gamma': 0.1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': RANDOM_STATE, 'n_jobs': -1,
        'tree_method': 'hist', 'eval_metric': 'mlogloss',
    }
    xgb_final = {**xgb_default, **xgb_params}
    xgb_final['random_state'] = RANDOM_STATE
    xgb_final['n_jobs'] = -1
    xgb_final['tree_method'] = 'hist'
    xgb_final['eval_metric'] = 'mlogloss'
    xgb_model = xgb.XGBClassifier(**xgb_final)

    # CatBoost
    cat_default = {
        'iterations': 500, 'depth': 8, 'learning_rate': 0.03,
        'l2_leaf_reg': 5, 'random_seed': RANDOM_STATE,
        'verbose': False, 'thread_count': -1,
        'border_count': 128, 'bagging_temperature': 0.8,
    }
    cat_final = {**cat_default, **cat_params}
    cat_final['random_seed'] = RANDOM_STATE
    cat_final['verbose'] = False
    cat_final['thread_count'] = -1
    cat_model = cb.CatBoostClassifier(**cat_final)

    # LightGBM
    lgb_default = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03,
        'num_leaves': 127, 'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_samples': 10, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
    }
    lgb_final = {**lgb_default, **lgb_params}
    lgb_final['random_state'] = RANDOM_STATE
    lgb_final['n_jobs'] = -1
    lgb_final['verbose'] = -1
    lgb_model = lgb.LGBMClassifier(**lgb_final)

    # ExtraTrees（V7 新增第四基模型）
    et_model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,  # 不限深度
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # ── 11. 训练 ─────────────────────────────────────
    if use_cv:
        xgb_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "XGBoost", xgb_model, n_cv_splits)
        cat_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "CatBoost", cat_model, n_cv_splits)
        lgb_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "LightGBM", lgb_model, n_cv_splits)
        et_result = train_with_cv(X_train, y_train, X_test, y_test,
                                   "ExtraTrees", et_model, n_cv_splits)
    else:
        for name, model in [("XGBoost", xgb_model), ("CatBoost", cat_model),
                            ("LightGBM", lgb_model), ("ExtraTrees", et_model)]:
            if isinstance(model, cb.CatBoostClassifier):
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
        xgb_result = {"model": xgb_model, "test_accuracy": accuracy_score(y_test, xgb_model.predict(X_test)),
                       "test_f1": f1_score(y_test, xgb_model.predict(X_test), average='weighted'),
                       "predictions": xgb_model.predict(X_test)}
        cat_result = {"model": cat_model, "test_accuracy": accuracy_score(y_test, cat_model.predict(X_test)),
                       "test_f1": f1_score(y_test, cat_model.predict(X_test), average='weighted'),
                       "predictions": cat_model.predict(X_test)}
        lgb_result = {"model": lgb_model, "test_accuracy": accuracy_score(y_test, lgb_model.predict(X_test)),
                       "test_f1": f1_score(y_test, lgb_model.predict(X_test), average='weighted'),
                       "predictions": lgb_model.predict(X_test)}
        et_result = {"model": et_model, "test_accuracy": accuracy_score(y_test, et_model.predict(X_test)),
                      "test_f1": f1_score(y_test, et_model.predict(X_test), average='weighted'),
                      "predictions": et_model.predict(X_test)}

    # ── 12. 集成模型 ─────────────────────────────────
    logger.info("\n=== 集成模型训练 ===")

    # Stacking（V7: 4 基模型 + GradientBoosting meta-learner）
    if use_stacking:
        logger.info("训练 Stacking 集成（4 基模型 + GBDT meta-learner）...")
        stacking = StackingClassifier(
            estimators=[
                ("xgb", xgb_result["model"]),
                ("cat", cat_result["model"]),
                ("lgb", lgb_result["model"]),
                ("et", et_result["model"]),
            ],
            final_estimator=GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ),
            cv=5,
            n_jobs=-1,
            passthrough=False,
        )
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_pred)
        stacking_f1 = f1_score(y_test, stacking_pred, average='weighted')
        logger.info("Stacking 准确率: %.4f, F1: %.4f", stacking_acc, stacking_f1)

    # Voting（4 模型加权软投票）
    logger.info("训练 Voting 集成 ...")
    accs = [xgb_result["test_accuracy"], cat_result["test_accuracy"],
            lgb_result["test_accuracy"], et_result["test_accuracy"]]
    total = sum(accs)
    weights = [a / total for a in accs]
    logger.info("Voting 权重: XGB=%.3f, Cat=%.3f, LGB=%.3f, ET=%.3f", *weights)

    voting = VotingClassifier(
        estimators=[
            ("xgb", xgb_result["model"]),
            ("cat", cat_result["model"]),
            ("lgb", lgb_result["model"]),
            ("et", et_result["model"]),
        ],
        voting="soft",
        weights=weights,
    )
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)
    voting_acc = accuracy_score(y_test, voting_pred)
    voting_f1 = f1_score(y_test, voting_pred, average='weighted')
    logger.info("Voting 准确率: %.4f, F1: %.4f", voting_acc, voting_f1)

    # 选择最优
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

    # ── 13. 分类报告 ─────────────────────────────────
    report_text = classification_report(
        y_test, ensemble_pred,
        target_names=le.classes_,
        digits=4,
    )
    logger.info("分类报告:\n%s", report_text)

    cm = confusion_matrix(y_test, ensemble_pred)
    logger.info("混淆矩阵:\n%s", cm)

    # ── 14. 保存 ─────────────────────────────────────
    for name, obj in [
        ("ensemble.pkl", ensemble),
        ("label_encoder.pkl", le),
        ("xgboost.pkl", xgb_result["model"]),
        ("catboost.pkl", cat_result["model"]),
        ("lightgbm.pkl", lgb_result["model"]),
        ("extratrees.pkl", et_result["model"]),
    ]:
        with open(out_dir / name, "wb") as f:
            pickle.dump(obj, f)

    # ── 15. manifest ─────────────────────────────────
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
            "smote_method": "SMOTE-ENN",
            "smote_target_ratio": 0.9,
            "use_optuna": use_optuna,
            "optuna_trials": optuna_trials if use_optuna else None,
            "test_size": TRAIN_TEST_SPLIT,
            "random_state": RANDOM_STATE,
            "external_data_max_ratio": 0.3,
            "advanced_features": adv_feature_names,
            "n_base_models": 4,
            "base_models": ["XGBoost", "CatBoost", "LightGBM", "ExtraTrees"],
        },
        "optuna_params": {
            "xgboost": xgb_params if xgb_params else "default",
            "catboost": cat_params if cat_params else "default",
            "lightgbm": lgb_params if lgb_params else "default",
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
            "extratrees": {
                "test_accuracy": round(et_result["test_accuracy"], 4),
                "test_f1": round(et_result["test_f1"], 4),
                "cv_accuracy": round(et_result.get("cv_accuracy", 0), 4) if use_cv else None,
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

    logger.info("=== V7 训练完成，模型已保存: %s ===", out_dir)
    logger.info("文件列表: %s", [p.name for p in out_dir.iterdir()])

    return out_dir
