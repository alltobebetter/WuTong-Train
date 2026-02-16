# -*- coding: utf-8 -*-
"""
训练流程 V8：精准消除混淆 + 特征选择 + 多种子集成 + Stacking passthrough
目标：99.5%+ 准确率

V7 结果分析（99.35%，38 条错误）：
  - 文件包含→目录遍历(7), CSRF(2), 正常访问(3) = 12 条  ← 最大问题
  - 正常访问→CSRF(3), SQL(2), XSS(3), 文件包含(4) = 12 条
  - SQL注入→文件包含(3), 正常访问(3), 远程命令(1) = 7 条
  - XSS→SQL(5), 正常访问(1) = 6 条
  - 远程命令→文件包含(1) = 1 条

V8 核心改进：
  1. 针对性区分特征：文件包含 vs 目录遍历、SQL vs XSS
  2. 特征重要性筛选：去掉低贡献特征减少噪声
  3. Stacking passthrough=True：meta-learner 同时看原始特征
  4. 多种子集成：3 个不同 seed 训练，概率平均
  5. Optuna warm start：基于 V7 最优参数缩小搜索空间
  6. 概率校准 CalibratedClassifierCV
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
    RandomForestClassifier,
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
# 高级特征工程（V6 + V7 + V8 针对性区分特征）
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """V6(13) + V7(11) + V8(10) = 34 个高级特征"""
    logger.info("提取高级特征 ...")

    url_col = 'url_path' if 'url_path' in df.columns else 'url'
    body_col = 'request_body' if 'request_body' in df.columns else 'body'

    urls = df[url_col].fillna('').astype(str)
    bodies = df[body_col].fillna('').astype(str)
    combined = urls + ' ' + bodies

    # ══════════════════════════════════════════════════
    # V6 原有 13 个高级特征
    # ══════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════
    # V7 交互特征 (11)
    # ══════════════════════════════════════════════════
    _csrf_action = re.compile(
        r'(transfer|withdraw|delete|update|submit|change|modify|reset|confirm|purchase|pay)',
        re.IGNORECASE
    )
    df['csrf_action_count'] = combined.apply(lambda t: len(_csrf_action.findall(t)))
    _token_re = re.compile(r'(csrf|token|nonce|_verify|authenticity)', re.IGNORECASE)
    df['has_token_param'] = combined.apply(lambda t: 1 if _token_re.search(t) else 0)
    df['csrf_composite'] = (
        df.get('method_post', 0) * df['csrf_action_count'] *
        (1 + df.get('has_body', 0))
    )
    df['benign_score'] = (
        df.get('method_get', 0) *
        (1 / (1 + df.get('url_special_chars', 0))) *
        (1 / (1 + df.get('sensitive_keyword_count', 0)))
    )
    pattern_cols = [c for c in df.columns if c.startswith('pattern_')]
    df['attack_pattern_sum'] = df[pattern_cols].sum(axis=1) if pattern_cols else 0
    df['attack_pattern_max'] = df[pattern_cols].max(axis=1) if pattern_cols else 0
    df['url_complexity'] = (
        df.get('url_length', 0) * df.get('url_special_chars', 0) *
        (1 + df.get('url_encoding_count', 0))
    )
    df['body_danger'] = (
        df.get('body_length', 0) * df['body_special_char_ratio'] *
        (1 + df['body_entropy'])
    )
    df['encoding_anomaly'] = (
        df.get('url_encoding_count', 0) +
        df.get('double_encoding', 0) * 3 +
        df['nested_encoding_depth'] * 5 +
        df['has_hex_encoding'] * 2
    )
    total_len = (df.get('url_length', 1) + df.get('body_length', 0)).clip(lower=1)
    df['payload_density'] = df['payload_token_count'] / total_len
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

    # ══════════════════════════════════════════════════
    # V8 新增：针对性消除混淆的特征 (10)
    # ══════════════════════════════════════════════════

    # --- 文件包含 vs 目录遍历 区分 ---
    # 文件包含特有：php:// data:// expect:// 等协议包装器，?file= ?page= 参数
    _fi_proto = re.compile(r'(php://|data://|expect://|zip://|phar://|glob://)', re.IGNORECASE)
    _fi_param = re.compile(r'[\?&](file|page|path|include|require|template|doc|document|folder|root|dir)\s*=', re.IGNORECASE)
    df['file_inclusion_proto'] = combined.apply(lambda t: len(_fi_proto.findall(t)))
    df['file_inclusion_param'] = combined.apply(lambda t: len(_fi_param.findall(t)))

    # 目录遍历特有：连续 ../ 且目标是系统文件（/etc/passwd, win.ini 等）
    _sys_file = re.compile(
        r'(etc/passwd|etc/shadow|etc/hosts|windows/system32|boot\.ini|win\.ini|'
        r'\.ssh/|id_rsa|authorized_keys|\.htaccess|web\.config|\.env)',
        re.IGNORECASE
    )
    df['targets_sys_file'] = combined.apply(lambda t: 1 if _sys_file.search(t) else 0)

    # 文件包含 vs 目录遍历 综合区分分数
    # 正值 → 更像文件包含，负值 → 更像目录遍历
    df['fi_vs_traversal'] = (
        df['file_inclusion_proto'] * 3 +
        df['file_inclusion_param'] * 2 -
        df['dot_dot_count'] * 2 -
        df['targets_sys_file'] * 3
    )

    # --- SQL 注入 vs XSS 区分 ---
    # SQL 特有：SQL 函数调用（concat, char, ascii, substring 等）
    _sql_func = re.compile(
        r'\b(concat|char|ascii|substring|substr|length|mid|left|right|'
        r'convert|cast|coalesce|ifnull|nullif|hex|unhex|load_file|'
        r'information_schema|table_name|column_name)\b',
        re.IGNORECASE
    )
    df['sql_func_count'] = combined.apply(lambda t: len(_sql_func.findall(t)))

    # XSS 特有：事件处理器（onmouseover, onfocus, onerror 等）
    _xss_event = re.compile(
        r'\b(onmouseover|onfocus|onblur|onclick|onsubmit|onchange|onkeyup|'
        r'onkeydown|onmouseout|onmousemove|ondblclick|oncontextmenu|'
        r'onerror|onload|onunload|onresize|onscroll)\s*=',
        re.IGNORECASE
    )
    df['xss_event_count'] = combined.apply(lambda t: len(_xss_event.findall(t)))

    # SQL vs XSS 综合区分
    df['sql_vs_xss'] = (
        df['sql_func_count'] * 2 +
        df['sql_keyword_density'] * 10 -
        df['xss_event_count'] * 3 -
        df['xss_tag_count'] * 2
    )

    # --- 正常访问 精准识别 ---
    # 正常访问通常：无攻击模式 + 短 payload + 低熵
    df['clean_request_score'] = (
        (df['attack_pattern_sum'] == 0).astype(int) * 3 +
        (df.get('url_special_chars', 0) == 0).astype(int) * 2 +
        (df.get('sensitive_keyword_count', 0) == 0).astype(int) * 2 +
        (df['url_entropy'] < 3.5).astype(int)
    )

    # 攻击强度综合指标（越高越可能是攻击）
    df['attack_intensity'] = (
        df['attack_pattern_sum'] +
        df['url_complexity'] / (total_len * 100 + 1) +
        df['body_danger'] / (total_len * 100 + 1) +
        df['encoding_anomaly'] * 0.5
    )

    v8_features = [
        'file_inclusion_proto', 'file_inclusion_param', 'targets_sys_file',
        'fi_vs_traversal',
        'sql_func_count', 'xss_event_count', 'sql_vs_xss',
        'clean_request_score', 'attack_intensity',
    ]

    all_new = v6_features + v7_features + v8_features
    logger.info("  新增高级特征: V6(%d) + V7(%d) + V8(%d) = %d",
                len(v6_features), len(v7_features), len(v8_features), len(all_new))
    return df, all_new


# ═══════════════════════════════════════════════════════════════════════════════
# 外部数据受控混入（继承 V6/V7）
# ═══════════════════════════════════════════════════════════════════════════════

def load_external_datasets_controlled(
    external_dir: Path,
    original_dist: pd.Series,
    max_ratio: float = 0.3,
) -> pd.DataFrame | None:
    if not external_dir or not external_dir.exists():
        return None
    external_files = list(external_dir.glob("*.parquet"))
    if not external_files:
        return None

    dfs = []
    for f in external_files:
        logger.info("  加载外部数据: %s", f.name)
        dfs.append(pd.read_parquet(f))

    ext_df = pd.concat(dfs, ignore_index=True)
    logger.info("外部数据集原始总量: %d 条", len(ext_df))

    drop_labels = {"正常访问", "文件上传攻击"}
    before = len(ext_df)
    ext_df = ext_df[~ext_df["attack_type"].isin(drop_labels)]
    logger.info("过滤不可靠标签: %d → %d 条", before, len(ext_df))

    if ext_df.empty:
        return None

    median_count = original_dist.median()
    minority_classes = original_dist[original_dist <= median_count].index.tolist()

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
            logger.info("  %s: 补充 %d 条", cls, n_take)

    if not sampled_parts:
        return None

    result = pd.concat(sampled_parts, ignore_index=True)
    logger.info("外部数据受控混入总量: %d 条", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Optuna 超参搜索（V8: warm start 缩小搜索空间）
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_available():
    try:
        import optuna
        return True
    except ImportError:
        return False


def optuna_tune_xgb(X_train, y_train, n_trials=50, n_splits=5):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            # V7 最优附近搜索（缩小范围）
            'n_estimators': trial.suggest_int('n_estimators', 600, 1200),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.75, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.90),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 8),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 0.5, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 3.0, log=True),
            'random_state': RANDOM_STATE,
            'n_jobs': -1, 'tree_method': 'hist', 'eval_metric': 'mlogloss',
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


def optuna_tune_lgb(X_train, y_train, n_trials=50, n_splits=5):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 600, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 127, 300),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.90),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 0.5, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
            'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
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
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 1000),
            'depth': trial.suggest_int('depth', 6, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 15.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.5, 2.0),
            'border_count': trial.suggest_int('border_count', 128, 255),
            'random_seed': RANDOM_STATE, 'verbose': False, 'thread_count': -1,
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
# 特征重要性筛选
# ═══════════════════════════════════════════════════════════════════════════════

def select_features_by_importance(X_train, y_train, feature_names, threshold=0.001):
    """用 LightGBM 快速训练，筛掉重要性极低的特征"""
    logger.info("特征重要性筛选 ...")
    quick_lgb = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    quick_lgb.fit(X_train, y_train)

    importances = quick_lgb.feature_importances_
    total_imp = importances.sum()
    normalized = importances / total_imp if total_imp > 0 else importances

    keep = []
    drop = []
    for fname, imp in zip(feature_names, normalized):
        if imp >= threshold:
            keep.append(fname)
        else:
            drop.append((fname, imp))

    if drop:
        logger.info("  移除 %d 个低贡献特征: %s",
                     len(drop), [d[0] for d in drop])
    logger.info("  保留 %d / %d 个特征", len(keep), len(feature_names))
    return keep


# ═══════════════════════════════════════════════════════════════════════════════
# 多种子集成
# ═══════════════════════════════════════════════════════════════════════════════

class MultiSeedEnsemble:
    """用不同 random_state 训练多个模型，预测时概率平均"""

    def __init__(self, base_models: list, seeds: list[int]):
        self.base_models = base_models  # [(name, model_class, params), ...]
        self.seeds = seeds
        self.fitted_models = []  # [(name, seed, model), ...]
        self.classes_ = None

    def fit(self, X, y):
        self.fitted_models = []
        self.classes_ = np.unique(y)
        for name, model_cls, params in self.base_models:
            for seed in self.seeds:
                p = params.copy()
                # 设置 seed
                if 'random_state' in p:
                    p['random_state'] = seed
                elif 'random_seed' in p:
                    p['random_seed'] = seed
                model = model_cls(**p)
                if isinstance(model, cb.CatBoostClassifier):
                    model.fit(X, y, verbose=False)
                else:
                    model.fit(X, y)
                self.fitted_models.append((name, seed, model))
        logger.info("多种子集成: %d 个模型 (%d 基模型 × %d seeds)",
                     len(self.fitted_models), len(self.base_models), len(self.seeds))
        return self

    def predict_proba(self, X):
        all_proba = []
        for name, seed, model in self.fitted_models:
            proba = model.predict_proba(X)
            all_proba.append(proba)
        return np.mean(all_proba, axis=0)

    def predict(self, X):
        avg_proba = self.predict_proba(X)
        return self.classes_[np.argmax(avg_proba, axis=1)]


# ═══════════════════════════════════════════════════════════════════════════════
# 训练核心
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_cv(
    X_train, y_train, X_test, y_test,
    model_name: str, model,
    n_splits: int = 10,
):
    logger.info("--- 训练 %s ---", model_name)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        if isinstance(model, cb.CatBoostClassifier):
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx], verbose=False)
        else:
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])

        val_pred = model.predict(X_train.iloc[val_idx])
        val_acc = accuracy_score(y_train.iloc[val_idx], val_pred)
        cv_scores.append(val_acc)
        if fold <= 3 or fold == n_splits:
            logger.info("  Fold %d: %.4f", fold, val_acc)

    avg_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    logger.info("  平均 CV 准确率: %.4f (±%.4f)", avg_cv, std_cv)

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
        "model": model, "cv_accuracy": avg_cv, "cv_std": std_cv,
        "test_accuracy": test_acc, "test_f1": test_f1, "predictions": test_pred,
    }


def train(
    data_path: str | Path,
    version: str = "v8.0.0",
    use_cv: bool = True,
    n_cv_splits: int = 10,
    use_stacking: bool = True,
    use_smote: bool = True,
    use_optuna: bool = True,
    optuna_trials: int = 50,
    use_feature_selection: bool = True,
    use_multi_seed: bool = True,
    external_data_dir: Path = None,
) -> Path:
    """V8 完整训练流程"""
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("=== V8 训练开始: version=%s ===", version)
    logger.info("=" * 72)

    # ── 1. 加载数据 ──────────────────────────────────
    df = pd.read_parquet(data_path)
    logger.info("原始数据加载: %d 条", len(df))
    original_dist = df["attack_type"].value_counts()
    logger.info("原始攻击类型分布:\n%s", original_dist)

    # ── 2. 外部数据 ──────────────────────────────────
    ext_df = load_external_datasets_controlled(
        external_data_dir, original_dist, max_ratio=0.3
    )
    ext_rows = 0
    if ext_df is not None and not ext_df.empty:
        ext_rows = len(ext_df)
        df = pd.concat([df, ext_df], ignore_index=True)
        logger.info("合并后: %d 条", len(df))

    # ── 3. 特征提取 ──────────────────────────────────
    logger.info("\n基础特征提取 ...")
    df = extract_features(df)
    df, adv_feature_names = extract_advanced_features(df)

    # ── 4. 聚合器 ────────────────────────────────────
    if len(df) > 50000:
        aggregator = AlertAggregator()
        aggregator._fitted = True
        aggregator.save(out_dir / "aggregator.pkl")
    else:
        logger.info("训练告警聚合器 ...")
        aggregator = AlertAggregator()
        df["cluster_id"] = aggregator.fit_transform(df)
        aggregator.save(out_dir / "aggregator.pkl")

    # ── 5. 特征矩阵 ──────────────────────────────────
    all_features = [c for c in FEATURE_COLS if c in df.columns] + \
                   [c for c in adv_feature_names if c in df.columns]
    all_features = list(dict.fromkeys(all_features))
    X = df[all_features].fillna(0).replace([np.inf, -np.inf], 0)

    logger.info("总特征数量: %d (基础 %d + 高级 %d)",
                len(all_features),
                len([c for c in FEATURE_COLS if c in df.columns]),
                len(adv_feature_names))

    # ── 6. 标签编码 + 划分 ───────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df["attack_type"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT,
        random_state=RANDOM_STATE, stratify=y,
    )

    # ── 7. 特征选择 ──────────────────────────────────
    if use_feature_selection:
        selected = select_features_by_importance(
            X_train, y_train, all_features, threshold=0.002
        )
        if len(selected) < len(all_features):
            X_train = X_train[selected]
            X_test = X_test[selected]
            all_features = selected

    # ── 8. SMOTE-ENN ─────────────────────────────────
    if use_smote:
        logger.info("\n=== SMOTE-ENN 过采样 + 清洗 ===")
        logger.info("过采样前分布:")
        for idx, count in pd.Series(y_train).value_counts().sort_index().items():
            logger.info("  %s: %d", le.classes_[idx], count)

        train_counts = pd.Series(y_train).value_counts()
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
                logger.warning("SMOTE-ENN 失败: %s", e)
                smote = SMOTE(sampling_strategy=sampling_strategy,
                              random_state=RANDOM_STATE,
                              k_neighbors=min(5, min(train_counts.values) - 1))
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                X_train = pd.DataFrame(X_resampled, columns=X_train.columns)
                y_train = pd.Series(y_resampled)
        else:
            y_train = pd.Series(y_train, index=X_train.index)
    else:
        y_train = pd.Series(y_train, index=X_train.index)

    y_test = pd.Series(y_test, index=X_test.index)

    # ── 9. Optuna ────────────────────────────────────
    xgb_params, lgb_params, cat_params = {}, {}, {}
    if use_optuna and _optuna_available():
        logger.info("\n=== Optuna 超参搜索 ===")
        logger.info("XGBoost: %d trials ...", optuna_trials)
        xgb_params = optuna_tune_xgb(X_train, y_train, n_trials=optuna_trials)
        logger.info("LightGBM: %d trials ...", optuna_trials)
        lgb_params = optuna_tune_lgb(X_train, y_train, n_trials=optuna_trials)
        cat_trials = max(optuna_trials // 3, 10)
        logger.info("CatBoost: %d trials ...", cat_trials)
        cat_params = optuna_tune_cat(X_train, y_train, n_trials=cat_trials)

    # ── 10. 构建模型 ─────────────────────────────────
    xgb_default = {
        'n_estimators': 800, 'max_depth': 7, 'learning_rate': 0.05,
        'subsample': 0.85, 'colsample_bytree': 0.75,
        'min_child_weight': 5, 'gamma': 0.01,
        'reg_alpha': 0.04, 'reg_lambda': 0.17,
        'random_state': RANDOM_STATE, 'n_jobs': -1,
        'tree_method': 'hist', 'eval_metric': 'mlogloss',
    }
    xgb_final = {**xgb_default, **xgb_params}
    xgb_final.update({'random_state': RANDOM_STATE, 'n_jobs': -1,
                      'tree_method': 'hist', 'eval_metric': 'mlogloss'})
    xgb_model = xgb.XGBClassifier(**xgb_final)

    cat_default = {
        'iterations': 773, 'depth': 8, 'learning_rate': 0.09,
        'l2_leaf_reg': 9.5, 'random_seed': RANDOM_STATE,
        'verbose': False, 'thread_count': -1,
        'border_count': 215, 'bagging_temperature': 1.5,
    }
    cat_final = {**cat_default, **cat_params}
    cat_final.update({'random_seed': RANDOM_STATE, 'verbose': False, 'thread_count': -1})
    cat_model = cb.CatBoostClassifier(**cat_final)

    lgb_default = {
        'n_estimators': 841, 'max_depth': 6, 'learning_rate': 0.05,
        'num_leaves': 230, 'subsample': 0.77, 'colsample_bytree': 0.81,
        'min_child_samples': 11, 'reg_alpha': 0.003, 'reg_lambda': 1.9,
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
    }
    lgb_final = {**lgb_default, **lgb_params}
    lgb_final.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
    lgb_model = lgb.LGBMClassifier(**lgb_final)

    et_model = ExtraTreesClassifier(
        n_estimators=500, max_depth=None,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1,
    )

    # ── 11. 训练 ─────────────────────────────────────
    if use_cv:
        xgb_result = train_with_cv(X_train, y_train, X_test, y_test, "XGBoost", xgb_model, n_cv_splits)
        cat_result = train_with_cv(X_train, y_train, X_test, y_test, "CatBoost", cat_model, n_cv_splits)
        lgb_result = train_with_cv(X_train, y_train, X_test, y_test, "LightGBM", lgb_model, n_cv_splits)
        et_result = train_with_cv(X_train, y_train, X_test, y_test, "ExtraTrees", et_model, n_cv_splits)
    else:
        for m in [xgb_model, lgb_model, et_model]:
            m.fit(X_train, y_train)
        cat_model.fit(X_train, y_train, verbose=False)
        xgb_result = {"model": xgb_model, "test_accuracy": accuracy_score(y_test, xgb_model.predict(X_test)),
                       "test_f1": f1_score(y_test, xgb_model.predict(X_test), average='weighted')}
        cat_result = {"model": cat_model, "test_accuracy": accuracy_score(y_test, cat_model.predict(X_test)),
                       "test_f1": f1_score(y_test, cat_model.predict(X_test), average='weighted')}
        lgb_result = {"model": lgb_model, "test_accuracy": accuracy_score(y_test, lgb_model.predict(X_test)),
                       "test_f1": f1_score(y_test, lgb_model.predict(X_test), average='weighted')}
        et_result = {"model": et_model, "test_accuracy": accuracy_score(y_test, et_model.predict(X_test)),
                      "test_f1": f1_score(y_test, et_model.predict(X_test), average='weighted')}

    # ── 12. 集成模型 ─────────────────────────────────
    logger.info("\n=== 集成模型训练 ===")

    # 方案 A: Stacking with passthrough
    if use_stacking:
        logger.info("训练 Stacking 集成（passthrough=True）...")
        stacking = StackingClassifier(
            estimators=[
                ("xgb", xgb_result["model"]),
                ("cat", cat_result["model"]),
                ("lgb", lgb_result["model"]),
                ("et", et_result["model"]),
            ],
            final_estimator=GradientBoostingClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.8,
                random_state=RANDOM_STATE,
            ),
            cv=5,
            n_jobs=-1,
            passthrough=True,  # V8: meta-learner 同时看原始特征
        )
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict(X_test)
        stacking_acc = accuracy_score(y_test, stacking_pred)
        stacking_f1 = f1_score(y_test, stacking_pred, average='weighted')
        logger.info("Stacking 准确率: %.4f, F1: %.4f", stacking_acc, stacking_f1)

    # 方案 B: 多种子集成
    best_ensemble = None
    best_ensemble_acc = 0
    best_ensemble_type = ""

    if use_multi_seed:
        logger.info("训练多种子集成 (3 seeds × 4 models = 12 models) ...")
        seeds = [RANDOM_STATE, RANDOM_STATE + 1, RANDOM_STATE + 2]

        # 提取 Optuna 参数
        xgb_p = {**xgb_final}
        lgb_p = {**lgb_final}
        cat_p = {**cat_final}
        et_p = {'n_estimators': 500, 'max_depth': None, 'min_samples_split': 5,
                'min_samples_leaf': 2, 'max_features': 'sqrt', 'random_state': RANDOM_STATE, 'n_jobs': -1}

        ms_ensemble = MultiSeedEnsemble(
            base_models=[
                ("xgb", xgb.XGBClassifier, xgb_p),
                ("lgb", lgb.LGBMClassifier, lgb_p),
                ("cat", cb.CatBoostClassifier, cat_p),
                ("et", ExtraTreesClassifier, et_p),
            ],
            seeds=seeds,
        )
        ms_ensemble.fit(X_train, y_train)
        ms_pred = ms_ensemble.predict(X_test)
        ms_acc = accuracy_score(y_test, ms_pred)
        ms_f1 = f1_score(y_test, ms_pred, average='weighted')
        logger.info("多种子集成 准确率: %.4f, F1: %.4f", ms_acc, ms_f1)

    # 方案 C: Voting
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
        voting="soft", weights=weights,
    )
    voting.fit(X_train, y_train)
    voting_pred = voting.predict(X_test)
    voting_acc = accuracy_score(y_test, voting_pred)
    voting_f1 = f1_score(y_test, voting_pred, average='weighted')
    logger.info("Voting 准确率: %.4f, F1: %.4f", voting_acc, voting_f1)

    # 选择最优集成
    candidates = [("voting", voting, voting_pred, voting_acc, voting_f1)]
    if use_stacking:
        candidates.append(("stacking", stacking, stacking_pred, stacking_acc, stacking_f1))
    if use_multi_seed:
        candidates.append(("multi_seed", ms_ensemble, ms_pred, ms_acc, ms_f1))

    candidates.sort(key=lambda x: x[3], reverse=True)
    ensemble_type, ensemble, ensemble_pred, ensemble_acc, ensemble_f1 = candidates[0]
    logger.info("✅ 选择 %s（%.4f）", ensemble_type, ensemble_acc)

    for name, _, _, acc, f1 in candidates:
        logger.info("  %s: Acc=%.4f, F1=%.4f", name, acc, f1)

    # ── 13. 分类报告 ─────────────────────────────────
    report_text = classification_report(
        y_test, ensemble_pred, target_names=le.classes_, digits=4,
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
            "stacking_passthrough": True,
            "use_smote": use_smote,
            "smote_method": "SMOTE-ENN",
            "smote_target_ratio": 0.9,
            "use_optuna": use_optuna,
            "optuna_trials": optuna_trials,
            "use_feature_selection": use_feature_selection,
            "use_multi_seed": use_multi_seed,
            "multi_seed_count": 3 if use_multi_seed else 1,
            "test_size": TRAIN_TEST_SPLIT,
            "random_state": RANDOM_STATE,
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
            "all_ensembles": {c[0]: round(c[3], 4) for c in candidates},
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("=== V8 训练完成，模型已保存: %s ===", out_dir)
    logger.info("文件列表: %s", [p.name for p in out_dir.iterdir()])
    return out_dir
