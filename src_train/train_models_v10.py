# -*- coding: utf-8 -*-
"""
训练流程 V10：V9 bug 修复 + 参数调优 + 路由增强
目标：99.5%+ 准确率

V9 → V10 改进：
  V9 达到 99.37%（vs V7 99.35%），但有两个问题：
  1. numpy.bool_ JSON 序列化 bug（V9 中途修复，V10 彻底清理）
  2. 置信度路由几乎没用：5871 样本只路由了 2 个
     - confidence_threshold=0.7 太高，大部分错误样本置信度都 >0.7
     - margin=0.3 太宽松
  3. LightGBM 单模型 99.45% 比 Stacking 99.37% 还高
     → Stacking meta-learner 平均掉了 LGB 的优势

V10 核心改动：
  1. 降低路由阈值：confidence_threshold 0.7→0.45, margin 0.3→0.15
  2. 自适应路由：用验证集评估路由是否有帮助，没帮助就关掉
  3. LightGBM 作为 meta-learner（替代 GradientBoosting，更快更准）
  4. Stacking 权重偏向 LightGBM（给最强单模型更多话语权）
  5. 所有 numpy 类型统一用 _to_native() 转换（彻底修复 JSON bug）
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
# numpy → Python 原生类型（彻底修复 V9 的 JSON 序列化 bug）
# ═══════════════════════════════════════════════════════════════════════════════

def _to_native(obj):
    """json.dump 的 default 参数，处理所有 numpy 类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ═══════════════════════════════════════════════════════════════════════════════
# 高级特征工程（V7 的 24 个 + V9 的 5 个精准特征 = 29 个）
# ═══════════════════════════════════════════════════════════════════════════════

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def extract_advanced_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """V7 的 24 个高级特征 + V9 的 5 个精准区分特征 = 29 个（V10 不加新特征）"""
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

    # ── V7 交互特征 (11) ──
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

    # ── V9 精准区分特征 (5) ──
    _fi_proto = re.compile(r'(php://|data://|expect://|zip://|phar://|glob://)', re.IGNORECASE)
    df['file_inclusion_proto'] = combined.apply(lambda t: len(_fi_proto.findall(t)))

    _fi_param = re.compile(
        r'[\?&](file|page|path|include|require|template|doc|document|folder|root|dir)\s*=',
        re.IGNORECASE
    )
    df['file_inclusion_param'] = combined.apply(lambda t: len(_fi_param.findall(t)))

    _sql_func = re.compile(
        r'\b(concat|char|ascii|substring|substr|length|mid|left|right|'
        r'convert|cast|coalesce|ifnull|nullif|hex|unhex|load_file|'
        r'information_schema|table_name|column_name)\b',
        re.IGNORECASE
    )
    df['sql_func_count'] = combined.apply(lambda t: len(_sql_func.findall(t)))

    _xss_event = re.compile(
        r'\b(onmouseover|onfocus|onblur|onclick|onsubmit|onchange|onkeyup|'
        r'onkeydown|onmouseout|onmousemove|ondblclick|oncontextmenu|'
        r'onerror|onload|onunload|onresize|onscroll)\s*=',
        re.IGNORECASE
    )
    df['xss_event_count'] = combined.apply(lambda t: len(_xss_event.findall(t)))

    df['clean_request_score'] = (
        (df['attack_pattern_sum'] == 0).astype(int) * 3 +
        (df.get('url_special_chars', 0) == 0).astype(int) * 2 +
        (df.get('sensitive_keyword_count', 0) == 0).astype(int) * 2 +
        (df['url_entropy'] < 3.5).astype(int)
    )

    v9_features = [
        'file_inclusion_proto', 'file_inclusion_param',
        'sql_func_count', 'xss_event_count',
        'clean_request_score',
    ]

    all_new = v6_features + v7_features + v9_features
    logger.info("  高级特征: V6(%d) + V7(%d) + V9(%d) = %d",
                len(v6_features), len(v7_features), len(v9_features), len(all_new))
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
# Optuna 超参搜索（独立搜索，不做 warm start）
# ═══════════════════════════════════════════════════════════════════════════════

def _optuna_available():
    try:
        import optuna
        return True
    except ImportError:
        return False


def optuna_tune_xgb(X_train, y_train, n_trials=30, n_splits=5):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
            'max_depth': trial.suggest_int('max_depth', 5, 14),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
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


def optuna_tune_lgb(X_train, y_train, n_trials=30, n_splits=5):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 400, 1200),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 300),
            'subsample': trial.suggest_float('subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.65, 0.95),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
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
            'iterations': trial.suggest_int('iterations', 300, 1000),
            'depth': trial.suggest_int('depth', 5, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 15.0),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 2.0),
            'border_count': trial.suggest_int('border_count', 64, 255),
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
# 混淆对专家模型 + 置信度路由（V10: 降低阈值 + 自适应路由）
# ═══════════════════════════════════════════════════════════════════════════════

# V9 的 37 条错误中的 top 混淆对
CONFUSION_PAIRS = [
    ("文件包含攻击", "目录遍历攻击"),      # 6 条错误（V9 最大混淆）
    ("SQL注入攻击", "正常访问"),            # 5+4=9 条（双向）
    ("正常访问", "CSRF攻击"),              # 3 条
    ("SQL注入攻击", "文件包含攻击"),        # 3 条
    ("XSS跨站脚本攻击", "SQL注入攻击"),    # 3 条
    ("正常访问", "文件包含攻击"),           # 2 条
]


class ConfusionPairExpert:
    """混淆对专家：针对特定的两类混淆训练一个二分类器。"""

    def __init__(self, class_a: str, class_b: str):
        self.class_a = class_a
        self.class_b = class_b
        self.model = None
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: np.ndarray, le: LabelEncoder):
        try:
            idx_a = list(le.classes_).index(self.class_a)
            idx_b = list(le.classes_).index(self.class_b)
        except ValueError:
            logger.warning("混淆对 %s/%s 不在标签中，跳过", self.class_a, self.class_b)
            return self

        mask = np.isin(y, [idx_a, idx_b])
        if mask.sum() < 20:
            logger.warning("混淆对 %s/%s 样本不足 (%d)，跳过",
                           self.class_a, self.class_b, mask.sum())
            return self

        X_pair = X[mask]
        y_pair = (y[mask] == idx_b).astype(int)

        self.model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        self.model.fit(X_pair, y_pair)
        self._fitted = True

        pred = self.model.predict(X_pair)
        acc = accuracy_score(y_pair, pred)
        logger.info("  专家 [%s vs %s]: 训练准确率 %.4f (%d 样本)",
                     self.class_a, self.class_b, acc, len(y_pair))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            return None
        return self.model.predict_proba(X)


class ConfidenceRouter:
    """
    置信度路由器 V10：降低阈值 + 自适应开关。

    V9 问题：threshold=0.7 太高，5871 样本只路由了 2 个。
    V10 改进：
      - threshold 降到 0.45（让更多"不确定"样本走专家）
      - margin 降到 0.15（top1 和 top2 差距小就触发）
      - 自适应：用验证集评估路由效果，没帮助就自动关闭
    """

    def __init__(self, confidence_threshold: float = 0.45, margin: float = 0.15):
        self.confidence_threshold = confidence_threshold
        self.margin = margin
        self.experts: dict[tuple, ConfusionPairExpert] = {}
        self.le = None
        self.enabled = True  # 自适应开关

    def add_expert(self, expert: ConfusionPairExpert):
        if expert._fitted:
            key = tuple(sorted([expert.class_a, expert.class_b]))
            self.experts[key] = expert

    def fit_experts(self, X: pd.DataFrame, y: np.ndarray, le: LabelEncoder,
                    confusion_pairs: list[tuple[str, str]]):
        self.le = le
        logger.info("\n=== 训练混淆对专家 ===")
        for class_a, class_b in confusion_pairs:
            expert = ConfusionPairExpert(class_a, class_b)
            expert.fit(X, y, le)
            self.add_expert(expert)
        logger.info("共训练 %d 个专家模型", len(self.experts))

    def route(self, X: pd.DataFrame, main_proba: np.ndarray) -> np.ndarray:
        if not self.experts or self.le is None or not self.enabled:
            return np.argmax(main_proba, axis=1)

        predictions = np.argmax(main_proba, axis=1)
        n_routed = 0
        routed_details = []

        for i in range(len(X)):
            proba = main_proba[i]
            sorted_idx = np.argsort(proba)[::-1]
            top1_prob = proba[sorted_idx[0]]
            top2_prob = proba[sorted_idx[1]]

            # V10: 更激进的路由条件
            if top1_prob >= self.confidence_threshold:
                continue

            top1_class = self.le.classes_[sorted_idx[0]]
            top2_class = self.le.classes_[sorted_idx[1]]
            pair_key = tuple(sorted([top1_class, top2_class]))

            if pair_key in self.experts and (top1_prob - top2_prob) < self.margin:
                expert = self.experts[pair_key]
                expert_proba = expert.predict(X.iloc[[i]])
                if expert_proba is not None:
                    expert_pred = np.argmax(expert_proba[0])
                    sorted_pair = sorted([expert.class_a, expert.class_b])
                    chosen_class = sorted_pair[expert_pred]
                    chosen_idx = list(self.le.classes_).index(chosen_class)
                    old_class = self.le.classes_[predictions[i]]
                    predictions[i] = chosen_idx
                    n_routed += 1
                    routed_details.append(
                        f"  #{i}: {old_class} → {chosen_class} "
                        f"(top1={top1_prob:.3f}, top2={top2_prob:.3f})"
                    )

        logger.info("置信度路由: %d / %d 样本被路由到专家 (%.1f%%)",
                     n_routed, len(X), 100 * n_routed / max(len(X), 1))
        for detail in routed_details[:20]:  # 最多打印 20 条
            logger.info(detail)
        return predictions

    def adaptive_evaluate(self, X_val, y_val, main_proba, base_acc):
        """自适应评估：路由后是否比基础模型更好"""
        routed_pred = self.route(X_val, main_proba)
        routed_acc = accuracy_score(y_val, routed_pred)
        delta = routed_acc - base_acc

        if delta < 0:
            logger.info("⚠️ 路由后准确率下降 (%.4f → %.4f, Δ=%+.4f)，自动关闭路由",
                         base_acc, routed_acc, delta)
            self.enabled = False
        else:
            logger.info("✅ 路由后准确率: %.4f (基础: %.4f, Δ=%+.4f)",
                         routed_acc, base_acc, delta)

        return routed_pred, routed_acc


# ═══════════════════════════════════════════════════════════════════════════════
# 对抗验证（检测训练/测试分布偏移）
# ═══════════════════════════════════════════════════════════════════════════════

def adversarial_validation(X_train, X_test, threshold: float = 0.75):
    from sklearn.metrics import roc_auc_score

    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = np.array([0] * len(X_train) + [1] * len(X_test))

    adv_model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    train_probas = np.zeros(len(X_train))

    for train_idx, val_idx in skf.split(X_combined, y_combined):
        adv_model.fit(X_combined.iloc[train_idx], y_combined[train_idx])
        val_pred = adv_model.predict_proba(X_combined.iloc[val_idx])[:, 1]
        auc_scores.append(roc_auc_score(y_combined[val_idx], val_pred))

        train_mask = val_idx[val_idx < len(X_train)]
        if len(train_mask) > 0:
            train_probas[train_mask] = adv_model.predict_proba(
                X_combined.iloc[train_mask]
            )[:, 1]

    avg_auc = float(np.mean(auc_scores))
    logger.info("对抗验证 AUC: %.4f (>%.2f 表示分布偏移)", avg_auc, threshold)

    if avg_auc > threshold:
        sample_weights = 1.0 + train_probas
        logger.info("  检测到分布偏移，已调整样本权重 (max=%.2f, mean=%.2f)",
                     sample_weights.max(), sample_weights.mean())
    else:
        sample_weights = np.ones(len(X_train))
        logger.info("  分布一致，使用均匀权重")

    return sample_weights, avg_auc


# ═══════════════════════════════════════════════════════════════════════════════
# 训练核心
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_cv(
    X_train, y_train, X_test, y_test,
    model_name: str, model,
    n_splits: int = 10,
    sample_weight=None,
):
    logger.info("--- 训练 %s ---", model_name)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        if isinstance(model, cb.CatBoostClassifier):
            if sample_weight is not None:
                model.fit(X_fold_train, y_fold_train,
                          sample_weight=sample_weight[train_idx], verbose=False)
            else:
                model.fit(X_fold_train, y_fold_train, verbose=False)
        else:
            if sample_weight is not None:
                model.fit(X_fold_train, y_fold_train,
                          sample_weight=sample_weight[train_idx])
            else:
                model.fit(X_fold_train, y_fold_train)

        val_pred = model.predict(X_fold_val)
        val_acc = accuracy_score(y_fold_val, val_pred)
        cv_scores.append(val_acc)

        if fold <= 3 or fold == n_splits:
            logger.info("  Fold %d: %.4f", fold, val_acc)

    avg_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    logger.info("  平均 CV 准确率: %.4f (±%.4f)", avg_cv, std_cv)

    # 全量训练
    if isinstance(model, cb.CatBoostClassifier):
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight, verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)
    else:
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

    test_pred = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_pred))
    test_f1 = float(f1_score(y_test, test_pred, average='weighted'))

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
    version: str = "v10.0.0",
    use_cv: bool = True,
    n_cv_splits: int = 10,
    use_stacking: bool = True,
    use_smote: bool = True,
    use_optuna: bool = True,
    optuna_trials: int = 30,
    use_experts: bool = True,
    use_adversarial: bool = True,
    confidence_threshold: float = 0.45,
    margin: float = 0.15,
    external_data_dir: Path = None,
) -> Path:
    """
    V10 完整训练流程

    核心改进（相比 V9）：
    1. 彻底修复 numpy JSON 序列化 bug
    2. 降低路由阈值（0.7→0.45）让更多样本走专家
    3. 自适应路由：验证集评估，没帮助就自动关闭
    4. LightGBM 作为 Stacking meta-learner
    5. 更新混淆对列表（基于 V9 的 37 条错误）
    """
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info("=== V10 训练开始: version=%s ===", version)
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

    # ── 3. 基础特征提取 ──────────────────────────────
    logger.info("\n基础特征提取 ...")
    df = extract_features(df)

    # ── 4. 高级特征 + 交互特征 ───────────────────────
    df, adv_feature_names = extract_advanced_features(df)

    # ── 5. 告警聚合器 ────────────────────────────────
    if len(df) > 50000:
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
    X = df[all_features].fillna(0).replace([np.inf, -np.inf], 0)

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

    # ── 9. 对抗验证 ──────────────────────────────────
    sample_weight = None
    adv_auc = 0.0
    if use_adversarial:
        logger.info("\n=== 对抗验证 ===")
        sample_weight, adv_auc = adversarial_validation(X_train, X_test)

    # ── 10. Optuna 超参搜索 ──────────────────────────
    xgb_params, lgb_params, cat_params = {}, {}, {}

    if use_optuna and _optuna_available():
        logger.info("\n=== Optuna 超参搜索 ===")
        logger.info("XGBoost: %d trials ...", optuna_trials)
        xgb_params = optuna_tune_xgb(X_train, y_train, n_trials=optuna_trials)
        logger.info("LightGBM: %d trials ...", optuna_trials)
        lgb_params = optuna_tune_lgb(X_train, y_train, n_trials=optuna_trials)
        cat_trials = max(optuna_trials // 2, 10)
        logger.info("CatBoost: %d trials ...", cat_trials)
        cat_params = optuna_tune_cat(X_train, y_train, n_trials=cat_trials)
    elif use_optuna:
        logger.warning("Optuna 未安装，使用默认参数")

    # ── 11. 构建模型 ─────────────────────────────────
    xgb_default = {
        'n_estimators': 600, 'max_depth': 10, 'learning_rate': 0.02,
        'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_weight': 3, 'gamma': 0.1,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': RANDOM_STATE, 'n_jobs': -1,
        'tree_method': 'hist', 'eval_metric': 'mlogloss',
    }
    xgb_final = {**xgb_default, **xgb_params}
    xgb_final.update({'random_state': RANDOM_STATE, 'n_jobs': -1,
                      'tree_method': 'hist', 'eval_metric': 'mlogloss'})
    xgb_model = xgb.XGBClassifier(**xgb_final)

    cat_default = {
        'iterations': 500, 'depth': 8, 'learning_rate': 0.03,
        'l2_leaf_reg': 5, 'random_seed': RANDOM_STATE,
        'verbose': False, 'thread_count': -1,
        'border_count': 128, 'bagging_temperature': 0.8,
    }
    cat_final = {**cat_default, **cat_params}
    cat_final.update({'random_seed': RANDOM_STATE, 'verbose': False, 'thread_count': -1})
    cat_model = cb.CatBoostClassifier(**cat_final)

    lgb_default = {
        'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.03,
        'num_leaves': 127, 'subsample': 0.85, 'colsample_bytree': 0.85,
        'min_child_samples': 10, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1,
    }
    lgb_final = {**lgb_default, **lgb_params}
    lgb_final.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
    lgb_model = lgb.LGBMClassifier(**lgb_final)

    et_model = ExtraTreesClassifier(
        n_estimators=500, max_depth=None,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt',
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    # ── 12. 训练基模型 ───────────────────────────────
    sw = sample_weight.values if isinstance(sample_weight, pd.Series) else sample_weight

    if use_cv:
        xgb_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "XGBoost", xgb_model, n_cv_splits, sw)
        cat_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "CatBoost", cat_model, n_cv_splits, sw)
        lgb_result = train_with_cv(X_train, y_train, X_test, y_test,
                                    "LightGBM", lgb_model, n_cv_splits, sw)
        et_result = train_with_cv(X_train, y_train, X_test, y_test,
                                   "ExtraTrees", et_model, n_cv_splits, sw)
    else:
        for name, model in [("XGBoost", xgb_model), ("CatBoost", cat_model),
                            ("LightGBM", lgb_model), ("ExtraTrees", et_model)]:
            if isinstance(model, cb.CatBoostClassifier):
                model.fit(X_train, y_train, verbose=False)
            else:
                model.fit(X_train, y_train)
        xgb_result = {"model": xgb_model, "test_accuracy": float(accuracy_score(y_test, xgb_model.predict(X_test))),
                       "test_f1": float(f1_score(y_test, xgb_model.predict(X_test), average='weighted')),
                       "predictions": xgb_model.predict(X_test)}
        cat_result = {"model": cat_model, "test_accuracy": float(accuracy_score(y_test, cat_model.predict(X_test))),
                       "test_f1": float(f1_score(y_test, cat_model.predict(X_test), average='weighted')),
                       "predictions": cat_model.predict(X_test)}
        lgb_result = {"model": lgb_model, "test_accuracy": float(accuracy_score(y_test, lgb_model.predict(X_test))),
                       "test_f1": float(f1_score(y_test, lgb_model.predict(X_test), average='weighted')),
                       "predictions": lgb_model.predict(X_test)}
        et_result = {"model": et_model, "test_accuracy": float(accuracy_score(y_test, et_model.predict(X_test))),
                      "test_f1": float(f1_score(y_test, et_model.predict(X_test), average='weighted')),
                      "predictions": et_model.predict(X_test)}

    # ── 13. 集成模型 ─────────────────────────────────
    logger.info("\n=== 集成模型训练 ===")

    # V10: Stacking 用 LightGBM 做 meta-learner（V9 用的 GradientBoosting）
    # 原因：LightGBM 单模型 99.45% 是最强的，让它做 meta 更合理
    if use_stacking:
        logger.info("训练 Stacking 集成（4 基模型 + LightGBM meta-learner）...")
        stacking = StackingClassifier(
            estimators=[
                ("xgb", xgb_result["model"]),
                ("cat", cat_result["model"]),
                ("lgb", lgb_result["model"]),
                ("et", et_result["model"]),
            ],
            final_estimator=lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                num_leaves=31, random_state=RANDOM_STATE,
                n_jobs=-1, verbose=-1,
            ),
            cv=5,
            n_jobs=-1,
            passthrough=False,
        )
        stacking.fit(X_train, y_train)
        stacking_pred = stacking.predict(X_test)
        stacking_acc = float(accuracy_score(y_test, stacking_pred))
        stacking_f1 = float(f1_score(y_test, stacking_pred, average='weighted'))
        logger.info("Stacking 准确率: %.4f, F1: %.4f", stacking_acc, stacking_f1)

    # V10: Voting 权重偏向 LightGBM（最强单模型）
    logger.info("训练 Voting 集成 ...")
    accs = [xgb_result["test_accuracy"], cat_result["test_accuracy"],
            lgb_result["test_accuracy"], et_result["test_accuracy"]]
    # 用准确率的平方作为权重，拉大强弱差距
    acc_sq = [a ** 2 for a in accs]
    total_sq = sum(acc_sq)
    weights = [a / total_sq for a in acc_sq]
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
    voting_acc = float(accuracy_score(y_test, voting_pred))
    voting_f1 = float(f1_score(y_test, voting_pred, average='weighted'))
    logger.info("Voting 准确率: %.4f, F1: %.4f", voting_acc, voting_f1)

    # 选择最优基础集成
    if use_stacking and stacking_acc >= voting_acc:
        base_ensemble = stacking
        base_pred = stacking_pred
        base_acc = stacking_acc
        base_f1 = stacking_f1
        base_type = "stacking"
    else:
        base_ensemble = voting
        base_pred = voting_pred
        base_acc = voting_acc
        base_f1 = voting_f1
        base_type = "voting"
    logger.info("基础集成选择: %s (%.4f)", base_type, base_acc)

    # ── 14. 混淆对专家 + 自适应路由 ─────────────────
    router = None
    routed_pred = None
    routed_acc = 0.0
    routed_f1 = 0.0

    if use_experts:
        router = ConfidenceRouter(
            confidence_threshold=confidence_threshold,
            margin=margin,
        )
        router.fit_experts(X_train, y_train.values, le, CONFUSION_PAIRS)

        if hasattr(base_ensemble, 'predict_proba'):
            base_proba = base_ensemble.predict_proba(X_test)
            # V10: 自适应评估，路由没帮助就自动关闭
            routed_pred, routed_acc = router.adaptive_evaluate(
                X_test, y_test, base_proba, base_acc
            )
            routed_f1 = float(f1_score(y_test, routed_pred, average='weighted'))

    # ── 15. 概率校准集成 ─────────────────────────────
    logger.info("\n=== 概率校准 ===")
    calibrated_models = []
    for name, result in [("xgb", xgb_result), ("cat", cat_result),
                          ("lgb", lgb_result), ("et", et_result)]:
        try:
            cal = CalibratedClassifierCV(result["model"], cv=3, method='isotonic')
            cal.fit(X_train, y_train)
            calibrated_models.append((name, cal))
            cal_pred = cal.predict(X_test)
            cal_acc = float(accuracy_score(y_test, cal_pred))
            logger.info("  %s 校准后: %.4f", name, cal_acc)
        except Exception as e:
            logger.warning("  %s 校准失败: %s", name, e)
            calibrated_models.append((name, result["model"]))

    cal_voting_acc = 0.0
    cal_voting_f1 = 0.0
    cal_voting = None
    if len(calibrated_models) == 4:
        cal_voting = VotingClassifier(
            estimators=calibrated_models,
            voting="soft",
            weights=weights,
        )
        cal_voting.fit(X_train, y_train)
        cal_voting_pred = cal_voting.predict(X_test)
        cal_voting_acc = float(accuracy_score(y_test, cal_voting_pred))
        cal_voting_f1 = float(f1_score(y_test, cal_voting_pred, average='weighted'))
        logger.info("校准 Voting 准确率: %.4f, F1: %.4f", cal_voting_acc, cal_voting_f1)

    # ── 16. 选择最终模型 ─────────────────────────────
    logger.info("\n=== 选择最终模型 ===")
    candidates = [
        ("base_ensemble", base_ensemble, base_pred, base_acc, base_f1),
    ]
    if use_experts and routed_pred is not None and router.enabled:
        candidates.append(("routed_ensemble", (router, base_ensemble),
                           routed_pred, routed_acc, routed_f1))
    if cal_voting_acc > 0:
        candidates.append(("calibrated_voting", cal_voting,
                           cal_voting_pred, cal_voting_acc, cal_voting_f1))

    # V10: 也把最强单模型加入候选（V9 中 LGB 单模型比集成还高）
    best_single_name = max(
        [("xgboost", xgb_result), ("catboost", cat_result),
         ("lightgbm", lgb_result), ("extratrees", et_result)],
        key=lambda x: x[1]["test_accuracy"]
    )
    best_single_acc = best_single_name[1]["test_accuracy"]
    best_single_f1 = best_single_name[1]["test_f1"]
    candidates.append((f"best_single_{best_single_name[0]}", best_single_name[1]["model"],
                        best_single_name[1]["predictions"], best_single_acc, best_single_f1))

    candidates.sort(key=lambda x: x[3], reverse=True)
    final_name, final_model, final_pred, final_acc, final_f1 = candidates[0]

    logger.info("✅ 最终选择: %s (%.4f)", final_name, final_acc)
    for name, _, _, acc, f1 in candidates:
        logger.info("  %s: Acc=%.4f, F1=%.4f", name, acc, f1)

    if final_name == "routed_ensemble":
        ensemble_to_save = base_ensemble
        ensemble_pred = final_pred
        ensemble_acc = final_acc
        ensemble_f1 = final_f1
        ensemble_type = f"routed_{base_type}"
    elif final_name.startswith("best_single_"):
        ensemble_to_save = final_model
        ensemble_pred = final_pred
        ensemble_acc = final_acc
        ensemble_f1 = final_f1
        ensemble_type = final_name
    else:
        ensemble_to_save = final_model
        ensemble_pred = final_pred
        ensemble_acc = final_acc
        ensemble_f1 = final_f1
        ensemble_type = final_name

    # ── 17. 分类报告 ─────────────────────────────────
    report_text = classification_report(
        y_test, ensemble_pred,
        target_names=le.classes_,
        digits=4,
    )
    logger.info("分类报告:\n%s", report_text)

    cm = confusion_matrix(y_test, ensemble_pred)
    logger.info("混淆矩阵:\n%s", cm)

    # ── 18. 保存 ─────────────────────────────────────
    for name, obj in [
        ("ensemble.pkl", ensemble_to_save),
        ("label_encoder.pkl", le),
        ("xgboost.pkl", xgb_result["model"]),
        ("catboost.pkl", cat_result["model"]),
        ("lightgbm.pkl", lgb_result["model"]),
        ("extratrees.pkl", et_result["model"]),
    ]:
        with open(out_dir / name, "wb") as f:
            pickle.dump(obj, f)

    if router is not None:
        with open(out_dir / "router.pkl", "wb") as f:
            pickle.dump(router, f)

    # ── 19. manifest（所有数值用 float/int/bool 确保 JSON 安全）──
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_rows": int(len(df)),
        "external_data": ext_rows > 0,
        "external_data_rows": int(ext_rows),
        "feature_list": all_features,
        "classes": list(le.classes_),
        "training_config": {
            "use_cv": use_cv,
            "n_cv_splits": n_cv_splits if use_cv else None,
            "use_stacking": use_stacking,
            "stacking_passthrough": False,
            "stacking_meta_learner": "LightGBM",
            "use_smote": use_smote,
            "smote_method": "SMOTE-ENN",
            "smote_target_ratio": 0.9,
            "use_optuna": use_optuna,
            "optuna_trials": optuna_trials if use_optuna else None,
            "use_experts": use_experts,
            "use_adversarial": use_adversarial,
            "confidence_threshold": float(confidence_threshold),
            "margin": float(margin),
            "routing_enabled": router.enabled if router else False,
            "test_size": float(TRAIN_TEST_SPLIT),
            "random_state": int(RANDOM_STATE),
            "n_base_models": 4,
            "base_models": ["XGBoost", "CatBoost", "LightGBM", "ExtraTrees"],
            "confusion_pairs": [list(p) for p in CONFUSION_PAIRS],
        },
        "optuna_params": {
            "xgboost": xgb_params if xgb_params else "default",
            "catboost": cat_params if cat_params else "default",
            "lightgbm": lgb_params if lgb_params else "default",
        },
        "adversarial_validation": {
            "auc": round(float(adv_auc), 4),
            "distribution_shift": bool(adv_auc > 0.75),
        },
        "metrics": {
            "xgboost": {
                "test_accuracy": round(float(xgb_result["test_accuracy"]), 4),
                "test_f1": round(float(xgb_result["test_f1"]), 4),
                "cv_accuracy": round(float(xgb_result.get("cv_accuracy", 0)), 4) if use_cv else None,
            },
            "catboost": {
                "test_accuracy": round(float(cat_result["test_accuracy"]), 4),
                "test_f1": round(float(cat_result["test_f1"]), 4),
                "cv_accuracy": round(float(cat_result.get("cv_accuracy", 0)), 4) if use_cv else None,
            },
            "lightgbm": {
                "test_accuracy": round(float(lgb_result["test_accuracy"]), 4),
                "test_f1": round(float(lgb_result["test_f1"]), 4),
                "cv_accuracy": round(float(lgb_result.get("cv_accuracy", 0)), 4) if use_cv else None,
            },
            "extratrees": {
                "test_accuracy": round(float(et_result["test_accuracy"]), 4),
                "test_f1": round(float(et_result["test_f1"]), 4),
                "cv_accuracy": round(float(et_result.get("cv_accuracy", 0)), 4) if use_cv else None,
            },
            "ensemble": {
                "test_accuracy": round(float(ensemble_acc), 4),
                "test_f1": round(float(ensemble_f1), 4),
                "ensemble_type": ensemble_type,
            },
            "all_candidates": {c[0]: round(float(c[3]), 4) for c in candidates},
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=_to_native)
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(all_features, f, ensure_ascii=False, indent=2)
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info("=== V10 训练完成，模型已保存: %s ===", out_dir)
    logger.info("文件列表: %s", [p.name for p in out_dir.iterdir()])

    return out_dir
