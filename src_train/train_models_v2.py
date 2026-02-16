# -*- coding: utf-8 -*-
"""
训练流程 V2：使用 XGBoost + CatBoost + LightGBM 提升准确率
目标准确率：98.5-99.5%
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

# Advanced ML models
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier

# 使用 wutong 包中的模块
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wutong.config import FEATURE_COLS, MODEL_DIR, RANDOM_STATE, TRAIN_TEST_SPLIT
from wutong.features import extract_features
from wutong.denoise import AlertAggregator

logger = logging.getLogger(__name__)


def train_with_cv(
    X_train, y_train, X_test, y_test, 
    model_name: str, model, 
    n_splits: int = 5
):
    """
    使用交叉验证训练模型并评估
    
    Parameters
    ----------
    X_train, y_train : 训练数据
    X_test, y_test : 测试数据
    model_name : 模型名称
    model : 模型实例
    n_splits : 交叉验证折数
    
    Returns
    -------
    dict : 包含模型、准确率、F1分数等指标
    """
    logger.info(f"--- 训练 {model_name} ---")
    
    # 交叉验证
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # 训练
        if isinstance(model, cb.CatBoostClassifier):
            model.fit(X_fold_train, y_fold_train, verbose=False)
        else:
            model.fit(X_fold_train, y_fold_train)
        
        # 验证
        val_pred = model.predict(X_fold_val)
        val_acc = accuracy_score(y_fold_val, val_pred)
        cv_scores.append(val_acc)
        logger.info(f"  Fold {fold}: {val_acc:.4f}")
    
    avg_cv_score = np.mean(cv_scores)
    logger.info(f"  平均 CV 准确率: {avg_cv_score:.4f} (±{np.std(cv_scores):.4f})")
    
    # 在全部训练集上重新训练
    if isinstance(model, cb.CatBoostClassifier):
        model.fit(X_train, y_train, verbose=False)
    else:
        model.fit(X_train, y_train)
    
    # 测试集评估
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')
    
    logger.info(f"  测试集准确率: {test_acc:.4f}")
    logger.info(f"  测试集 F1: {test_f1:.4f}")
    
    return {
        "model": model,
        "cv_accuracy": avg_cv_score,
        "cv_std": np.std(cv_scores),
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "predictions": test_pred,
    }


def train(
    data_path: str | Path,
    version: str = "v2.0.0",
    use_cv: bool = True,
    n_cv_splits: int = 5,
) -> Path:
    """
    完整训练流程 V2
    
    Parameters
    ----------
    data_path : 标准化 parquet 文件路径
    version : 模型版本号
    use_cv : 是否使用交叉验证
    n_cv_splits : 交叉验证折数
    
    Returns
    -------
    Path : 模型输出目录
    """
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== 训练开始 V2: version=%s ===", version)
    
    # ── 加载数据 ──────────────────────────────────────
    df = pd.read_parquet(data_path)
    logger.info("数据加载: %d 条", len(df))
    logger.info("攻击类型分布:\n%s", df["attack_type"].value_counts())
    
    # ── 特征提取 ──────────────────────────────────────
    logger.info("特征提取 ...")
    df = extract_features(df)
    
    # ── 告警聚合器训练 ────────────────────────────────
    logger.info("训练告警聚合器 ...")
    aggregator = AlertAggregator()
    cluster_labels = aggregator.fit_transform(df)
    df["cluster_id"] = cluster_labels
    aggregator.save(out_dir / "aggregator.pkl")
    
    # ── 准备特征矩阵 ─────────────────────────────────
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_features].fillna(0)
    logger.info("特征数量: %d", len(available_features))
    
    # ── 多分类（攻击类型）────────────────────────────
    logger.info("\n=== 多分类训练 ===")
    le = LabelEncoder()
    y_multi = le.fit_transform(df["attack_type"])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, 
        test_size=TRAIN_TEST_SPLIT, 
        random_state=RANDOM_STATE, 
        stratify=y_multi,
    )
    
    # 转换为 pandas Series 以便交叉验证
    y_train = pd.Series(y_train, index=X_train.index)
    y_test = pd.Series(y_test, index=X_test.index)
    
    # ── 模型 1: XGBoost ───────────────────────────────
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
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
    
    # ── 模型 2: CatBoost ──────────────────────────────
    cat_model = cb.CatBoostClassifier(
        iterations=200,
        depth=8,
        learning_rate=0.1,
        l2_leaf_reg=3,
        random_seed=RANDOM_STATE,
        verbose=False,
        thread_count=-1,
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
    
    # ── 模型 3: LightGBM ──────────────────────────────
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
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
    
    # ── 集成模型（Soft Voting）──────────────────────
    logger.info("\n=== 集成模型训练 ===")
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_result["model"]),
            ("cat", cat_result["model"]),
            ("lgb", lgb_result["model"]),
        ],
        voting="soft",
        weights=[1, 1, 1],  # 可以根据单模型表现调整权重
    )
    
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred, average='weighted')
    
    logger.info(f"集成模型准确率: {ensemble_acc:.4f}")
    logger.info(f"集成模型 F1: {ensemble_f1:.4f}")
    
    # ── 分类报告 ──────────────────────────────────────
    report_text = classification_report(
        y_test, ensemble_pred, 
        target_names=le.classes_,
        digits=4
    )
    logger.info("分类报告:\n%s", report_text)
    
    # ── 混淆矩阵 ──────────────────────────────────────
    cm = confusion_matrix(y_test, ensemble_pred)
    logger.info("混淆矩阵:\n%s", cm)
    
    # ── 保存模型工件 ──────────────────────────────────
    with open(out_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    
    # 单独保存各个模型（可选）
    with open(out_dir / "xgboost.pkl", "wb") as f:
        pickle.dump(xgb_result["model"], f)
    with open(out_dir / "catboost.pkl", "wb") as f:
        pickle.dump(cat_result["model"], f)
    with open(out_dir / "lightgbm.pkl", "wb") as f:
        pickle.dump(lgb_result["model"], f)
    
    # ── 生成 manifest ─────────────────────────────────
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_rows": len(df),
        "feature_list": available_features,
        "classes": list(le.classes_),
        "training_config": {
            "use_cv": use_cv,
            "n_cv_splits": n_cv_splits if use_cv else None,
            "test_size": TRAIN_TEST_SPLIT,
            "random_state": RANDOM_STATE,
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
            },
        },
    }
    
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    # 特征列表单独保存
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(available_features, f, ensure_ascii=False, indent=2)
    
    # 保存分类报告
    with open(out_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info("=== 训练完成，模型已保存: %s ===", out_dir)
    logger.info("文件列表: %s", [p.name for p in out_dir.iterdir()])
    
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    from wutong.config import STAGING_DIR
    
    # 查找 staging 中的 parquet
    parquets = list(STAGING_DIR.glob("*.parquet"))
    if not parquets:
        print("请先执行 ingest 导入数据")
        raise SystemExit(1)
    
    train(parquets[0], version="v2.0.0", use_cv=True, n_cv_splits=5)
