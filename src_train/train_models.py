# -*- coding: utf-8 -*-
"""
训练流程：从标准化数据训练二分类 + 多分类 + 集成模型，导出到 models/<version>/
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 使用 wutong 包中的模块
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wutong.config import FEATURE_COLS, MODEL_DIR, RANDOM_STATE, TRAIN_TEST_SPLIT
from wutong.features import extract_features
from wutong.denoise import AlertAggregator

logger = logging.getLogger(__name__)


def train(
    data_path: str | Path,
    version: str = "v1.0.0",
) -> Path:
    """
    完整训练流程。

    Parameters
    ----------
    data_path : 标准化 parquet 文件路径
    version : 模型版本号

    Returns
    -------
    Path : 模型输出目录
    """
    out_dir = MODEL_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== 训练开始: version=%s ===", version)

    # ── 加载数据 ──────────────────────────────────────
    df = pd.read_parquet(data_path)
    logger.info("数据加载: %d 条", len(df))

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

    # ── 二分类（正常 vs 攻击）────────────────────────
    logger.info("--- 二分类训练 ---")
    y_binary = (df["attack_type"] != "正常访问").astype(int)

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
        X, y_binary, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_binary,
    )

    rf_binary = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_split=5, min_samples_leaf=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_binary.fit(X_train_b, y_train_b)
    rf_pred_b = rf_binary.predict(X_test_b)
    binary_acc = accuracy_score(y_test_b, rf_pred_b)
    logger.info("二分类准确率: %.4f", binary_acc)

    with open(out_dir / "rf_binary.pkl", "wb") as f:
        pickle.dump(rf_binary, f)

    # ── 多分类（攻击类型）────────────────────────────
    logger.info("--- 多分类训练 ---")
    le = LabelEncoder()
    y_multi = le.fit_transform(df["attack_type"])

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X, y_multi, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_multi,
    )

    rf_multi = RandomForestClassifier(
        n_estimators=150, max_depth=20, min_samples_split=3, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    rf_multi.fit(X_train_m, y_train_m)
    rf_acc_m = accuracy_score(y_test_m, rf_multi.predict(X_test_m))
    logger.info("RF 多分类准确率: %.4f", rf_acc_m)

    gb_multi = GradientBoostingClassifier(
        n_estimators=100, max_depth=10, learning_rate=0.1, random_state=RANDOM_STATE,
    )
    gb_multi.fit(X_train_m, y_train_m)
    gb_acc_m = accuracy_score(y_test_m, gb_multi.predict(X_test_m))
    logger.info("GB 多分类准确率: %.4f", gb_acc_m)

    # ── 集成模型 ──────────────────────────────────────
    logger.info("--- 集成模型训练 ---")
    ensemble = VotingClassifier(
        estimators=[("rf", rf_multi), ("gb", gb_multi)],
        voting="soft",
    )
    ensemble.fit(X_train_m, y_train_m)
    ensemble_pred = ensemble.predict(X_test_m)
    ensemble_acc = accuracy_score(y_test_m, ensemble_pred)
    logger.info("集成模型准确率: %.4f", ensemble_acc)

    report_text = classification_report(y_test_m, ensemble_pred, target_names=le.classes_)
    logger.info("分类报告:\n%s", report_text)

    # ── 保存模型工件 ──────────────────────────────────
    with open(out_dir / "ensemble.pkl", "wb") as f:
        pickle.dump(ensemble, f)
    with open(out_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    # ── 生成 manifest ─────────────────────────────────
    manifest = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(data_path),
        "data_rows": len(df),
        "feature_list": available_features,
        "classes": list(le.classes_),
        "metrics": {
            "binary_accuracy": round(binary_acc, 4),
            "rf_multiclass_accuracy": round(rf_acc_m, 4),
            "gb_multiclass_accuracy": round(gb_acc_m, 4),
            "ensemble_accuracy": round(ensemble_acc, 4),
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 特征列表单独保存
    with open(out_dir / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(available_features, f, ensure_ascii=False, indent=2)

    logger.info("=== 训练完成，模型已保存: %s ===", out_dir)
    logger.info("文件列表: %s", [p.name for p in out_dir.iterdir()])
    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from wutong.config import STAGING_DIR

    # 查找 staging 中的 parquet
    parquets = list(STAGING_DIR.glob("*.parquet"))
    if not parquets:
        print("请先执行 ingest 导入数据")
        raise SystemExit(1)

    train(parquets[0], version="v1.0.0")
