# -*- coding: utf-8 -*-
"""告警聚合降噪模块 — 基于 TF-IDF + DBSCAN"""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class AlertAggregator:
    """
    告警聚合器：基于 TF-IDF 向量化 + DBSCAN 聚类实现相似告警自动聚合。

    Parameters
    ----------
    eps : DBSCAN 邻域半径（越小聚类越紧密）
    min_samples : 最小样本数
    """

    def __init__(self, eps: float = 0.3, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.cluster_labels = None
        self._fitted = False

    # ── 内部 ──────────────────────────────────────────

    def _prepare_text(self, df: pd.DataFrame) -> list[str]:
        url_col = "url_path" if "url_path" in df.columns else "url"
        body_col = "request_body" if "request_body" in df.columns else "body"
        attack_col = "attack_type" if "attack_type" in df.columns else None

        texts = []
        for _, row in df.iterrows():
            parts = [str(row.get(url_col, "")), str(row.get(body_col, ""))]
            if attack_col and attack_col in row:
                parts.append(str(row[attack_col]))
            texts.append(" ".join(parts))
        return texts

    # ── 公开 API ──────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame):
        """执行告警聚合，返回聚类标签数组"""
        logger.info("开始告警聚合 ...")
        texts = self._prepare_text(df)
        logger.info("  处理 %d 条告警", len(texts))

        tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info("  TF-IDF 特征维度: %s", tfidf_matrix.shape)

        clustering = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, metric="cosine"
        )
        self.cluster_labels = clustering.fit_predict(tfidf_matrix)
        self._fitted = True

        n_clusters = len(set(self.cluster_labels)) - (
            1 if -1 in self.cluster_labels else 0
        )
        n_noise = list(self.cluster_labels).count(-1)
        logger.info("  聚类数量: %d", n_clusters)
        logger.info("  噪声点（独立告警）: %d", n_noise)

        return self.cluster_labels

    def get_aggregation_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cluster_labels is None:
            raise ValueError("请先执行 fit_transform")

        tmp = df.copy()
        tmp["cluster_id"] = self.cluster_labels

        stats = []
        for cid in set(self.cluster_labels):
            if cid == -1:
                continue
            grp = tmp[tmp["cluster_id"] == cid]
            stats.append(
                {
                    "cluster_id": cid,
                    "count": len(grp),
                    "attack_types": grp["attack_type"].unique().tolist()
                    if "attack_type" in grp.columns
                    else [],
                    "src_ips": grp["src_ip"].nunique()
                    if "src_ip" in grp.columns
                    else 0,
                }
            )
        return pd.DataFrame(stats)

    # ── 持久化 ────────────────────────────────────────

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("聚合器已保存: %s", path)

    @classmethod
    def load(cls, path: Path) -> "AlertAggregator":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("聚合器已加载: %s", path)
        return obj
