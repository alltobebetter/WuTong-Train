# -*- coding: utf-8 -*-
"""全局配置"""

from pathlib import Path

# ── 目录 ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
STAGING_DIR = PROJECT_ROOT / "data" / "staging"
MODEL_DIR = PROJECT_ROOT / "models"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

for _d in (RAW_DIR, STAGING_DIR, MODEL_DIR, EXTERNAL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── 训练参数 ──────────────────────────────────────────
RANDOM_STATE = 42
TRAIN_TEST_SPLIT = 0.2

# ── 基础特征列（25 维） ──────────────────────────────
FEATURE_COLS = [
    # 基础
    "url_length", "body_length", "has_body",
    # HTTP 方法
    "method_get", "method_post", "method_put", "method_delete",
    # URL 结构
    "url_param_count", "url_depth", "url_special_chars",
    # 攻击模式匹配
    "pattern_sql_injection", "pattern_xss", "pattern_path_traversal",
    "pattern_command_injection", "pattern_file_inclusion",
    "pattern_file_upload", "pattern_java_deserialization", "pattern_csrf",
    # 编码
    "url_encoding_count", "double_encoding",
    # 敏感词
    "sensitive_keyword_count",
    # User-Agent
    "ua_is_bot", "ua_is_mobile",
    # 时间
    "is_night", "is_weekend",
]
