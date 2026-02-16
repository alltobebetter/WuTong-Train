# -*- coding: utf-8 -*-
"""特征提取模块 — 从原始告警记录中提取 25 维基础特征"""

import logging
import re
from urllib.parse import unquote

import pandas as pd

logger = logging.getLogger(__name__)

# ── 攻击特征正则 ─────────────────────────────────────
ATTACK_PATTERNS: dict[str, list[str]] = {
    "sql_injection": [
        r"(?i)(union\s+select|select\s+.*\s+from|insert\s+into|delete\s+from|drop\s+table)",
        r"(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1|'\s*or\s*'|'\s*=\s*')",
        r"(?i)(exec\s*\(|execute\s*\(|sp_executesql)",
        r"(?i)(%27|%22|--|%23|#)",
        r"(?i)(benchmark\s*\(|sleep\s*\(|waitfor\s+delay)",
    ],
    "xss": [
        r"(?i)(<script|</script|javascript:|onerror\s*=|onload\s*=)",
        r"(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()",
        r"(?i)(document\.cookie|document\.location|document\.write)",
        r"(?i)(<img[^>]+onerror|<svg[^>]+onload)",
        r"(?i)(%3cscript|%3c/script)",
    ],
    "path_traversal": [
        r"(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|\.\.%2f)",
        r"(?i)(/etc/passwd|/etc/shadow|/windows/system32)",
        r"(?i)(\.ssh/|id_rsa|authorized_keys)",
        r"(?i)(boot\.ini|win\.ini|system\.ini)",
    ],
    "command_injection": [
        r"(;|\||\$\(|`|&&|\|\|)",
        r"(?i)(ping\s+-|wget\s+|curl\s+|nc\s+-|bash\s+-)",
        r"(?i)(/bin/sh|/bin/bash|cmd\.exe|powershell)",
        r"(?i)(whoami|id\s|uname|cat\s+/etc)",
    ],
    "file_inclusion": [
        r"(?i)(include\s*=|require\s*=|file\s*=)",
        r"(?i)(php://|data://|expect://|zip://)",
        r"(?i)(\?file=|\?page=|\?path=|\?include=)",
    ],
    "file_upload": [
        r"(?i)(\.php|\.jsp|\.asp|\.aspx|\.exe|\.sh)",
        r"(?i)(multipart/form-data)",
        r"(?i)(filename=.*\.(php|jsp|asp|exe|sh))",
    ],
    "java_deserialization": [
        r"(rO0AB|aced0005)",
        r"(?i)(java\.lang\.Runtime|java\.lang\.ProcessBuilder)",
        r"(?i)(ObjectInputStream|readObject)",
    ],
    "csrf": [
        r"(?i)(transfer|withdraw|delete|update).*amount",
        r"(?i)(action=|method=post).*token",
    ],
}

SENSITIVE_KEYWORDS = [
    "admin", "root", "password", "passwd", "login", "shell", "cmd", "exec",
]


def _decode_url(text) -> str:
    if pd.isna(text) or text == "":
        return ""
    try:
        decoded = str(text)
        for _ in range(3):
            new = unquote(decoded)
            if new == decoded:
                break
            decoded = new
        return decoded.lower()
    except Exception:
        return str(text).lower()


def _count_pattern_matches(text: str, patterns: list[str]) -> int:
    count = 0
    for pat in patterns:
        try:
            if re.search(pat, text, re.IGNORECASE):
                count += 1
        except re.error:
            pass
    return count


def _extract_row_features(row) -> dict:
    """从单条记录提取 25 维特征"""
    f: dict = {}

    url_decoded = _decode_url(row.get("url_path", ""))
    body_decoded = _decode_url(row.get("request_body", ""))
    combined = url_decoded + " " + body_decoded

    # 基础
    f["url_length"] = len(str(row.get("url_path", "")))
    f["body_length"] = len(str(row.get("request_body", ""))) if row.get("request_body") else 0
    f["has_body"] = 1 if row.get("request_body") else 0

    # HTTP 方法
    method = str(row.get("method", "")).upper()
    f["method_get"] = 1 if method == "GET" else 0
    f["method_post"] = 1 if method == "POST" else 0
    f["method_put"] = 1 if method == "PUT" else 0
    f["method_delete"] = 1 if method == "DELETE" else 0

    # URL 结构
    f["url_param_count"] = url_decoded.count("=")
    f["url_depth"] = url_decoded.count("/")
    f["url_special_chars"] = len(re.findall(r"[<>\"';(){}\[\]]", url_decoded))

    # 攻击模式匹配
    for attack_type, patterns in ATTACK_PATTERNS.items():
        f[f"pattern_{attack_type}"] = _count_pattern_matches(combined, patterns)

    # 编码
    f["url_encoding_count"] = str(row.get("url_path", "")).count("%")
    f["double_encoding"] = 1 if "%25" in str(row.get("url_path", "")) else 0

    # 敏感词
    f["sensitive_keyword_count"] = sum(1 for kw in SENSITIVE_KEYWORDS if kw in combined)

    # User-Agent
    ua = str(row.get("user_agent", "")).lower()
    f["ua_is_bot"] = 1 if any(b in ua for b in ["bot", "spider", "crawler", "curl", "wget"]) else 0
    f["ua_is_mobile"] = 1 if any(m in ua for m in ["mobile", "android", "iphone"]) else 0

    # 时间
    ts = row.get("timestamp")
    if pd.notna(ts):
        try:
            ts = pd.Timestamp(ts)
            f["is_night"] = 1 if ts.hour < 6 or ts.hour > 22 else 0
            f["is_weekend"] = 1 if ts.weekday() >= 5 else 0
        except Exception:
            f["is_night"] = 0
            f["is_weekend"] = 0
    else:
        f["is_night"] = 0
        f["is_weekend"] = 0

    return f


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    批量提取特征，返回带有新特征列的 DataFrame。

    Parameters
    ----------
    df : 原始 / 增强后的 DataFrame（需包含 url_path, request_body, method 等列）

    Returns
    -------
    pd.DataFrame : 原始列 + 25 个特征列
    """
    total = len(df)
    logger.info("开始特征提取，共 %d 条记录 ...", total)

    features_list = []
    for idx, (_, row) in enumerate(df.iterrows()):
        features_list.append(_extract_row_features(row))
        if (idx + 1) % 2000 == 0:
            logger.info("  特征提取进度: %d / %d", idx + 1, total)

    features_df = pd.DataFrame(features_list, index=df.index)

    # 合并到原始 DataFrame
    for col in features_df.columns:
        df[col] = features_df[col]

    logger.info("特征提取完成: %d 个特征", len(features_df.columns))
    return df
