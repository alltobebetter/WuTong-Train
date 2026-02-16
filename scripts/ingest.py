#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据导入脚本 — 将原始 xlsx/csv 转换为标准化 parquet

用法:
    python scripts/ingest.py data/raw/原始告警信息样例数据集V1.2.xlsx
    python scripts/ingest.py              # 自动扫描 data/raw/
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wutong.config import RAW_DIR, STAGING_DIR

logger = logging.getLogger(__name__)

# 原始数据集中攻击类型列的可能名称
_ATTACK_COL_CANDIDATES = [
    "attack_type",
    "attack_type(结果列，需参赛选手判断输出)",
]


def ingest(source: Path, output_dir: Path = STAGING_DIR) -> Path:
    """读取 xlsx / csv，标准化列名，保存为 parquet。"""
    logger.info("读取 %s ...", source.name)

    if source.suffix in (".xlsx", ".xls"):
        df = pd.read_excel(source)
    elif source.suffix == ".csv":
        df = pd.read_csv(source)
    else:
        raise ValueError(f"不支持的文件格式: {source.suffix}")

    # 标准化攻击类型列名
    for col in _ATTACK_COL_CANDIDATES:
        if col in df.columns and col != "attack_type":
            df = df.rename(columns={col: "attack_type"})
            break

    # 填充缺失
    if "request_body" in df.columns:
        df["request_body"] = df["request_body"].fillna("")

    logger.info("读取 %s 成功，形状: %s", source.name, df.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source.stem}.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("已保存标准化数据: %s  (%d 条)", out_path, len(df))
    return out_path


def main():
    parser = argparse.ArgumentParser(description="导入原始告警数据")
    parser.add_argument("source", nargs="?", help="xlsx / csv 文件路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.source:
        src = Path(args.source)
    else:
        # 自动扫描 data/raw
        candidates = list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.csv"))
        if not candidates:
            print(f"❌ 未在 {RAW_DIR} 找到数据文件，请指定路径")
            return 1
        src = candidates[0]

    if not src.exists():
        print(f"❌ 文件不存在: {src}")
        return 1

    out = ingest(src)
    print(f"✔ 已生成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
