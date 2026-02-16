#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI: 训练模型"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_train.train_models import train
from wutong.config import STAGING_DIR


def main():
    parser = argparse.ArgumentParser(description="训练攻击分类模型")
    parser.add_argument("-d", "--data", help="parquet 数据路径（默认使用 staging 中最新文件）")
    parser.add_argument("-v", "--version", default="v1.0.0", help="模型版本号")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.data:
        data_path = Path(args.data)
    else:
        parquets = sorted(STAGING_DIR.glob("*.parquet"))
        if not parquets:
            print("错误: data/staging/ 中无 parquet 文件，请先运行 ingest")
            sys.exit(1)
        data_path = parquets[-1]

    result_dir = train(data_path, version=args.version)
    print(f"✔ 模型已保存: {result_dir}")


if __name__ == "__main__":
    main()
