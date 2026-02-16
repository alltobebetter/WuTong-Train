#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI: 训练模型 V2 (XGBoost + CatBoost + LightGBM)"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_train.train_models_v2 import train
from wutong.config import STAGING_DIR


def main():
    parser = argparse.ArgumentParser(
        description="训练攻击分类模型 V2 (XGBoost + CatBoost + LightGBM)"
    )
    parser.add_argument(
        "-d", "--data", 
        help="parquet 数据路径（默认使用 staging 中最新文件）"
    )
    parser.add_argument(
        "-v", "--version", 
        default="v2.0.0", 
        help="模型版本号（默认: v2.0.0）"
    )
    parser.add_argument(
        "--no-cv", 
        action="store_true",
        help="禁用交叉验证（加快训练速度）"
    )
    parser.add_argument(
        "--cv-splits", 
        type=int, 
        default=5,
        help="交叉验证折数（默认: 5）"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.data:
        data_path = Path(args.data)
    else:
        parquets = sorted(STAGING_DIR.glob("*.parquet"))
        if not parquets:
            print("错误: data/staging/ 中无 parquet 文件，请先运行 ingest")
            sys.exit(1)
        data_path = parquets[-1]

    print(f"使用数据: {data_path}")
    print(f"模型版本: {args.version}")
    print(f"交叉验证: {'否' if args.no_cv else f'是 ({args.cv_splits} 折)'}")
    
    result_dir = train(
        data_path, 
        version=args.version,
        use_cv=not args.no_cv,
        n_cv_splits=args.cv_splits,
    )
    
    print(f"\n✔ 模型已保存: {result_dir}")
    print(f"✔ 查看结果: {result_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
