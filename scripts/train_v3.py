#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI: 训练模型 V3 (增强版 - 冲击 99%)
- 使用数据增强
- Stacking 集成
- 针对性优化
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src_train.train_models_v3 import train
from wutong.config import STAGING_DIR


def main():
    parser = argparse.ArgumentParser(
        description="训练攻击分类模型 V3 (增强版 - 冲击 99%)"
    )
    parser.add_argument(
        "-d", "--data",
        help="parquet 数据路径（默认使用 staging 中最新文件）"
    )
    parser.add_argument(
        "-v", "--version",
        default="v3.0.0",
        help="模型版本号（默认: v3.0.0）"
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="禁用交叉验证（加快训练速度）"
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=10,
        help="交叉验证折数（默认: 10，更稳定）"
    )
    parser.add_argument(
        "--use-stacking",
        action="store_true",
        default=True,
        help="使用 Stacking 集成（默认: True）"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    if args.data:
        data_path = Path(args.data)
    else:
        # 优先使用增强后的数据
        parquets = sorted(STAGING_DIR.glob("*augmented*.parquet"))
        if not parquets:
            parquets = sorted(STAGING_DIR.glob("*.parquet"))
        if not parquets:
            print("错误: data/staging/ 中无 parquet 文件，请先运行 ingest")
            sys.exit(1)
        data_path = parquets[-1]

    print(f"使用数据: {data_path}")
    print(f"模型版本: {args.version}")
    print(f"交叉验证: {'否' if args.no_cv else f'是 ({args.cv_splits} 折)'}")
    print(f"Stacking: {'是' if args.use_stacking else '否'}")

    result_dir = train(
        data_path,
        version=args.version,
        use_cv=not args.no_cv,
        n_cv_splits=args.cv_splits,
        use_stacking=args.use_stacking,
    )

    print(f"\n✔ 模型已保存: {result_dir}")
    print(f"✔ 查看结果: {result_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
