#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V6 训练脚本 - 数据质量优先 + 高级特征工程 + 精细化集成
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.train.train_models_v6 import train
    from src.wutong.config import STAGING_DIR
except ImportError:
    from src_train.train_models_v6 import train
    from wutong.config import STAGING_DIR


def main():
    parser = argparse.ArgumentParser(description="训练 V6 模型")
    parser.add_argument("--version", type=str, default="v6.0.0",
                        help="模型版本号（默认: v6.0.0）")
    parser.add_argument("--cv-splits", type=int, default=10,
                        help="交叉验证折数（默认: 10）")
    parser.add_argument("--no-cv", action="store_true",
                        help="禁用交叉验证（快速训练）")
    parser.add_argument("--no-stacking", action="store_true",
                        help="禁用 Stacking")
    parser.add_argument("--no-smote", action="store_true",
                        help="禁用 SMOTE-ENN 过采样")
    parser.add_argument("-i", "--input", type=str,
                        help="输入 parquet 文件路径")
    parser.add_argument("--external-data", type=str,
                        help="外部数据集目录（默认: data/external）")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # 查找输入文件
    if args.input:
        data_path = Path(args.input)
        if not data_path.exists():
            print(f"❌ 文件不存在: {data_path}")
            return 1
    else:
        parquets = list(STAGING_DIR.glob("*augmented*.parquet"))
        if not parquets:
            parquets = list(STAGING_DIR.glob("*.parquet"))
        if not parquets:
            print("❌ 未找到数据文件！")
            print("请先运行: python scripts/ingest.py <excel_file>")
            print("然后运行: python scripts/augment_data.py")
            return 1
        data_path = parquets[0]

    # 外部数据集目录
    external_dir = None
    if args.external_data:
        external_dir = Path(args.external_data)
        if not external_dir.exists():
            print(f"⚠️ 外部数据集目录不存在: {external_dir}")
            external_dir = None
    else:
        external_dir = Path("data/external")
        if not external_dir.exists():
            external_dir = None

    cv_label = f"是 ({args.cv_splits} 折)" if not args.no_cv else "否"
    print(f"使用数据: {data_path}")
    print(f"模型版本: {args.version}")
    print(f"交叉验证: {cv_label}")
    print(f"Stacking: {'是' if not args.no_stacking else '否'}")
    print(f"SMOTE-ENN: {'是' if not args.no_smote else '否'}")
    print(f"外部数据集: {'受控混入' if external_dir else '否'}")

    train(
        data_path=data_path,
        version=args.version,
        use_cv=not args.no_cv,
        n_cv_splits=args.cv_splits,
        use_stacking=not args.no_stacking,
        use_smote=not args.no_smote,
        external_data_dir=external_dir,
    )

    print(f"\n✔ 模型已保存: models/{args.version}")
    print(f"✔ 查看结果: models/{args.version}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
