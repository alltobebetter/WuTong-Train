#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V8 训练脚本"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.train.train_models_v8 import train
    from src.wutong.config import STAGING_DIR
except ImportError:
    from src_train.train_models_v8 import train
    from wutong.config import STAGING_DIR


def main():
    parser = argparse.ArgumentParser(description="训练 V8 模型")
    parser.add_argument("--version", type=str, default="v8.0.0")
    parser.add_argument("--cv-splits", type=int, default=10)
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-stacking", action="store_true")
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-optuna", action="store_true")
    parser.add_argument("--no-feature-selection", action="store_true")
    parser.add_argument("--no-multi-seed", action="store_true")
    parser.add_argument("--optuna-trials", type=int, default=50)
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("--external-data", type=str)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
            return 1
        data_path = parquets[0]

    external_dir = Path(args.external_data) if args.external_data else Path("data/external")
    if not external_dir.exists():
        external_dir = None

    print(f"使用数据: {data_path}")
    print(f"模型版本: {args.version}")
    print(f"Optuna: {'是 (' + str(args.optuna_trials) + ' trials)' if not args.no_optuna else '否'}")
    print(f"特征选择: {'是' if not args.no_feature_selection else '否'}")
    print(f"多种子集成: {'是' if not args.no_multi_seed else '否'}")
    print(f"外部数据集: {'受控混入' if external_dir else '否'}")

    train(
        data_path=data_path,
        version=args.version,
        use_cv=not args.no_cv,
        n_cv_splits=args.cv_splits,
        use_stacking=not args.no_stacking,
        use_smote=not args.no_smote,
        use_optuna=not args.no_optuna,
        optuna_trials=args.optuna_trials,
        use_feature_selection=not args.no_feature_selection,
        use_multi_seed=not args.no_multi_seed,
        external_data_dir=external_dir,
    )

    print(f"\n✔ 模型已保存: models/{args.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
