#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10 训练脚本 - V9 bug 修复 + 路由增强 + LGB meta-learner
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from src.train.train_models_v10 import train
    from src.wutong.config import STAGING_DIR
except ImportError:
    from src_train.train_models_v10 import train
    from wutong.config import STAGING_DIR


def main():
    parser = argparse.ArgumentParser(description="训练 V10 模型")
    parser.add_argument("--version", type=str, default="v10.0.0",
                        help="模型版本号")
    parser.add_argument("--cv-splits", type=int, default=10,
                        help="交叉验证折数")
    parser.add_argument("--no-cv", action="store_true")
    parser.add_argument("--no-stacking", action="store_true")
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-optuna", action="store_true",
                        help="禁用 Optuna 超参搜索")
    parser.add_argument("--no-experts", action="store_true",
                        help="禁用混淆对专家模型")
    parser.add_argument("--no-adversarial", action="store_true",
                        help="禁用对抗验证")
    parser.add_argument("--optuna-trials", type=int, default=30,
                        help="Optuna 每个模型的搜索次数")
    parser.add_argument("--confidence-threshold", type=float, default=0.45,
                        help="置信度路由阈值（默认: 0.45，V9 是 0.7）")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="路由 margin 阈值（默认: 0.15，V9 是 0.3）")
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("--external-data", type=str)

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

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

    external_dir = None
    if args.external_data:
        external_dir = Path(args.external_data)
        if not external_dir.exists():
            external_dir = None
    else:
        external_dir = Path("data/external")
        if not external_dir.exists():
            external_dir = None

    print(f"使用数据: {data_path}")
    print(f"模型版本: {args.version}")
    print(f"交叉验证: {'是 (' + str(args.cv_splits) + ' 折)' if not args.no_cv else '否'}")
    print(f"Stacking: {'是 (LightGBM meta-learner)' if not args.no_stacking else '否'}")
    print(f"SMOTE-ENN: {'是' if not args.no_smote else '否'}")
    print(f"Optuna: {'是 (' + str(args.optuna_trials) + ' trials)' if not args.no_optuna else '否'}")
    print(f"混淆对专家: {'是 (threshold=' + str(args.confidence_threshold) + ', margin=' + str(args.margin) + ')' if not args.no_experts else '否'}")
    print(f"对抗验证: {'是' if not args.no_adversarial else '否'}")
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
        use_experts=not args.no_experts,
        use_adversarial=not args.no_adversarial,
        confidence_threshold=args.confidence_threshold,
        margin=args.margin,
        external_data_dir=external_dir,
    )

    print(f"\n✔ 模型已保存: models/{args.version}")
    print(f"✔ 查看结果: models/{args.version}/manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
