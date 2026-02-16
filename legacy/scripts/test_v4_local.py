#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
本地快速测试 V4 训练（使用小数据集，验证代码正确性）
"""

import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train.train_models_v4 import train
from src.wutong.config import STAGING_DIR


def main():
    """快速测试 V4 训练流程"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    print("=" * 80)
    print("V4 本地快速测试")
    print("=" * 80)
    print("\n⚠️ 注意：这是快速测试模式，不带交叉验证")
    print("   完整训练请使用: python scripts/train_v4.py")
    print("   或上传到 Colab 使用 GPU 训练\n")
    
    # 查找输入文件
    parquets = list(STAGING_DIR.glob("*augmented*.parquet"))
    if not parquets:
        parquets = list(STAGING_DIR.glob("*.parquet"))
    
    if not parquets:
        print("❌ 未找到数据文件！")
        print("\n请先运行:")
        print("  1. python scripts/ingest.py <excel_file>")
        print("  2. python scripts/augment_data.py")
        return 1
    
    data_path = parquets[0]
    print(f"使用数据: {data_path}")
    print(f"模型版本: v4.0.0-test")
    print(f"配置: 快速模式（无 CV）+ SMOTE + 优化超参数\n")
    
    # 快速训练（不带交叉验证）
    try:
        train(
            data_path=data_path,
            version="v4.0.0-test",
            use_cv=False,           # 快速模式：不带交叉验证
            n_cv_splits=10,
            use_stacking=True,
            use_smote=True,         # 启用 SMOTE
        )
        
        print("\n" + "=" * 80)
        print("✅ V4 快速测试完成！")
        print("=" * 80)
        print("\n模型已保存: models/v4.0.0-test/")
        print("\n查看结果:")
        print("  python -c \"import json; print(json.dumps(json.load(open('models/v4.0.0-test/manifest.json')), indent=2, ensure_ascii=False))\"")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
