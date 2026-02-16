#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V3 vs V4 对比分析
"""

import json
from pathlib import Path


def load_manifest(version: str):
    """加载模型 manifest"""
    manifest_path = Path(f"models/{version}/manifest.json")
    if not manifest_path.exists():
        return None
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 80)
    print("V3 vs V4 模型对比分析")
    print("=" * 80)
    
    v3 = load_manifest("v3.0.0")
    v4 = load_manifest("v4.0.0")
    
    if not v3:
        print("\n⚠️ V3 模型未找到，请先训练 V3 模型")
        print("   运行: python scripts/train_v3.py")
        return
    
    if not v4:
        print("\n⚠️ V4 模型未找到，请先训练 V4 模型")
        print("   运行: python scripts/train_v4.py")
        return
    
    # 配置对比
    print("\n" + "=" * 80)
    print("📋 训练配置对比")
    print("=" * 80)
    
    v3_config = v3.get("training_config", {})
    v4_config = v4.get("training_config", {})
    
    print(f"\n{'配置项':<20} {'V3':<30} {'V4':<30}")
    print("-" * 80)
    print(f"{'数据量':<20} {v3['data_rows']:<30} {v4['data_rows']:<30}")
    print(f"{'特征数':<20} {len(v3['feature_list']):<30} {len(v4['feature_list']):<30}")
    print(f"{'交叉验证':<20} {v3_config.get('n_cv_splits', 'N/A')} 折{'':<25} {v4_config.get('n_cv_splits', 'N/A')} 折")
    print(f"{'集成方式':<20} {'Stacking':<30} {'Stacking':<30}")
    print(f"{'SMOTE 过采样':<20} {'否':<30} {'是 ⭐':<30}")
    
    # 性能对比
    print("\n" + "=" * 80)
    print("🎯 模型性能对比")
    print("=" * 80)
    
    v3_metrics = v3.get("metrics", {})
    v4_metrics = v4.get("metrics", {})
    
    print(f"\n{'模型':<15} {'V3 准确率':<20} {'V4 准确率':<20} {'提升':<15}")
    print("-" * 80)
    
    for model_name in ["xgboost", "catboost", "lightgbm"]:
        v3_acc = v3_metrics.get(model_name, {}).get("test_accuracy", 0)
        v4_acc = v4_metrics.get(model_name, {}).get("test_accuracy", 0)
        improvement = (v4_acc - v3_acc) * 100
        
        print(f"{model_name.upper():<15} {v3_acc:.4f} ({v3_acc*100:.2f}%){'':<3} {v4_acc:.4f} ({v4_acc*100:.2f}%){'':<3} {improvement:+.2f}%")
    
    # 集成模型对比
    v3_ensemble = v3_metrics.get("ensemble", {})
    v4_ensemble = v4_metrics.get("ensemble", {})
    
    v3_acc = v3_ensemble.get("test_accuracy", 0)
    v4_acc = v4_ensemble.get("test_accuracy", 0)
    improvement = (v4_acc - v3_acc) * 100
    
    print("-" * 80)
    print(f"{'集成模型':<15} {v3_acc:.4f} ({v3_acc*100:.2f}%){'':<3} {v4_acc:.4f} ({v4_acc*100:.2f}%){'':<3} {improvement:+.2f}%")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 总结")
    print("=" * 80)
    
    print(f"\nV3 准确率: {v3_acc:.4f} ({v3_acc*100:.2f}%)")
    print(f"V4 准确率: {v4_acc:.4f} ({v4_acc*100:.2f}%)")
    print(f"绝对提升: {improvement:+.2f}%")
    
    if v4_acc >= 0.995:
        print("\n🎉🎉🎉 恭喜！V4 达到 99.5% 目标！")
    elif v4_acc >= 0.99:
        print("\n🎉 恭喜！V4 达到 99% 目标！")
    elif v4_acc > v3_acc:
        print(f"\n✨ V4 相比 V3 有提升，继续优化可达到 99%+")
    else:
        print(f"\n⚠️ V4 未达到预期，建议检查训练日志")
    
    # 关键改进
    print("\n" + "=" * 80)
    print("🔑 V4 关键改进")
    print("=" * 80)
    print("\n1. ⭐⭐⭐⭐⭐ SMOTE 过采样")
    print("   - 解决类别不平衡问题（CSRF 样本少）")
    print("   - 自动平衡所有类别的样本数")
    print("   - 预期提升: +0.8-1.2%")
    
    print("\n2. ⭐⭐⭐⭐ XGBoost 深度优化")
    print("   - 树的数量: 300 → 500")
    print("   - 树的深度: 10 → 12")
    print("   - 学习率: 0.05 → 0.03")
    print("   - 预期提升: +0.3-0.5%")
    
    print("\n3. ⭐⭐⭐ 集成权重优化")
    print("   - meta-learner 迭代: 1000 → 2000")
    print("   - 添加正则化: C=0.5")
    print("   - CV 折数: 5 → 10")
    print("   - 预期提升: +0.1-0.2%")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
