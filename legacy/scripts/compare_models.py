#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比不同版本模型的性能"""

import json
import sys
from pathlib import Path
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wutong.config import MODEL_DIR


def load_manifest(version: str) -> dict:
    """加载模型 manifest"""
    manifest_path = MODEL_DIR / version / "manifest.json"
    if not manifest_path.exists():
        return None
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_models():
    """对比所有模型版本"""
    versions = [d.name for d in MODEL_DIR.iterdir() if d.is_dir()]
    
    if not versions:
        print("未找到任何模型版本")
        return
    
    print(f"找到 {len(versions)} 个模型版本: {', '.join(versions)}\n")
    
    # 收集数据
    comparison_data = []
    
    for version in sorted(versions):
        manifest = load_manifest(version)
        if not manifest:
            continue
        
        metrics = manifest.get("metrics", {})
        
        # V1 格式
        if "ensemble_accuracy" in metrics:
            row = {
                "版本": version,
                "训练时间": manifest.get("trained_at", "N/A")[:19],
                "数据量": manifest.get("data_rows", "N/A"),
                "模型类型": "RF+GB",
                "准确率": f"{metrics.get('ensemble_accuracy', 0):.4f}",
                "F1分数": "N/A",
            }
        # V2 格式
        elif "ensemble" in metrics:
            ensemble = metrics["ensemble"]
            xgb = metrics.get("xgboost", {})
            cat = metrics.get("catboost", {})
            lgb = metrics.get("lightgbm", {})
            
            row = {
                "版本": version,
                "训练时间": manifest.get("trained_at", "N/A")[:19],
                "数据量": manifest.get("data_rows", "N/A"),
                "模型类型": "XGB+Cat+LGB",
                "准确率": f"{ensemble.get('test_accuracy', 0):.4f}",
                "F1分数": f"{ensemble.get('test_f1', 0):.4f}",
            }
            
            # 添加单模型性能
            comparison_data.append(row)
            
            # XGBoost
            if xgb:
                comparison_data.append({
                    "版本": f"  └─ XGBoost",
                    "训练时间": "",
                    "数据量": "",
                    "模型类型": "",
                    "准确率": f"{xgb.get('test_accuracy', 0):.4f}",
                    "F1分数": f"{xgb.get('test_f1', 0):.4f}",
                })
            
            # CatBoost
            if cat:
                comparison_data.append({
                    "版本": f"  └─ CatBoost",
                    "训练时间": "",
                    "数据量": "",
                    "模型类型": "",
                    "准确率": f"{cat.get('test_accuracy', 0):.4f}",
                    "F1分数": f"{cat.get('test_f1', 0):.4f}",
                })
            
            # LightGBM
            if lgb:
                comparison_data.append({
                    "版本": f"  └─ LightGBM",
                    "训练时间": "",
                    "数据量": "",
                    "模型类型": "",
                    "准确率": f"{lgb.get('test_accuracy', 0):.4f}",
                    "F1分数": f"{lgb.get('test_f1', 0):.4f}",
                })
            
            continue
        else:
            row = {
                "版本": version,
                "训练时间": manifest.get("trained_at", "N/A")[:19],
                "数据量": manifest.get("data_rows", "N/A"),
                "模型类型": "Unknown",
                "准确率": "N/A",
                "F1分数": "N/A",
            }
        
        comparison_data.append(row)
    
    # 打印表格
    if comparison_data:
        print(tabulate(comparison_data, headers="keys", tablefmt="grid"))
    else:
        print("无法读取模型性能数据")
    
    # 打印详细信息
    print("\n=== 详细信息 ===")
    for version in sorted(versions):
        manifest = load_manifest(version)
        if not manifest:
            continue
        
        print(f"\n【{version}】")
        print(f"  训练时间: {manifest.get('trained_at', 'N/A')}")
        print(f"  数据量: {manifest.get('data_rows', 'N/A')} 条")
        print(f"  特征数: {len(manifest.get('feature_list', []))} 个")
        print(f"  类别数: {len(manifest.get('classes', []))} 类")
        
        if "training_config" in manifest:
            config = manifest["training_config"]
            print(f"  交叉验证: {config.get('use_cv', False)}")
            if config.get('use_cv'):
                print(f"  CV折数: {config.get('n_cv_splits', 'N/A')}")


if __name__ == "__main__":
    try:
        compare_models()
    except ImportError:
        print("提示: 需要安装 tabulate 库")
        print("运行: pip install tabulate")
        sys.exit(1)
