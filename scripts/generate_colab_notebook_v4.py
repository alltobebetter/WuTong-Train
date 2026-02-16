#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 V4 Colab 训练 Notebook（SOTA 优化版）
"""

import json
from pathlib import Path


def generate_v4_notebook():
    """生成 V4 训练 notebook"""
    
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4"
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "cells": []
    }

    # Cell 1: 标题和说明
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {"id": "title"},
        "source": [
            "# 梧桐杯 AI 安全告警智能研判系统 - V4 SOTA 优化版（冲击 99.5%+）\n",
            "\n",
            "## 🎯 V4 核心优化\n",
            "\n",
            "1. ⭐⭐⭐⭐⭐ **SMOTE 过采样**：解决类别不平衡，预期 +0.8-1.2%\n",
            "2. ⭐⭐⭐⭐ **XGBoost 深度优化**：500 树 + 深度 12，预期 +0.3-0.5%\n",
            "3. ⭐⭐⭐ **集成权重优化**：优化 meta-learner，预期 +0.1-0.2%\n",
            "4. ✨ **10 折交叉验证**：更稳定的性能评估\n",
            "\n",
            "## 📋 训练流程\n",
            "\n",
            "1. ✅ 克隆 GitHub 仓库\n",
            "2. ✅ 安装依赖（含 imbalanced-learn）\n",
            "3. ✅ 检查 GPU\n",
            "4. ✅ 检查数据\n",
            "5. ✅ 数据预处理\n",
            "6. ✅ 数据增强\n",
            "7. ✅ **训练 V4 模型（SMOTE + 优化超参数）**\n",
            "8. ✅ 查看结果\n",
            "9. ✅ 下载模型\n",
            "\n",
            "## 🎯 目标\n",
            "\n",
            "- **准确率目标**: 99.5%+\n",
            "- **V3 基线**: 98.36%\n",
            "- **预期提升**: +1.1-1.6%\n",
            "- **训练时间**: 约 30-40 分钟\n",
            "- **模型版本**: v4.0.0\n",
            "\n",
            "## ⚙️ 运行前准备\n",
            "\n",
            "1. 确保运行时类型设置为 **GPU**（Runtime > Change runtime type > GPU）\n",
            "2. 按顺序执行每个单元格\n",
            "\n",
            "---"
        ]
    })

    # Cell 2: 克隆仓库
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "clone"},
        "source": [
            "# 1. 克隆 GitHub 仓库\n",
            "!git clone https://github.com/alltobebetter/WuTong.git\n",
            "%cd WuTong\n",
            "\n",
            "# 查看项目结构\n",
            "!ls -la"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 3: 安装依赖
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "install"},
        "source": [
            "# 2. 安装依赖（含 imbalanced-learn）\n",
            "print(\"📦 安装依赖包...\")\n",
            "!pip install -q -r requirements.txt\n",
            "\n",
            "print(\"\\n✅ 依赖安装完成！\")\n",
            "\n",
            "# 验证关键包\n",
            "import xgboost as xgb\n",
            "import catboost as cb\n",
            "import lightgbm as lgb\n",
            "import pandas as pd\n",
            "from imblearn.over_sampling import SMOTE\n",
            "\n",
            "print(f\"XGBoost 版本: {xgb.__version__}\")\n",
            "print(f\"CatBoost 版本: {cb.__version__}\")\n",
            "print(f\"LightGBM 版本: {lgb.__version__}\")\n",
            "print(f\"Pandas 版本: {pd.__version__}\")\n",
            "print(f\"✅ SMOTE 已安装\")"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 4: 检查 GPU
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "gpu"},
        "source": [
            "# 3. 检查 GPU 可用性\n",
            "import torch\n",
            "\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"✅ GPU 可用: {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB\")\n",
            "else:\n",
            "    print(\"⚠️ GPU 不可用，将使用 CPU 训练（速度较慢）\")\n",
            "    print(\"   建议: Runtime > Change runtime type > GPU\")"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 5: 检查数据
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "data"},
        "source": [
            "# 4. 检查数据集\n",
            "import pandas as pd\n",
            "from pathlib import Path\n",
            "\n",
            "data_files = list(Path('data').rglob('*.xlsx'))\n",
            "print(f\"找到 {len(data_files)} 个数据文件:\")\n",
            "for f in data_files:\n",
            "    print(f\"  - {f}\")\n",
            "\n",
            "if data_files:\n",
            "    df = pd.read_excel(data_files[0])\n",
            "    print(f\"\\n数据集大小: {len(df)} 条\")\n",
            "    print(f\"\\n攻击类型分布:\")\n",
            "    print(df.iloc[:, -1].value_counts())\n",
            "else:\n",
            "    print(\"\\n⚠️ 未找到数据文件！\")"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 6: 数据预处理
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "preprocess"},
        "source": [
            "# 5. 数据预处理\n",
            "print(\"🔄 开始数据预处理...\")\n",
            "\n",
            "import glob\n",
            "excel_files = glob.glob('data/**/*.xlsx', recursive=True)\n",
            "\n",
            "if excel_files:\n",
            "    data_file = excel_files[0]\n",
            "    print(f\"使用数据文件: {data_file}\")\n",
            "    !python scripts/ingest.py \"{data_file}\"\n",
            "    print(\"\\n✅ 数据预处理完成！\")\n",
            "\n",
            "    parquet_files = glob.glob('data/staging/*.parquet')\n",
            "    print(f\"\\n生成的 parquet 文件: {parquet_files}\")\n",
            "else:\n",
            "    print(\"❌ 未找到 Excel 数据文件！\")"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 7: 数据增强
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "augment"},
        "source": [
            "# 6. 数据增强\n",
            "print(\"=\"*60)\n",
            "print(\"🚀 数据增强开始\")\n",
            "print(\"=\"*60)\n",
            "print(\"\\n策略:\")\n",
            "print(\"  - SQL 注入: 关键词替换、注释变换、空格编码\")\n",
            "print(\"  - XSS 攻击: 标签大小写、事件处理器、编码变换\")\n",
            "print(\"  - 命令注入: 分隔符变换、命令组合\")\n",
            "print(\"  - URL 路径: 大小写、编码、路径分隔符\")\n",
            "print(\"\\n目标: 11,000 → 30,000+ 条 (2.7x)\\n\")\n",
            "\n",
            "!python scripts/augment_data.py --target-size 30000 --ratio 2.5\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"✅ 数据增强完成！\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# 查看增强后的数据\n",
            "import pandas as pd\n",
            "import glob\n",
            "\n",
            "augmented_files = glob.glob('data/staging/*augmented*.parquet')\n",
            "if augmented_files:\n",
            "    df_aug = pd.read_parquet(augmented_files[0])\n",
            "    print(f\"\\n📊 增强后统计:\")\n",
            "    print(f\"  总数据量: {len(df_aug)} 条\")\n",
            "    print(f\"  增强倍数: {len(df_aug) / 11000:.2f}x\")\n",
            "    print(f\"\\n类别分布:\")\n",
            "    print(df_aug['attack_type'].value_counts())"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 8: 训练 V4 模型
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "train"},
        "source": [
            "# 7. 训练 V4 模型（SMOTE + 优化超参数）\n",
            "print(\"=\"*60)\n",
            "print(\"🚀 V4 模型训练开始（SOTA 优化）\")\n",
            "print(\"=\"*60)\n",
            "print(\"\\n配置:\")\n",
            "print(\"  - 模型: XGBoost + CatBoost + LightGBM\")\n",
            "print(\"  - 集成方式: Stacking (优化 meta-learner)\")\n",
            "print(\"  - 交叉验证: 10 折\")\n",
            "print(\"  - ⭐ SMOTE 过采样: 是（解决类别不平衡）\")\n",
            "print(\"  - ⭐ XGBoost 优化: 500 树 + 深度 12\")\n",
            "print(\"  - 预计时间: 30-40 分钟\\n\")\n",
            "\n",
            "!python scripts/train_v4.py --version v4.0.0 --cv-splits 10\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"✅ V4 训练完成！\")\n",
            "print(\"=\"*60)"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 9: 查看结果
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "results"},
        "source": [
            "# 8. 查看 V4 训练结果\n",
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "manifest_path = Path('models/v4.0.0/manifest.json')\n",
            "\n",
            "if manifest_path.exists():\n",
            "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
            "        manifest = json.load(f)\n",
            "\n",
            "    print(\"=\"*70)\n",
            "    print(\"📊 V4 训练结果（SOTA 优化）\")\n",
            "    print(\"=\"*70)\n",
            "    print(f\"\\n版本: {manifest['version']}\")\n",
            "    print(f\"训练时间: {manifest['trained_at']}\")\n",
            "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
            "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
            "    print(f\"类别数: {len(manifest['classes'])} 类\")\n",
            "\n",
            "    config = manifest.get('training_config', {})\n",
            "    print(f\"\\n训练配置:\")\n",
            "    print(f\"  - 交叉验证: {config.get('n_cv_splits', 'N/A')} 折\")\n",
            "    print(f\"  - 集成方式: {config.get('use_stacking', False) and 'Stacking' or 'Voting'}\")\n",
            "    print(f\"  - SMOTE 过采样: {config.get('use_smote', False) and '是' or '否'}\")\n",
            "\n",
            "    print(\"\\n\" + \"=\"*70)\n",
            "    print(\"🎯 模型性能\")\n",
            "    print(\"=\"*70)\n",
            "\n",
            "    metrics = manifest['metrics']\n",
            "\n",
            "    # 单模型性能\n",
            "    for model_name in ['xgboost', 'catboost', 'lightgbm']:\n",
            "        if model_name in metrics:\n",
            "            m = metrics[model_name]\n",
            "            print(f\"\\n{model_name.upper()}:\")\n",
            "            print(f\"  测试准确率: {m['test_accuracy']:.4f} ({m['test_accuracy']*100:.2f}%)\")\n",
            "            print(f\"  测试 F1: {m['test_f1']:.4f}\")\n",
            "            if m.get('cv_accuracy'):\n",
            "                print(f\"  CV 准确率: {m['cv_accuracy']:.4f} (±{m.get('cv_std', 0):.4f})\")\n",
            "\n",
            "    # 集成模型性能\n",
            "    if 'ensemble' in metrics:\n",
            "        e = metrics['ensemble']\n",
            "        print(f\"\\n{'='*70}\")\n",
            "        print(f\"🏆 V4 集成模型（最终模型）\")\n",
            "        print(f\"{'='*70}\")\n",
            "        print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
            "        print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
            "        print(f\"  集成方式: {e.get('ensemble_type', 'N/A')}\")\n",
            "\n",
            "        # 与 V3 对比\n",
            "        v3_acc = 0.9836  # V3 的准确率\n",
            "        improvement = (e['test_accuracy'] - v3_acc) * 100\n",
            "        print(f\"\\n  📈 相比 V3 提升: {improvement:+.2f}%\")\n",
            "\n",
            "        if e['test_accuracy'] >= 0.995:\n",
            "            print(f\"\\n  🎉🎉🎉 恭喜！达到 99.5% 目标！\")\n",
            "        elif e['test_accuracy'] >= 0.99:\n",
            "            print(f\"\\n  🎉 恭喜！达到 99% 目标！\")\n",
            "        elif e['test_accuracy'] >= 0.985:\n",
            "            print(f\"\\n  ✨ 非常接近 99%，表现优秀！\")\n",
            "\n",
            "    print(\"\\n\" + \"=\"*70)\n",
            "\n",
            "    # 详细分类报告\n",
            "    report_path = Path('models/v4.0.0/classification_report.txt')\n",
            "    if report_path.exists():\n",
            "        print(\"\\n📋 详细分类报告:\")\n",
            "        print(\"=\"*70)\n",
            "        with open(report_path, 'r', encoding='utf-8') as f:\n",
            "            print(f.read())\n",
            "else:\n",
            "    print(\"❌ 未找到 V4 训练结果文件！\")\n",
            "    print(\"请先运行训练单元格\")"
        ],
        "execution_count": None,
        "outputs": []
    })

    # Cell 10: 下载模型
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {"id": "download"},
        "source": [
            "# 9. 打包并下载模型\n",
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "model_dir = Path('models/v4.0.0')\n",
            "if model_dir.exists():\n",
            "    print(\"📦 打包模型文件...\")\n",
            "    shutil.make_archive('models_v4.0.0', 'zip', 'models', 'v4.0.0')\n",
            "    print(\"✅ 打包完成: models_v4.0.0.zip\")\n",
            "    \n",
            "    # 在 Colab 中下载\n",
            "    from google.colab import files\n",
            "    files.download('models_v4.0.0.zip')\n",
            "    print(\"\\n✅ 下载已开始！\")\n",
            "else:\n",
            "    print(\"❌ 模型目录不存在！\")"
        ],
        "execution_count": None,
        "outputs": []
    })

    return notebook


def main():
    """主函数"""
    notebook = generate_v4_notebook()
    
    output_path = Path(__file__).parent.parent / "jupyter" / "colab_train_v4.ipynb"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print(f"✅ V4 Notebook 已生成: {output_path}")
    print(f"📝 可以上传到 Google Colab 进行训练")


if __name__ == "__main__":
    main()
