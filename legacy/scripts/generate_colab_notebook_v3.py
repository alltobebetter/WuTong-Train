#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 Colab 训练 Notebook V3 - 冲击 99%"""

import json
from pathlib import Path


def create_v3_notebook():
    """创建 V3 Jupyter Notebook"""
    
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
    
    # Cell 1: 标题
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 梧桐杯 AI 安全告警智能研判系统 - V3 增强版（冲击 99%）\n",
            "\n",
            "## 🎯 V3 改进点\n",
            "\n",
            "1. ✨ **数据增强**：11,000 → 30,000+ 条（2.7x）\n",
            "2. ✨ **Stacking 集成**：替代简单 Voting，学习各模型优势\n",
            "3. ✨ **10 折交叉验证**：更稳定，减少过拟合\n",
            "4. ✨ **超参数优化**：更深的树、更多迭代\n",
            "\n",
            "## 📋 训练流程\n",
            "\n",
            "1. ✅ 克隆 GitHub 仓库\n",
            "2. ✅ 安装依赖\n",
            "3. ✅ 检查 GPU\n",
            "4. ✅ 检查数据\n",
            "5. ✅ 数据预处理\n",
            "6. ✅ **数据增强（V3 核心）**\n",
            "7. ✅ 训练 V3 模型\n",
            "8. ✅ 查看结果\n",
            "9. ✅ 下载模型\n",
            "\n",
            "## 🎯 目标\n",
            "\n",
            "- **准确率目标**: 99%+\n",
            "- **训练时间**: 约 20-30 分钟\n",
            "- **模型版本**: v3.0.0\n",
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
        "metadata": {},
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
        "metadata": {},
        "source": [
            "# 2. 安装依赖\n",
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
            "\n",
            "print(f\"XGBoost 版本: {xgb.__version__}\")\n",
            "print(f\"CatBoost 版本: {cb.__version__}\")\n",
            "print(f\"LightGBM 版本: {lgb.__version__}\")\n",
            "print(f\"Pandas 版本: {pd.__version__}\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 4: 检查 GPU
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
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
        "metadata": {},
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
        "metadata": {},
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
            "    \n",
            "    parquet_files = glob.glob('data/staging/*.parquet')\n",
            "    print(f\"\\n生成的 parquet 文件: {parquet_files}\")\n",
            "else:\n",
            "    print(\"❌ 未找到 Excel 数据文件！\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 7: 数据增强（V3 核心）
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 6. 数据增强（V3 核心 - 冲击 99%）\n",
            "print(\"=\"*60)\n",
            "print(\"🚀 V3 数据增强开始\")\n",
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
    
    # Cell 8: 训练 V3 模型
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 7. 训练 V3 模型（Stacking + 10折CV）\n",
            "print(\"=\"*60)\n",
            "print(\"🚀 V3 模型训练开始\")\n",
            "print(\"=\"*60)\n",
            "print(\"\\n配置:\")\n",
            "print(\"  - 模型: XGBoost + CatBoost + LightGBM\")\n",
            "print(\"  - 集成方式: Stacking (Logistic Regression)\")\n",
            "print(\"  - 交叉验证: 10 折\")\n",
            "print(\"  - 超参数: 优化版（更深的树、更多迭代）\")\n",
            "print(\"  - 预计时间: 20-30 分钟\\n\")\n",
            "\n",
            "!python scripts/train_v3.py --version v3.0.0 --cv-splits 10\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"✅ V3 训练完成！\")\n",
            "print(\"=\"*60)"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 9: 快速训练（可选）
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 7-alternative. 快速训练（不带交叉验证，约 10-15 分钟）\n",
            "# 如果想快速测试，可以运行这个单元格代替上面的单元格\n",
            "\n",
            "# print(\"⚡ 快速训练模式（不带交叉验证）...\")\n",
            "# !python scripts/train_v3.py --version v3.0.0-fast --no-cv\n",
            "# print(\"\\n✅ 快速训练完成！\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 10: 查看结果
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 8. 查看 V3 训练结果\n",
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "manifest_path = Path('models/v3.0.0/manifest.json')\n",
            "\n",
            "if manifest_path.exists():\n",
            "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
            "        manifest = json.load(f)\n",
            "    \n",
            "    print(\"=\"*70)\n",
            "    print(\"📊 V3 训练结果\")\n",
            "    print(\"=\"*70)\n",
            "    print(f\"\\n版本: {manifest['version']}\")\n",
            "    print(f\"训练时间: {manifest['trained_at']}\")\n",
            "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
            "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
            "    print(f\"类别数: {len(manifest['classes'])} 类\")\n",
            "    \n",
            "    config = manifest.get('training_config', {})\n",
            "    print(f\"\\n训练配置:\")\n",
            "    print(f\"  - 交叉验证: {config.get('n_cv_splits', 'N/A')} 折\")\n",
            "    print(f\"  - 集成方式: {config.get('use_stacking', False) and 'Stacking' or 'Voting'}\")\n",
            "    \n",
            "    print(\"\\n\" + \"=\"*70)\n",
            "    print(\"🎯 模型性能\")\n",
            "    print(\"=\"*70)\n",
            "    \n",
            "    metrics = manifest['metrics']\n",
            "    \n",
            "    # 单模型性能\n",
            "    for model_name in ['xgboost', 'catboost', 'lightgbm']:\n",
            "        if model_name in metrics:\n",
            "            m = metrics[model_name]\n",
            "            print(f\"\\n{model_name.upper()}:\")\n",
            "            print(f\"  测试准确率: {m['test_accuracy']:.4f} ({m['test_accuracy']*100:.2f}%)\")\n",
            "            print(f\"  测试 F1: {m['test_f1']:.4f}\")\n",
            "            if m.get('cv_accuracy'):\n",
            "                print(f\"  CV 准确率: {m['cv_accuracy']:.4f} (±{m.get('cv_std', 0):.4f})\")\n",
            "    \n",
            "    # 集成模型性能\n",
            "    if 'ensemble' in metrics:\n",
            "        e = metrics['ensemble']\n",
            "        print(f\"\\n{'='*70}\")\n",
            "        print(f\"🏆 V3 集成模型（最终模型）\")\n",
            "        print(f\"{'='*70}\")\n",
            "        print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
            "        print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
            "        print(f\"  集成方式: {e.get('ensemble_type', 'N/A')}\")\n",
            "        \n",
            "        # 与 V2 对比\n",
            "        v2_acc = 0.9759  # V2 的准确率\n",
            "        improvement = (e['test_accuracy'] - v2_acc) * 100\n",
            "        print(f\"\\n  📈 相比 V2 提升: {improvement:+.2f}%\")\n",
            "        \n",
            "        if e['test_accuracy'] >= 0.99:\n",
            "            print(f\"\\n  🎉 恭喜！达到 99% 目标！\")\n",
            "        elif e['test_accuracy'] >= 0.985:\n",
            "            print(f\"\\n  ✨ 非常接近 99%，表现优秀！\")\n",
            "    \n",
            "    print(\"\\n\" + \"=\"*70)\n",
            "    \n",
            "    # 详细分类报告\n",
            "    report_path = Path('models/v3.0.0/classification_report.txt')\n",
            "    if report_path.exists():\n",
            "        print(\"\\n📋 详细分类报告:\")\n",
            "        print(\"=\"*70)\n",
            "        with open(report_path, 'r', encoding='utf-8') as f:\n",
            "            print(f.read())\n",
            "else:\n",
            "    print(\"❌ 未找到 V3 训练结果文件！\")\n",
            "    print(\"请先运行训练单元格\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 11: 对比所有版本
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 9. 对比 V2 vs V3\n",
            "print(\"📊 模型版本对比\\n\")\n",
            "!python scripts/compare_models.py"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 12: 打包下载
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 10. 打包 V3 模型文件\n",
            "print(\"📦 打包 V3 模型文件...\")\n",
            "\n",
            "!zip -r models_v3.0.0.zip models/v3.0.0/\n",
            "\n",
            "print(\"\\n✅ 打包完成！\")\n",
            "print(\"\\n文件大小:\")\n",
            "!ls -lh models_v3.0.0.zip"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 13: 下载
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 11. 下载 V3 模型到本地\n",
            "from google.colab import files\n",
            "\n",
            "print(\"⬇️ 开始下载 V3 模型文件...\")\n",
            "print(\"（下载完成后解压到本地项目的 models/ 目录）\\n\")\n",
            "\n",
            "files.download('models_v3.0.0.zip')\n",
            "\n",
            "print(\"\\n✅ 下载完成！\")\n",
            "print(\"\\n📝 后续步骤:\")\n",
            "print(\"1. 解压 models_v3.0.0.zip\")\n",
            "print(\"2. 将 models/v3.0.0/ 目录复制到本地项目\")\n",
            "print(\"3. 运行推理: python scripts/infer.py data/xxx.xlsx --model-version v3.0.0\")\n",
            "print(\"4. V3 模型在 CPU 上运行完全没问题！\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 14: 总结
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "\n",
            "## ✅ V3 训练完成！\n",
            "\n",
            "### 🎯 V3 改进总结\n",
            "\n",
            "| 项目 | V2 | V3 | 提升 |\n",
            "|------|----|----|------|\n",
            "| 数据量 | 11,000 | 30,000+ | 2.7x |\n",
            "| 集成方式 | Voting | Stacking | ✨ |\n",
            "| 交叉验证 | 5 折 | 10 折 | ✨ |\n",
            "| 超参数 | 基础 | 优化 | ✨ |\n",
            "| 预期准确率 | 97.5% | 99%+ | +1.5%+ |\n",
            "\n",
            "### 🔑 关键改进\n",
            "\n",
            "1. **数据增强**（最重要）\n",
            "   - SQL 注入：关键词替换、注释变换\n",
            "   - XSS 攻击：标签大小写、事件处理器\n",
            "   - 命令注入：分隔符变换\n",
            "   - URL 路径：编码变换、路径分隔符\n",
            "\n",
            "2. **Stacking 集成**\n",
            "   - 学习各模型的优势领域\n",
            "   - 对难分类样本（如 CSRF）效果更好\n",
            "\n",
            "3. **超参数优化**\n",
            "   - 树深度：8 → 10\n",
            "   - 迭代次数：200 → 300\n",
            "   - 学习率：0.1 → 0.05\n",
            "\n",
            "### 📦 V3 模型文件\n",
            "\n",
            "```\n",
            "models/v3.0.0/\n",
            "├── ensemble.pkl              # Stacking 集成模型\n",
            "├── xgboost.pkl               # XGBoost (300 棵树)\n",
            "├── catboost.pkl              # CatBoost (300 轮)\n",
            "├── lightgbm.pkl              # LightGBM (300 棵树)\n",
            "├── label_encoder.pkl         # 标签编码器\n",
            "├── aggregator.pkl            # 告警聚合器\n",
            "├── manifest.json             # 模型元数据\n",
            "├── feature_list.json         # 特征列表\n",
            "└── classification_report.txt # 详细分类报告\n",
            "```\n",
            "\n",
            "### 💡 使用建议\n",
            "\n",
            "1. **本地推理**：V3 模型在 CPU 上运行完全没问题\n",
            "2. **API 服务**：直接替换 V2 模型即可\n",
            "3. **比赛提交**：只提交模型文件，不包含原始数据\n",
            "\n",
            "### 🚀 如果还想提升\n",
            "\n",
            "1. **补充公开数据集**：CSIC 2010、CICIDS2017\n",
            "2. **针对 CSRF 优化**：单独训练二分类器\n",
            "3. **特征工程**：N-gram、熵值、编码检测\n",
            "\n",
            "---\n",
            "\n",
            "**项目地址**: https://github.com/alltobebetter/WuTong\n",
            "\n",
            "**祝比赛顺利！冲击 99%！** 🎉\n"
        ]
    })
    
    return notebook


def main():
    """生成并保存 V3 notebook"""
    notebook = create_v3_notebook()
    
    output_path = Path(__file__).parent.parent / "colab_train_v3.ipynb"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print(f"✅ V3 Notebook 已生成: {output_path}")
    print(f"📝 文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\n🎯 V3 特点:")
    print(f"  - 数据增强: 11k → 30k+")
    print(f"  - Stacking 集成")
    print(f"  - 10 折交叉验证")
    print(f"  - 目标准确率: 99%+")
    print(f"\n使用方法:")
    print(f"1. 上传 colab_train_v3.ipynb 到 Google Colab")
    print(f"2. 设置运行时为 GPU")
    print(f"3. 按顺序执行所有单元格")
    print(f"4. 训练时间约 20-30 分钟")


if __name__ == "__main__":
    main()
