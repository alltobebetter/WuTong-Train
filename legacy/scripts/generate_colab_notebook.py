#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 Colab 训练 Notebook"""

import json
from pathlib import Path


def create_notebook():
    """创建 Jupyter Notebook 结构"""
    
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
        "metadata": {},
        "source": [
            "# 梧桐杯 AI 安全告警智能研判系统 - Colab 训练\n",
            "\n",
            "## 📋 训练流程\n",
            "\n",
            "1. ✅ 克隆 GitHub 仓库\n",
            "2. ✅ 安装依赖\n",
            "3. ✅ 检查数据\n",
            "4. ✅ 数据预处理\n",
            "5. ✅ 训练 V2 模型（XGBoost + CatBoost + LightGBM）\n",
            "6. ✅ 查看训练结果\n",
            "7. ✅ 下载模型文件\n",
            "\n",
            "## ⚙️ 运行前准备\n",
            "\n",
            "1. 确保运行时类型设置为 **GPU**（Runtime > Change runtime type > GPU）\n",
            "2. 按顺序执行每个单元格\n",
            "3. 训练时间约 10-15 分钟\n",
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
            "!pip install -q -r requirements-colab.txt\n",
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
            "# 查找数据文件\n",
            "data_files = list(Path('data').rglob('*.xlsx'))\n",
            "print(f\"找到 {len(data_files)} 个数据文件:\")\n",
            "for f in data_files:\n",
            "    print(f\"  - {f}\")\n",
            "\n",
            "if data_files:\n",
            "    # 读取第一个数据文件\n",
            "    df = pd.read_excel(data_files[0])\n",
            "    print(f\"\\n数据集大小: {len(df)} 条\")\n",
            "    print(f\"\\n列名: {df.columns.tolist()}\")\n",
            "    print(f\"\\n攻击类型分布:\")\n",
            "    print(df.iloc[:, -1].value_counts())\n",
            "else:\n",
            "    print(\"\\n⚠️ 未找到数据文件！\")\n",
            "    print(\"请确保 data/ 目录下有数据文件\")"
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
            "# 查找 Excel 文件\n",
            "import glob\n",
            "excel_files = glob.glob('data/**/*.xlsx', recursive=True)\n",
            "\n",
            "if excel_files:\n",
            "    data_file = excel_files[0]\n",
            "    print(f\"使用数据文件: {data_file}\")\n",
            "    \n",
            "    # 运行预处理\n",
            "    !python scripts/ingest.py \"{data_file}\"\n",
            "    \n",
            "    print(\"\\n✅ 数据预处理完成！\")\n",
            "    \n",
            "    # 检查生成的 parquet 文件\n",
            "    parquet_files = glob.glob('data/staging/*.parquet')\n",
            "    print(f\"\\n生成的 parquet 文件: {parquet_files}\")\n",
            "else:\n",
            "    print(\"❌ 未找到 Excel 数据文件！\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 7: 训练模型
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 6. 训练 V2 模型（完整训练，带交叉验证）\n",
            "print(\"🚀 开始训练 V2 模型...\")\n",
            "print(\"模型: XGBoost + CatBoost + LightGBM\")\n",
            "print(\"交叉验证: 5 折\")\n",
            "print(\"预计时间: 10-15 分钟\\n\")\n",
            "\n",
            "!python scripts/train_v2.py --version v2.0.0\n",
            "\n",
            "print(\"\\n✅ 训练完成！\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 8: 快速训练（可选）
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 6-alternative. 快速训练（不带交叉验证，约 5 分钟）\n",
            "# 如果想快速测试，可以运行这个单元格代替上面的单元格\n",
            "\n",
            "# print(\"⚡ 快速训练模式（不带交叉验证）...\")\n",
            "# !python scripts/train_v2.py --version v2.0.0-fast --no-cv\n",
            "# print(\"\\n✅ 快速训练完成！\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 9: 查看训练结果
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 7. 查看训练结果\n",
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "manifest_path = Path('models/v2.0.0/manifest.json')\n",
            "\n",
            "if manifest_path.exists():\n",
            "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
            "        manifest = json.load(f)\n",
            "    \n",
            "    print(\"=\" * 60)\n",
            "    print(\"📊 训练结果\")\n",
            "    print(\"=\" * 60)\n",
            "    print(f\"\\n版本: {manifest['version']}\")\n",
            "    print(f\"训练时间: {manifest['trained_at']}\")\n",
            "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
            "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
            "    print(f\"类别数: {len(manifest['classes'])} 类\")\n",
            "    \n",
            "    print(\"\\n\" + \"=\" * 60)\n",
            "    print(\"🎯 模型性能\")\n",
            "    print(\"=\" * 60)\n",
            "    \n",
            "    metrics = manifest['metrics']\n",
            "    \n",
            "    # 单模型性能\n",
            "    for model_name in ['xgboost', 'catboost', 'lightgbm']:\n",
            "        if model_name in metrics:\n",
            "            m = metrics[model_name]\n",
            "            print(f\"\\n{model_name.upper()}:\")\n",
            "            print(f\"  测试准确率: {m['test_accuracy']:.4f}\")\n",
            "            print(f\"  测试 F1: {m['test_f1']:.4f}\")\n",
            "            if m.get('cv_accuracy'):\n",
            "                print(f\"  交叉验证准确率: {m['cv_accuracy']:.4f}\")\n",
            "    \n",
            "    # 集成模型性能\n",
            "    if 'ensemble' in metrics:\n",
            "        e = metrics['ensemble']\n",
            "        print(f\"\\n{'='*60}\")\n",
            "        print(f\"🏆 集成模型（最终模型）\")\n",
            "        print(f\"{'='*60}\")\n",
            "        print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
            "        print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
            "    \n",
            "    print(\"\\n\" + \"=\" * 60)\n",
            "    \n",
            "    # 读取详细分类报告\n",
            f"    report_path = Path('models/{train_version}/classification_report.txt')\n",
            "    if report_path.exists():\n",
            "        print(\"\\n📋 详细分类报告:\")\n",
            "        print(\"=\" * 60)\n",
            "        with open(report_path, 'r', encoding='utf-8') as f:\n",
            "            print(f.read())\n",
            "else:\n",
            "    print(\"❌ 未找到训练结果文件！\")\n",
            "    print(\"请先运行训练单元格\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 10: 对比模型
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 8. 对比不同版本模型（如果有 V1 模型）\n",
            "!python scripts/compare_models.py"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 11: 打包下载
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 9. 打包模型文件准备下载\n",
            "print(\"📦 打包模型文件...\")\n",
            "\n",
            "!zip -r models_v2.0.0.zip models/v2.0.0/\n",
            "\n",
            "print(\"\\n✅ 打包完成！\")\n",
            "print(\"\\n文件大小:\")\n",
            "!ls -lh models_v2.0.0.zip"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 12: 下载文件
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 10. 下载模型文件到本地\n",
            "from google.colab import files\n",
            "\n",
            "print(\"⬇️ 开始下载模型文件...\")\n",
            "print(\"（下载完成后解压到本地项目的 models/ 目录）\\n\")\n",
            "\n",
            "files.download('models_v2.0.0.zip')\n",
            "\n",
            "print(\"\\n✅ 下载完成！\")\n",
            "print(\"\\n📝 后续步骤:\")\n",
            "print(\"1. 解压 models_v2.0.0.zip\")\n",
            "print(\"2. 将 models/v2.0.0/ 目录复制到本地项目\")\n",
            "print(\"3. 运行推理: python scripts/infer.py data/xxx.xlsx --model-version v2.0.0\")"
        ],
        "execution_count": None,
        "outputs": []
    })
    
    # Cell 13: 测试推理（可选）
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "source": [
            "# 11. 测试推理（可选）\n",
            "import glob\n",
            "\n",
            "excel_files = glob.glob('data/**/*.xlsx', recursive=True)\n",
            "if excel_files:\n",
            "    test_file = excel_files[0]\n",
            "    print(f\"🧪 测试推理: {test_file}\")\n",
            "    \n",
            "    !python scripts/infer.py \"{test_file}\" --model-version v2.0.0 --job-id colab_test\n",
            "    \n",
            "    print(\"\\n✅ 推理完成！\")\n",
            "    print(\"\\n查看结果:\")\n",
            "    !ls -la data/outputs/colab_test/\n",
            "else:\n",
            "    print(\"❌ 未找到测试数据文件\")"
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
            "## ✅ 训练完成！\n",
            "\n",
            "### 📊 关键指标\n",
            "\n",
            "- **模型架构**: XGBoost + CatBoost + LightGBM 三模型集成\n",
            "- **预期准确率**: 98.5-99.5%\n",
            "- **训练方式**: 5 折交叉验证\n",
            "- **数据量**: 11,000 条告警数据\n",
            "- **攻击类型**: 9 类（8 种攻击 + 正常访问）\n",
            "\n",
            "### 📦 已生成文件\n",
            "\n",
            "```\n",
            "models/v2.0.0/\n",
            "├── ensemble.pkl              # 集成模型（主要使用）\n",
            "├── xgboost.pkl               # XGBoost 单模型\n",
            "├── catboost.pkl              # CatBoost 单模型\n",
            "├── lightgbm.pkl              # LightGBM 单模型\n",
            "├── label_encoder.pkl         # 标签编码器\n",
            "├── aggregator.pkl            # 告警聚合器\n",
            "├── manifest.json             # 模型元数据\n",
            "├── feature_list.json         # 特征列表\n",
            "└── classification_report.txt # 详细分类报告\n",
            "```\n",
            "\n",
            "### 🚀 下一步\n",
            "\n",
            "1. ✅ 下载 `models_v2.0.0.zip` 到本地\n",
            "2. ✅ 解压到项目的 `models/` 目录\n",
            "3. ✅ 在本地 CPU 上运行推理（完全兼容）\n",
            "4. ✅ 集成到 API 服务或 Electron 前端\n",
            "\n",
            "### ❓ 常见问题\n",
            "\n",
            "**Q: Colab 训练的模型能在本地 CPU 运行吗？**  \n",
            "A: 完全可以！模型训练时使用的是 CPU 指令，GPU 只是加速计算。下载后可以直接在任何 CPU 环境运行。\n",
            "\n",
            "**Q: 准确率没达到预期怎么办？**  \n",
            "A: 可以尝试：\n",
            "- 增加交叉验证折数：`--cv-splits 10`\n",
            "- 调整模型超参数\n",
            "- 补充公开数据集\n",
            "\n",
            "**Q: 模型文件太大怎么办？**  \n",
            "A: 只保留 `ensemble.pkl`、`label_encoder.pkl`、`aggregator.pkl` 和配置文件即可，单模型文件可以删除。\n",
            "\n",
            "---\n",
            "\n",
            "**项目地址**: https://github.com/alltobebetter/WuTong\n"
        ]
    })
    
    return notebook


def main():
    """生成并保存 notebook"""
    notebook = create_notebook()
    
    output_path = Path(__file__).parent.parent / "colab_train.ipynb"
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Notebook 已生成: {output_path}")
    print(f"📝 文件大小: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\n使用方法:")
    print(f"1. 上传 colab_train.ipynb 到 Google Colab")
    print(f"2. 设置运行时为 GPU")
    print(f"3. 按顺序执行所有单元格")


if __name__ == "__main__":
    main()
