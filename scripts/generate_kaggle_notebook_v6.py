#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Kaggle 一键训练 Notebook（V6）
输出: jupyter/kaggle_train_v6.ipynb
"""

import json
from pathlib import Path


def _md_cell(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def _code_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "code", "metadata": {},
        "source": lines, "execution_count": None, "outputs": [],
    }


def generate_notebook() -> dict:
    nb = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "cells": [],
    }

    nb["cells"].append(_md_cell([
        "# WuTong V6 Kaggle 一键训练\n",
        "\n",
        "V6 核心改进：\n",
        "- 外部数据受控混入（不再淹没原始分布）\n",
        "- +13 个高级特征（信息熵、编码深度、关键词密度等）\n",
        "- SMOTE-ENN（合成 + 清洗边界噪声）\n",
        "- 精调模型参数 + 双集成对比\n",
        "\n",
        "直接 **Run All** 即可。\n",
    ]))

    # Cell 1: Clone repo
    nb["cells"].append(_code_cell([
        "# 1) 克隆/更新代码\n",
        "from pathlib import Path\n",
        "import os\n",
        "\n",
        "repo_dir = Path('/kaggle/working/WuTong')\n",
        "if repo_dir.exists():\n",
        "    print('Repo exists, pulling latest...')\n",
        "    os.system(f'cd {repo_dir} && git pull')\n",
        "else:\n",
        "    print('Cloning repo...')\n",
        "    os.system('git clone https://github.com/alltobebetter/WuTong.git /kaggle/working/WuTong')\n",
        "\n",
        "%cd /kaggle/working/WuTong\n",
        "!git log --oneline -n 3\n",
    ]))

    # Cell 2: Install deps
    nb["cells"].append(_code_cell([
        "# 2) 安装依赖\n",
        "!pip -q install -r requirements.txt\n",
        "!pip -q install imbalanced-learn  # SMOTE-ENN\n",
        "\n",
        "import torch\n",
        "print('CUDA available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('GPU:', torch.cuda.get_device_name(0))\n",
    ]))

    # Cell 3: Copy data
    nb["cells"].append(_code_cell([
        "# 3) 从 /kaggle/input 拷贝数据\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "\n",
        "raw_dir = Path('data/raw')\n",
        "raw_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "candidates = list(Path('/kaggle/input').rglob('*.xlsx')) + list(Path('/kaggle/input').rglob('*.csv'))\n",
        "print(f'Found {len(candidates)} candidate files in /kaggle/input')\n",
        "for src in candidates[:20]:\n",
        "    dst = raw_dir / src.name\n",
        "    if not dst.exists():\n",
        "        shutil.copy2(src, dst)\n",
        "\n",
        "print('data/raw files:')\n",
        "for p in raw_dir.glob('*'):\n",
        "    print(' -', p)\n",
    ]))

    # Cell 4: Preprocess pipeline
    nb["cells"].append(_code_cell([
        "# 4) 预处理 + 增强 + 外部数据下载\n",
        "import glob, os\n",
        "\n",
        "xlsx_files = glob.glob('data/raw/*.xlsx')\n",
        "csv_files = glob.glob('data/raw/*.csv')\n",
        "source_files = xlsx_files + csv_files\n",
        "if not source_files:\n",
        "    raise RuntimeError('data/raw 没有可用数据')\n",
        "\n",
        "src = source_files[0]\n",
        "print('Using source file:', src)\n",
        "ret = os.system(f'python -u scripts/ingest.py \"{src}\"')\n",
        "if ret != 0: raise RuntimeError('ingest 失败')\n",
        "\n",
        "ret = os.system('python -u scripts/augment_data.py --target-size 30000 --ratio 2.5')\n",
        "if ret != 0: raise RuntimeError('augment 失败')\n",
        "\n",
        "# 下载外部数据集（V6 会受控混入，不会淹没原始数据）\n",
        "ret = os.system('python -u scripts/integrate_csic2010.py')\n",
        "if ret != 0:\n",
        "    print('⚠️ 外部数据集下载失败，将仅使用原始+增强数据训练')\n",
        "\n",
        "print('preprocess pipeline done')\n",
    ]))

    # Cell 5: Train V6
    nb["cells"].append(_code_cell([
        "# 5) V6 训练\n",
        "import subprocess, sys\n",
        "\n",
        "print('='*80)\n",
        "print('🚀 V6 训练开始')\n",
        "print('='*80)\n",
        "print('\\n核心改进:')\n",
        "print('  - 外部数据受控混入（每类最多补充 30%）')\n",
        "print('  - +13 个高级特征（信息熵、编码深度等）')\n",
        "print('  - SMOTE-ENN（合成 + 清洗边界噪声）')\n",
        "print('  - 精调参数 + 双集成对比\\n')\n",
        "\n",
        "cmd = [\n",
        "    sys.executable, '-u', 'scripts/train_v6.py',\n",
        "    '--version', 'v6.0.0-kaggle',\n",
        "    '--cv-splits', '10',\n",
        "]\n",
        "\n",
        "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "for line in proc.stdout:\n",
        "    print(line, end='')\n",
        "\n",
        "ret = proc.wait()\n",
        "if ret != 0:\n",
        "    raise RuntimeError(f'训练失败，退出码: {ret}')\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('✅ V6 训练完成！')\n",
        "print('='*80)\n",
    ]))

    # Cell 6: Results
    nb["cells"].append(_code_cell([
        "# 6) 查看训练结果\n",
        "from pathlib import Path\n",
        "import json\n",
        "\n",
        "manifest_path = Path('models/v6.0.0-kaggle/manifest.json')\n",
        "\n",
        "if manifest_path.exists():\n",
        "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
        "        manifest = json.load(f)\n",
        "\n",
        "    print('='*80)\n",
        "    print('📊 V6 训练结果')\n",
        "    print('='*80)\n",
        "    print(f\"\\n版本: {manifest['version']}\")\n",
        "    print(f\"训练时间: {manifest['trained_at']}\")\n",
        "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
        "    print(f\"外部数据: {'受控混入 ' + str(manifest.get('external_data_rows', 0)) + ' 条' if manifest.get('external_data') else '否'}\")\n",
        "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
        "    print(f\"类别数: {len(manifest['classes'])} 类\")\n",
        "\n",
        "    print('\\n' + '='*80)\n",
        "    print('🎯 模型性能')\n",
        "    print('='*80)\n",
        "\n",
        "    metrics = manifest['metrics']\n",
        "    for name in ['xgboost', 'catboost', 'lightgbm']:\n",
        "        m = metrics[name]\n",
        "        cv_str = f\", CV: {m['cv_accuracy']:.4f}\" if m.get('cv_accuracy') else ''\n",
        "        print(f\"  {name}: Acc={m['test_accuracy']:.4f}, F1={m['test_f1']:.4f}{cv_str}\")\n",
        "\n",
        "    e = metrics['ensemble']\n",
        "    print(f\"\\n🏆 集成模型（{e['ensemble_type']}）\")\n",
        "    print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
        "    print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
        "\n",
        "    if e['test_accuracy'] >= 0.998:\n",
        "        print('\\n  🎉🎉🎉 达到 99.8% 目标！')\n",
        "    elif e['test_accuracy'] >= 0.995:\n",
        "        print('\\n  🎉🎉 达到 99.5% 目标！')\n",
        "    elif e['test_accuracy'] >= 0.99:\n",
        "        print('\\n  🎉 达到 99% 目标！')\n",
        "\n",
        "    print('\\n' + '='*80)\n",
        "else:\n",
        "    print('❌ 未找到训练结果文件！')\n",
    ]))

    # Cell 7: Package
    nb["cells"].append(_code_cell([
        "# 7) 打包下载\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "\n",
        "model_dir = Path('models/v6.0.0-kaggle')\n",
        "if model_dir.exists():\n",
        "    print('📦 打包模型文件...')\n",
        "    archive = shutil.make_archive(\n",
        "        '/kaggle/working/models_v6.0.0-kaggle', 'zip',\n",
        "        'models', 'v6.0.0-kaggle'\n",
        "    )\n",
        "    print(f'✅ 打包完成: {archive}')\n",
        "    print(f'\\n文件大小: {Path(archive).stat().st_size / 1024 / 1024:.2f} MB')\n",
        "    print('\\n💡 在 Kaggle 右侧 Output 面板下载')\n",
        "else:\n",
        "    print('❌ 模型目录不存在！')\n",
    ]))

    nb["cells"].append(_md_cell([
        "## V6 vs V5 改进说明\n",
        "\n",
        "| 维度 | V5 | V6 |\n",
        "|------|-----|-----|\n",
        "| 外部数据 | 全量混入（61k 淹没 29k） | 受控混入（每类最多 30%，过滤不可靠标签） |\n",
        "| 特征数 | 25 | 38（+13 高级特征） |\n",
        "| 过采样 | SMOTE（全量拉平） | SMOTE-ENN（合成+清洗，目标 80%） |\n",
        "| XGBoost 深度 | 12 | 10（防过拟合） |\n",
        "| 集成策略 | 固定 Stacking | Stacking vs Voting 对比取优 |\n",
        "| 预期准确率 | 77.46%（实际） | 99%+（修复数据问题后） |\n",
    ]))

    return nb


def main() -> None:
    notebook = generate_notebook()
    output = Path(__file__).resolve().parent.parent / "jupyter" / "kaggle_train_v6.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print(f"Kaggle notebook generated: {output}")


if __name__ == "__main__":
    main()
