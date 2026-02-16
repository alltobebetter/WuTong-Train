#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Kaggle 一键训练 Notebook（V7）
输出: jupyter/kaggle_train_v7.ipynb
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
        "# WuTong V7 Kaggle 一键训练\n",
        "\n",
        "V7 核心改进（目标 99%+）：\n",
        "- Optuna 贝叶斯超参搜索（自动找最优参数）\n",
        "- +11 个交互特征（针对 CSRF/正常访问 混淆）\n",
        "- ExtraTrees 第四基模型增加集成多样性\n",
        "- Stacking meta-learner 升级为 GradientBoosting\n",
        "\n",
        "直接 **Run All** 即可。预计 25-35 分钟（含 Optuna 调参）。\n",
    ]))

    # Cell 1: Clone
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
        "!git log --oneline -n 5\n",
    ]))

    # Cell 2: Install deps
    nb["cells"].append(_code_cell([
        "# 2) 安装依赖\n",
        "!pip -q install -r requirements.txt\n",
        "!pip -q install imbalanced-learn optuna\n",
        "\n",
        "import torch\n",
        "print('CUDA available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('GPU:', torch.cuda.get_device_name(0))\n",
    ]))

    # Cell 3: Copy data
    nb["cells"].append(_code_cell([
        "# 3) 准备数据\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "\n",
        "raw_dir = Path('data/raw')\n",
        "raw_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "repo_data = list(Path('data_raw').glob('*.xlsx')) + list(Path('data_raw').glob('*.csv'))\n",
        "if repo_data:\n",
        "    print(f'从仓库 data_raw/ 找到 {len(repo_data)} 个数据文件')\n",
        "    for src in repo_data:\n",
        "        dst = raw_dir / src.name\n",
        "        if not dst.exists():\n",
        "            shutil.copy2(src, dst)\n",
        "            print(f'  复制: {src.name}')\n",
        "else:\n",
        "    candidates = list(Path('/kaggle/input').rglob('*.xlsx')) + list(Path('/kaggle/input').rglob('*.csv'))\n",
        "    print(f'从 /kaggle/input 找到 {len(candidates)} 个候选文件')\n",
        "    for src in candidates[:20]:\n",
        "        dst = raw_dir / src.name\n",
        "        if not dst.exists():\n",
        "            shutil.copy2(src, dst)\n",
        "\n",
        "print('\\ndata/raw files:')\n",
        "for p in sorted(raw_dir.glob('*')):\n",
        "    print(f'  - {p.name} ({p.stat().st_size / 1024:.0f} KB)')\n",
        "\n",
        "if not list(raw_dir.glob('*')):\n",
        "    raise RuntimeError('未找到任何数据文件！')\n",
    ]))

    # Cell 4: Preprocess
    nb["cells"].append(_code_cell([
        "# 4) 预处理 + 增强 + 外部数据\n",
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
        "ret = os.system('python -u scripts/integrate_csic2010.py')\n",
        "if ret != 0:\n",
        "    print('⚠️ 外部数据集下载失败，将仅使用原始+增强数据训练')\n",
        "\n",
        "print('preprocess pipeline done')\n",
    ]))

    # Cell 5: Train V7
    nb["cells"].append(_code_cell([
        "# 5) V7 训练（含 Optuna 自动调参）\n",
        "import subprocess, sys\n",
        "\n",
        "print('='*80)\n",
        "print('🚀 V7 训练开始')\n",
        "print('='*80)\n",
        "print('\\n核心改进（vs V6 98.55%）:')\n",
        "print('  - Optuna 贝叶斯超参搜索（30 trials/模型）')\n",
        "print('  - +11 个交互特征（CSRF/正常访问 区分）')\n",
        "print('  - ExtraTrees 第四基模型')\n",
        "print('  - GradientBoosting meta-learner\\n')\n",
        "\n",
        "cmd = [\n",
        "    sys.executable, '-u', 'scripts/train_v7.py',\n",
        "    '--version', 'v7.0.0-kaggle',\n",
        "    '--cv-splits', '10',\n",
        "    '--optuna-trials', '30',\n",
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
        "print('✅ V7 训练完成！')\n",
        "print('='*80)\n",
    ]))

    # Cell 6: Results
    nb["cells"].append(_code_cell([
        "# 6) 查看训练结果\n",
        "from pathlib import Path\n",
        "import json\n",
        "\n",
        "manifest_path = Path('models/v7.0.0-kaggle/manifest.json')\n",
        "\n",
        "if manifest_path.exists():\n",
        "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
        "        manifest = json.load(f)\n",
        "\n",
        "    print('='*80)\n",
        "    print('📊 V7 训练结果')\n",
        "    print('='*80)\n",
        "    print(f\"\\n版本: {manifest['version']}\")\n",
        "    print(f\"训练时间: {manifest['trained_at']}\")\n",
        "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
        "    print(f\"外部数据: {'受控混入 ' + str(manifest.get('external_data_rows', 0)) + ' 条' if manifest.get('external_data') else '否'}\")\n",
        "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
        "    print(f\"类别数: {len(manifest['classes'])} 类\")\n",
        "    print(f\"Optuna: {'是' if manifest['training_config'].get('use_optuna') else '否'}\")\n",
        "    print(f\"基模型数: {manifest['training_config'].get('n_base_models', 3)}\")\n",
        "\n",
        "    print('\\n' + '='*80)\n",
        "    print('🎯 模型性能')\n",
        "    print('='*80)\n",
        "\n",
        "    metrics = manifest['metrics']\n",
        "    for name in ['xgboost', 'catboost', 'lightgbm', 'extratrees']:\n",
        "        if name not in metrics: continue\n",
        "        m = metrics[name]\n",
        "        cv_str = f\", CV: {m['cv_accuracy']:.4f}\" if m.get('cv_accuracy') else ''\n",
        "        print(f\"  {name}: Acc={m['test_accuracy']:.4f}, F1={m['test_f1']:.4f}{cv_str}\")\n",
        "\n",
        "    e = metrics['ensemble']\n",
        "    print(f\"\\n🏆 集成模型（{e['ensemble_type']}）\")\n",
        "    print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
        "    print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
        "\n",
        "    v6_acc = 0.9855\n",
        "    delta = e['test_accuracy'] - v6_acc\n",
        "    print(f\"\\n  vs V6: {'+' if delta >= 0 else ''}{delta*100:.2f}% ({'↑' if delta > 0 else '↓'})\")\n",
        "\n",
        "    if e['test_accuracy'] >= 0.998:\n",
        "        print('\\n  🎉🎉🎉 达到 99.8% 目标！')\n",
        "    elif e['test_accuracy'] >= 0.995:\n",
        "        print('\\n  🎉🎉 达到 99.5% 目标！')\n",
        "    elif e['test_accuracy'] >= 0.99:\n",
        "        print('\\n  🎉 达到 99% 目标！')\n",
        "\n",
        "    # Optuna 参数\n",
        "    if 'optuna_params' in manifest:\n",
        "        print('\\n' + '='*80)\n",
        "        print('🔧 Optuna 最优参数')\n",
        "        print('='*80)\n",
        "        for name, params in manifest['optuna_params'].items():\n",
        "            if params != 'default':\n",
        "                print(f'  {name}: {json.dumps(params, indent=4)}')\n",
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
        "model_dir = Path('models/v7.0.0-kaggle')\n",
        "if model_dir.exists():\n",
        "    print('📦 打包模型文件...')\n",
        "    archive = shutil.make_archive(\n",
        "        '/kaggle/working/models_v7.0.0-kaggle', 'zip',\n",
        "        'models', 'v7.0.0-kaggle'\n",
        "    )\n",
        "    print(f'✅ 打包完成: {archive}')\n",
        "    print(f'\\n文件大小: {Path(archive).stat().st_size / 1024 / 1024:.2f} MB')\n",
        "    print('\\n💡 在 Kaggle 右侧 Output 面板下载')\n",
        "else:\n",
        "    print('❌ 模型目录不存在！')\n",
    ]))

    nb["cells"].append(_md_cell([
        "## V7 vs V6 改进说明\n",
        "\n",
        "| 维度 | V6 (98.55%) | V7 |\n",
        "|------|-------------|-----|\n",
        "| 超参搜索 | 手动调参 | Optuna 贝叶斯搜索 (30 trials) |\n",
        "| 特征数 | 38 | 49 (+11 交互特征) |\n",
        "| 基模型 | 3 (XGB+Cat+LGB) | 4 (+ExtraTrees) |\n",
        "| Meta-learner | LogisticRegression | GradientBoosting |\n",
        "| SMOTE 目标 | 80% | 90% |\n",
        "| 针对性优化 | 无 | CSRF/正常访问 区分特征 |\n",
    ]))

    return nb


def main() -> None:
    notebook = generate_notebook()
    output = Path(__file__).resolve().parent.parent / "jupyter" / "kaggle_train_v7.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print(f"Kaggle notebook generated: {output}")


if __name__ == "__main__":
    main()
