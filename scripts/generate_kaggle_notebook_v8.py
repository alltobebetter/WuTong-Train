#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成 Kaggle 一键训练 Notebook（V8）"""

import json
from pathlib import Path


def _md_cell(lines): return {"cell_type": "markdown", "metadata": {}, "source": lines}
def _code_cell(lines): return {"cell_type": "code", "metadata": {}, "source": lines, "execution_count": None, "outputs": []}


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
        "# WuTong V8 Kaggle 一键训练\n",
        "\n",
        "V8 核心改进（目标 99.5%+）：\n",
        "- 针对性区分特征：文件包含 vs 目录遍历、SQL vs XSS\n",
        "- 特征重要性筛选（去噪声）\n",
        "- Stacking passthrough（meta-learner 看原始特征）\n",
        "- 多种子集成（3 seeds × 4 models = 12 models 概率平均）\n",
        "- Optuna 50 trials（基于 V7 最优参数 warm start）\n",
        "\n",
        "直接 **Run All**，预计 40-50 分钟。\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 1) 克隆/更新代码\n",
        "from pathlib import Path\n",
        "import os\n",
        "repo_dir = Path('/kaggle/working/WuTong')\n",
        "if repo_dir.exists():\n",
        "    os.system(f'cd {repo_dir} && git pull')\n",
        "else:\n",
        "    os.system('git clone https://github.com/alltobebetter/WuTong-Train.git /kaggle/working/WuTong')\n",
        "%cd /kaggle/working/WuTong\n",
        "!git log --oneline -n 5\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 2) 安装依赖\n",
        "!pip -q install -r requirements.txt\n",
        "!pip -q install imbalanced-learn optuna\n",
        "import torch\n",
        "print('CUDA available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available(): print('GPU:', torch.cuda.get_device_name(0))\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 3) 准备数据\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "raw_dir = Path('data/raw')\n",
        "raw_dir.mkdir(parents=True, exist_ok=True)\n",
        "repo_data = list(Path('data_raw').glob('*.xlsx')) + list(Path('data_raw').glob('*.csv'))\n",
        "if repo_data:\n",
        "    print(f'从仓库 data_raw/ 找到 {len(repo_data)} 个数据文件')\n",
        "    for src in repo_data:\n",
        "        dst = raw_dir / src.name\n",
        "        if not dst.exists(): shutil.copy2(src, dst)\n",
        "else:\n",
        "    candidates = list(Path('/kaggle/input').rglob('*.xlsx')) + list(Path('/kaggle/input').rglob('*.csv'))\n",
        "    for src in candidates[:20]:\n",
        "        dst = raw_dir / src.name\n",
        "        if not dst.exists(): shutil.copy2(src, dst)\n",
        "print('\\ndata/raw files:')\n",
        "for p in sorted(raw_dir.glob('*')): print(f'  - {p.name} ({p.stat().st_size/1024:.0f} KB)')\n",
        "if not list(raw_dir.glob('*')): raise RuntimeError('未找到数据文件！')\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 4) 预处理 + 增强 + 外部数据\n",
        "import glob, os\n",
        "source_files = glob.glob('data/raw/*.xlsx') + glob.glob('data/raw/*.csv')\n",
        "if not source_files: raise RuntimeError('data/raw 没有可用数据')\n",
        "src = source_files[0]\n",
        "print('Using:', src)\n",
        "ret = os.system(f'python -u scripts/ingest.py \"{src}\"')\n",
        "if ret != 0: raise RuntimeError('ingest 失败')\n",
        "ret = os.system('python -u scripts/augment_data.py --target-size 30000 --ratio 2.5')\n",
        "if ret != 0: raise RuntimeError('augment 失败')\n",
        "ret = os.system('python -u scripts/integrate_csic2010.py')\n",
        "if ret != 0: print('⚠️ 外部数据下载失败')\n",
        "print('preprocess done')\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 5) V8 训练\n",
        "import subprocess, sys\n",
        "print('='*80)\n",
        "print('🚀 V8 训练开始')\n",
        "print('='*80)\n",
        "print('\\n核心改进（vs V7 99.35%）:')\n",
        "print('  - 针对性区分特征（文件包含/目录遍历、SQL/XSS）')\n",
        "print('  - 特征重要性筛选')\n",
        "print('  - Stacking passthrough + 多种子集成')\n",
        "print('  - Optuna 50 trials\\n')\n",
        "cmd = [sys.executable, '-u', 'scripts/train_v8.py',\n",
        "       '--version', 'v8.0.0-kaggle', '--cv-splits', '10', '--optuna-trials', '50']\n",
        "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "for line in proc.stdout: print(line, end='')\n",
        "ret = proc.wait()\n",
        "if ret != 0: raise RuntimeError(f'训练失败: {ret}')\n",
        "print('\\n' + '='*80)\n",
        "print('✅ V8 训练完成！')\n",
        "print('='*80)\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 6) 查看结果\n",
        "from pathlib import Path\n",
        "import json\n",
        "manifest_path = Path('models/v8.0.0-kaggle/manifest.json')\n",
        "if manifest_path.exists():\n",
        "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
        "        manifest = json.load(f)\n",
        "    print('='*80)\n",
        "    print('📊 V8 训练结果')\n",
        "    print('='*80)\n",
        "    print(f\"\\n版本: {manifest['version']}\")\n",
        "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
        "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
        "    print(f\"Optuna: {manifest['training_config'].get('use_optuna')}\")\n",
        "    print(f\"多种子: {manifest['training_config'].get('use_multi_seed')}\")\n",
        "    print('\\n' + '='*80)\n",
        "    print('🎯 模型性能')\n",
        "    print('='*80)\n",
        "    metrics = manifest['metrics']\n",
        "    for name in ['xgboost', 'catboost', 'lightgbm', 'extratrees']:\n",
        "        if name not in metrics: continue\n",
        "        m = metrics[name]\n",
        "        cv_str = f\", CV: {m['cv_accuracy']:.4f}\" if m.get('cv_accuracy') else ''\n",
        "        print(f\"  {name}: Acc={m['test_accuracy']:.4f}, F1={m['test_f1']:.4f}{cv_str}\")\n",
        "    e = metrics['ensemble']\n",
        "    print(f\"\\n🏆 集成模型（{e['ensemble_type']}）\")\n",
        "    print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
        "    print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
        "    v7_acc = 0.9935\n",
        "    delta = e['test_accuracy'] - v7_acc\n",
        "    print(f\"\\n  vs V7: {'+' if delta >= 0 else ''}{delta*100:.2f}%\")\n",
        "    if 'all_ensembles' in metrics:\n",
        "        print('\\n  所有集成方案:')\n",
        "        for k, v in metrics['all_ensembles'].items():\n",
        "            print(f'    {k}: {v:.4f}')\n",
        "    if e['test_accuracy'] >= 0.995: print('\\n  🎉🎉 达到 99.5% 目标！')\n",
        "    elif e['test_accuracy'] >= 0.99: print('\\n  🎉 达到 99% 目标！')\n",
        "    print('\\n' + '='*80)\n",
        "else:\n",
        "    print('❌ 未找到训练结果')\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 7) 打包下载\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "model_dir = Path('models/v8.0.0-kaggle')\n",
        "if model_dir.exists():\n",
        "    archive = shutil.make_archive('/kaggle/working/models_v8.0.0-kaggle', 'zip', 'models', 'v8.0.0-kaggle')\n",
        "    print(f'✅ 打包完成: {archive}')\n",
        "    print(f'文件大小: {Path(archive).stat().st_size/1024/1024:.2f} MB')\n",
        "else:\n",
        "    print('❌ 模型目录不存在')\n",
    ]))

    return nb


def main():
    notebook = generate_notebook()
    output = Path(__file__).resolve().parent.parent / "jupyter" / "kaggle_train_v8.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print(f"Kaggle notebook generated: {output}")


if __name__ == "__main__":
    main()
