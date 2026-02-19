#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Kaggle 一键训练 Notebook（V10）
输出: jupyter/kaggle_train_v10.ipynb
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
        "# WuTong V10 Kaggle 一键训练\n",
        "\n",
        "V10 改进（基于 V9 99.37%）：\n",
        "- 修复 numpy JSON 序列化 bug\n",
        "- 路由阈值大幅降低：threshold 0.7→0.45, margin 0.3→0.15\n",
        "- 自适应路由：验证集评估，没帮助自动关闭\n",
        "- LightGBM 作为 Stacking meta-learner（替代 GradientBoosting）\n",
        "- Voting 权重偏向最强单模型\n",
        "- 最强单模型也加入最终候选\n",
        "\n",
        "直接 **Run All** 即可。预计 30-40 分钟。\n",
    ]))

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
        "    os.system('git clone https://github.com/alltobebetter/WuTong-Train.git /kaggle/working/WuTong')\n",
        "\n",
        "%cd /kaggle/working/WuTong\n",
        "!git log --oneline -n 5\n",
    ]))

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
    ]))

    nb["cells"].append(_code_cell([
        "# 4) 预处理 + 增强 + 外部数据\n",
        "import glob, os\n",
        "\n",
        "source_files = glob.glob('data/raw/*.xlsx') + glob.glob('data/raw/*.csv')\n",
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

    nb["cells"].append(_code_cell([
        "# 5) V10 训练\n",
        "import subprocess, sys\n",
        "\n",
        "print('='*80)\n",
        "print('🚀 V10 训练开始')\n",
        "print('='*80)\n",
        "print('\\n核心改进（vs V9 99.37%）:')\n",
        "print('  - 路由阈值: threshold 0.7→0.45, margin 0.3→0.15')\n",
        "print('  - 自适应路由: 验证集评估，没帮助自动关闭')\n",
        "print('  - LightGBM meta-learner（替代 GradientBoosting）')\n",
        "print('  - 最强单模型加入最终候选\\n')\n",
        "\n",
        "cmd = [\n",
        "    sys.executable, '-u', 'scripts/train_v10.py',\n",
        "    '--version', 'v10.0.0-kaggle',\n",
        "    '--cv-splits', '10',\n",
        "    '--optuna-trials', '30',\n",
        "    '--confidence-threshold', '0.45',\n",
        "    '--margin', '0.15',\n",
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
        "print('✅ V10 训练完成！')\n",
        "print('='*80)\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 6) 查看训练结果\n",
        "from pathlib import Path\n",
        "import json\n",
        "\n",
        "manifest_path = Path('models/v10.0.0-kaggle/manifest.json')\n",
        "\n",
        "if manifest_path.exists():\n",
        "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
        "        manifest = json.load(f)\n",
        "\n",
        "    print('='*80)\n",
        "    print('📊 V10 训练结果')\n",
        "    print('='*80)\n",
        "    print(f\"\\n版本: {manifest['version']}\")\n",
        "    print(f\"训练时间: {manifest['trained_at']}\")\n",
        "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
        "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
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
        "    print(f\"\\n🏆 最终模型（{e['ensemble_type']}）\")\n",
        "    print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
        "    print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
        "\n",
        "    if 'all_candidates' in metrics:\n",
        "        print('\\n  所有候选:')\n",
        "        for name, acc in metrics['all_candidates'].items():\n",
        "            print(f'    {name}: {acc:.4f}')\n",
        "\n",
        "    cfg = manifest.get('training_config', {})\n",
        "    print(f\"\\n  路由自适应: {'开启' if cfg.get('routing_enabled') else '已关闭（路由无帮助）'}\")\n",
        "    print(f\"  Meta-learner: {cfg.get('stacking_meta_learner', 'N/A')}\")\n",
        "\n",
        "    adv = manifest.get('adversarial_validation', {})\n",
        "    if adv:\n",
        "        print(f\"  对抗验证 AUC: {adv.get('auc', 'N/A')}\")\n",
        "        print(f\"  分布偏移: {'是' if adv.get('distribution_shift') else '否'}\")\n",
        "\n",
        "    v9_acc = 0.9937\n",
        "    delta = e['test_accuracy'] - v9_acc\n",
        "    print(f\"\\n  vs V9: {'+' if delta >= 0 else ''}{delta*100:.2f}% ({'↑' if delta > 0 else '↓' if delta < 0 else '→'})\")\n",
        "    v7_acc = 0.9935\n",
        "    delta7 = e['test_accuracy'] - v7_acc\n",
        "    print(f\"  vs V7: {'+' if delta7 >= 0 else ''}{delta7*100:.2f}% ({'↑' if delta7 > 0 else '↓' if delta7 < 0 else '→'})\")\n",
        "else:\n",
        "    print('❌ 未找到训练结果文件！')\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 7) 打包下载\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "\n",
        "model_dir = Path('models/v10.0.0-kaggle')\n",
        "if model_dir.exists():\n",
        "    print('📦 打包模型文件...')\n",
        "    archive = shutil.make_archive(\n",
        "        '/kaggle/working/models_v10.0.0-kaggle', 'zip',\n",
        "        'models', 'v10.0.0-kaggle'\n",
        "    )\n",
        "    print(f'✅ 打包完成: {archive}')\n",
        "    print(f'\\n文件大小: {Path(archive).stat().st_size / 1024 / 1024:.2f} MB')\n",
        "    print('\\n💡 在 Kaggle 右侧 Output 面板下载')\n",
        "else:\n",
        "    print('❌ 模型目录不存在！')\n",
    ]))

    nb["cells"].append(_md_cell([
        "## V10 vs V9 改进说明\n",
        "\n",
        "| 维度 | V9 (99.37%) | V10 |\n",
        "|------|-------------|------|\n",
        "| JSON bug | numpy.bool_ 崩溃 | 彻底修复 _to_native() |\n",
        "| 路由阈值 | threshold=0.7 (仅 2 样本路由) | threshold=0.45 |\n",
        "| 路由 margin | 0.3 | 0.15 |\n",
        "| 自适应路由 | 无 | 验证集评估，无帮助自动关闭 |\n",
        "| Meta-learner | GradientBoosting | LightGBM |\n",
        "| Voting 权重 | 线性 | 平方加权（偏向强模型） |\n",
        "| 最终候选 | 3 个 | 4 个（+最强单模型） |\n",
        "| 混淆对 | 5 对 | 6 对（+SQL↔文件包含, +XSS↔SQL） |\n",
    ]))

    return nb


def main() -> None:
    notebook = generate_notebook()
    output = Path(__file__).resolve().parent.parent / "jupyter" / "kaggle_train_v10.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print(f"Kaggle notebook generated: {output}")


if __name__ == "__main__":
    main()
