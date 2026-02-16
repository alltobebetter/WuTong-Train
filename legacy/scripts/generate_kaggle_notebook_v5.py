#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成 Kaggle 一键训练 Notebook（V5）
输出: jupyter/kaggle_train_v5.ipynb
"""

import json
from pathlib import Path


def _md_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines,
    }


def _code_cell(lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": lines,
        "execution_count": None,
        "outputs": [],
    }


def generate_notebook() -> dict:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [],
    }

    nb["cells"].append(_md_cell([
        "# WuTong V5 Kaggle 一键训练\n",
        "\n",
        "这个 Notebook 面向 Kaggle：\n",
        "- 自动准备代码与依赖\n",
        "- 自动从 `/kaggle/input` 扫描数据\n",
        "- 后台启动 V5 训练（可断开页面继续）\n",
        "- 实时查看日志、完成后打包模型\n",
        "\n",
        "建议：直接 **Run All**。\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 1) 克隆/更新代码到 /kaggle/working/WuTong\n",
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
        "!git log --oneline -n 2\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 2) 安装依赖 + 检查环境\n",
        "!pip -q install -r requirements.txt\n",
        "\n",
        "import torch\n",
        "print('CUDA available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('GPU:', torch.cuda.get_device_name(0))\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 3) 从 /kaggle/input 拷贝数据到项目 data/raw\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "\n",
        "raw_dir = Path('data/raw')\n",
        "raw_dir.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "candidates = list(Path('/kaggle/input').rglob('*.xlsx')) + list(Path('/kaggle/input').rglob('*.csv'))\n",
        "print(f'Found {len(candidates)} candidate files in /kaggle/input')\n",
        "for p in candidates[:20]:\n",
        "    print(' -', p)\n",
        "\n",
        "for src in candidates:\n",
        "    dst = raw_dir / src.name\n",
        "    if not dst.exists():\n",
        "        shutil.copy2(src, dst)\n",
        "\n",
        "print('data/raw files:')\n",
        "for p in raw_dir.glob('*'):\n",
        "    print(' -', p)\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 4) 一键预处理 + 增强 + 外部数据整合\n",
        "import glob, os\n",
        "\n",
        "xlsx_files = glob.glob('data/raw/*.xlsx')\n",
        "csv_files = glob.glob('data/raw/*.csv')\n",
        "source_files = xlsx_files + csv_files\n",
        "if not source_files:\n",
        "    raise RuntimeError('data/raw 没有可用数据，请先在 Kaggle Add Data 后重试')\n",
        "\n",
        "src = source_files[0]\n",
        "print('Using source file:', src)\n",
        "ret = os.system(f'python -u scripts/ingest.py \"{src}\"')\n",
        "if ret != 0:\n",
        "    raise RuntimeError('ingest 失败')\n",
        "\n",
        "ret = os.system('python -u scripts/augment_data.py --target-size 30000 --ratio 2.5')\n",
        "if ret != 0:\n",
        "    raise RuntimeError('augment 失败')\n",
        "\n",
        "ret = os.system('python -u scripts/integrate_csic2010.py')\n",
        "if ret != 0:\n",
        "    raise RuntimeError('integrate_csic2010 失败')\n",
        "\n",
        "print('preprocess pipeline done')\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 5) 前台训练（适合 Save and Run All）\n",
        "# 注意：这会阻塞 notebook 直到训练完成（约 40-60 分钟）\n",
        "# 如果想快速测试，可以添加 --no-cv 或减少 --cv-splits\n",
        "\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "print('='*80)\n",
        "print('🚀 V5 训练开始（前台模式，适合 Run All）')\n",
        "print('='*80)\n",
        "print('\\n配置:')\n",
        "print('  - 模型版本: v5.0.0-kaggle')\n",
        "print('  - 交叉验证: 10 折')\n",
        "print('  - 外部数据集: CSIC 2010')\n",
        "print('  - SMOTE 过采样: 是')\n",
        "print('  - 预计时间: 40-60 分钟\\n')\n",
        "\n",
        "cmd = [\n",
        "    sys.executable, '-u', 'scripts/train_v5.py',\n",
        "    '--version', 'v5.0.0-kaggle',\n",
        "    '--cv-splits', '10'\n",
        "]\n",
        "\n",
        "# 实时输出训练日志\n",
        "proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)\n",
        "for line in proc.stdout:\n",
        "    print(line, end='')\n",
        "\n",
        "ret = proc.wait()\n",
        "if ret != 0:\n",
        "    raise RuntimeError(f'训练失败，退出码: {ret}')\n",
        "\n",
        "print('\\n' + '='*80)\n",
        "print('✅ V5 训练完成！')\n",
        "print('='*80)\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 6) 查看训练结果\n",
        "from pathlib import Path\n",
        "import json\n",
        "\n",
        "manifest_path = Path('models/v5.0.0-kaggle/manifest.json')\n",
        "\n",
        "if manifest_path.exists():\n",
        "    with open(manifest_path, 'r', encoding='utf-8') as f:\n",
        "        manifest = json.load(f)\n",
        "\n",
        "    print('='*80)\n",
        "    print('📊 V5 训练结果')\n",
        "    print('='*80)\n",
        "    print(f\"\\n版本: {manifest['version']}\")\n",
        "    print(f\"训练时间: {manifest['trained_at']}\")\n",
        "    print(f\"数据量: {manifest['data_rows']} 条\")\n",
        "    print(f\"外部数据集: {'是' if manifest.get('external_data') else '否'}\")\n",
        "    if manifest.get('external_data'):\n",
        "        print(f\"外部数据量: {manifest.get('external_data_rows', 0)} 条\")\n",
        "    print(f\"特征数: {len(manifest['feature_list'])} 个\")\n",
        "    print(f\"类别数: {len(manifest['classes'])} 类\")\n",
        "\n",
        "    print('\\n' + '='*80)\n",
        "    print('🎯 模型性能')\n",
        "    print('='*80)\n",
        "\n",
        "    metrics = manifest['metrics']\n",
        "    \n",
        "    # 集成模型性能\n",
        "    if 'ensemble' in metrics:\n",
        "        e = metrics['ensemble']\n",
        "        print(f\"\\n🏆 集成模型（最终模型）\")\n",
        "        print(f\"  准确率: {e['test_accuracy']:.4f} ({e['test_accuracy']*100:.2f}%)\")\n",
        "        print(f\"  F1 分数: {e['test_f1']:.4f}\")\n",
        "        \n",
        "        if e['test_accuracy'] >= 0.998:\n",
        "            print(f\"\\n  🎉🎉🎉 恭喜！达到 99.8% 目标！\")\n",
        "        elif e['test_accuracy'] >= 0.995:\n",
        "            print(f\"\\n  🎉🎉 恭喜！达到 99.5% 目标！\")\n",
        "        elif e['test_accuracy'] >= 0.99:\n",
        "            print(f\"\\n  🎉 恭喜！达到 99% 目标！\")\n",
        "    \n",
        "    print('\\n' + '='*80)\n",
        "else:\n",
        "    print('❌ 未找到训练结果文件！')\n",
    ]))

    nb["cells"].append(_code_cell([
        "# 7) 打包并准备下载\n",
        "from pathlib import Path\n",
        "import shutil\n",
        "\n",
        "model_dir = Path('models/v5.0.0-kaggle')\n",
        "if model_dir.exists():\n",
        "    print('📦 打包模型文件...')\n",
        "    archive = shutil.make_archive(\n",
        "        '/kaggle/working/models_v5.0.0-kaggle',\n",
        "        'zip',\n",
        "        'models',\n",
        "        'v5.0.0-kaggle'\n",
        "    )\n",
        "    print(f'✅ 打包完成: {archive}')\n",
        "    print(f'\\n文件大小: {Path(archive).stat().st_size / 1024 / 1024:.2f} MB')\n",
        "    print('\\n💡 提示: 在 Kaggle 右侧 Output 面板可以下载此文件')\n",
        "else:\n",
        "    print('❌ 模型目录不存在！')\n",
    ]))

    nb["cells"].append(_md_cell([
        "## 说明\n",
        "\n",
        "### 训练模式\n",
        "\n",
        "本 Notebook 使用**前台训练模式**，适合 Kaggle 的 \"Save and Run All\" 功能：\n",
        "- ✅ 训练过程会实时显示在 notebook 中\n",
        "- ✅ 训练完成后自动继续执行后续 cell\n",
        "- ✅ 适合一次性完整训练\n",
        "- ⏱️ 预计耗时：40-60 分钟（10折交叉验证）\n",
        "\n",
        "### 快速测试\n",
        "\n",
        "如果想快速测试（5-10分钟），可以修改第5个cell的命令：\n",
        "```python\n",
        "cmd = [\n",
        "    sys.executable, '-u', 'scripts/train_v5.py',\n",
        "    '--version', 'v5.0.0-kaggle-fast',\n",
        "    '--no-cv'  # 禁用交叉验证\n",
        "]\n",
        "```\n",
        "\n",
        "### 后台训练模式（不推荐用于 Run All）\n",
        "\n",
        "如果需要后台训练（可以关闭页面），请手动执行以下命令：\n",
        "```python\n",
        "!python scripts/kaggle_bg_train_v5.py start --version v5.0.0-kaggle --cv-splits 10\n",
        "!python scripts/kaggle_bg_train_v5.py tail -n 100  # 查看日志\n",
        "```\n",
        "\n",
        "注意：后台模式不适合 \"Save and Run All\"，因为 kernel 退出会杀死后台进程。\n",
    ]))

    return nb


def main() -> None:
    notebook = generate_notebook()
    output = Path(__file__).resolve().parent.parent / "jupyter" / "kaggle_train_v5.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)
    print(f"Kaggle notebook generated: {output}")


if __name__ == "__main__":
    main()
