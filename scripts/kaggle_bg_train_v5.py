#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Manage V5 training as a background job on Kaggle/Linux.

Usage examples:
  python scripts/kaggle_bg_train_v5.py start --version v5.0.0-kaggle --no-cv
  python scripts/kaggle_bg_train_v5.py status
  python scripts/kaggle_bg_train_v5.py tail -n 80
  python scripts/kaggle_bg_train_v5.py stop
"""

import argparse
import os
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "data" / "outputs" / "kaggle_bg_v5"
PID_FILE = RUN_DIR / "train.pid"
LOG_FILE = RUN_DIR / "train.log"


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def cmd_start(args: argparse.Namespace) -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    pid = _read_pid()
    if pid and _is_running(pid):
        print(f"已有训练进程在运行，PID={pid}")
        print(f"日志: {LOG_FILE}")
        return 0

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_v5.py"),
        "--version",
        args.version,
        "--cv-splits",
        str(args.cv_splits),
    ]

    if args.input:
        cmd.extend(["--input", args.input])
    if args.external_data is not None:
        cmd.extend(["--external-data", args.external_data])
    if args.no_cv:
        cmd.append("--no-cv")
    if args.no_stacking:
        cmd.append("--no-stacking")
    if args.no_smote:
        cmd.append("--no-smote")

    with open(LOG_FILE, "a", encoding="utf-8") as logf:
        logf.write("\n" + "=" * 80 + "\n")
        logf.write(f"[{datetime.now().isoformat()}] START CMD: {' '.join(cmd)}\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    print(f"已启动后台训练，PID={proc.pid}")
    print(f"日志: {LOG_FILE}")
    print("查看日志: python scripts/kaggle_bg_train_v5.py tail -n 80")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    pid = _read_pid()
    if not pid:
        print("未找到 PID 文件，当前没有后台任务。")
        return 0
    if _is_running(pid):
        print(f"后台训练运行中，PID={pid}")
    else:
        print(f"PID 文件存在但进程已结束，PID={pid}")
    print(f"日志: {LOG_FILE}")
    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    if not LOG_FILE.exists():
        print(f"日志文件不存在: {LOG_FILE}")
        return 1
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    n = max(1, args.n)
    for line in lines[-n:]:
        print(line)
    return 0


def cmd_stop(_: argparse.Namespace) -> int:
    pid = _read_pid()
    if not pid:
        print("未找到 PID 文件，当前没有后台任务。")
        return 0
    if not _is_running(pid):
        print(f"进程已结束，PID={pid}")
        return 0

    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"已发送停止信号到进程组 PID={pid}")
    except Exception:
        os.kill(pid, signal.SIGTERM)
        print(f"已发送停止信号到 PID={pid}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kaggle 后台训练 V5 任务管理")
    sub = parser.add_subparsers(dest="action", required=True)

    p_start = sub.add_parser("start", help="启动后台训练")
    p_start.add_argument("--version", default="v5.0.0-kaggle", help="模型版本")
    p_start.add_argument("--cv-splits", type=int, default=10, help="CV 折数")
    p_start.add_argument("--input", type=str, help="输入 parquet 路径")
    p_start.add_argument("--external-data", type=str, help="外部数据目录")
    p_start.add_argument("--no-cv", action="store_true", help="禁用交叉验证")
    p_start.add_argument("--no-stacking", action="store_true", help="使用 Voting")
    p_start.add_argument("--no-smote", action="store_true", help="禁用 SMOTE")
    p_start.set_defaults(func=cmd_start)

    p_status = sub.add_parser("status", help="查看任务状态")
    p_status.set_defaults(func=cmd_status)

    p_tail = sub.add_parser("tail", help="查看日志末尾")
    p_tail.add_argument("-n", type=int, default=60, help="显示最后 n 行")
    p_tail.set_defaults(func=cmd_tail)

    p_stop = sub.add_parser("stop", help="停止后台任务")
    p_stop.set_defaults(func=cmd_stop)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
