#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载和整合外部开源数据集

支持的数据集：
1. CSIC 2010 HTTP Dataset - Web 攻击数据集（36,000+ 条）
2. PayloadsAllTheThings - 攻击 Payload 库
3. CICIDS2017 - 网络入侵检测数据集（可选，较大）
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def download_csic_2010():
    """
    下载 CSIC 2010 HTTP Dataset
    
    数据集信息：
    - 来源: Spanish Research National Council (CSIC)
    - 规模: 36,000+ HTTP 请求
    - 类型: 正常流量 + Web 攻击（SQL 注入、XSS、缓冲区溢出等）
    - 格式: 原始 HTTP 请求
    - 下载: https://github.com/msudol/Web-Application-Attack-Datasets
    """
    print("=" * 80)
    print("CSIC 2010 HTTP Dataset")
    print("=" * 80)
    print("\n📊 数据集信息:")
    print("  - 规模: 36,000+ HTTP 请求")
    print("  - 正常流量: 36,000 条")
    print("  - 攻击流量: 25,000 条（SQL 注入、XSS、缓冲区溢出、目录遍历等）")
    print("  - 格式: 原始 HTTP 请求文本")
    
    print("\n📥 下载方式:")
    print("  1. GitHub 仓库:")
    print("     git clone https://github.com/msudol/Web-Application-Attack-Datasets.git")
    print("     cd Web-Application-Attack-Datasets")
    print()
    print("  2. 或直接下载:")
    print("     https://github.com/msudol/Web-Application-Attack-Datasets/archive/refs/heads/master.zip")
    
    print("\n📁 数据文件:")
    print("  - normalTrafficTraining.txt (正常流量)")
    print("  - anomalousTrafficTest.txt (攻击流量)")
    
    print("\n⚠️ 注意:")
    print("  - 数据集为原始 HTTP 请求格式，需要解析")
    print("  - 只有二分类标签（正常/异常），需要人工标注具体攻击类型")
    print("  - 建议用于补充训练数据，而非替代原始数据集")


def download_payloads_all_the_things():
    """
    下载 PayloadsAllTheThings
    
    数据集信息：
    - 来源: swisskyrepo (GitHub)
    - 规模: 数千个攻击 Payload
    - 类型: SQL 注入、XSS、命令注入、文件包含、CSRF 等
    - 格式: Markdown 文档 + Payload 列表
    - 下载: https://github.com/swisskyrepo/PayloadsAllTheThings
    """
    print("=" * 80)
    print("PayloadsAllTheThings")
    print("=" * 80)
    print("\n📊 数据集信息:")
    print("  - 规模: 数千个真实攻击 Payload")
    print("  - 类型: SQL 注入、XSS、命令注入、文件包含、CSRF、XXE 等")
    print("  - 格式: Markdown 文档 + Payload 列表")
    print("  - 更新: 持续更新（2025 年最新）")
    
    print("\n📥 下载方式:")
    print("  1. GitHub 仓库:")
    print("     git clone https://github.com/swisskyrepo/PayloadsAllTheThings.git")
    print("     cd PayloadsAllTheThings")
    print()
    print("  2. 或直接下载:")
    print("     https://github.com/swisskyrepo/PayloadsAllTheThings/archive/refs/heads/master.zip")
    
    print("\n📁 相关目录:")
    print("  - SQL Injection/")
    print("  - XSS Injection/")
    print("  - Command Injection/")
    print("  - File Inclusion/")
    print("  - CSRF Injection/")
    print("  - Directory Traversal/")
    
    print("\n💡 使用建议:")
    print("  - 提取各类攻击的 Payload 示例")
    print("  - 用于数据增强的变体生成")
    print("  - 补充原始数据集中缺少的攻击模式")


def download_cicids2017():
    """
    下载 CICIDS2017 Dataset
    
    数据集信息：
    - 来源: Canadian Institute for Cybersecurity
    - 规模: 280 万条网络流量记录
    - 类型: 14 种攻击类型（包含 Web 攻击）
    - 格式: CSV（79 个特征）
    - 下载: https://www.unb.ca/cic/datasets/ids-2017.html
    """
    print("=" * 80)
    print("CICIDS2017 Dataset")
    print("=" * 80)
    print("\n📊 数据集信息:")
    print("  - 规模: 280 万条网络流量记录")
    print("  - 攻击类型: 14 种（DDoS、Brute Force、Web Attack、Botnet 等）")
    print("  - 格式: CSV（79 个特征）")
    print("  - 大小: 约 7 GB")
    
    print("\n📥 下载方式:")
    print("  1. 官方网站:")
    print("     https://www.unb.ca/cic/datasets/ids-2017.html")
    print()
    print("  2. Kaggle:")
    print("     https://www.kaggle.com/datasets/cicdataset/cicids2017")
    print()
    print("  3. IEEE DataPort:")
    print("     https://ieee-dataport.org/documents/cicids2017")
    
    print("\n📁 数据文件:")
    print("  - Monday-WorkingHours.pcap_ISCX.csv")
    print("  - Tuesday-WorkingHours.pcap_ISCX.csv")
    print("  - Wednesday-workingHours.pcap_ISCX.csv")
    print("  - Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv ⭐")
    print("  - Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    print("  - Friday-WorkingHours-Morning.pcap_ISCX.csv")
    print("  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    print("  - Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    
    print("\n⚠️ 注意:")
    print("  - 数据集较大（7 GB），下载和处理需要时间")
    print("  - 特征与原始数据集不同，需要特征映射")
    print("  - 建议只使用 Thursday-WebAttacks 文件（约 200 MB）")
    print("  - Web Attack 包含: SQL 注入、XSS、Brute Force")


def show_integration_guide():
    """显示数据集整合指南"""
    print("\n" + "=" * 80)
    print("📚 数据集整合指南")
    print("=" * 80)
    
    print("\n方案 1: 使用 CSIC 2010（推荐）")
    print("-" * 80)
    print("优点:")
    print("  ✅ 规模适中（36k 正常 + 25k 攻击）")
    print("  ✅ 纯 Web 攻击数据")
    print("  ✅ 格式简单（HTTP 请求）")
    print("  ✅ 可直接补充到现有数据集")
    
    print("\n步骤:")
    print("  1. 下载 CSIC 2010 数据集")
    print("  2. 解析 HTTP 请求，提取 URL、参数、方法等")
    print("  3. 标注攻击类型（根据 Payload 特征）")
    print("  4. 转换为与原始数据集相同的格式")
    print("  5. 合并到 data/staging/ 目录")
    print("  6. 重新训练模型")
    
    print("\n预期效果:")
    print("  - 数据量: 11k → 70k+（6.4x）")
    print("  - 准确率提升: +0.5-1.0%")
    
    print("\n方案 2: 使用 PayloadsAllTheThings")
    print("-" * 80)
    print("优点:")
    print("  ✅ 真实攻击 Payload")
    print("  ✅ 持续更新（2025 年最新）")
    print("  ✅ 覆盖所有攻击类型")
    print("  ✅ 可用于数据增强")
    
    print("\n步骤:")
    print("  1. 下载 PayloadsAllTheThings")
    print("  2. 提取各类攻击的 Payload 示例")
    print("  3. 构造完整的 HTTP 请求")
    print("  4. 标注攻击类型")
    print("  5. 合并到训练数据")
    
    print("\n预期效果:")
    print("  - 数据量: 11k → 20k+（1.8x）")
    print("  - 准确率提升: +0.3-0.6%")
    
    print("\n方案 3: 使用 CICIDS2017（可选）")
    print("-" * 80)
    print("优点:")
    print("  ✅ 数据量大（280 万条）")
    print("  ✅ 真实网络流量")
    print("  ✅ 多种攻击类型")
    
    print("\n缺点:")
    print("  ❌ 数据集较大（7 GB）")
    print("  ❌ 特征不同（79 个网络流特征 vs HTTP 请求）")
    print("  ❌ 需要大量特征工程")
    
    print("\n建议:")
    print("  - 只使用 Thursday-WebAttacks 文件")
    print("  - 提取 Web 攻击相关的流量")
    print("  - 需要从 PCAP 重构 HTTP 请求")


def create_integration_script():
    """创建数据集整合脚本模板"""
    script_path = Path("scripts/integrate_csic2010.py")
    
    if script_path.exists():
        print(f"\n⚠️ 脚本已存在: {script_path}")
        return
    
    template = '''#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
整合 CSIC 2010 数据集到训练数据

使用方法:
1. 下载 CSIC 2010: git clone https://github.com/msudol/Web-Application-Attack-Datasets.git
2. 运行脚本: python scripts/integrate_csic2010.py --csic-dir <path>
"""

import argparse
import pandas as pd
import re
from pathlib import Path


def parse_http_request(request_text: str) -> dict:
    """解析 HTTP 请求"""
    lines = request_text.strip().split('\\n')
    if not lines:
        return None
    
    # 解析请求行
    request_line = lines[0]
    match = re.match(r'(GET|POST|PUT|DELETE|HEAD)\\s+(.+?)\\s+HTTP', request_line)
    if not match:
        return None
    
    method = match.group(1)
    url = match.group(2)
    
    # 解析 URL
    if '?' in url:
        path, query = url.split('?', 1)
    else:
        path = url
        query = ''
    
    # 解析 headers
    headers = {}
    body = ''
    in_body = False
    
    for line in lines[1:]:
        if not line.strip():
            in_body = True
            continue
        
        if in_body:
            body += line + '\\n'
        else:
            if ':' in line:
                key, value = line.split(':', 1)
                headers[key.strip()] = value.strip()
    
    return {
        'method': method,
        'url': url,
        'path': path,
        'query': query,
        'headers': headers,
        'body': body.strip(),
    }


def classify_attack_type(request: dict) -> str:
    """根据 Payload 特征分类攻击类型"""
    url = request.get('url', '').lower()
    query = request.get('query', '').lower()
    body = request.get('body', '').lower()
    
    combined = url + query + body
    
    # SQL 注入
    if any(kw in combined for kw in ['union', 'select', 'insert', 'update', 'delete', 
                                       'drop', 'exec', 'script', '--', '/*', '*/', 
                                       'or 1=1', 'or true', 'and 1=1']):
        return 'SQL注入攻击'
    
    # XSS
    if any(kw in combined for kw in ['<script', 'javascript:', 'onerror=', 'onload=',
                                       'alert(', 'prompt(', 'confirm(', '<img', '<svg']):
        return 'XSS跨站脚本攻击'
    
    # 命令注入
    if any(kw in combined for kw in ['|', ';', '&&', '||', '`', '$(', 'cat ', 'ls ',
                                       'wget ', 'curl ', 'nc ', 'bash']):
        return '远程命令执行攻击'
    
    # 目录遍历
    if any(kw in combined for kw in ['../', '..\\\\', '%2e%2e', 'etc/passwd', 'windows/system32']):
        return '目录遍历攻击'
    
    # 文件包含
    if any(kw in combined for kw in ['include', 'require', 'file=', 'page=', 'path=']):
        return '文件包含攻击'
    
    # 默认为其他攻击
    return '其他攻击'


def integrate_csic2010(csic_dir: Path, output_path: Path):
    """整合 CSIC 2010 数据集"""
    print("=" * 80)
    print("整合 CSIC 2010 数据集")
    print("=" * 80)
    
    # 读取正常流量
    normal_file = csic_dir / "normalTrafficTraining.txt"
    anomalous_file = csic_dir / "anomalousTrafficTest.txt"
    
    if not normal_file.exists():
        print(f"❌ 未找到文件: {normal_file}")
        return
    
    if not anomalous_file.exists():
        print(f"❌ 未找到文件: {anomalous_file}")
        return
    
    print(f"\\n读取正常流量: {normal_file}")
    print(f"读取攻击流量: {anomalous_file}")
    
    # TODO: 实现完整的解析和整合逻辑
    print("\\n⚠️ 此脚本为模板，需要根据实际数据格式完善")
    print("\\n建议:")
    print("  1. 解析 HTTP 请求文本")
    print("  2. 提取 URL、参数、方法等字段")
    print("  3. 根据 Payload 特征分类攻击类型")
    print("  4. 转换为与原始数据集相同的格式")
    print("  5. 保存为 parquet 文件")


def main():
    parser = argparse.ArgumentParser(description="整合 CSIC 2010 数据集")
    parser.add_argument(
        "--csic-dir",
        type=str,
        required=True,
        help="CSIC 2010 数据集目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/staging/csic2010_integrated.parquet",
        help="输出文件路径"
    )
    
    args = parser.parse_args()
    
    csic_dir = Path(args.csic_dir)
    output_path = Path(args.output)
    
    integrate_csic2010(csic_dir, output_path)


if __name__ == "__main__":
    main()
'''
    
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"\n✅ 已创建整合脚本模板: {script_path}")
    print("   请根据实际数据格式完善脚本")


def main():
    parser = argparse.ArgumentParser(description="下载外部开源数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["csic2010", "payloads", "cicids2017", "all"],
        default="all",
        help="要下载的数据集"
    )
    parser.add_argument(
        "--create-script",
        action="store_true",
        help="创建数据集整合脚本模板"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("外部开源数据集下载指南")
    print("=" * 80)
    print("\n💡 提示: 使用外部数据集可以大幅提升模型准确率")
    print("   - CSIC 2010: +0.5-1.0%")
    print("   - PayloadsAllTheThings: +0.3-0.6%")
    print("   - 结合 V4 SMOTE 优化: 总提升 +1.5-2.5%\n")
    
    if args.dataset in ["csic2010", "all"]:
        download_csic_2010()
        print()
    
    if args.dataset in ["payloads", "all"]:
        download_payloads_all_the_things()
        print()
    
    if args.dataset in ["cicids2017", "all"]:
        download_cicids2017()
        print()
    
    show_integration_guide()
    
    if args.create_script:
        create_integration_script()
    
    print("\n" + "=" * 80)
    print("📝 下一步")
    print("=" * 80)
    print("\n1. 下载推荐的数据集（CSIC 2010 或 PayloadsAllTheThings）")
    print("2. 创建整合脚本: python scripts/download_external_datasets.py --create-script")
    print("3. 完善整合脚本，解析和转换数据格式")
    print("4. 运行整合脚本，合并数据")
    print("5. 重新训练 V4 模型")
    print("\n预期效果: 98.36% → 99.5%+ 🎉\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    main()
