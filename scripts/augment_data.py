#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据增强脚本 - 针对 Web 攻击特征
目标：将 11,000 条数据扩充到 30,000+ 条，提升模型泛化能力
"""

import argparse
import logging
import random
import re
import sys
import urllib.parse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wutong.config import STAGING_DIR

logger = logging.getLogger(__name__)

# 设置随机种子
random.seed(42)


class WebAttackAugmenter:
    """Web 攻击数据增强器"""
    
    def __init__(self):
        # SQL 注入同义词
        self.sql_synonyms = {
            "OR": ["OR", "||"],
            "AND": ["AND", "&&"],
            "1=1": ["1=1", "'1'='1'", "2>1", "1<2"],
            "UNION": ["UNION", "UNION ALL"],
            "SELECT": ["SELECT", "SeLeCt"],
        }
        
        # XSS 标签变体
        self.xss_tags = [
            "<script>", "<ScRiPt>", "<SCRIPT>",
            "<img src=x>", "<svg/onload=alert(1)>",
            "<iframe>", "<body onload=alert(1)>",
        ]
        
        # 编码方式
        self.encodings = ["none", "url", "double_url", "hex"]
    
    def augment_url_path(self, url_path: str) -> list[str]:
        """URL 路径增强"""
        variants = [url_path]
        
        # 1. 大小写变换
        if url_path and len(url_path) > 1:
            variants.append(url_path.capitalize())
            variants.append(url_path.upper())
            variants.append(url_path.lower())
        
        # 2. 路径分隔符变换
        if "/" in url_path:
            variants.append(url_path.replace("/", "//"))
            variants.append(url_path.replace("/", "/./"))
        
        # 3. URL 编码
        variants.append(urllib.parse.quote(url_path))
        
        # 4. 添加空字节
        if not url_path.endswith("%00"):
            variants.append(url_path + "%00")
        
        return list(set(variants))[:3]  # 最多返回3个变体
    
    def augment_sql_injection(self, text: str) -> list[str]:
        """SQL 注入增强"""
        if not text or not any(kw in text.upper() for kw in ["SELECT", "UNION", "OR", "AND"]):
            return [text]
        
        variants = [text]
        
        # 替换关键词
        for original, replacements in self.sql_synonyms.items():
            if original in text.upper():
                for replacement in replacements[:2]:
                    new_text = re.sub(
                        re.escape(original), 
                        replacement, 
                        text, 
                        flags=re.IGNORECASE
                    )
                    variants.append(new_text)
        
        # 添加注释
        if "--" not in text:
            variants.append(text + " --")
            variants.append(text + " /**/")
        
        # 空格变换
        variants.append(text.replace(" ", "/**/"))
        variants.append(text.replace(" ", "+"))
        
        return list(set(variants))[:4]
    
    def augment_xss(self, text: str) -> list[str]:
        """XSS 攻击增强"""
        if not text or "<" not in text:
            return [text]
        
        variants = [text]
        
        # 标签大小写变换
        for tag in ["script", "img", "svg", "iframe"]:
            if f"<{tag}" in text.lower():
                variants.append(text.replace(f"<{tag}", f"<{tag.upper()}"))
                variants.append(text.replace(f"<{tag}", f"<{tag.capitalize()}"))
        
        # 添加事件处理器
        if "onload" not in text.lower():
            variants.append(text.replace(">", " onload=alert(1)>"))
        
        # 编码变换
        variants.append(text.replace("<", "&lt;"))
        variants.append(text.replace("<", "\\x3c"))
        
        return list(set(variants))[:4]
    
    def augment_command_injection(self, text: str) -> list[str]:
        """命令注入增强"""
        if not text or not any(cmd in text for cmd in [";", "|", "&", "`", "$("]):
            return [text]
        
        variants = [text]
        
        # 命令分隔符变换
        separators = [";", "&&", "||", "|"]
        for sep in separators:
            if sep in text:
                for new_sep in separators:
                    if new_sep != sep:
                        variants.append(text.replace(sep, new_sep))
        
        # 添加命令
        common_cmds = ["ls", "cat", "whoami", "id"]
        for cmd in common_cmds:
            if cmd not in text:
                variants.append(f"{text};{cmd}")
        
        return list(set(variants))[:3]
    
    def augment_row(self, row: pd.Series, attack_type: str) -> list[pd.Series]:
        """增强单行数据"""
        augmented = []
        
        # 根据攻击类型选择增强策略
        if "SQL" in attack_type:
            url_variants = self.augment_url_path(row["url_path"])
            body_variants = self.augment_sql_injection(row["request_body"])
            
            for url in url_variants[:2]:
                for body in body_variants[:2]:
                    new_row = row.copy()
                    new_row["url_path"] = url
                    new_row["request_body"] = body
                    augmented.append(new_row)
        
        elif "XSS" in attack_type:
            url_variants = self.augment_url_path(row["url_path"])
            body_variants = self.augment_xss(row["request_body"])
            
            for url in url_variants[:2]:
                for body in body_variants[:2]:
                    new_row = row.copy()
                    new_row["url_path"] = url
                    new_row["request_body"] = body
                    augmented.append(new_row)
        
        elif "命令执行" in attack_type:
            body_variants = self.augment_command_injection(row["request_body"])
            for body in body_variants[:3]:
                new_row = row.copy()
                new_row["request_body"] = body
                augmented.append(new_row)
        
        else:
            # 其他攻击类型：简单的 URL 变换
            url_variants = self.augment_url_path(row["url_path"])
            for url in url_variants[:2]:
                new_row = row.copy()
                new_row["url_path"] = url
                augmented.append(new_row)
        
        return augmented


def augment_dataset(
    input_path: Path,
    output_path: Path,
    target_size: int = 30000,
    augment_ratio: float = 2.0,
) -> None:
    """
    增强数据集
    
    Parameters
    ----------
    input_path : 输入 parquet 文件
    output_path : 输出 parquet 文件
    target_size : 目标数据量
    augment_ratio : 增强倍数（每条数据生成多少个变体）
    """
    logger.info("=== 数据增强开始 ===")
    logger.info(f"输入文件: {input_path}")
    logger.info(f"目标数据量: {target_size}")
    
    # 读取数据
    df = pd.read_parquet(input_path)
    original_size = len(df)
    logger.info(f"原始数据量: {original_size}")
    
    # 统计各类别数量
    attack_counts = df["attack_type"].value_counts()
    logger.info(f"原始类别分布:\n{attack_counts}")
    
    # 初始化增强器
    augmenter = WebAttackAugmenter()
    
    # 增强数据
    augmented_rows = []
    
    for idx, row in df.iterrows():
        attack_type = row["attack_type"]
        
        # 保留原始数据
        augmented_rows.append(row)
        
        # 对攻击类型进行增强（正常访问不增强或少量增强）
        if attack_type == "正常访问":
            # 正常访问只增强 20%
            if random.random() < 0.2:
                variants = augmenter.augment_row(row, attack_type)
                augmented_rows.extend(variants[:1])
        else:
            # 攻击类型增强
            variants = augmenter.augment_row(row, attack_type)
            # 根据原始类别数量决定增强数量
            num_variants = min(
                len(variants),
                int(augment_ratio * (1 + 1000 / attack_counts[attack_type]))
            )
            augmented_rows.extend(variants[:num_variants])
        
        if (idx + 1) % 2000 == 0:
            logger.info(f"  处理进度: {idx + 1} / {original_size}")
    
    # 转换为 DataFrame
    df_augmented = pd.DataFrame(augmented_rows)
    
    # 如果超过目标大小，随机采样
    if len(df_augmented) > target_size:
        logger.info(f"数据量 {len(df_augmented)} 超过目标 {target_size}，进行采样")
        df_augmented = df_augmented.sample(n=target_size, random_state=42)
    
    # 打乱顺序
    df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存
    df_augmented.to_parquet(output_path, index=False)
    
    logger.info(f"\n=== 数据增强完成 ===")
    logger.info(f"原始数据量: {original_size}")
    logger.info(f"增强后数据量: {len(df_augmented)}")
    logger.info(f"增强倍数: {len(df_augmented) / original_size:.2f}x")
    logger.info(f"\n增强后类别分布:\n{df_augmented['attack_type'].value_counts()}")
    logger.info(f"\n输出文件: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Web 攻击数据增强")
    parser.add_argument(
        "-i", "--input",
        help="输入 parquet 文件路径"
    )
    parser.add_argument(
        "-o", "--output",
        help="输出 parquet 文件路径"
    )
    parser.add_argument(
        "-t", "--target-size",
        type=int,
        default=30000,
        help="目标数据量（默认: 30000）"
    )
    parser.add_argument(
        "-r", "--ratio",
        type=float,
        default=2.0,
        help="增强倍数（默认: 2.0）"
    )
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    
    # 确定输入文件
    if args.input:
        input_path = Path(args.input)
    else:
        parquets = sorted(STAGING_DIR.glob("*.parquet"))
        if not parquets:
            print("错误: 未找到 parquet 文件")
            sys.exit(1)
        input_path = parquets[-1]
    
    # 确定输出文件
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = STAGING_DIR / f"{input_path.stem}_augmented.parquet"
    
    augment_dataset(
        input_path,
        output_path,
        target_size=args.target_size,
        augment_ratio=args.ratio,
    )
    
    print(f"\n✔ 增强完成: {output_path}")


if __name__ == "__main__":
    main()
