#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from Bio import SeqIO
import argparse
import sys

def run_build(args):
    
    # 提取物理特征 
    print("▶ 1/4 提取 GC 含量...")
    gc_data = []
    for record in SeqIO.parse(args.fasta, "fasta"):
        seq = str(record.seq).upper()
        seq_len = len(seq)

        gc = (seq.count('G') + seq.count('C')) / seq_len if seq_len > 0 else 0

        gc_data.append({
            'contig': record.id,
            'gc_content': gc,
            'length': seq_len
        })

    df_gc = pd.DataFrame(gc_data)

    # 提取 PHROG 功能特征 (func_0-8)
    print("▶ 2/4 提取 PHROG 功能向量...")
    CATEGORIES = [
        'head and packaging', 'connector', 'tail', 
        'integration and excision', 'lysis', 
        'DNA, RNA and nucleotide metabolism', 'transcription regulation', 
        'moron, auxiliary metabolic gene and host takeover', 'other'
    ]
    cat_to_idx = {cat: i for i, cat in enumerate(CATEGORIES)}
    
    phrog_raw = pd.read_csv(args.phrog, sep='\t')
    
    def compute_phrog(group):  #对每个contig统计PHROG
        vec = np.zeros(9)
        for _, row in group.iterrows():
            cat = row['phrog_category']
            if cat in cat_to_idx:
                vec[cat_to_idx[cat]] += row.get('score', 1.0)
        return pd.Series({f'func_{i}': vec[i] for i in range(9)})

    df_phrog = phrog_raw.groupby('contig').apply(compute_phrog).reset_index()

    
    print("▶ 3/4 加载预测分数...")
    df_ppr = pd.read_csv(args.ppr)
    label_col = [c for c in ['label', 'true_label'] if c in df_ppr.columns]
    keep_cols = ['contig', 'predict_score'] + label_col
    df_ppr = df_ppr[keep_cols]

    #  以 K-mer ID 顺序为基准进行合并
    print("▶ 4/4 组装 Master Table 并对齐 row_id...")
    df_master = pd.read_csv(args.kmer_id)
    df_master['row_id'] = df_master.index

    
    df_master = df_master.merge(df_gc, on='contig', how='left')
    df_master = df_master.merge(df_ppr, on='contig', how='left')
    df_master = df_master.merge(df_phrog, on='contig', how='left')

    # 填充缺失值 
    f_cols = [f"func_{i}" for i in range(9)]
    df_master[f_cols] = df_master[f_cols].fillna(0)
    df_master['predict_score'] = df_master['predict_score'].fillna(0.5) 

    # 检查
    n_missing_gc = df_master['gc_content'].isna().sum()
    if n_missing_gc > 0:
        print(f" 警告: 有 {n_missing_gc} 个 contig 在 FASTA 中缺失 GC 信息")
    n_missing_len = df_master['length'].isna().sum()
    if n_missing_len > 0:
        print(f" 警告: 有 {n_missing_len} 个 contig 缺失 length")

    df_master.to_csv(args.output, index=False)
    print(f" 任务完成！Master Table 已保存至: {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build a unified Master Table for downstream graph refinement.")
    # 输入
    parser.add_argument("--fasta", required=True, help="Input FASTA file path")
    parser.add_argument("--phrog", required=True, help="Input PHROG hit TSV path")
    parser.add_argument("--ppr", required=True, help="Input prediction CSV (contains contig, predict_score)")
    parser.add_argument("--kmer_id", required=True, help="Input 3-mer ID CSV to ensure row alignment")
    # 输出
    parser.add_argument("--output", required=True, help="Output master table CSV path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_build(args)
