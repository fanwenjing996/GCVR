#!/usr/bin/env python3

"""基于序列相似性与生物学约束的图构建"""
import os
import multiprocessing

def get_safe_threads():
    cpu_count = multiprocessing.cpu_count()
    # 设定安全上限为64，避免 多进程 × BLAS多线程 → 线程爆炸
    return min(cpu_count, 64)

SAFE_THREADS = get_safe_threads()
os.environ["OPENBLAS_NUM_THREADS"] = str(SAFE_THREADS)
os.environ["MKL_NUM_THREADS"] = str(SAFE_THREADS)
os.environ["OMP_NUM_THREADS"] = str(SAFE_THREADS)

import argparse
import pandas as pd
import numpy as np
import gc
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def build_mknn_graph(args):
   
    print(f"[*] Reading master table: {args.master}")
    df = pd.read_csv(args.master)
    
    print(f"[*] Loading k-mer matrix: {args.kmer}")
    
    kmer_matrix = np.load(args.kmer).astype(np.float32)
    
    n = len(df)
    contigs = df["contig"].values
    gc_values = df["gc_content"].values

    #检查对齐：保证每个 contig 对应一行 k-mer 向量
    if len(kmer_matrix) != n:
        print(f"[!] Warning: Matrix rows ({len(kmer_matrix)}) != Master table rows ({n})")

    #  计算 KNN
    print(f"[*] Computing KNN (k={args.topk}, threads={args.n_jobs})...")
    nn = NearestNeighbors(n_neighbors=args.topk + 1, metric='cosine', n_jobs=args.n_jobs)
    nn.fit(kmer_matrix)
    
    # 获取距离和索引
    distances, indices = nn.kneighbors(kmer_matrix)
    
    knn_idx = indices[:, 1:].astype(np.int32)
    knn_sims = (1.0 - distances[:, 1:]).astype(np.float32)
    
    # 内存回收
    del distances, indices
    gc.collect()

    # 构建 Mutual KNN 图逻辑
    print("[*] Filtering Mutual KNN edges and applying GC constraints...")
    edges = []
    
    
    neighbor_lookup = [set(neighbors) for neighbors in knn_idx]

    for i in tqdm(range(n), desc="Building Edges"):
        # 遍历节点 i 的所有候选邻居 j
        for rank, j in enumerate(knn_idx[i]):
            
            if i >= j:
                continue
         
            if i in neighbor_lookup[j]:
                # 约束 A: GC 含量差异
                if abs(gc_values[i] - gc_values[j]) > args.gc_diff:
                    continue
                
                # 约束 B: 相似度权重
                weight = knn_sims[i, rank]
                if weight >= args.w_min:
                    edges.append({
                        "contig_u": contigs[i],
                        "contig_v": contigs[j],
                        "weight": round(float(weight), 4)
                    })

 
    print(f"[*] Finalizing {len(edges)} edges...")
    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(args.output, sep='\t', index=False)
    
    print("-" * 40)
    print(f"Graph Construction Summary:")
    print(f"  - Nodes: {n}")
    print(f"  - Edges: {len(edge_df)}")
    print(f"  - Output: {args.output}")
    print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Mutual-KNN Graph Constructor")
    
    parser.add_argument("--master", required=True, help="Path to master table (CSV)")
    parser.add_argument("--kmer", required=True, help="Path to k-mer matrix (NPY)")
    parser.add_argument("--output", default="edges.tsv", help="Output edge list filename")
    
    # 构图核心参数
    parser.add_argument("--topk", type=int, default=15, help="Number of neighbors per node (default: 15)")
    parser.add_argument("--w_min", type=float, default=0.6, help="Minimum similarity threshold (default: 0.6)")
    parser.add_argument("--gc_diff", type=float, default=0.08, help="Max GC content difference (default: 0.08)")
    
    
    parser.add_argument("--n_jobs", type=int, default=SAFE_THREADS, 
                        help=f"Threads for computation (default: {SAFE_THREADS})")

    args = parser.parse_args()
    
    # 二次保护：防止手动输入过大的 n_jobs 导致 OpenBLAS 报错
    if args.n_jobs > 128:
        print(f"[!] Warning: Capping n_jobs at 128 for OpenBLAS stability.")
        args.n_jobs = 128

    build_mknn_graph(args)

if __name__ == "__main__":
    main()
