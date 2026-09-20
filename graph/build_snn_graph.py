#!/usr/bin/env python3
"""SNN (Shared Nearest Neighbors) graph constructor for ensemble with MKNN."""
import argparse, pandas as pd, numpy as np, gc
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def main():
    p = argparse.ArgumentParser(description="SNN Graph Constructor")
    p.add_argument("--master", required=True)
    p.add_argument("--kmer", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--k_knn", type=int, default=30)
    p.add_argument("--k_snn", type=int, default=15)
    p.add_argument("--min_shared", type=int, default=3)
    p.add_argument("--min_jaccard", type=float, default=0.15)
    p.add_argument("--min_sim", type=float, default=0.5)
    p.add_argument("--gc_diff", type=float, default=0.10)
    p.add_argument("--n_jobs", type=int, default=10)
    args = p.parse_args()

    df = pd.read_csv(args.master)
    kmer = np.load(args.kmer).astype(np.float32)
    n = len(df)
    contigs = df["contig"].values
    gc_values = df["gc_content"].values

    k_knn = min(args.k_knn + 1, n)
    print(f"[*] Computing KNN (k={args.k_knn})...")
    nn = NearestNeighbors(n_neighbors=k_knn, metric='cosine', n_jobs=args.n_jobs)
    nn.fit(kmer)
    distances, indices = nn.kneighbors(kmer)
    knn_idx = indices[:, 1:args.k_knn+1].astype(np.int32)
    knn_sims = (1.0 - distances[:, 1:args.k_knn+1]).astype(np.float32)
    del distances, indices; gc.collect()

    # Neighbor sets for SNN
    neighbor_sets = [set(knn_idx[i, :args.k_snn].tolist()) for i in range(n)]

    edges = []
    for i in tqdm(range(n), desc="SNN Edges"):
        for rank, j in enumerate(knn_idx[i]):
            if i >= j:
                continue
            shared = len(neighbor_sets[i] & neighbor_sets[j])
            union = len(neighbor_sets[i] | neighbor_sets[j])
            jaccard = shared / (union + 1e-9)
            if shared < args.min_shared and jaccard < args.min_jaccard:
                continue
            gc_diff = abs(gc_values[i] - gc_values[j])
            if gc_diff > args.gc_diff:
                continue
            sim = float(knn_sims[i, rank])
            if sim < args.min_sim:
                continue
            weight = 0.6 * sim + 0.4 * jaccard
            if weight >= 0.25:
                edges.append({"contig_u": contigs[i], "contig_v": contigs[j], "weight": round(weight, 4)})

    edge_df = pd.DataFrame(edges)
    print(f"[*] {len(edge_df)} edges, avg degree: {2*len(edge_df)/n:.1f}")
    edge_df.to_csv(args.output, sep='\t', index=False)
    print(f"  Saved: {args.output}")

if __name__ == "__main__":
    main()
