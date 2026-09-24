#!/usr/bin/env python3

import os
import multiprocessing


def get_safe_threads():
    cpu_count = multiprocessing.cpu_count()
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


def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def build_mknn_graph(args):
    print(f"[*] Reading master table: {args.master}")
    df = pd.read_csv(args.master)

    print(f"[*] Loading k-mer matrix: {args.kmer}")
    kmer_matrix = np.load(args.kmer).astype(np.float32)

    n = len(df)
    contigs = df["contig"].values
    gc_values = df["gc_content"].values

    # Extract lengths for adaptive KNN
    if "length" in df.columns:
        lengths = df["length"].values
    else:
        lengths = np.full(n, 5000.0)  # fallback

    if len(kmer_matrix) != n:
        raise ValueError(
            f"K-mer matrix rows ({len(kmer_matrix)}) != master rows ({n})"
        )

    f_cols = [f"func_{i}" for i in range(9)]
    viral_specific_cats = [0, 1, 2, 5]  # head, connector, tail, lysis
    has_phrog_data = all(c in df.columns for c in f_cols)

    if has_phrog_data:
        phrog_mat = df[f_cols].values.astype(np.float64)
        phrog_has_specific = (
            phrog_mat[:, viral_specific_cats].sum(axis=1) > 0
        )

        # Normalise PHROG vectors for cosine similarity
        phrog_norms = np.linalg.norm(phrog_mat, axis=1) + 1e-9
        phrog_normalised = phrog_mat / phrog_norms[:, None]

        phrog_coverage = (phrog_mat.sum(axis=1) > 0).mean()
        specific_frac = phrog_has_specific.mean()

        print(
            f"[*] PHROG available for edge boosting — "
            f"coverage={phrog_coverage:.1%}, "
            f"viral-specific={specific_frac:.1%}"
        )
    else:
        phrog_normalised = None
        phrog_has_specific = None
        print("[*] No PHROG data — skipping PHROG edge boost")

    if "predict_score" in df.columns:
        ppr_scores = df["predict_score"].values.astype(np.float64)
        has_ppr_scores = True
    else:
        ppr_scores = None
        has_ppr_scores = False

    # Short contigs: larger k
    # Long contigs: smaller k
    short_mask = lengths < 3000
    mid_mask = (lengths >= 3000) & (lengths < 5000)
    long_mask = lengths >= 5000

    k_short = args.topk + 8    # e.g. 15+8 = 23
    k_mid = args.topk + 3      # e.g. 15+3 = 18
    k_long = args.topk         # e.g. 15

    n_short = short_mask.sum()
    n_mid = mid_mask.sum()
    n_long = long_mask.sum()

    print(
        f"[*] Length distribution: "
        f"<3kb={n_short}, 3-5kb={n_mid}, >5kb={n_long}"
    )
    print(
        f"[*] Adaptive K: "
        f"short={k_short}, mid={k_mid}, long={k_long}"
    )

    # Run KNN with the largest k, then slice per contig
    max_k = max(k_short, k_mid, k_long) + 1  # +1 for self

    print(
        f"[*] Computing KNN "
        f"(k={max_k}, threads={args.n_jobs})..."
    )

    nn = NearestNeighbors(
        n_neighbors=min(max_k, n),
        metric="cosine",
        n_jobs=args.n_jobs
    )

    nn.fit(kmer_matrix)
    distances, indices = nn.kneighbors(kmer_matrix)

    # Convert to similarity and strip self-neighbour
    knn_sims_full = (1.0 - distances[:, 1:]).astype(np.float32)
    knn_idx_full = indices[:, 1:].astype(np.int32)

    del distances, indices
    gc.collect()

    # ── Build edges with per-contig adaptive k ─────────────────────
    print(
        "[*] Building edges "
        "(Mutual KNN + soft GC + sigmoid weighting)..."
    )

    # Build reverse lookup for mutual-KNN check
    neighbor_lookup = [set() for _ in range(n)]

    for i in range(n):
        if lengths[i] < 3000:
            ki = k_short
        elif lengths[i] < 5000:
            ki = k_mid
        else:
            ki = k_long

        ki = min(ki, knn_idx_full.shape[1])

        for j in knn_idx_full[i, :ki]:
            neighbor_lookup[i].add(int(j))

    edges = []

    for i in tqdm(range(n), desc="Building Edges"):
        li = lengths[i]

        if li < 3000:
            ki = k_short
            gc_tol = args.gc_diff * 1.3   # wider GC tolerance for short contigs
        elif li < 5000:
            ki = k_mid
            gc_tol = args.gc_diff * 1.1
        else:
            ki = k_long
            gc_tol = args.gc_diff

        ki = min(ki, knn_idx_full.shape[1])

        for rank in range(ki):
            j = int(knn_idx_full[i, rank])

            if i >= j:
                continue

            # Mutual KNN check (using j's own adaptive k)
            if i not in neighbor_lookup[j]:
                continue

            gc_diff = abs(gc_values[i] - gc_values[j])

            if gc_diff > gc_tol * 1.2:
                continue

            # GC similarity factor
            # (1.0 at GC_diff=0, decays to ~0.5 at gc_tol)
            gc_factor = sigmoid(
                (gc_tol - gc_diff) /
                (gc_tol * 0.2 + 1e-9)
            )

            sim = float(knn_sims_full[i, rank])

            soft_weight = sigmoid(
                (sim - args.w_min + 0.05) / 0.04
            )

            # Combine: sequence similarity × GC compatibility
            weight = sim * soft_weight * gc_factor

            if has_ppr_scores:
                ppr_agreement = (
                    1.0 - abs(ppr_scores[i] - ppr_scores[j])
                )
                ppr_boost = 0.75 + 0.25 * ppr_agreement
                weight *= ppr_boost

            if (
                phrog_normalised is not None
                and phrog_has_specific is not None
            ):
                has_i = phrog_has_specific[i]
                has_j = phrog_has_specific[j]

                if has_i and has_j:
                    # Both have viral-specific PHROG → functional connection
                    phrog_cos = float(
                        np.dot(
                            phrog_normalised[i],
                            phrog_normalised[j]
                        )
                    )

                    phrog_boost = (
                        1.0 + 0.20 * max(0.0, phrog_cos)
                    )

                    weight = min(
                        weight * phrog_boost,
                        0.995
                    )

            if weight >= 0.15:  # very loose floor to filter only garbage
                edges.append(
                    {
                        "contig_u": contigs[i],
                        "contig_v": contigs[j],
                        "weight": round(float(weight), 4),
                    }
                )

    print(f"[*] Finalising {len(edges)} edges...")

    edge_df = pd.DataFrame(edges)

    if len(edge_df) > 0:
        print(
            f"[*] Edge weight stats: "
            f"min={edge_df['weight'].min():.4f}, "
            f"median={edge_df['weight'].median():.4f}, "
            f"max={edge_df['weight'].max():.4f}"
        )

    edge_df.to_csv(
        args.output,
        sep="\t",
        index=False
    )

    n_isolated = sum(
        1 for s in neighbor_lookup if len(s) == 0
    )

    print("-" * 50)
    print("Graph Construction Summary:")
    print(f"  Nodes              : {n}")
    print(f"  Edges              : {len(edge_df)}")
    print(f"  Isolated nodes     : {n_isolated}")
    print(f"  Avg degree         : {2 * len(edge_df) / n:.1f}")

    print(
        f"  Short (<3kb) deg   : "
        f"{sum(len(neighbor_lookup[i]) for i in range(n) if lengths[i] < 3000) / max(1, n_short):.1f}"
    )

    print(
        f"  Mid   (3-5kb) deg  : "
        f"{sum(len(neighbor_lookup[i]) for i in range(n) if 3000 <= lengths[i] < 5000) / max(1, n_mid):.1f}"
    )

    print(
        f"  Long  (>5kb) deg   : "
        f"{sum(len(neighbor_lookup[i]) for i in range(n) if lengths[i] >= 5000) / max(1, n_long):.1f}"
    )

    print(f"  Output             : {args.output}")
    print("-" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Adaptive Mutual-KNN Graph Constructor (improved)"
    )

    parser.add_argument(
        "--master",
        required=True,
        help="Path to master table (CSV)"
    )

    parser.add_argument(
        "--kmer",
        required=True,
        help="Path to k-mer matrix (NPY)"
    )

    parser.add_argument(
        "--output",
        default="edges.tsv",
        help="Output edge list filename"
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=15,
        help="Base number of neighbours per node (default: 15)"
    )

    parser.add_argument(
        "--w_min",
        type=float,
        default=0.55,
        help="Sigmoid centre for similarity soft-threshold (default: 0.55)"
    )

    parser.add_argument(
        "--gc_diff",
        type=float,
        default=0.08,
        help="Base max GC content difference (default: 0.08)"
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=SAFE_THREADS,
        help=f"Threads for computation (default: {SAFE_THREADS})"
    )

    args = parser.parse_args()

    if args.n_jobs > 128:
        print(
            "[!] Warning: Capping n_jobs at 128 "
            "for OpenBLAS stability."
        )
        args.n_jobs = 128

    build_mknn_graph(args)


if __name__ == "__main__":
    main()
