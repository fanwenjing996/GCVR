#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCVR — Ensemble of MKNN and SNN propagation results.

Fuse the refined scores of the two graphs (MKNN and SNN) via a fixed weight:
    S_final = w * S_snn + (1 - w) * S_mknn

The default weight w = 0.5 corresponds to the arithmetic mean of the two
graphs (the "Full GCVR" method reported in the paper).

Usage:
    python ensemble.py --mknn mknn_result.tsv --snn snn_result.tsv \
        --output final_result.tsv [--weight 0.5]
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_metrics(y_true, y_score, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    pre = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * pre * rec / (pre + rec + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(tp * tn - fp * fn) / (denom + 1e-9)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')
    return pre, rec, f1, acc, mcc, auc


def parse_args():
    p = argparse.ArgumentParser(description="GCVR — MKNN+SNN ensemble")
    p.add_argument("--mknn", required=True, help="MKNN propagation result TSV")
    p.add_argument("--snn", required=True, help="SNN propagation result TSV")
    p.add_argument("--output", required=True, help="Output final TSV")
    p.add_argument("--weight", type=float, default=0.5,
                   help="Weight of SNN score, w in [0,1] (default 0.5)")
    return p.parse_args()


def main():
    args = parse_args()
    w = args.weight

    mknn = pd.read_csv(args.mknn, sep="\t")
    snn = pd.read_csv(args.snn, sep="\t")

    # Align the two results on contig id
    mknn = mknn.set_index("contig")
    snn = snn.set_index("contig")
    common = mknn.index.intersection(snn.index)
    mknn = mknn.loc[common]
    snn = snn.loc[common]

    # Ensemble score (fixed weight)
    S_final = w * snn["refined_score"].values + (1 - w) * mknn["refined_score"].values

    result = pd.DataFrame({
        "contig": common,
        "initial_score": mknn["initial_score"].values,
        "refined_score": S_final,
    })
    result["refined_label"] = (result["refined_score"] >= 0.5).astype(int)

    # Evaluate if true labels are available
    if "true_label" in mknn.columns:
        result["true_label"] = mknn["true_label"].values
        m = compute_metrics(
            result["true_label"].values,
            result["refined_score"].values,
            result["refined_label"].values,
        )
        print("\n" + "=" * 60)
        print(f"Ensemble (w={w}) — Full GCVR")
        print("=" * 60)
        print(f"  Precision : {m[0]:.4f}")
        print(f"  Recall    : {m[1]:.4f}")
        print(f"  F1-score  : {m[2]:.4f}")
        print(f"  Accuracy  : {m[3]:.4f}")
        print(f"  MCC       : {m[4]:.4f}")
        print(f"  AUC       : {m[5]:.4f}")
        print("=" * 60)

    result.to_csv(args.output, sep="\t", index=False)
    print(f"\n✅ Saved ensemble result to: {args.output}")


if __name__ == "__main__":
    main()
