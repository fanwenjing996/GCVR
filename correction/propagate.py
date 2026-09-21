#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCVR: Confidence-Aware Viral Identification through Graph Propagation.

"""

import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="GCVR — Graph Propagation")

    parser.add_argument("--master", required=True, help="Master table (CSV)")
    parser.add_argument("--edges", required=True, help="Edge list file (TSV)")
    parser.add_argument("--output", required=True, help="Output refined TSV")

    parser.add_argument("--iter", type=int, default=8,
                        help="Number of propagation iterations (default: 8)")
    parser.add_argument("--gamma_phrog", type=float, default=0.15,
                        help="Strength of PHROG functional prior (default: 0.15)")
    parser.add_argument("--gamma_push", type=float, default=0.10,
                        help="Strength of neighbour consensus push (default: 0.10)")
    parser.add_argument("--temperature", type=float, default=0.55,
                        help="Score sharpening temperature (<1 sharpens, >1 softens, default: 0.55)")
    parser.add_argument("--phrog_weights", type=str, default="",
                        help="Comma-separated custom PHROG weights (9 values), e.g. '0.68,0.80,...'")
    parser.add_argument("--no_phrog", action="store_true",
                        help="Ablation: disable PHROG functional prior")
    parser.add_argument("--no_confidence", action="store_true",
                        help="Ablation: use fixed q=0.5 (naive graph propagation, no confidence/purity/push)")

    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def compute_metrics(yt, yp):
    tp = ((yt == 1) & (yp == 1)).sum()
    fp = ((yt == 0) & (yp == 1)).sum()
    fn = ((yt == 1) & (yp == 0)).sum()
    tn = ((yt == 0) & (yp == 0)).sum()

    pre = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * pre * rec / (pre + rec + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    # MCC
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(tp * tn - fp * fn) / (denom + 1e-9)

    return pre, rec, f1, acc, mcc


def run_propagation():
    args = parse_args()

    print(f"\n Starting GCVR Propagation")
    print(f"[*] Master : {args.master}")
    print(f"[*] Edges  : {args.edges}")
    print(f"[*] iter={args.iter}, gamma_phrog={args.gamma_phrog}, gamma_push={args.gamma_push}")

    # ── Load master table ──────────────────────────────────────────
    df = pd.read_csv(args.master)

    nodes = df["contig"].tolist()
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n_nodes = len(nodes)

    # Initial scores & labels
    score_dict = dict(zip(df["contig"], df["predict_score"]))
    label_col = "label" if "label" in df.columns else None
    label_dict = dict(zip(df["contig"], df[label_col])) if label_col else None

    # Length map (for length-aware confidence)
    if "length" in df.columns:
        length_map = dict(zip(df["contig"], df["length"]))
    else:
        length_map = {n: 3000.0 for n in nodes}  # fallback

    # ── PHROG signal with category specificity ─────────────────────
    # # PHROG category weights used in the default configuration
    w_p = np.array([0.86, 0.94, 0.79, 0.63, 0.55, 0.75, 0.64, 0.34, 0.43])
    # Dataset-adaptive weights can be passed via --phrog_weights
    if hasattr(args, 'phrog_weights') and args.phrog_weights:
        w_p = np.array([float(x) for x in args.phrog_weights.split(',')])
        print(f"[*] Custom PHROG weights: {np.round(w_p, 3)}")
    # Virus-specificity per category: head(0), connector(1), tail(2), lysis(5)
    # are highly diagnostic of viruses. moron(7), other(8) are ambiguous.
    virus_specificity_weights = np.array([1.0, 1.0, 1.0, 0.5, 0.3, 0.8, 0.4, 0.15, 0.1])
    # Hard gate: categories 0,1,2,5 are the ONLY reliable viral indicators
    viral_specific_cat_indices = [0, 1, 2, 5]  # head, connector, tail, lysis
    f_cols = [f"func_{i}" for i in range(9)]

    has_phrog = all(c in df.columns for c in f_cols)
    if has_phrog:
        phrog_vals = df[f_cols].values
        phrog_total = phrog_vals.sum(axis=1)
        # Base virus signal (original weighting)
        p_signal_vec = (phrog_vals @ w_p) / (phrog_total + 1e-9)
        # Specificity score
        p_specificity_vec = (phrog_vals @ virus_specificity_weights) / (phrog_total + 1e-9)
        # Combined signal
        p_combined = p_signal_vec * np.clip(p_specificity_vec / 0.6, 0.0, 1.0)
        # Hard reliability gate: contig has hits in viral-specific categories
        phrog_has_specific = (phrog_vals[:, viral_specific_cat_indices].sum(axis=1) > 0)

        virus_signal = dict(zip(df["contig"], p_signal_vec))
        virus_specificity = dict(zip(df["contig"], p_specificity_vec))
        virus_signal_combined = dict(zip(df["contig"], p_combined))
        virus_has_specific = dict(zip(df["contig"], phrog_has_specific))

        phrog_coverage = (phrog_total > 0).mean()
        specific_frac = phrog_has_specific.mean()
        print(f"[*] PHROG coverage: {phrog_coverage:.1%} (viral-specific: {specific_frac:.1%})")
        if label_dict:
            v_spec = p_specificity_vec[[i for i, n in enumerate(nodes) if label_dict[n] == 1]]
            nv_spec = p_specificity_vec[[i for i, n in enumerate(nodes) if label_dict[n] == 0]]
            v_gate = phrog_has_specific[[i for i, n in enumerate(nodes) if label_dict[n] == 1]]
            nv_gate = phrog_has_specific[[i for i, n in enumerate(nodes) if label_dict[n] == 0]]
            print(f"[*] PHROG specificity — Virus: {np.mean(v_spec):.3f}, NonVirus: {np.mean(nv_spec):.3f}")
            print(f"[*] PHROG reliable    — Virus: {v_gate.mean():.1%}, NonVirus: {nv_gate.mean():.1%}")
    else:
        virus_signal = {n: 0.0 for n in nodes}
        virus_specificity = {n: 0.0 for n in nodes}
        virus_signal_combined = {n: 0.0 for n in nodes}
        virus_has_specific = {n: False for n in nodes}
        print("[*] No PHROG columns found — running without functional prior")

    # ── Ablation: disable PHROG prior ──────────────────────────────
    if args.no_phrog:
        has_phrog = False
        virus_signal = {n: 0.0 for n in nodes}
        virus_specificity = {n: 0.0 for n in nodes}
        virus_signal_combined = {n: 0.0 for n in nodes}
        virus_has_specific = {n: False for n in nodes}
        print("[*] Ablation: PHROG prior DISABLED")

    # ── Build graph ─────────────────────────────────────────────────
    edges = pd.read_csv(args.edges, sep="\t")
    neighbors = defaultdict(list)
    for _, row in tqdm(edges.iterrows(), total=len(edges), desc="Loading Graph"):
        u, v, w = row["contig_u"], row["contig_v"], row["weight"]
        if u in node_to_idx and v in node_to_idx:
            neighbors[u].append((v, w))
            neighbors[v].append((u, w))

    # ── Initialise ──────────────────────────────────────────────────
    S0 = np.array([score_dict[n] for n in nodes], dtype=np.float64)
    S = S0.copy()
    lengths = np.array([length_map.get(n, 3000.0) for n in nodes], dtype=np.float64)

    # Global prior for data-adaptive push centering
    global_prior = float(np.mean(S0))
    print(f"[*] Global prior (mean S0): {global_prior:.4f}")

    # Length-aware confidence BOOST: only increase q for long contigs (anchors),
    # never decrease for short ones. Long contigs = more reliable = keep S0.
    # len_boost: 1.0 at ≤3kb, ~1.3 at 5kb+
    len_boost = np.clip(1.0 + 0.35 * sigmoid((lengths - 3500.0) / 1000.0), 1.0, 1.4)
    conf_0_raw = np.abs(2.0 * S0 - 1.0)

    # ── PHROG-informed confidence adjustment ──────────────────────
    # If PHROG is reliable AND supports PPR → boost confidence (anchor)
    # If PHROG is reliable AND contradicts PPR → reduce confidence (allow correction)
    phrog_conf_adj = np.ones(n_nodes, dtype=np.float64)
    if has_phrog:
        for i in range(n_nodes):
            phrog_ok = virus_has_specific.get(nodes[i], False)
            p_val = virus_signal.get(nodes[i], 0.0)
            if phrog_ok:
                ppr_viral = S0[i] > 0.5
                phrog_viral = p_val > 0.5
                if ppr_viral and phrog_viral:
                    # PHROG supports PPR → increase confidence (strong anchor)
                    phrog_conf_adj[i] = 1.15
                elif not ppr_viral and phrog_viral:
                    # PHROG contradicts PPR → reduce confidence (allow correction)
                    phrog_conf_adj[i] = 0.65

    conf_0 = np.clip(conf_0_raw * len_boost * phrog_conf_adj, 0.05, 0.98)

    # ── Ablation: disable confidence (fixed q=0.5) ────────────────
    if args.no_confidence:
        conf_0 = np.full(n_nodes, 0.5)
        print("[*] Ablation: confidence DISABLED (fixed q=0.5)")

    # ── Two-phase anchor selection ────────────────────────────────
    anchor_mask = np.zeros(n_nodes, dtype=bool)
    anchor_phase_start = max(1, args.iter // 2)  # Start anchor phase at halfway

    # ── Propagation ─────────────────────────────────────────────────
    iter_scores = [S.copy()]  # Store scores from each iteration for ensemble
    for it in range(args.iter):
        S_new = S.copy()

        if it >= anchor_phase_start and it == anchor_phase_start:
            # Select anchors: high confidence + high score + PHROG reliable
            for idx in range(n_nodes):
                node = nodes[idx]
                phrog_ok = virus_has_specific.get(node, False)
                high_conf = conf_0[idx] > 0.6
                high_score = S[idx] > 0.65
                if high_conf and high_score and phrog_ok:
                    anchor_mask[idx] = True
            n_anchors = anchor_mask.sum()
            print(f"   [Anchor Phase] Selected {n_anchors} anchors "
                  f"({n_anchors/n_nodes*100:.1f}% of nodes)")

        for i, node in enumerate(tqdm(nodes, desc=f"Iter {it+1}/{args.iter}", leave=False)):
            neighs = neighbors.get(node, [])
            if not neighs:
                continue

            n_idxs = [node_to_idx[nb[0]] for nb in neighs]
            n_weights_raw = np.array([nb[1] for nb in neighs], dtype=np.float64)
            n_scores = S[n_idxs]

            # Boost edges FROM anchors to non-anchors
            if it >= anchor_phase_start:
                n_is_anchor = anchor_mask[n_idxs]
                if n_is_anchor.any():
                    # Anchors get 1.5x edge weight
                    n_weights_raw = n_weights_raw * np.where(n_is_anchor, 1.5, 1.0)

            # ── Per-node L1-normalise edge weights ─────────────────
            w_eff = n_weights_raw / (n_weights_raw.sum() + 1e-9)

            # ── Weighted neighbour average ─────────────────────────
            avg_n = np.sum(w_eff * n_scores)

            # ── Ablation: naive graph propagation (no confidence) ─
            if args.no_confidence:
                S_new[i] = np.clip(0.5 * S0[i] + 0.5 * avg_n, 0.0, 1.0)
                continue

            # ── Neighbour consistency (purity) ────────────────────
            if len(n_scores) > 1:
                n_std = np.std(n_scores)
            else:
                n_std = 0.0
            purity = np.clip(1.0 - 0.5 * n_std, 0.05, 1.0)

            # ── Neighbourhood consensus ───────────────────────────
            # Fall back to S0 when neighbours disagree (low purity)
            m_i = purity * avg_n + (1.0 - purity) * S0[i]

            # ── Effective confidence q_i ───────────────────────────
            q_i = conf_0[i]

            # ── Anchor boost: anchors retain more of their score ───
            if it >= anchor_phase_start and anchor_mask[i]:
                q_i = min(q_i * 1.2, 0.98)

            # PHROG signals for this node
            p_val = virus_signal.get(node, 0.0)
            p_comb = virus_signal_combined.get(node, 0.0)
            phrog_reliable = virus_has_specific.get(node, False)
            has_any_phrog = (p_val > 1e-9)

            # ── PHROG contradiction detection ──────────────────────
            if has_phrog and phrog_reliable:
                ppr_is_viral = S0[i] > 0.5
                phrog_is_viral = p_val > 0.5
                if ppr_is_viral != phrog_is_viral:
                    contradiction_strength = np.abs(S0[i] - p_val)
                    q_i *= (1.0 - 0.35 * contradiction_strength)

            # ── Trust graph MORE for contigs without PHROG ─────────
            # No PHROG = rely on graph neighbours to provide signal
            if not has_any_phrog:
                q_i = min(q_i, 0.88)  # Slightly relax self-retention

            # ── PHROG bias: ONLY for contigs with viral-specific hits ─
            # Categories 0,1,2,5 (head, connector, tail, lysis) = truly viral
            # Without these, PHROG bias is ZERO — prevents false pushes
            if has_phrog and phrog_reliable:
                p_bias = args.gamma_phrog * p_comb
            else:
                p_bias = 0.0

            # ── Precision-first push: strict dual-signal requirement ──
            # Push only when: (a) neighbours clearly above prior AND
            #                 (b) PHROG is reliable OR purity is very high
            push_centre = max(global_prior * 2.5, 0.35)
            push_raw = np.tanh(5.0 * (avg_n - push_centre))
            push_candidate = args.gamma_push * purity * max(0.0, push_raw)

            if purity > 0.5 and (phrog_reliable or purity > 0.85):
                push = push_candidate
            else:
                push = 0.0

            # ── Precision safeguard: if being pushed above 0.5 ────────
            # without reliable PHROG support, increase self-retention
            proposed = q_i * S0[i] + (1.0 - q_i) * (m_i + p_bias + push)
            if proposed > 0.5 and S0[i] < 0.5 and not phrog_reliable:
                q_i = min(q_i + 0.2, 0.98)

            q_i = np.clip(q_i, 0.05, 0.98)

            # ── Unified update ─────────────────────────────────────
            S_new[i] = np.clip(
                q_i * S0[i] + (1.0 - q_i) * (m_i + p_bias + push),
                0.0, 1.0
            )

        S = S_new
        iter_scores.append(S.copy())
        v_mean = float(np.mean(S))
        v_std = float(np.std(S))
        n_viral = int((S > 0.5).sum())
        print(f"   Iteration {it+1}/{args.iter} done — "
              f"mean={v_mean:.4f}, std={v_std:.4f}, n_viral={n_viral}")

    # ── Iteration ensemble: weighted average of last iterations ────
    n_avg = min(4, len(iter_scores))
    if n_avg > 1:
        ensemble_weights = np.array([0.1, 0.2, 0.3, 0.4][-n_avg:])
        ensemble_weights = ensemble_weights / ensemble_weights.sum()
        S_ensemble = np.zeros_like(S)
        for k in range(n_avg):
            S_ensemble += ensemble_weights[k] * iter_scores[-(n_avg - k)]
        print(f"[*] Iter-ensemble: averaging last {n_avg} iterations (weights: {ensemble_weights})")
    else:
        S_ensemble = S

    # ── Score tempering (sharpen distribution) ────────────────────
    T = args.temperature
    if T != 1.0:
        S_tempered = 1.0 / (1.0 + np.exp(-(S_ensemble - 0.5) / T))
        print(f"[*] Score tempering applied (T={T})")
    else:
        S_tempered = S_ensemble

    # ── Output ──────────────────────────────────────────────────────
    result = pd.DataFrame({
        "contig": nodes,
        "initial_score": S0,
        "refined_score": S_tempered,
    })
    result["refined_label"] = (result["refined_score"] >= 0.5).astype(int)

    # ── Evaluation ──────────────────────────────────────────────────
    if label_dict:
        result["true_label"] = [label_dict[n] for n in nodes]

        y_true = result["true_label"].values
        y_init = (S0 >= 0.5).astype(int)
        y_refi = result["refined_label"].values

        m0 = compute_metrics(y_true, y_init)
        m1 = compute_metrics(y_true, y_refi)

        print("\n" + "=" * 75)
        print(f"{'Metric':<14} | {'Initial':>10} | {'Refined':>10} | {'Δ':>10}")
        print("-" * 75)
        for name, v0, v1 in [
            ("Precision", m0[0], m1[0]),
            ("Recall", m0[1], m1[1]),
            ("F1-score", m0[2], m1[2]),
            ("Accuracy", m0[3], m1[3]),
            ("MCC", m0[4], m1[4]),
        ]:
            print(f"{name:<14} | {v0:>10.4f} | {v1:>10.4f} | {v1-v0:>+10.4f}")
        print("=" * 75)

        # ── Length-stratified evaluation ───────────────────────────
        if "length" in df.columns:
            result["length"] = result["contig"].map(length_map)
            bins = [0, 3000, 5000, np.inf]
            labels = ["1-3kb", "3-5kb", ">5kb"]
            result["range"] = pd.cut(result["length"], bins=bins, labels=labels)

            print(f"\n{'Range':<10} | {'P_Init':>8} | {'P_Ref':>8} | "
                  f"{'R_Init':>8} | {'R_Ref':>8} | {'F1_Init':>8} | {'F1_Ref':>8} | {'N':>6}")
            print("-" * 85)

            for rng in labels:
                sub = result[result["range"] == rng]
                if len(sub) == 0:
                    continue
                pi, ri, fi, _, _ = compute_metrics(
                    sub["true_label"].values, (sub["initial_score"] >= 0.5).astype(int))
                pr, rr, fr, _, _ = compute_metrics(
                    sub["true_label"].values, sub["refined_label"].values)
                print(f"{rng:<10} | {pi:>8.4f} | {pr:>8.4f} | "
                      f"{ri:>8.4f} | {rr:>8.4f} | {fi:>8.4f} | {fr:>8.4f} | {len(sub):>6}")

    result.to_csv(args.output, sep="\t", index=False)
    print(f"\n Done! Saved to: {args.output}")


if __name__ == "__main__":
    run_propagation()
