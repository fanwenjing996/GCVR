#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为每一个预测出来的蛋白质（ORF）打上“身份标签”（PHROG 类别），并生成一个干净的、按 Contig 归类的特征表
"""
import re

# =========================
# 输入文件
# =========================
tbl = "phrog_hits.tbl"
phrog_annot = "phrog_annot_v4.tsv"
faa_file = "prodigal/proteins.faa"

# 输出
out_file = "phrog_hits_best_per_orf.tsv"


# =========================
#  读取 PHROG 注释
# =========================
phrog_map = {}
with open(phrog_annot) as fh:
    header = fh.readline()
    for line in fh:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        pid = int(parts[0])   # PHROG ID 数字
        category = parts[3]
        phrog_map[pid] = category


# =========================
# 读取所有 ORF
# =========================
def load_all_orfs(faa):
    orfs = set()
    with open(faa) as f:
        for line in f:
            if line.startswith(">"):
                orf = line[1:].strip().split()[0]
                orfs.add(orf)
    return orfs

all_orfs = load_all_orfs(faa_file)
print(f"[INFO] Total ORFs: {len(all_orfs)}")


# =========================
#  提取 best hit
# =========================
best_hit = {}

with open(tbl) as fh:
    for line in fh:
        if not line.strip() or line.startswith("#"):
            continue

        toks = line.strip().split()

        orf_id = toks[0]        # target
        phrog_str = toks[2]     # query (phrog_xxxx)

        # 解析 phrog_id
        try:
            pid = int(phrog_str.replace("phrog_", ""))
        except:
            continue

        # 读取 E-value (科学计数法) 和 Score
        evalue = float(toks[4]) if toks[4] != "-" else float("inf")
        score = float(toks[5])

        if orf_id not in best_hit:
            best_hit[orf_id] = (evalue, score, pid)
        else:
            prev_e, prev_s, _ = best_hit[orf_id]
            # 如果新命中的 E-value 更小，或者 E-value 相同但 Score 更高，则替换
            if evalue < prev_e or (evalue == prev_e and score > prev_s):
                best_hit[orf_id] = (evalue, score, pid)

print(f"[INFO] ORFs with PHROG hits: {len(best_hit)}")


# =========================
with open(out_file, "w") as f:
    f.write("orf_id\tcontig\torf_index\tphrog_id\tphrog_category\tevalue\tscore\n")

    for orf in all_orfs:

        # 解析 contig & index
        try:
            contig, idx = orf.rsplit("_", 1)
            idx = int(idx)
        except:
            contig = orf
            idx = -1

        if orf in best_hit:
            ev, sc, pid = best_hit[orf]
            cat = phrog_map.get(pid, "NA")
        else:
            # 未命中
            pid = "NA"
            cat = "unannotated"
            ev = "NA"
            sc = "NA"

        f.write(f"{orf}\t{contig}\t{idx}\t{pid}\t{cat}\t{ev}\t{sc}\n")


print(f"[INFO] Output written to: {out_file}")
