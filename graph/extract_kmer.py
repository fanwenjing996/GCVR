import numpy as np
import pandas as pd
from Bio import SeqIO
import os
from multiprocessing import Pool, cpu_count

BATCH_SIZE = 2000 
NUM_CORES = min(15, cpu_count())  

base2int = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}

def seq_3mer_count(seq):
   
    arr = np.array([base2int.get(b, -100) for b in seq], dtype=np.int16)
    L = len(arr)
    if L < 3:
        return np.zeros(64, dtype=np.float32)

    
    idx = arr[:-2]*16 + arr[1:-1]*4 + arr[2:]
    
    mask = (arr[:-2] >= 0) & (arr[1:-1] >= 0) & (arr[2:] >= 0)
    valid_idx = idx[mask]

    if len(valid_idx) == 0:
        return np.zeros(64, dtype=np.float32)

    counts = np.bincount(valid_idx, minlength=64).astype(np.float32)
    return counts / len(valid_idx)

def process_batch(records):
    matrix = np.zeros((len(records), 64), dtype=np.float32)
    ids = []
    for i, rec in enumerate(records):
        matrix[i] = seq_3mer_count(str(rec.seq))
        ids.append(rec.id)
    return matrix, ids

def chunked_iterable(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

def extract_3mer(fasta_path):
    print(f"\n Processing: {fasta_path}")
    
    prefix = os.path.splitext(fasta_path)[0]
    all_matrices = []
    all_ids = []

    with Pool(NUM_CORES) as pool:
        
        records = SeqIO.parse(fasta_path, "fasta")
        chunks = chunked_iterable(records, BATCH_SIZE)
        
        for matrix, ids in pool.imap(process_batch, chunks):
            all_matrices.append(matrix)
            all_ids.extend(ids)
            print(f"  Finished batch of {len(ids)} sequences...")

    final_matrix = np.vstack(all_matrices)
    
    np.save(prefix + "_3mer.npy", final_matrix)
    pd.DataFrame({'contig': all_ids}).to_csv(prefix + "_3mer_ids.csv", index=False)
    print(f" Done! Saved to {prefix}_3mer.npy")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extract_3mer(sys.argv[1])
