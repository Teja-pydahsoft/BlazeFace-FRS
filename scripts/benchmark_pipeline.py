"""
Simple benchmark for the embedding similarity pipeline

Measures average time to compute similarities between a query encoding and gallery of N encodings.
Outputs simple metrics to stdout and a CSV.
"""

import argparse
import os
import sqlite3
import time
import numpy as np
import csv
import io


def load_gallery(conn, limit=None):
    cur = conn.cursor()
    q = "SELECT id, encoding FROM face_encodings"
    if limit:
        q += f" LIMIT {int(limit)}"
    cur.execute(q)
    rows = cur.fetchall()
    vecs = []
    for rid, enc_bytes in rows:
        try:
            buf = io.BytesIO(enc_bytes)
            arr = np.load(buf, allow_pickle=False)
        except Exception:
            try:
                arr = np.frombuffer(enc_bytes, dtype=np.float64).astype(np.float32)
            except Exception:
                continue
        if arr is None or arr.size == 0:
            continue
        vecs.append(arr.astype(np.float32))
    return vecs


def main():
    parser = argparse.ArgumentParser(description='Benchmark embedding similarity throughput')
    parser.add_argument('--db', default='database/blazeface_frs.db')
    parser.add_argument('--gallery-size', type=int, default=1000)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--out', default='benchmark_results.csv')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)
    gallery = load_gallery(conn, limit=args.gallery_size)
    if not gallery:
        print("No gallery encodings found")
        return

    # Normalize gallery
    gmat = np.stack([v / np.linalg.norm(v) for v in gallery], axis=0)

    # Create random query vectors
    query = np.random.rand(gmat.shape[1]).astype(np.float32)
    query = query / np.linalg.norm(query)

    times = []
    for i in range(args.iterations):
        t0 = time.time()
        sims = gmat.dot(query)
        idx = int(np.argmax(sims))
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)  # ms

    avg_ms = sum(times) / len(times)
    print(f"Average similarity compute time over {args.iterations} iterations: {avg_ms:.3f} ms")

    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['iteration','ms'])
        for i, ms in enumerate(times):
            w.writerow([i+1, ms])
    print(f"Wrote benchmark results to {args.out}")

if __name__ == '__main__':
    main()
