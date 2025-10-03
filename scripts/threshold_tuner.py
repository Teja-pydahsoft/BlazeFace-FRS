"""
Threshold tuning harness

This script loads a labeled test set of face encodings (CSV mapping test_id, student_id, encoding_path or raw .npy files)
and evaluates recognition performance against the gallery stored in the DB (table face_encodings).
It sweeps thresholds for each encoding_type (embedder) found in the DB and populates the
`performance_metrics` table with TP/FP/TN/FN and derived metrics for each threshold value.

Usage example:
    python scripts/threshold_tuner.py --db database/blazeface_frs.db --test-csv test_encodings.csv --thresholds 0.5,0.6,0.7,0.8,0.85,0.9 --output results.csv

Test CSV format (header required):
    test_id,student_id,encoding_path,encoding_type
    t1,student_1,test_data/t1.npy,insightface

The script attempts to normalize encodings and match by cosine similarity. It only compares test encodings
with gallery encodings of the same encoding_type (embedding dim must match). For each threshold, the script
computes TP/FP/TN/FN and writes a row into `performance_metrics`.

Note: This script is intended for offline evaluation. It does not alter production data.
"""

import argparse
import csv
import io
import os
import sqlite3
import sys
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict


def load_test_encodings(csv_path: str) -> List[Tuple[str, str, np.ndarray, str]]:
    """Load test encodings from a CSV. Returns list of tuples (test_id, student_id, vector, encoding_type)"""
    rows = []
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            test_id = r.get('test_id') or r.get('id')
            student_id = r.get('student_id')
            encoding_path = r.get('encoding_path') or r.get('filepath')
            enc_type = r.get('encoding_type') or r.get('embedder') or 'unknown'
            if not encoding_path or not os.path.exists(encoding_path):
                print(f"Warning: skipping missing encoding file for test {test_id}: {encoding_path}")
                continue
            try:
                arr = np.load(encoding_path)
            except Exception as e:
                print(f"Failed loading {encoding_path}: {e}")
                continue
            if arr is None or arr.size == 0:
                continue
            rows.append((test_id, student_id, arr.astype(np.float32), enc_type))
    return rows


def load_gallery(conn: sqlite3.Connection) -> Dict[str, List[Tuple[int, str, np.ndarray]]]:
    """Load gallery encodings from DB and return dict by encoding_type -> list of (id, student_id, vector)"""
    cur = conn.cursor()
    cur.execute("SELECT id, student_id, encoding, encoding_type FROM face_encodings")
    rows = cur.fetchall()
    gallery = {}
    for rid, sid, enc_bytes, enc_type in rows:
        if not enc_bytes:
            continue
        try:
            buf = io.BytesIO(enc_bytes)
            arr = np.load(buf, allow_pickle=False)
        except Exception:
            try:
                arr = np.frombuffer(enc_bytes, dtype=np.float64).astype(np.float32)
            except Exception:
                print(f"Skipping gallery id {rid} due to decode error")
                continue
        if arr is None or arr.size == 0:
            continue
        # normalize
        norm = np.linalg.norm(arr)
        if norm == 0:
            continue
        vec = arr.astype(np.float32) / norm
        gallery.setdefault(enc_type or 'unknown', []).append((rid, sid, vec))
    return gallery


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def evaluate_for_thresholds(test_set: List[Tuple[str, str, np.ndarray, str]],
                            gallery: Dict[str, List[Tuple[int, str, np.ndarray]]],
                            thresholds: List[float]) -> Dict[str, List[Dict]]:
    """For each encoding_type, evaluate performance for each threshold. Returns mapping enc_type->[metrics rows]."""
    results = {}
    for enc_type in set([t[3] for t in test_set]):
        gallery_vecs = gallery.get(enc_type, [])
        if not gallery_vecs:
            print(f"No gallery encodings for type {enc_type}, skipping")
            continue
        # Precompute gallery matrix and ids
        g_ids = [g[0] for g in gallery_vecs]
        g_sids = [g[1] for g in gallery_vecs]
        g_vecs = np.stack([g[2] for g in gallery_vecs], axis=0)

        rows = []
        for thr in thresholds:
            tp = fp = tn = fn = 0
            for test_id, true_sid, vec, ttype in [t for t in test_set if t[3] == enc_type]:
                # normalize test vec
                v = vec
                norm = np.linalg.norm(v)
                if norm == 0:
                    # treat as no match
                    predicted_sid = None
                    max_sim = -1.0
                else:
                    v = (v / norm).astype(np.float32)
                    # Ensure shapes match
                    if v.shape != g_vecs.shape[1:]:
                        # skip comparisons where dims mismatch
                        predicted_sid = None
                        max_sim = -1.0
                    else:
                        sims = g_vecs.dot(v)
                        max_idx = int(np.argmax(sims))
                        max_sim = float(sims[max_idx])
                        if max_sim >= thr:
                            predicted_sid = g_sids[max_idx]
                        else:
                            predicted_sid = None

                if predicted_sid is None:
                    if true_sid is None or true_sid.strip() == 'unknown' or true_sid == '':
                        tn += 1
                    else:
                        fn += 1
                else:
                    if true_sid and predicted_sid == true_sid:
                        tp += 1
                    else:
                        fp += 1

            accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
            precision = tp / max(1, (tp + fp))
            recall = tp / max(1, (tp + fn))
            f1 = 2 * precision * recall / max(1e-9, (precision + recall))

            rows.append({
                'encoding_type': enc_type,
                'threshold_value': float(thr),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn),
                'accuracy': float(accuracy),
                'precision_score': float(precision),
                'recall_score': float(recall),
                'f1_score': float(f1),
                'dataset_size': int(tp + tn + fp + fn)
            })
        results[enc_type] = rows
    return results


def persist_metrics(conn: sqlite3.Connection, rows: List[Dict]):
    cur = conn.cursor()
    now = datetime.now().date().isoformat()
    for r in rows:
        cur.execute("""
            INSERT INTO performance_metrics (
                embedder_type, threshold_value, true_positives, false_positives,
                true_negatives, false_negatives, accuracy, precision_score, recall_score,
                f1_score, test_date, dataset_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r['encoding_type'], r['threshold_value'], r['true_positives'], r['false_positives'],
            r['true_negatives'], r['false_negatives'], r['accuracy'], r['precision_score'],
            r['recall_score'], r['f1_score'], now, r['dataset_size']
        ))
    conn.commit()


def write_csv(out_path: str, results: Dict[str, List[Dict]]):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['encoding_type','threshold_value','true_positives','false_positives','true_negatives','false_negatives','accuracy','precision_score','recall_score','f1_score','dataset_size'])
        for enc_type, rows in results.items():
            for r in rows:
                writer.writerow([r['encoding_type'], r['threshold_value'], r['true_positives'], r['false_positives'], r['true_negatives'], r['false_negatives'], r['accuracy'], r['precision_score'], r['recall_score'], r['f1_score'], r['dataset_size']])


def parse_thresholds(s: str) -> List[float]:
    parts = [p.strip() for p in s.split(',') if p.strip()]
    return [float(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description='Threshold tuning harness')
    parser.add_argument('--db', default='database/blazeface_frs.db')
    parser.add_argument('--test-csv', required=True, help='CSV with test encodings')
    parser.add_argument('--thresholds', required=True, help='Comma-separated thresholds to test')
    parser.add_argument('--output', default='threshold_results.csv', help='Output CSV')

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        sys.exit(1)
    if not os.path.exists(args.test_csv):
        print(f"Test CSV not found: {args.test_csv}")
        sys.exit(1)

    thresholds = parse_thresholds(args.thresholds)

    conn = sqlite3.connect(args.db)

    test_set = load_test_encodings(args.test_csv)
    gallery = load_gallery(conn)

    results = evaluate_for_thresholds(test_set, gallery, thresholds)

    # Persist to DB and write CSV
    all_rows = []
    for enc_type, rows in results.items():
        all_rows.extend(rows)

    if all_rows:
        persist_metrics(conn, all_rows)
        write_csv(args.output, results)
        print(f"Wrote results to {args.output} and persisted {len(all_rows)} metric rows to DB")
    else:
        print("No results to persist (no matching encoding types between test set and gallery)")


if __name__ == '__main__':
    main()
