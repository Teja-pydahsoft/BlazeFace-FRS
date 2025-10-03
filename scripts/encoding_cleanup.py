"""
Encoding cleanup and pruning utility

Scans all encodings per student and recommends encodings to remove based on:
 - low quality score (using EncodingQualityChecker)
 - near-duplicate encodings (similarity above duplicate threshold)

Outputs a CSV with recommendations and can optionally apply deletions with --apply (admin must confirm).

Usage:
    python scripts/encoding_cleanup.py --db database/blazeface_frs.db --out cleanup_report.csv
    python scripts/encoding_cleanup.py --db database/blazeface_frs.db --out cleanup_report.csv --apply
"""

import argparse
import csv
import io
import os
import sqlite3
import numpy as np
from app.utils.encoding_quality_checker import EncodingQualityChecker


def load_encodings(conn):
    cur = conn.cursor()
    cur.execute("SELECT id, student_id, encoding, encoding_type, image_path FROM face_encodings ORDER BY student_id")
    rows = cur.fetchall()
    encs = {}
    for rid, sid, enc_bytes, enc_type, img_path in rows:
        try:
            buf = io.BytesIO(enc_bytes)
            arr = np.load(buf, allow_pickle=False)
        except Exception:
            try:
                arr = np.frombuffer(enc_bytes, dtype=np.float64).astype(np.float32)
            except Exception:
                print(f"Skipping encoding id {rid} (decode failed)")
                continue
        if arr is None or arr.size == 0:
            continue
        norm = np.linalg.norm(arr)
        if norm == 0:
            continue
        vec = arr.astype(np.float32)
        encs.setdefault(sid, []).append((rid, vec, enc_type, img_path))
    return encs


def compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a2 = a / np.linalg.norm(a)
    b2 = b / np.linalg.norm(b)
    return float(np.dot(a2, b2))


def main():
    parser = argparse.ArgumentParser(description='Encoding cleanup utility')
    parser.add_argument('--db', default='database/blazeface_frs.db')
    parser.add_argument('--out', default='encoding_cleanup_report.csv')
    parser.add_argument('--apply', action='store_true', help='Actually delete recommended encodings (admin only)')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)
    encs = load_encodings(conn)
    checker = EncodingQualityChecker()

    recommendations = []  # rows: student_id, encoding_id, reason, image_path

    for sid, items in encs.items():
        # If only 1 encoding, skip
        if len(items) <= 1:
            continue

        # Evaluate each encoding's quality score relative to others
        vectors = [it[1] for it in items]
        quality_scores = []
        for i, (rid, vec, etype, img) in enumerate(items):
            # Build existing_encodings (other encs for same person)
            existing = [v for j, v in enumerate(vectors) if j != i]
            # Other people's encodings sample (not exhaustive): take one encoding per other student
            other_sample = []
            for other_sid, other_items in encs.items():
                if other_sid == sid:
                    continue
                # take first encoding per other student
                other_sample.append((other_sid, other_items[0][1]))
            ok, reason, metrics = checker.check_new_encoding_quality(vec, existing, other_sample, sid)
            qscore = metrics.get('quality_score', 0.0)
            quality_scores.append((rid, qscore, reason))

        # Mark low-quality encodings for removal
        for rid, qscore, reason in quality_scores:
            if qscore < checker.min_encoding_quality:
                recommendations.append((sid, rid, 'low_quality', qscore, reason))

        # Detect duplicates within same student (very high mutual similarity)
        for i in range(len(items)):
            rid_i, vec_i, _, img_i = items[i]
            for j in range(i+1, len(items)):
                rid_j, vec_j, _, img_j = items[j]
                try:
                    sim = compute_similarity(vec_i, vec_j)
                except Exception:
                    continue
                if sim >= checker.duplicate_threshold:
                    # Recommend keeping the older one? We don't have timestamps here; mark both for review
                    recommendations.append((sid, rid_i, 'duplicate_with_'+str(rid_j), sim, f'duplicate pair {rid_i} vs {rid_j}'))

    # Write report
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['student_id','encoding_id','reason','score_or_similarity','notes'])
        for row in recommendations:
            writer.writerow(row)

    print(f"Written cleanup recommendations to {args.out} ({len(recommendations)} items)")

    if args.apply and recommendations:
        confirm = input('Apply deletions? This will DELETE rows from the database. Type YES to confirm: ')
        if confirm.strip() == 'YES':
            cur = conn.cursor()
            to_delete = set(r[1] for r in recommendations)
            for rid in to_delete:
                cur.execute('DELETE FROM face_encodings WHERE id = ?', (rid,))
            conn.commit()
            print(f"Deleted {len(to_delete)} encodings from DB")
        else:
            print('Aborted deletion')

    conn.close()

if __name__ == '__main__':
    main()
