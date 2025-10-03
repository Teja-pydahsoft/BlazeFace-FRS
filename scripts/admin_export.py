"""
Admin export utility

Exports recognition logs, attendance records, and performance metrics to CSV or JSON for a given date range.

Usage:
    python scripts/admin_export.py --db database/blazeface_frs.db --from 2025-09-01 --to 2025-09-30 --out exports/ --formats csv,json

"""
import argparse
import os
import sqlite3
import csv
import json
from datetime import datetime


def ensure_out_dir(path):
    os.makedirs(path, exist_ok=True)


def query_rows(conn, query, params):
    cur = conn.cursor()
    cur.execute(query, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows


def write_csv(path, cols, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in rows:
            writer.writerow(r)


def write_json(path, cols, rows):
    arr = []
    for r in rows:
        arr.append({cols[i]: r[i] for i in range(len(cols))})
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(arr, f, indent=2, default=str)


def export_recognition_logs(conn, out_dir, date_from, date_to, formats):
    query = "SELECT * FROM recognition_logs WHERE DATE(created_at) >= ? AND DATE(created_at) <= ? ORDER BY created_at"
    cols, rows = query_rows(conn, query, (date_from, date_to))
    base = os.path.join(out_dir, f"recognition_logs_{date_from}_{date_to}")
    if 'csv' in formats:
        write_csv(base + '.csv', cols, rows)
    if 'json' in formats:
        write_json(base + '.json', cols, rows)
    print(f"Exported recognition logs to {out_dir}")


def export_attendance(conn, out_dir, date_from, date_to, formats):
    query = "SELECT a.*, s.name FROM attendance a JOIN students s ON a.student_id = s.student_id WHERE DATE(a.created_at) >= ? AND DATE(a.created_at) <= ? ORDER BY a.created_at"
    cols, rows = query_rows(conn, query, (date_from, date_to))
    base = os.path.join(out_dir, f"attendance_{date_from}_{date_to}")
    if 'csv' in formats:
        write_csv(base + '.csv', cols, rows)
    if 'json' in formats:
        write_json(base + '.json', cols, rows)
    print(f"Exported attendance to {out_dir}")


def export_performance_metrics(conn, out_dir, date_from, date_to, formats):
    query = "SELECT * FROM performance_metrics WHERE DATE(test_date) >= ? AND DATE(test_date) <= ? ORDER BY test_date"
    cols, rows = query_rows(conn, query, (date_from, date_to))
    base = os.path.join(out_dir, f"performance_metrics_{date_from}_{date_to}")
    if 'csv' in formats:
        write_csv(base + '.csv', cols, rows)
    if 'json' in formats:
        write_json(base + '.json', cols, rows)
    print(f"Exported performance metrics to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Admin export utility')
    parser.add_argument('--db', default='database/blazeface_frs.db')
    parser.add_argument('--from', dest='date_from', required=True)
    parser.add_argument('--to', dest='date_to', required=True)
    parser.add_argument('--out', dest='out_dir', default='exports')
    parser.add_argument('--formats', default='csv', help='Comma-separated formats: csv,json')

    args = parser.parse_args()

    date_from = args.date_from
    date_to = args.date_to
    out_dir = args.out_dir
    formats = [f.strip() for f in args.formats.split(',') if f.strip()]

    ensure_out_dir(out_dir)

    if not os.path.exists(args.db):
        print(f"DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)

    export_recognition_logs(conn, out_dir, date_from, date_to, formats)
    export_attendance(conn, out_dir, date_from, date_to, formats)
    export_performance_metrics(conn, out_dir, date_from, date_to, formats)

    conn.close()

if __name__ == '__main__':
    main()
