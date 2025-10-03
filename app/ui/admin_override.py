"""
Admin Override UI

Simple Tkinter dialog to review recent recognition logs and apply manual overrides to attendance records.
This provides a way for admins to correct false auto-marks.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
import sqlite3
import os

from ..core.database import DatabaseManager

class AdminOverrideDialog:
    def __init__(self, parent, db: DatabaseManager):
        self.parent = parent
        self.db = db
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Admin Overrides")
        self.dialog.geometry('800x500')
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._setup_ui()
        self._load_recent_logs()

    def _setup_ui(self):
        frame = ttk.Frame(self.dialog, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        # Logs tree
        cols = ('id', 'student_id', 'confidence_score', 'embedder_type', 'recognition_result', 'created_at')
        self.tree = ttk.Treeview(frame, columns=cols, show='headings', height=15)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120)
        self.tree.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        ttk.Button(btn_frame, text='Mark as Present', command=self._mark_present).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='Mark as Absent', command=self._mark_absent).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text='Refresh', command=self._load_recent_logs).pack(side=tk.RIGHT, padx=5)

    def _load_recent_logs(self):
        # load last 200 recognition logs
        try:
            conn = sqlite3.connect(self.db.db_path)
            cur = conn.cursor()
            cur.execute("SELECT id, student_id, confidence_score, embedder_type, recognition_result, created_at FROM recognition_logs ORDER BY created_at DESC LIMIT 200")
            rows = cur.fetchall()
            conn.close()

            for item in self.tree.get_children():
                self.tree.delete(item)

            for r in rows:
                self.tree.insert('', 'end', values=r)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load logs: {e}')

    def _get_selected_log(self) -> Optional[dict]:
        sel = self.tree.selection()
        if not sel:
            messagebox.showwarning('Select', 'Please select a log row to override')
            return None
        vals = self.tree.item(sel[0])['values']
        return {'id': vals[0], 'student_id': vals[1], 'confidence_score': vals[2], 'embedder_type': vals[3], 'recognition_result': vals[4], 'created_at': vals[5]}

    def _apply_override(self, action: str):
        log = self._get_selected_log()
        if not log:
            return
        student_id = log['student_id']
        if not student_id:
            messagebox.showwarning('No student', 'Selected log has no student to apply override')
            return

        # Apply override: create an attendance record with override note and log the override
        try:
            conn = sqlite3.connect(self.db.db_path)
            cur = conn.cursor()

            # Insert attendance record
            now = cur.execute("SELECT datetime('now')").fetchone()[0]
            note = f"Admin override from log {log['id']}"
            if action == 'present':
                cur.execute(
                    "INSERT INTO attendance (student_id, date, time, status, confidence, detection_type, notes) VALUES (?, DATE('now'), TIME('now'), ?, ?, ?, ?)",
                    (student_id, 'present', float(log['confidence_score'] or 0.0), 'manual_override', note)
                )
            else:
                cur.execute(
                    "INSERT INTO attendance (student_id, date, time, status, confidence, detection_type, notes) VALUES (?, DATE('now'), TIME('now'), ?, ?, ?, ?)",
                    (student_id, 'absent', None, 'manual_override', note)
                )

            # Log manual override
            cur.execute("INSERT INTO manual_overrides (student_id, original_result, override_action, override_reason, admin_user) VALUES (?, ?, ?, ?, ?)", (student_id, log['recognition_result'], action, f'Override via Admin UI on {now}', os.getlogin()))
            conn.commit()
            conn.close()

            messagebox.showinfo('Success', f'Override {action} applied for {student_id}')
            self._load_recent_logs()
        except Exception as e:
            messagebox.showerror('Error', f'Failed to apply override: {e}')

    def _mark_present(self):
        self._apply_override('present')

    def _mark_absent(self):
        self._apply_override('absent')
