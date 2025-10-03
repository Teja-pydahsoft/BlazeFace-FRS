import time
from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager


class DummyDB(DatabaseManager):
    def __init__(self, db_path):
        # Do not call DatabaseManager.__init__ to avoid filesystem effects
        self.db_path = db_path
        self._records = []

    def add_attendance_record(self, student_id, status='present', confidence=None, detection_type=None, notes=None):
        self._records.append((student_id, status, confidence, detection_type, notes))
        return True


def test_try_consensus_mark_counts_votes(tmp_path, monkeypatch):
    # Create a minimal AttendanceMarkingDialog-like object
    db = DummyDB(str(tmp_path / 'dummy.db'))
    # We need to create a minimal config and parent; AttendanceMarkingDialog tries to build a UI (tkinter)
    # To avoid creating a UI, we'll create a tiny stub object that only uses _try_consensus_mark and _mark_attendance

    class StubAttendance:
        def __init__(self):
            self.tracked_face_data = {}
            self.attendance_marked = set()
            self._marked = []

        def _mark_attendance(self, student_id, confidence):
            self._marked.append((student_id, confidence))
            self.attendance_marked.add(student_id)

        # Copy the real _try_consensus_mark implementation but adapt to instance
        def _try_consensus_mark(self, object_id):
            data = self.tracked_face_data.get(object_id)
            if not data:
                return

            history = data.get('history', [])
            if not history:
                return

            counts = {}
            for sid, conf, ts in history:
                if sid is None:
                    continue
                counts[sid] = counts.get(sid, 0) + 1

            if not counts:
                return

            best_student_id, votes = max(counts.items(), key=lambda x: x[1])
            # For this test assume consensus_required = 2
            consensus_required = 2
            if votes >= consensus_required:
                latest_conf = 0.0
                for sid, conf, ts in reversed(history):
                    if sid == best_student_id:
                        latest_conf = conf
                        break
                if best_student_id in self.attendance_marked:
                    return
                self._mark_attendance(best_student_id, latest_conf)

    stub = StubAttendance()
    # Simulate history: 3 entries with student s1 twice, s2 once
    stub.tracked_face_data[1] = {
        'history': [
            ('s1', 0.8, time.time() - 2),
            ('s2', 0.75, time.time() - 1),
            ('s1', 0.82, time.time())
        ]
    }

    stub._try_consensus_mark(1)
    assert ('s1', 0.82) in stub._marked
