"""
Test unknown face display and attendance table loading
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager

def test_unknown_face_and_attendance():
    """Test unknown face display and attendance table loading"""
    try:
        print("Testing Unknown Face Display and Attendance Table...")
        
        # Initialize database
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Check current attendance
        from datetime import datetime
        today = datetime.now().date().strftime('%Y-%m-%d')
        today_records = db_manager.get_attendance_records(date_from=today, date_to=today)
        print(f"Current attendance records for today: {len(today_records)}")
        for record in today_records:
            print(f"  - {record['student_id']} ({record['name']}): {record['status']} at {record['time']}")
        
        # Create main window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Dummy config
        config = {
            "camera_index": 0,
            "face_data_path": "face_data",
            "models_path": "models"
        }
        
        # Open attendance marking dialog
        dialog = AttendanceMarkingDialog(root, db_manager, config)
        
        # Handle dialog closing
        def on_dialog_close():
            try:
                dialog.dialog.destroy()
                root.destroy()
            except:
                root.destroy()
        
        dialog.dialog.protocol("WM_DELETE_WINDOW", on_dialog_close)
        
        print("✓ Attendance marking dialog opened")
        print("Features to test:")
        print("  1. Attendance table should show existing records")
        print("  2. Green boxes for recognized faces (Teja)")
        print("  3. Red boxes for unknown faces (try with someone not registered)")
        print("  4. Statistics should show correct counts")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error in unknown face test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_unknown_face_and_attendance()
