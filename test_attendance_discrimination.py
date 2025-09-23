"""
Test the complete attendance system with improved discrimination
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager

def test_attendance_discrimination():
    """Test the complete attendance system with improved discrimination"""
    try:
        print("Testing Complete Attendance System with Improved Discrimination...")
        
        # Initialize database
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Check current students
        students = db_manager.get_all_students()
        print(f"Current students in database: {len(students)}")
        for student in students:
            print(f"  - {student['student_id']}: {student['name']}")
        
        # Check today's attendance
        from datetime import datetime
        today = datetime.now().date().strftime('%Y-%m-%d')
        today_records = db_manager.get_attendance_records(date_from=today, date_to=today)
        print(f"Today's attendance records: {len(today_records)}")
        
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
        print("\nIMPORTANT TESTING INSTRUCTIONS:")
        print("1. Click 'Start Attendance' to begin recognition")
        print("2. Test with REGISTERED student (Teja) - should show GREEN box with 'Recognized: Teja'")
        print("3. Test with UNREGISTERED person - should show RED box with 'Unknown Face'")
        print("4. The system now uses improved discrimination with higher thresholds")
        print("5. Check the console output for detailed similarity scores")
        print("6. The attendance table should show existing records")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error in attendance discrimination test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attendance_discrimination()
