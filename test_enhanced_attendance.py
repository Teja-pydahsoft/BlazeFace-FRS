"""
Test the enhanced attendance marking with visual feedback
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager

def test_enhanced_attendance():
    """Test enhanced attendance marking"""
    try:
        print("Testing Enhanced Attendance Marking...")
        
        # Initialize database
        db_manager = DatabaseManager("database/test_enhanced_attendance.db")
        
        # Add a test student if not exists
        test_student = {
            'student_id': 'TEST001',
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '123-456-7890',
            'department': 'Computer Science',
            'year': '2024'
        }
        
        # Check if student exists
        existing_student = db_manager.get_student('TEST001')
        if not existing_student:
            db_manager.add_student(test_student)
            print("✓ Test student added")
        else:
            print("✓ Test student already exists")
        
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
        
        print("✓ Enhanced attendance marking dialog opened")
        print("Features to test:")
        print("  - Green boxes for recognized faces with student names")
        print("  - Red boxes for unknown faces")
        print("  - Real-time recognition status display")
        print("  - Enhanced statistics showing unique students present")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error in enhanced attendance test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_attendance()
