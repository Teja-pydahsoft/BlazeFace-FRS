"""
Test script for attendance features
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager

def test_attendance_features():
    """Test the attendance features"""
    try:
        # Create a simple test window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Initialize database
        db_manager = DatabaseManager("database/test_attendance.db")
        
        # Load config
        config = {
            'camera_index': 0,
            'detection_confidence': 0.7,
            'facenet_model_path': None
        }
        
        print("Testing Attendance Features...")
        
        # Test attendance marking dialog
        print("1. Testing Attendance Marking Dialog...")
        from app.ui.attendance_marking import AttendanceMarkingDialog
        marking_dialog = AttendanceMarkingDialog(root, db_manager, config)
        
        # Test attendance history dialog
        print("2. Testing Attendance History Dialog...")
        from app.ui.attendance_history import AttendanceHistoryDialog
        history_dialog = AttendanceHistoryDialog(root, db_manager)
        
        print("Attendance features test completed successfully!")
        print("Both dialogs should be open and functional.")
        
        # Show the dialogs
        root.deiconify()  # Show the main window
        root.mainloop()
        
    except Exception as e:
        print(f"Error testing attendance features: {e}")
        messagebox.showerror("Test Error", f"Failed to test attendance features: {str(e)}")

if __name__ == "__main__":
    test_attendance_features()
