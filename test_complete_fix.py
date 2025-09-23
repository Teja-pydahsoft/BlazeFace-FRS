"""
Test the complete attendance system with proper encoding comparison
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager

def test_complete_fix():
    """Test the complete attendance system with proper encoding comparison"""
    try:
        print("Testing Complete Attendance System with Proper Encoding Comparison...")
        
        # Initialize database
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Check current students and encodings
        students = db_manager.get_all_students()
        encodings = db_manager.get_face_encodings()
        print(f"Current students: {len(students)}")
        print(f"Current encodings: {len(encodings)}")
        
        for student in students:
            print(f"  - {student['student_id']}: {student['name']}")
        
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
        print("2. Set 'Min Face Confidence' to 0.7 or higher")
        print("3. Set 'Recognition Threshold' to 0.6 or higher")
        print("4. Test with REGISTERED student (Teja) - should show GREEN box with 'Recognized: Teja'")
        print("5. Test with UNREGISTERED person - should show RED box with 'Unknown Face'")
        print("6. Check console output for detailed encoding comparison results")
        print("\nExpected behavior:")
        print("- Low confidence faces (<0.7): Red box with 'Low Confidence'")
        print("- High confidence faces but no match: Red box with 'Unknown Face'")
        print("- High confidence faces with match: Green box with 'Recognized: [Name]'")
        print("- Console should show: '✓ MATCH FOUND' or '✗ NO MATCH' with similarity scores")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error in complete fix test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_complete_fix()
