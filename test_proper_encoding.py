"""
Test proper face encoding comparison for attendance marking
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager

def test_proper_encoding():
    """Test proper face encoding comparison"""
    try:
        print("Testing Proper Face Encoding Comparison...")
        
        # Initialize database
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Check current students and encodings
        students = db_manager.get_all_students()
        encodings = db_manager.get_face_encodings()
        print(f"Current students: {len(students)}")
        print(f"Current encodings: {len(encodings)}")
        
        for student in students:
            print(f"  - {student['student_id']}: {student['name']}")
        
        # Show encoding details
        if encodings:
            print(f"\nEncoding details:")
            for i, (student_id, encoding, encoding_type) in enumerate(encodings[:3]):  # Show first 3
                print(f"  Encoding {i}: Student {student_id}, Type: {encoding_type}, Shape: {encoding.shape}")
        
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
        print("\nCRITICAL TESTING INSTRUCTIONS:")
        print("1. Click 'Start Attendance' to begin recognition")
        print("2. Set 'Min Face Confidence' to 0.7 or higher")
        print("3. Set 'Recognition Threshold' to 0.6 or higher")
        print("4. Test with REGISTERED student (Teja) - should show GREEN box ONLY if face encoding matches")
        print("5. Test with UNREGISTERED person - should show RED box with 'Unknown Face'")
        print("6. Check console output for detailed encoding comparison")
        print("\nExpected behavior:")
        print("- Only faces with matching encodings should show 'Recognized: Teja'")
        print("- All other faces should show 'Unknown Face'")
        print("- Console should show similarity scores and match/no-match decisions")
        print("\nConsole output should show:")
        print("- 'Face X: Processing recognition with confidence Y.YY'")
        print("- 'Face X: Embedding shape: (128,), norm: X.XXXX'")
        print("- 'Face X: Available students: [\"1234\"]'")
        print("- 'Using UI threshold: X.XX, comparison threshold: 0.80'")
        print("- 'Checking student 1234 with X encodings'")
        print("- 'Encoding X: is_same=False, similarity=X.XXXX'")
        print("- '✗ NO MATCH: Best confidence X.XXXX below 0.9 threshold'")
        print("- 'Face X: NO MATCH found - will show as Unknown Face'")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error in proper encoding test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_proper_encoding()
