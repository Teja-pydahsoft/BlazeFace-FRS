"""
Test the confidence threshold fix for face recognition
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.attendance_marking import AttendanceMarkingDialog
from app.core.database import DatabaseManager

def test_confidence_fix():
    """Test the confidence threshold fix"""
    try:
        print("Testing Confidence Threshold Fix...")
        
        # Initialize database
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
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
        print("2. Adjust 'Min Face Confidence' slider to 0.7 or higher")
        print("3. Test with low confidence faces - should show 'Low Confidence' instead of recognition")
        print("4. Test with high confidence faces - should show proper recognition or 'Unknown Face'")
        print("5. Check console output for detailed confidence filtering")
        print("\nExpected behavior:")
        print("- Faces with confidence < 0.7: Red box with 'Low Confidence'")
        print("- Faces with confidence >= 0.7 but no match: Red box with 'Unknown Face'")
        print("- Faces with confidence >= 0.7 and match: Green box with 'Recognized: [Name]'")
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error in confidence fix test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_confidence_fix()
