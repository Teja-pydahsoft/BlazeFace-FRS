"""
Test script for student registration dialog
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager

def test_registration_dialog():
    """Test the student registration dialog"""
    try:
        # Create a simple test window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Initialize database
        db_manager = DatabaseManager("database/test_registration.db")
        
        # Load config
        config = {
            'camera_index': 0,
            'detection_confidence': 0.7,
            'facenet_model_path': None
        }
        
        # Import and create registration dialog
        from app.ui.student_registration import StudentRegistrationDialog
        
        print("Opening student registration dialog...")
        dialog = StudentRegistrationDialog(root, db_manager, config)
        
        # Show the dialog
        root.deiconify()  # Show the main window
        dialog.dialog.mainloop()
        
        print("Registration dialog test completed")
        
    except Exception as e:
        print(f"Error testing registration dialog: {e}")
        messagebox.showerror("Test Error", f"Failed to test registration dialog: {str(e)}")

if __name__ == "__main__":
    test_registration_dialog()
