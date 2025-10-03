"""
Student Management Dialog for BlazeFace-FRS System
Handles student deletion, re-registration, and management
"""

import tkinter as tk
from tkinter import ttk, messagebox
import os
from typing import Dict, Any, List
import logging

from ..core.database import DatabaseManager

class StudentManagementDialog:
    def __init__(self, parent, database_manager: DatabaseManager):
        """
        Initialize student management dialog
        
        Args:
            parent: Parent window
            database_manager: Database manager instance
        """
        self.parent = parent
        self.database_manager = database_manager
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Student Management - BlazeFace-FRS")
        self.dialog.geometry("800x600")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self._setup_ui()
        
        # Load students
        self._load_students()
        
        # Center dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup the student management UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Student Management", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Warning message
        warning_frame = ttk.Frame(main_frame)
        warning_frame.pack(fill=tk.X, pady=(0, 10))
        
        warning_label = ttk.Label(warning_frame, 
                                 text="⚠️ WARNING: Deleting students will remove all their face encodings and attendance records!",
                                 foreground='red', font=('Arial', 10, 'bold'))
        warning_label.pack()
        
        # Students list
        list_frame = ttk.LabelFrame(main_frame, text="Registered Students", padding="10")
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview for students list
        columns = ('Student ID', 'Name', 'Email', 'Department', 'Year', 'Encodings', 'Last Updated')
        self.students_tree = ttk.Treeview(list_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        column_widths = {
            'Student ID': 100,
            'Name': 150,
            'Email': 150,
            'Department': 120,
            'Year': 80,
            'Encodings': 80,
            'Last Updated': 120
        }
        
        for col in columns:
            self.students_tree.heading(col, text=col)
            self.students_tree.column(col, width=column_widths[col], anchor=tk.CENTER)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack treeview and scrollbar
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Left buttons
        left_buttons = ttk.Frame(button_frame)
        left_buttons.pack(side=tk.LEFT)
        
        ttk.Button(left_buttons, text="Refresh List", 
                  command=self._refresh_students).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(left_buttons, text="Bulk Register",
                  command=self._open_bulk_registration).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(left_buttons, text="Delete Selected", 
                  command=self._delete_selected_student).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(left_buttons, text="Delete All Students", 
                  command=self._delete_all_students).pack(side=tk.LEFT, padx=(0, 5))
        
        # Right buttons
        right_buttons = ttk.Frame(button_frame)
        right_buttons.pack(side=tk.RIGHT)
        
        ttk.Button(right_buttons, text="Re-register Selected", 
                  command=self._reregister_selected_student).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(right_buttons, text="Close", 
                  command=self._close_dialog).pack(side=tk.LEFT)
        
        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready", 
                                     font=('Arial', 9), foreground='blue')
        self.status_label.pack(fill=tk.X, pady=(10, 0))
    
    def _load_students(self):
        """Load students from database"""
        try:
            # Clear existing items
            for item in self.students_tree.get_children():
                self.students_tree.delete(item)
            
            # Get all students
            print("Loading students from database...")
            students = self.database_manager.get_all_students()
            print(f"Found {len(students)} students in database")
            
            # Get face encoding counts for each student
            encodings = self.database_manager.get_face_encodings()
            print(f"Found {len(encodings)} face encodings in database")
            encoding_counts = {}
            for student_id, _, _, _ in encodings:  # Fixed: unpack 4 values
                encoding_counts[student_id] = encoding_counts.get(student_id, 0) + 1
            
            # Add students to treeview
            for student in students:
                student_id = student['student_id']
                encoding_count = encoding_counts.get(student_id, 0)
                last_updated = student.get('updated_at', student.get('created_at', 'Unknown'))
                
                # Format last updated date
                if last_updated != 'Unknown':
                    try:
                        from datetime import datetime
                        if isinstance(last_updated, str):
                            last_updated = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                        last_updated = last_updated.strftime('%Y-%m-%d %H:%M')
                    except:
                        last_updated = str(last_updated)
                
                self.students_tree.insert('', 'end', values=(
                    student_id,
                    student['name'],
                    student.get('email', ''),
                    student.get('department', ''),
                    student.get('year', ''),
                    encoding_count,
                    last_updated
                ))
            
            self.status_label.config(text=f"Loaded {len(students)} students")
            
        except Exception as e:
            error_msg = f"Failed to load students: {str(e)}"
            print(f"ERROR: {error_msg}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", error_msg)
            self.status_label.config(text="Error loading students", foreground='red')
    
    def _refresh_students(self):
        """Refresh the students list"""
        self._load_students()
    
    def _delete_selected_student(self):
        """Delete the selected student"""
        try:
            selected_item = self.students_tree.selection()
            if not selected_item:
                messagebox.showwarning("Warning", "Please select a student to delete")
                return
            
            # Get student info
            item = self.students_tree.item(selected_item[0])
            student_id = item['values'][0]
            student_name = item['values'][1]
            
            # Confirm deletion
            if messagebox.askyesno("Confirm Deletion", 
                                 f"Are you sure you want to delete student '{student_name}' (ID: {student_id})?\n\n"
                                 f"This will permanently remove:\n"
                                 f"- Student information\n"
                                 f"- All face encodings (pickle-based)\n"
                                 f"- All attendance records"):
                
                # Delete student
                success = self.database_manager.delete_student(student_id)
                if success:
                    messagebox.showinfo("Success", f"Student '{student_name}' deleted successfully")
                    self._load_students()
                    self.status_label.config(text=f"Student '{student_name}' deleted", foreground='green')
                else:
                    messagebox.showerror("Error", "Failed to delete student")
                    self.status_label.config(text="Failed to delete student", foreground='red')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete student: {str(e)}")
            self.status_label.config(text="Error deleting student", foreground='red')
    
    def _delete_all_students(self):
        """Delete all students"""
        try:
            # Get all students
            students = self.database_manager.get_all_students()
            if not students:
                messagebox.showinfo("Info", "No students to delete")
                return
            
            # Confirm deletion
            if messagebox.askyesno("Confirm Deletion", 
                                 f"Are you sure you want to delete ALL {len(students)} students?\n\n"
                                 f"This will permanently remove:\n"
                                 f"- All student information\n"
                                 f"- All face encodings (pickle-based)\n"
                                 f"- All attendance records\n\n"
                                 f"This action cannot be undone!"):
                
                # Delete all students
                deleted_count = 0
                for student in students:
                    # Delete from database
                    success = self.database_manager.delete_student(student['student_id'])
                    if success:
                        deleted_count += 1
                
                if deleted_count > 0:
                    messagebox.showinfo("Success", f"Deleted {deleted_count} students successfully")
                    self._load_students()
                    self.status_label.config(text=f"Deleted {deleted_count} students", foreground='green')
                else:
                    messagebox.showerror("Error", "Failed to delete any students")
                    self.status_label.config(text="Failed to delete students", foreground='red')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete students: {str(e)}")
            self.status_label.config(text="Error deleting students", foreground='red')
    
    def _reregister_selected_student(self):
        """Re-register the selected student"""
        try:
            selected_item = self.students_tree.selection()
            if not selected_item:
                messagebox.showwarning("Warning", "Please select a student to re-register")
                return
            
            # Get student info
            item = self.students_tree.item(selected_item[0])
            student_id = item['values'][0]
            student_name = item['values'][1]
            
            # Confirm re-registration
            if messagebox.askyesno("Confirm Re-registration", 
                                 f"Are you sure you want to re-register student '{student_name}' (ID: {student_id})?\n\n"
                                 f"This will:\n"
                                 f"- Delete existing face encodings\n"
                                 f"- Open registration dialog for new face capture\n"
                                 f"- Keep existing student information"):
                
                # Delete existing face encodings
                success = self.database_manager.delete_student(student_id)
                if success:
                    # Open registration dialog
                    self._open_registration_dialog(student_id, student_name)
                else:
                    messagebox.showerror("Error", "Failed to delete existing face encodings")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to re-register student: {str(e)}")

    def _open_bulk_registration(self):
        """Open the bulk registration dialog"""
        try:
            from .bulk_registration_dialog import BulkRegistrationDialog
            config = {
                'face_data_path': 'face_data',
                'models_path': 'models'
            }
            bulk_dialog = BulkRegistrationDialog(self.dialog, self.database_manager, config)
            self.dialog.wait_window(bulk_dialog.dialog)
            self._refresh_students()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open bulk registration: {str(e)}")
    
    def _open_registration_dialog(self, student_id: str, student_name: str):
        """Open registration dialog for re-registration"""
        try:
            from .student_registration import StudentRegistrationDialog
            
            # Create config
            config = {
                'camera_index': 0,
                'detection_confidence': 0.7,
                'face_data_path': 'face_data',
                'models_path': 'models'
            }
            
            # Open registration dialog
            registration_dialog = StudentRegistrationDialog(
                self.dialog, self.database_manager, config
            )
            
            # Pre-fill student information
            registration_dialog.student_id_var.set(student_id)
            registration_dialog.name_var.set(student_name)
            
            # Refresh students list when dialog closes
            def on_registration_close():
                self._load_students()
            
            registration_dialog.dialog.protocol("WM_DELETE_WINDOW", on_registration_close)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open registration dialog: {str(e)}")
    
    def _close_dialog(self):
        """Close the dialog"""
        self.dialog.destroy()
    
    def _center_dialog(self):
        """Center the dialog on the parent window"""
        try:
            self.dialog.update_idletasks()
            width = self.dialog.winfo_width()
            height = self.dialog.winfo_height()
            x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
            y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
            self.dialog.geometry(f'{width}x{height}+{x}+{y}')
        except:
            pass
