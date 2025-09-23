"""
Attendance History Viewer for BlazeFace-FRS
View and manage attendance records
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import os

from ..core.database import DatabaseManager

class AttendanceHistoryDialog:
    def __init__(self, parent, database_manager: DatabaseManager):
        """
        Initialize attendance history dialog
        
        Args:
            parent: Parent window
            database_manager: Database manager instance
        """
        self.parent = parent
        self.database_manager = database_manager
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Attendance History - BlazeFace-FRS")
        self.dialog.geometry("1200x800")
        self.dialog.resizable(True, True)
        
        # Make dialog modal
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Setup UI
        self._setup_ui()
        
        # Load initial data
        self._load_attendance_data()
        
        # Center dialog
        self._center_dialog()
    
    def _setup_ui(self):
        """Setup the attendance history UI"""
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top panel - Filters and Controls
        self._setup_filter_panel(main_frame)
        
        # Middle panel - Attendance List
        self._setup_attendance_panel(main_frame)
        
        # Bottom panel - Statistics and Actions
        self._setup_action_panel(main_frame)
    
    def _setup_filter_panel(self, parent):
        """Setup filter and control panel"""
        filter_frame = ttk.LabelFrame(parent, text="Filters & Controls", padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Date range selection
        date_frame = ttk.Frame(filter_frame)
        date_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(date_frame, text="Date Range:").pack(side=tk.LEFT, padx=(0, 10))
        
        # From date
        ttk.Label(date_frame, text="From:").pack(side=tk.LEFT)
        self.from_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))
        from_date_entry = ttk.Entry(date_frame, textvariable=self.from_date_var, width=12)
        from_date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        # To date
        ttk.Label(date_frame, text="To:").pack(side=tk.LEFT)
        self.to_date_var = tk.StringVar(value=datetime.now().strftime('%Y-%m-%d'))
        to_date_entry = ttk.Entry(date_frame, textvariable=self.to_date_var, width=12)
        to_date_entry.pack(side=tk.LEFT, padx=(5, 10))
        
        # Student filter
        ttk.Label(date_frame, text="Student ID:").pack(side=tk.LEFT, padx=(20, 5))
        self.student_id_var = tk.StringVar()
        student_id_entry = ttk.Entry(date_frame, textvariable=self.student_id_var, width=15)
        student_id_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status filter
        ttk.Label(date_frame, text="Status:").pack(side=tk.LEFT, padx=(10, 5))
        self.status_var = tk.StringVar(value="All")
        status_combo = ttk.Combobox(date_frame, textvariable=self.status_var, 
                                   values=["All", "Present", "Absent", "Late"], 
                                   state="readonly", width=10)
        status_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Filter button
        ttk.Button(date_frame, text="Apply Filters", 
                  command=self._apply_filters).pack(side=tk.LEFT, padx=(10, 0))
        
        # Quick date buttons
        quick_frame = ttk.Frame(filter_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(quick_frame, text="Quick Select:").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(quick_frame, text="Today", 
                  command=lambda: self._set_date_range(0, 0)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Last 7 Days", 
                  command=lambda: self._set_date_range(7, 0)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="Last 30 Days", 
                  command=lambda: self._set_date_range(30, 0)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(quick_frame, text="This Month", 
                  command=self._set_this_month).pack(side=tk.LEFT, padx=(0, 5))
    
    def _setup_attendance_panel(self, parent):
        """Setup attendance list panel"""
        attendance_frame = ttk.LabelFrame(parent, text="Attendance Records", padding="10")
        attendance_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create treeview for attendance records
        columns = ('Date', 'Time', 'Student ID', 'Name', 'Department', 'Status', 'Confidence', 'Notes')
        self.attendance_tree = ttk.Treeview(attendance_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        column_widths = {
            'Date': 100, 'Time': 80, 'Student ID': 100, 'Name': 150, 
            'Department': 120, 'Status': 80, 'Confidence': 80, 'Notes': 200
        }
        
        for col in columns:
            self.attendance_tree.heading(col, text=col)
            self.attendance_tree.column(col, width=column_widths[col], anchor=tk.CENTER)
        
        # Scrollbars for treeview
        v_scrollbar = ttk.Scrollbar(attendance_frame, orient=tk.VERTICAL, command=self.attendance_tree.yview)
        h_scrollbar = ttk.Scrollbar(attendance_frame, orient=tk.HORIZONTAL, command=self.attendance_tree.xview)
        self.attendance_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack treeview and scrollbars
        self.attendance_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Bind double-click event
        self.attendance_tree.bind('<Double-1>', self._on_record_double_click)
    
    def _setup_action_panel(self, parent):
        """Setup action and statistics panel"""
        action_frame = ttk.Frame(parent)
        action_frame.pack(fill=tk.X)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(action_frame, text="Statistics", padding="10")
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.total_records_label = ttk.Label(stats_frame, text="Total Records: 0")
        self.total_records_label.pack(anchor=tk.W)
        
        self.present_count_label = ttk.Label(stats_frame, text="Present: 0")
        self.present_count_label.pack(anchor=tk.W)
        
        self.absent_count_label = ttk.Label(stats_frame, text="Absent: 0")
        self.absent_count_label.pack(anchor=tk.W)
        
        self.attendance_rate_label = ttk.Label(stats_frame, text="Attendance Rate: 0%")
        self.attendance_rate_label.pack(anchor=tk.W)
        
        # Actions frame
        actions_frame = ttk.LabelFrame(action_frame, text="Actions", padding="10")
        actions_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(actions_frame, text="Export to CSV", 
                  command=self._export_to_csv).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Export to Excel", 
                  command=self._export_to_excel).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Print Report", 
                  command=self._print_report).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Refresh Data", 
                  command=self._load_attendance_data).pack(fill=tk.X, pady=2)
        ttk.Button(actions_frame, text="Close", 
                  command=self._close_dialog).pack(fill=tk.X, pady=(20, 2))
    
    def _set_date_range(self, days_back: int, days_forward: int):
        """Set date range for quick selection"""
        try:
            from_date = datetime.now() - timedelta(days=days_back)
            to_date = datetime.now() + timedelta(days=days_forward)
            
            self.from_date_var.set(from_date.strftime('%Y-%m-%d'))
            self.to_date_var.set(to_date.strftime('%Y-%m-%d'))
            
            self._apply_filters()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set date range: {str(e)}")
    
    def _set_this_month(self):
        """Set date range to current month"""
        try:
            now = datetime.now()
            from_date = now.replace(day=1)
            to_date = now
            
            self.from_date_var.set(from_date.strftime('%Y-%m-%d'))
            self.to_date_var.set(to_date.strftime('%Y-%m-%d'))
            
            self._apply_filters()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to set this month: {str(e)}")
    
    def _apply_filters(self):
        """Apply filters and reload data"""
        try:
            self._load_attendance_data()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filters: {str(e)}")
    
    def _load_attendance_data(self):
        """Load attendance data based on current filters"""
        try:
            # Get filter values
            from_date = self.from_date_var.get()
            to_date = self.to_date_var.get()
            student_id = self.student_id_var.get().strip()
            status = self.status_var.get()
            
            # Validate dates
            try:
                from datetime import datetime
                datetime.strptime(from_date, '%Y-%m-%d')
                datetime.strptime(to_date, '%Y-%m-%d')
            except ValueError:
                messagebox.showerror("Error", "Invalid date format. Use YYYY-MM-DD")
                return
            
            # Get records from database
            records = self.database_manager.get_attendance_records(
                student_id=student_id if student_id else None,
                date_from=from_date,
                date_to=to_date
            )
            
            # Filter by status if not "All"
            if status != "All":
                records = [r for r in records if r['status'].lower() == status.lower()]
            
            # Clear existing items
            for item in self.attendance_tree.get_children():
                self.attendance_tree.delete(item)
            
            # Add records to treeview
            for record in records:
                status_text = record['status'].title()
                confidence = f"{record['confidence']:.2f}" if record['confidence'] else "N/A"
                notes = record.get('notes', '')
                
                self.attendance_tree.insert('', 'end', values=(
                    record['date'],
                    record['time'],
                    record['student_id'],
                    record['name'],
                    record.get('department', ''),
                    status_text,
                    confidence,
                    notes
                ))
            
            # Update statistics
            self._update_statistics(records)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load attendance data: {str(e)}")
    
    def _update_statistics(self, records: List[Dict[str, Any]]):
        """Update statistics display"""
        try:
            total_records = len(records)
            present_count = len([r for r in records if r['status'].lower() == 'present'])
            absent_count = len([r for r in records if r['status'].lower() == 'absent'])
            
            attendance_rate = (present_count / total_records * 100) if total_records > 0 else 0
            
            self.total_records_label.config(text=f"Total Records: {total_records}")
            self.present_count_label.config(text=f"Present: {present_count}")
            self.absent_count_label.config(text=f"Absent: {absent_count}")
            self.attendance_rate_label.config(text=f"Attendance Rate: {attendance_rate:.1f}%")
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def _on_record_double_click(self, event):
        """Handle double-click on attendance record"""
        try:
            selection = self.attendance_tree.selection()
            if selection:
                item = self.attendance_tree.item(selection[0])
                values = item['values']
                
                # Show record details
                details = f"Attendance Record Details:\n\n"
                details += f"Date: {values[0]}\n"
                details += f"Time: {values[1]}\n"
                details += f"Student ID: {values[2]}\n"
                details += f"Name: {values[3]}\n"
                details += f"Department: {values[4]}\n"
                details += f"Status: {values[5]}\n"
                details += f"Confidence: {values[6]}\n"
                details += f"Notes: {values[7]}"
                
                messagebox.showinfo("Record Details", details)
                
        except Exception as e:
            print(f"Error handling double-click: {e}")
    
    def _export_to_csv(self):
        """Export attendance data to CSV file"""
        try:
            # Get current filter values
            from_date = self.from_date_var.get()
            to_date = self.to_date_var.get()
            
            # Ask for save location
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialname=f"attendance_{from_date}_to_{to_date}.csv"
            )
            
            if filename:
                # Get all records from treeview
                records = []
                for item in self.attendance_tree.get_children():
                    values = self.attendance_tree.item(item)['values']
                    records.append({
                        'Date': values[0],
                        'Time': values[1],
                        'Student ID': values[2],
                        'Name': values[3],
                        'Department': values[4],
                        'Status': values[5],
                        'Confidence': values[6],
                        'Notes': values[7]
                    })
                
                # Write CSV file
                with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                    fieldnames = ['Date', 'Time', 'Student ID', 'Name', 'Department', 'Status', 'Confidence', 'Notes']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    
                    writer.writeheader()
                    writer.writerows(records)
                
                messagebox.showinfo("Success", f"Data exported to {filename}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to CSV: {str(e)}")
    
    def _export_to_excel(self):
        """Export attendance data to Excel file"""
        try:
            messagebox.showinfo("Info", "Excel export feature coming soon!\nUse CSV export for now.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export to Excel: {str(e)}")
    
    def _print_report(self):
        """Print attendance report"""
        try:
            messagebox.showinfo("Info", "Print report feature coming soon!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to print report: {str(e)}")
    
    def _close_dialog(self):
        """Close the dialog"""
        try:
            self.dialog.destroy()
        except Exception as e:
            print(f"Error closing dialog: {e}")
    
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
