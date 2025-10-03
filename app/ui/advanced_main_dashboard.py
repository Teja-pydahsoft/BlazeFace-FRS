"""
Advanced Main Dashboard for BlazeFace-FRS
Modern desktop application with full window interface and comprehensive features
"""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import json
import os

from ..core.database import DatabaseManager
from ..core.dual_pipeline import DualPipeline
from ..utils.camera_utils import CameraManager
from .bulk_registration_dialog import BulkRegistrationDialog
from .advanced_analytics_dashboard import AdvancedAnalyticsDashboard
from .user_management_dialog import UserManagementDialog
from .system_monitoring_dashboard import SystemMonitoringDashboard

class AdvancedMainDashboard:
    def __init__(self):
        """Initialize advanced main dashboard"""
        self.root = tk.Tk()
        self.root.title("BlazeFace-FRS - Advanced Face Recognition System")
        self.root.state('zoomed')  # Maximize window on Windows
        self.root.configure(bg='#f0f0f0')
        
        # Configuration and database
        self.config = self._load_config()
        self.database_manager = DatabaseManager(self.config.get('database_path', 'database/blazeface_frs.db'))
        
        # System components
        self.camera_manager = None
        self.pipeline = None
        self.is_detection_running = False
        
        # UI state
        self.current_user = None
        self.last_stats_update = time.time()
        
        # Setup modern UI
        self._setup_modern_ui()
        
        # Load initial data
        self._load_initial_data()
        
        # Start periodic updates
        self._start_periodic_updates()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration"""
        try:
            with open('app/config.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _setup_modern_ui(self):
        """Setup modern UI with full window layout"""
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Top navigation bar
        self._setup_navigation_bar(main_container)
        
        # Main content area
        self._setup_main_content(main_container)
        
        # Status bar
        self._setup_status_bar(main_container)
    
    def _setup_navigation_bar(self, parent):
        """Setup top navigation bar"""
        nav_frame = tk.Frame(parent, bg='#2c3e50', height=60)
        nav_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        nav_frame.grid_propagate(False)
        
        # Logo and title
        title_frame = tk.Frame(nav_frame, bg='#2c3e50')
        title_frame.pack(side=tk.LEFT, padx=20, pady=15)
        
        title_label = tk.Label(title_frame, text="BlazeFace-FRS", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(side=tk.LEFT)
        
        subtitle_label = tk.Label(title_frame, text="Advanced Face Recognition System", 
                                 font=('Arial', 12), fg='#bdc3c7', bg='#2c3e50')
        subtitle_label.pack(side=tk.LEFT, padx=(10, 0))
        
        # Navigation buttons
        nav_buttons_frame = tk.Frame(nav_frame, bg='#2c3e50')
        nav_buttons_frame.pack(side=tk.RIGHT, padx=20, pady=10)
        
        # Dashboard button (active)
        self.dashboard_btn = tk.Button(nav_buttons_frame, text="Dashboard", 
                                      command=self._show_dashboard,
                                      bg='#3498db', fg='white', font=('Arial', 10, 'bold'),
                                      relief='flat', padx=20, pady=8)
        self.dashboard_btn.pack(side=tk.LEFT, padx=5)
        
        # Analytics button
        self.analytics_btn = tk.Button(nav_buttons_frame, text="Analytics", 
                                      command=self._show_analytics,
                                      bg='#34495e', fg='white', font=('Arial', 10),
                                      relief='flat', padx=20, pady=8)
        self.analytics_btn.pack(side=tk.LEFT, padx=5)
        
        # System button
        self.system_btn = tk.Button(nav_buttons_frame, text="System", 
                                   command=self._show_system,
                                   bg='#34495e', fg='white', font=('Arial', 10),
                                   relief='flat', padx=20, pady=8)
        self.system_btn.pack(side=tk.LEFT, padx=5)
        
        # User management button
        self.users_btn = tk.Button(nav_buttons_frame, text="Users", 
                                  command=self._show_users,
                                  bg='#34495e', fg='white', font=('Arial', 10),
                                  relief='flat', padx=20, pady=8)
        self.users_btn.pack(side=tk.LEFT, padx=5)
    
    def _setup_main_content(self, parent):
        """Setup main content area"""
        # Content container with notebook for tabs
        self.content_notebook = ttk.Notebook(parent)
        self.content_notebook.grid(row=1, column=0, sticky='nsew', pady=(0, 10))
        
        # Dashboard tab
        self._setup_dashboard_tab()
        
        # Analytics tab
        self._setup_analytics_tab()
        
        # System tab
        self._setup_system_tab()
        
        # Users tab
        self._setup_users_tab()
    
    def _setup_dashboard_tab(self):
        """Setup dashboard tab"""
        dashboard_frame = ttk.Frame(self.content_notebook)
        self.content_notebook.add(dashboard_frame, text="Dashboard")
        
        # Configure grid
        dashboard_frame.grid_rowconfigure(0, weight=1)
        dashboard_frame.grid_columnconfigure(0, weight=2)
        dashboard_frame.grid_columnconfigure(1, weight=1)
        
        # Left panel - Live camera and detection
        left_panel = ttk.LabelFrame(dashboard_frame, text="Live Detection", padding="10")
        left_panel.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        self._setup_camera_panel(left_panel)
        
        # Right panel - Quick actions and stats
        right_panel = ttk.Frame(dashboard_frame)
        right_panel.grid(row=0, column=1, sticky='nsew')
        
        self._setup_quick_actions_panel(right_panel)
        self._setup_stats_panel(right_panel)
    
    def _setup_camera_panel(self, parent):
        """Setup camera panel"""
        # Camera canvas
        self.camera_canvas = tk.Canvas(parent, width=800, height=600, bg='black')
        self.camera_canvas.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Camera controls
        controls_frame = tk.Frame(parent)
        controls_frame.pack(fill=tk.X)
        
        # Detection controls
        self.start_detection_btn = tk.Button(controls_frame, text="Start Detection", 
                                           command=self._start_detection,
                                           bg='#27ae60', fg='white', font=('Arial', 10, 'bold'),
                                           relief='flat', padx=20, pady=8)
        self.start_detection_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_detection_btn = tk.Button(controls_frame, text="Stop Detection", 
                                          command=self._stop_detection,
                                          bg='#e74c3c', fg='white', font=('Arial', 10, 'bold'),
                                          relief='flat', padx=20, pady=8, state='disabled')
        self.stop_detection_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Camera source selection
        ttk.Label(controls_frame, text="Camera:").pack(side=tk.LEFT, padx=(20, 5))
        self.camera_var = tk.StringVar(value="webcam")
        camera_combo = ttk.Combobox(controls_frame, textvariable=self.camera_var, 
                                   values=["webcam", "nvr_camera", "ip_camera"], width=12)
        camera_combo.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status indicators
        status_frame = tk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.detection_status = tk.Label(status_frame, text="Status: Ready", 
                                        font=('Arial', 10), fg='#27ae60')
        self.detection_status.pack(side=tk.LEFT)
        
        self.fps_label = tk.Label(status_frame, text="FPS: 0", 
                                 font=('Arial', 10), fg='#7f8c8d')
        self.fps_label.pack(side=tk.RIGHT)
    
    def _setup_quick_actions_panel(self, parent):
        """Setup quick actions panel"""
        actions_frame = ttk.LabelFrame(parent, text="Quick Actions", padding="10")
        actions_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Action buttons with modern styling
        actions = [
            ("Bulk Registration", self._open_bulk_registration, '#3498db'),
            ("Student Management", self._open_student_management, '#9b59b6'),
            ("Attendance History", self._open_attendance_history, '#f39c12'),
            ("Export Data", self._open_export_dialog, '#1abc9c'),
            ("System Settings", self._open_system_settings, '#95a5a6'),
        ]
        
        for text, command, color in actions:
            btn = tk.Button(actions_frame, text=text, command=command,
                           bg=color, fg='white', font=('Arial', 10, 'bold'),
                           relief='flat', padx=15, pady=10)
            btn.pack(fill=tk.X, pady=2)
    
    def _setup_stats_panel(self, parent):
        """Setup statistics panel"""
        stats_frame = ttk.LabelFrame(parent, text="System Statistics", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        # Statistics labels
        self.total_students_label = tk.Label(stats_frame, text="Total Students: 0", 
                                           font=('Arial', 12), fg='#2c3e50')
        self.total_students_label.pack(anchor=tk.W, pady=2)
        
        self.total_encodings_label = tk.Label(stats_frame, text="Face Encodings: 0", 
                                            font=('Arial', 12), fg='#2c3e50')
        self.total_encodings_label.pack(anchor=tk.W, pady=2)
        
        self.today_attendance_label = tk.Label(stats_frame, text="Today's Attendance: 0", 
                                             font=('Arial', 12), fg='#27ae60')
        self.today_attendance_label.pack(anchor=tk.W, pady=2)
        
        self.attendance_rate_label = tk.Label(stats_frame, text="Attendance Rate: 0%", 
                                            font=('Arial', 12), fg='#e74c3c')
        self.attendance_rate_label.pack(anchor=tk.W, pady=2)
        
        # Performance metrics
        ttk.Separator(stats_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        perf_label = tk.Label(stats_frame, text="Performance", 
                            font=('Arial', 10, 'bold'), fg='#34495e')
        perf_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.recognition_accuracy_label = tk.Label(stats_frame, text="Recognition Accuracy: 0%", 
                                                 font=('Arial', 10), fg='#7f8c8d')
        self.recognition_accuracy_label.pack(anchor=tk.W, pady=2)
        
        self.system_uptime_label = tk.Label(stats_frame, text="System Uptime: 00:00:00", 
                                          font=('Arial', 10), fg='#7f8c8d')
        self.system_uptime_label.pack(anchor=tk.W, pady=2)
    
    def _setup_analytics_tab(self):
        """Setup analytics tab"""
        analytics_frame = ttk.Frame(self.content_notebook)
        self.content_notebook.add(analytics_frame, text="Analytics")
        
        # Placeholder for analytics dashboard
        placeholder = tk.Label(analytics_frame, text="Advanced Analytics Dashboard\n(Coming Soon)", 
                             font=('Arial', 16), fg='#7f8c8d')
        placeholder.pack(expand=True)
    
    def _setup_system_tab(self):
        """Setup system tab"""
        system_frame = ttk.Frame(self.content_notebook)
        self.content_notebook.add(system_frame, text="System")
        
        # Placeholder for system monitoring
        placeholder = tk.Label(system_frame, text="System Monitoring Dashboard\n(Coming Soon)", 
                             font=('Arial', 16), fg='#7f8c8d')
        placeholder.pack(expand=True)
    
    def _setup_users_tab(self):
        """Setup users tab"""
        users_frame = ttk.Frame(self.content_notebook)
        self.content_notebook.add(users_frame, text="Users")
        
        # Placeholder for user management
        placeholder = tk.Label(users_frame, text="User Management System\n(Coming Soon)", 
                             font=('Arial', 16), fg='#7f8c8d')
        placeholder.pack(expand=True)
    
    def _setup_status_bar(self, parent):
        """Setup status bar"""
        status_frame = tk.Frame(parent, bg='#34495e', height=30)
        status_frame.grid(row=2, column=0, sticky='ew')
        status_frame.grid_propagate(False)
        
        # Status message
        self.status_label = tk.Label(status_frame, text="Ready", 
                                   font=('Arial', 9), fg='white', bg='#34495e')
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Current time
        self.time_label = tk.Label(status_frame, text="", 
                                 font=('Arial', 9), fg='white', bg='#34495e')
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def _load_initial_data(self):
        """Load initial data and update statistics"""
        try:
            self._update_statistics()
            self._update_time()
        except Exception as e:
            print(f"Error loading initial data: {e}")
    
    def _start_periodic_updates(self):
        """Start periodic UI updates"""
        self._update_time()
        self._update_statistics()
        self.root.after(1000, self._start_periodic_updates)
    
    def _update_time(self):
        """Update time display"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.time_label.config(text=current_time)
        except Exception as e:
            print(f"Error updating time: {e}")
    
    def _update_statistics(self):
        """Update system statistics"""
        try:
            # Only update stats every 5 seconds to avoid performance issues
            current_time = time.time()
            if current_time - self.last_stats_update < 5:
                return
            
            self.last_stats_update = current_time
            
            stats = self.database_manager.get_statistics()
            
            self.total_students_label.config(text=f"Total Students: {stats.get('total_students', 0)}")
            self.total_encodings_label.config(text=f"Face Encodings: {stats.get('total_face_encodings', 0)}")
            self.today_attendance_label.config(text=f"Today's Attendance: {stats.get('today_attendance', 0)}")
            
            # Calculate attendance rate
            total_students = stats.get('total_students', 0)
            today_attendance = stats.get('today_attendance', 0)
            if total_students > 0:
                rate = (today_attendance / total_students) * 100
                self.attendance_rate_label.config(text=f"Attendance Rate: {rate:.1f}%")
            else:
                self.attendance_rate_label.config(text="Attendance Rate: 0%")
            
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def _start_detection(self):
        """Start face detection"""
        try:
            camera_source = self.camera_var.get()
            camera_index = self.config.get('camera_sources', {}).get(camera_source, 0)
            
            self.camera_manager = CameraManager(camera_index)
            if not self.camera_manager.start_camera():
                messagebox.showerror("Error", "Failed to start camera")
                return
            
            # Initialize detection pipeline
            self.pipeline = DualPipeline(self.camera_manager, self.database_manager, self.config)
            self.pipeline.start()
            
            self.is_detection_running = True
            self.start_detection_btn.config(state='disabled')
            self.stop_detection_btn.config(state='normal')
            self.detection_status.config(text="Status: Detection Running", fg='#27ae60')
            
            # Start camera preview
            self._update_camera_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def _stop_detection(self):
        """Stop face detection"""
        try:
            self.is_detection_running = False
            
            if self.pipeline:
                self.pipeline.stop()
                self.pipeline = None
            
            if self.camera_manager:
                self.camera_manager.stop_camera()
                self.camera_manager = None
            
            self.start_detection_btn.config(state='normal')
            self.stop_detection_btn.config(state='disabled')
            self.detection_status.config(text="Status: Stopped", fg='#e74c3c')
            
            # Clear camera canvas
            self.camera_canvas.delete("all")
            self.camera_canvas.create_text(400, 300, text="Camera Stopped", 
                                         fill='white', font=('Arial', 16))
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop detection: {str(e)}")
    
    def _update_camera_preview(self):
        """Update camera preview"""
        try:
            if not self.is_detection_running or not self.camera_manager:
                return
            
            frame = self.camera_manager.get_frame()
            if frame is not None:
                # Get detection results
                if self.pipeline:
                    results = self.pipeline.get_latest_results()
                    if results:
                        frame = self._draw_detections(frame, results)
                
                # Display frame
                self._display_frame(frame)
            
            # Schedule next update
            if self.is_detection_running:
                self.root.after(30, self._update_camera_preview)
                
        except Exception as e:
            print(f"Error updating camera preview: {e}")
            if self.is_detection_running:
                self.root.after(1000, self._update_camera_preview)
    
    def _display_frame(self, frame):
        """Display frame on canvas"""
        try:
            # Resize frame to fit canvas
            canvas_width = self.camera_canvas.winfo_width()
            canvas_height = self.camera_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                height, width = frame.shape[:2]
                scale = min(canvas_width / width, canvas_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                
                resized = cv2.resize(frame, (new_width, new_height))
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                photo = self._cv2_to_photo(rgb_frame)
                
                # Clear canvas and display image
                self.camera_canvas.delete("all")
                self.camera_canvas.create_image(canvas_width//2, canvas_height//2, 
                                              image=photo, anchor=tk.CENTER)
                
        except Exception as e:
            print(f"Error displaying frame: {e}")
    
    def _draw_detections(self, frame, results):
        """Draw detection results on frame"""
        try:
            # Draw face detections
            if 'faces' in results:
                for face in results['faces']:
                    x, y, w, h = face['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw student name if recognized
                    if 'student_id' in face:
                        cv2.putText(frame, face['student_id'], (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return frame
            
        except Exception as e:
            print(f"Error drawing detections: {e}")
            return frame
    
    def _cv2_to_photo(self, cv2_image):
        """Convert OpenCV image to PhotoImage"""
        try:
            from PIL import Image, ImageTk
            pil_image = Image.fromarray(cv2_image)
            return ImageTk.PhotoImage(pil_image)
        except Exception as e:
            print(f"Error converting image: {e}")
            return None
    
    # Navigation methods
    def _show_dashboard(self):
        """Show dashboard tab"""
        self.content_notebook.select(0)
        self._update_nav_buttons('dashboard')
    
    def _show_analytics(self):
        """Show analytics tab"""
        self.content_notebook.select(1)
        self._update_nav_buttons('analytics')
    
    def _show_system(self):
        """Show system tab"""
        self.content_notebook.select(2)
        self._update_nav_buttons('system')
    
    def _show_users(self):
        """Show users tab"""
        self.content_notebook.select(3)
        self._update_nav_buttons('users')
    
    def _update_nav_buttons(self, active):
        """Update navigation button states"""
        buttons = {
            'dashboard': self.dashboard_btn,
            'analytics': self.analytics_btn,
            'system': self.system_btn,
            'users': self.users_btn
        }
        
        for name, btn in buttons.items():
            if name == active:
                btn.config(bg='#3498db')
            else:
                btn.config(bg='#34495e')
    
    # Action methods
    def _open_bulk_registration(self):
        """Open bulk registration dialog"""
        try:
            BulkRegistrationDialog(self.root, self.database_manager, self.config)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open bulk registration: {str(e)}")
    
    def _open_student_management(self):
        """Open student management dialog"""
        try:
            from .student_management import StudentManagementDialog
            StudentManagementDialog(self.root, self.database_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open student management: {str(e)}")
    
    def _open_attendance_history(self):
        """Open attendance history dialog"""
        try:
            from .attendance_history import AttendanceHistoryDialog
            AttendanceHistoryDialog(self.root, self.database_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open attendance history: {str(e)}")
    
    def _open_export_dialog(self):
        """Open export dialog"""
        try:
            from .export_dialog import ExportDialog
            ExportDialog(self.root, self.database_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open export dialog: {str(e)}")
    
    def _open_system_settings(self):
        """Open system settings dialog"""
        try:
            from .system_settings_dialog import SystemSettingsDialog
            SystemSettingsDialog(self.root, self.config)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open system settings: {str(e)}")
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self._cleanup()
        except Exception as e:
            print(f"Error running application: {e}")
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        try:
            self._stop_detection()
            if self.database_manager:
                self.database_manager.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

if __name__ == "__main__":
    app = AdvancedMainDashboard()
    app.run()
