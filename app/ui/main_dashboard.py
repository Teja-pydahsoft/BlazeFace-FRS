"""
Main Dashboard for BlazeFace-FRS System
Combined interface for human and face detection
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
import os
from typing import Dict, Any, Optional

from ..core.dual_pipeline import DualPipeline
from ..core.database import DatabaseManager
from ..utils.camera_utils import CameraManager

class MainDashboard:
    def __init__(self, root: tk.Tk):
        """
        Initialize main dashboard
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("BlazeFace-FRS - Dual Detection System")
        self.root.state('zoomed')
        self.root.configure(bg='#f0f0f0')
        self.root.resizable(True, True) # Make the window resizable
        self.root.minsize(1024, 768) # Set a minimum size for the window
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        camera_source = self.config.get('camera_sources', {}).get('webcam', self.config.get('camera_index', 0))
        self.camera_manager = CameraManager(camera_source)
        self.database_manager = DatabaseManager(self.config.get('database_path', 'database/blazeface_frs.db'))
        self.pipeline = None
        
        # UI state
        self.is_running = False
        self.current_frame = None
        self.detection_results = {}
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        
        # Start camera preview
        self._start_camera_preview()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}
    
    def _setup_ui(self):
        """Setup the main UI components"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel - Controls
        self._setup_control_panel(main_frame)
        
        # Right panel - Camera and Detection
        self._setup_camera_panel(main_frame)
        
        # Status bar
        self._setup_status_bar()
    
    def _setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Detection controls
        detection_frame = ttk.LabelFrame(control_frame, text="Detection Controls", padding="5")
        detection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Start/Stop buttons
        self.start_btn = ttk.Button(detection_frame, text="Start Detection", 
                                   command=self._start_detection)
        self.start_btn.pack(fill=tk.X, pady=2)
        
        self.stop_btn = ttk.Button(detection_frame, text="Stop Detection", 
                                  command=self._stop_detection, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        
        # Detection type selection
        self.detection_var = tk.StringVar(value="both")
        ttk.Label(detection_frame, text="Detection Type:").pack(anchor=tk.W)
        
        ttk.Radiobutton(detection_frame, text="Face Only", variable=self.detection_var, 
                       value="face").pack(anchor=tk.W)
        ttk.Radiobutton(detection_frame, text="Human Only", variable=self.detection_var, 
                       value="human").pack(anchor=tk.W)
        ttk.Radiobutton(detection_frame, text="Both", variable=self.detection_var, 
                       value="both").pack(anchor=tk.W)
        
        # Confidence threshold
        conf_frame = ttk.Frame(detection_frame)
        conf_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(conf_frame, text="Confidence:").pack(side=tk.LEFT)
        self.confidence_var = tk.DoubleVar(value=0.7)
        self.confidence_scale = ttk.Scale(conf_frame, from_=0.1, to=1.0, 
                                        variable=self.confidence_var, orient=tk.HORIZONTAL)
        self.confidence_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Settings panel
        settings_frame = ttk.LabelFrame(control_frame, text="Settings", padding="5")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Camera selection
        ttk.Label(settings_frame, text="Camera:").pack(anchor=tk.W)
        self.camera_var = tk.StringVar(value="0")
        camera_combo = ttk.Combobox(settings_frame, textvariable=self.camera_var, 
                                   values=["0", "1", "2", "3"], state="readonly")
        camera_combo.pack(fill=tk.X, pady=(0, 5))
        
        # Max faces
        ttk.Label(settings_frame, text="Max Faces:").pack(anchor=tk.W)
        self.max_faces_var = tk.IntVar(value=10)
        max_faces_spin = ttk.Spinbox(settings_frame, from_=1, to=50, 
                                   textvariable=self.max_faces_var, width=10)
        max_faces_spin.pack(fill=tk.X, pady=(0, 5))
        
        # Statistics panel
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics", padding="5")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.faces_count_label = ttk.Label(stats_frame, text="Faces: 0")
        self.faces_count_label.pack(anchor=tk.W)
        
        self.humans_count_label = ttk.Label(stats_frame, text="Humans: 0")
        self.humans_count_label.pack(anchor=tk.W)
        
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0")
        self.fps_label.pack(anchor=tk.W)
        
        # Action buttons
        action_frame = ttk.LabelFrame(control_frame, text="Actions", padding="5")
        action_frame.pack(fill=tk.X)
        
        ttk.Button(action_frame, text="Register Student", 
                  command=self._open_student_registration).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Manage Students", 
                  command=self._open_student_management).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Take Attendance", 
                  command=self._open_attendance_marking).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="View Attendance", 
                  command=self._open_attendance_history).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Settings", 
                  command=self._open_settings).pack(fill=tk.X, pady=2)
    
    def _setup_camera_panel(self, parent):
        """Setup camera display panel"""
        camera_frame = ttk.LabelFrame(parent, text="Camera Feed", padding="10")
        camera_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Camera display
        self.camera_label = ttk.Label(camera_frame, text="Camera not available", 
                                     anchor=tk.CENTER, background='black', foreground='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True)
        
        # Detection info
        info_frame = ttk.Frame(camera_frame)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.detection_info = tk.Text(info_frame, height=8, width=50, wrap=tk.WORD)
        self.detection_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Scrollbar for detection info
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.detection_info.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.detection_info.configure(yscrollcommand=scrollbar.set)
    
    def _setup_status_bar(self):
        """Setup status bar"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(self.status_bar, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.time_label = ttk.Label(self.status_bar, text="", relief=tk.SUNKEN, anchor=tk.E)
        self.time_label.pack(side=tk.RIGHT)
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Session", command=self._new_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._exit_application)
        
        # Detection menu
        detection_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Detection", menu=detection_menu)
        detection_menu.add_command(label="Start Detection", command=self._start_detection)
        detection_menu.add_command(label="Stop Detection", command=self._stop_detection)
        detection_menu.add_separator()
        detection_menu.add_command(label="Reset Camera", command=self._reset_camera)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _start_camera_preview(self):
        """Start camera preview"""
        try:
            if self.camera_manager.is_camera_available():
                self._update_camera_preview()
            else:
                self.camera_label.config(text="Camera not available")
        except Exception as e:
            print(f"Error starting camera preview: {e}")
    
    def _update_camera_preview(self):
        """Update camera preview"""
        try:
            if self.camera_manager.is_camera_available():
                ret, frame = self.camera_manager.get_frame()
                if ret and frame is not None:
                    # Process frame through pipeline if running
                    if self.is_running and self.pipeline:
                        results = self.pipeline.process_frame(frame)
                        frame = self.pipeline.draw_detections(frame)
                        self._update_detection_info(results)
                    
                    # Convert frame for display
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    frame_tk = ImageTk.PhotoImage(frame_pil)
                    
                    # Update display
                    self.camera_label.config(image=frame_tk, text="")
                    self.camera_label.image = frame_tk
                    
                    # Update statistics
                    self._update_statistics()
                else:
                    # Only show "No camera feed" if we've been trying for a while
                    if not hasattr(self, '_no_feed_count'):
                        self._no_feed_count = 0
                    self._no_feed_count += 1
                    
                    if self._no_feed_count > 10:  # Show error after 10 failed attempts
                        self.camera_label.config(text="No camera feed")
            else:
                # Camera not available, try to reinitialize much less frequently
                if not hasattr(self, '_reinit_count'):
                    self._reinit_count = 0
                if not hasattr(self, '_last_reinit_time'):
                    self._last_reinit_time = 0
                
                current_time = time.time()
                self._reinit_count += 1
                
                # Only try to reinitialize every 60 seconds and after 30 failed attempts
                if (self._reinit_count > 30 and 
                    current_time - self._last_reinit_time > 60):
                    print("Camera not available, attempting to reinitialize...")
                    try:
                        self.camera_manager.release()
                        camera_source = self.config.get('camera_sources', {}).get('webcam', self.config.get('camera_index', 0))
                        from ..utils.camera_utils import CameraManager
                        self.camera_manager = CameraManager(camera_source)
                        self._reinit_count = 0
                        self._last_reinit_time = current_time
                    except Exception as e:
                        print(f"Failed to reinitialize camera: {e}")
                        self._last_reinit_time = current_time
            
            # Schedule next update
            self.root.after(30, self._update_camera_preview)
            
        except Exception as e:
            print(f"Error updating camera preview: {e}")
            self.root.after(1000, self._update_camera_preview)
    
    def _start_detection(self):
        """Start detection pipeline"""
        try:
            if self.is_running:
                return
            
            # Initialize pipeline
            self.pipeline = DualPipeline(self.config)
            self.pipeline.start_pipeline()
            
            self.is_running = True
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            
            self.status_label.config(text="Detection running...")
            self._log_message("Detection started")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start detection: {str(e)}")
    
    def _stop_detection(self):
        """Stop detection pipeline"""
        try:
            if not self.is_running:
                return
            
            if self.pipeline:
                self.pipeline.stop_pipeline()
                self.pipeline = None
            
            self.is_running = False
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            
            self.status_label.config(text="Detection stopped")
            self._log_message("Detection stopped")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to stop detection: {str(e)}")
    
    def _update_detection_info(self, results: Dict[str, Any]):
        """Update detection information display"""
        try:
            info_text = f"Detection Results:\n"
            info_text += f"Faces: {len(results.get('faces', []))}\n"
            info_text += f"Humans: {len(results.get('humans', []))}\n"
            info_text += f"Embeddings: {len(results.get('embeddings', []))}\n"
            info_text += f"Timestamp: {results.get('timestamp', 0):.2f}\n\n"
            
            # Add face details
            if results.get('faces'):
                info_text += "Face Details:\n"
                for i, face in enumerate(results['faces']):
                    x, y, w, h, conf = face
                    info_text += f"  Face {i+1}: ({x},{y}) {w}x{h} conf={conf:.2f}\n"
            
            # Add human details
            if results.get('humans'):
                info_text += "\nHuman Details:\n"
                for i, human in enumerate(results['humans']):
                    x, y, w, h, conf = human
                    info_text += f"  Human {i+1}: ({x},{y}) {w}x{h} conf={conf:.2f}\n"
            
            self.detection_info.delete(1.0, tk.END)
            self.detection_info.insert(1.0, info_text)
            
        except Exception as e:
            print(f"Error updating detection info: {e}")
    
    def _update_statistics(self):
        """Update statistics display"""
        try:
            if self.pipeline:
                results = self.pipeline.get_detection_results()
                self.faces_count_label.config(text=f"Faces: {len(results.get('faces', []))}")
                self.humans_count_label.config(text=f"Humans: {len(results.get('humans', []))}")
                
                # Update time
                current_time = time.strftime("%H:%M:%S")
                self.time_label.config(text=current_time)
                
        except Exception as e:
            print(f"Error updating statistics: {e}")
    
    def _log_message(self, message: str):
        """Log message to detection info"""
        try:
            timestamp = time.strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}\n"
            self.detection_info.insert(tk.END, log_message)
            self.detection_info.see(tk.END)
        except Exception as e:
            print(f"Error logging message: {e}")
    
    def _open_student_registration(self):
        """Open enhanced student registration dialog"""
        try:
            from .enhanced_registration import EnhancedRegistrationDialog
            # Pass the existing camera manager to avoid conflicts
            dialog = EnhancedRegistrationDialog(self.root, self.database_manager, self.config, self.camera_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open student registration: {str(e)}")
    
    def _open_attendance_marking(self):
        """Open attendance marking dialog"""
        try:
            from .attendance_marking import AttendanceMarkingDialog
            dialog = AttendanceMarkingDialog(self.root, self.database_manager, self.config)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open attendance marking: {str(e)}")
    
    def _open_student_management(self):
        """Open student management dialog"""
        try:
            from .student_management import StudentManagementDialog
            dialog = StudentManagementDialog(self.root, self.database_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open student management: {str(e)}")
    
    def _open_attendance_history(self):
        """Open attendance history dialog"""
        try:
            from .attendance_history import AttendanceHistoryDialog
            dialog = AttendanceHistoryDialog(self.root, self.database_manager)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open attendance history: {str(e)}")
    
    def _open_settings(self):
        """Open settings dialog"""
        # Placeholder for settings
        messagebox.showinfo("Info", "Settings feature coming soon!")
    
    def _new_session(self):
        """Start new session"""
        if self.is_running:
            self._stop_detection()
        self._log_message("New session started")
    
    def _reset_camera(self):
        """Reset camera"""
        try:
            self.camera_manager.release()
            self.camera_manager = CameraManager(self.config.get('camera_index', 0))
            self._log_message("Camera reset")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset camera: {str(e)}")
    
    def _show_about(self):
        """Show about dialog"""
        about_text = "BlazeFace-FRS v1.0\n\n"
        about_text += "Dual Detection System\n"
        about_text += "BlazeFace + FaceNet + Human Detection\n\n"
        about_text += "Developed for efficient real-time detection"
        
        messagebox.showinfo("About", about_text)
    
    def _exit_application(self):
        """Exit application"""
        try:
            if self.is_running:
                self._stop_detection()
            self.camera_manager.release()
            self.root.quit()
        except Exception as e:
            print(f"Error exiting application: {e}")
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        except Exception as e:
            print(f"Error running application: {e}")
        finally:
            self._exit_application()
