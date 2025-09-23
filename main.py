"""
BlazeFace-FRS System
Dual Detection System using BlazeFace and FaceNet
Main application entry point
"""

import sys
import os
import tkinter as tk
from tkinter import messagebox
import logging
import json
from pathlib import Path

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.ui.main_dashboard import MainDashboard
from app.core.database import DatabaseManager

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('blazeface_frs.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import cv2
        import numpy as np
        import mediapipe as mp
        import tensorflow as tf
        from PIL import Image, ImageTk
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages using: pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'database',
        'face_data',
        'models',
        'logs',
        'assets'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """Main application entry point"""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting BlazeFace-FRS System")
        
        # Check dependencies
        if not check_dependencies():
            return 1
        
        # Create necessary directories
        create_directories()
        
        # Initialize database
        try:
            db_manager = DatabaseManager()
            logger.info("Database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            messagebox.showerror("Database Error", f"Failed to initialize database: {e}")
            return 1
        
        # Create and run the main application
        root = tk.Tk()
        
        # Set application icon if available
        try:
            icon_path = os.path.join('assets', 'app_icon.ico')
            if os.path.exists(icon_path):
                root.iconbitmap(icon_path)
        except:
            pass
        
        # Create main dashboard
        app = MainDashboard(root)
        
        # Handle window closing
        def on_closing():
            try:
                app._exit_application()
                root.destroy()
            except:
                root.destroy()
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Run the application
        logger.info("Application started successfully")
        app.run()
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        messagebox.showerror("Fatal Error", f"Application failed to start: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
