"""
Setup script for BlazeFace-FRS System
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    try:
        if not os.path.exists("venv"):
            print("Creating virtual environment...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✓ Virtual environment created")
        else:
            print("✓ Virtual environment already exists")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def get_pip_command():
    """Get the correct pip command for the platform"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip.exe")
    else:
        return os.path.join("venv", "bin", "pip")

def install_dependencies():
    """Install required dependencies"""
    try:
        pip_cmd = get_pip_command()
        
        print("Installing dependencies...")
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False
    except FileNotFoundError:
        print("Error: pip not found. Please ensure Python is properly installed.")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        "database",
        "face_data", 
        "models",
        "logs",
        "assets"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Created directory: {directory}")

def test_installation():
    """Test the installation"""
    try:
        print("Testing installation...")
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Installation test passed")
            return True
        else:
            print("✗ Installation test failed")
            print("Error output:", result.stderr)
            return False
    except Exception as e:
        print(f"Error running test: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 50)
    print("BlazeFace-FRS System Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Create virtual environment
    if not create_virtual_environment():
        return 1
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("Warning: Some dependencies may not have installed correctly")
        print("You can try running: pip install -r requirements.txt manually")
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 50)
        print("Setup completed successfully!")
        print("=" * 50)
        print("To run the application:")
        print("  Windows: run.bat")
        print("  Linux/Mac: python main.py")
        print("\nOr activate the virtual environment and run:")
        print("  python main.py")
        return 0
    else:
        print("\n" + "=" * 50)
        print("Setup completed with warnings")
        print("=" * 50)
        print("Some tests failed, but you can still try running the application")
        return 1

if __name__ == "__main__":
    sys.exit(main())
