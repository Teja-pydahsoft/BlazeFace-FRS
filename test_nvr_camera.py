"""
Test script for NVR camera functionality
"""

import sys
import os
import cv2
import time

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager

def test_camera_sources():
    """Test different camera sources"""
    print("=" * 60)
    print("TESTING CAMERA SOURCES")
    print("=" * 60)
    
    # Test different camera sources
    test_sources = [
        ("Webcam", "0"),
        ("RTSP Stream", "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"),
        ("HTTP Stream", "http://192.168.3.235:8080/video"),
        ("Video File", "test_video.mp4")
    ]
    
    for name, source in test_sources:
        print(f"\n--- Testing {name}: {source} ---")
        
        try:
            # Initialize camera manager
            camera_manager = CameraManager(source)
            
            if camera_manager.is_initialized:
                print(f"✅ {name} initialized successfully")
                print(f"   Camera type: {camera_manager.camera_type}")
                
                # Try to capture a frame
                camera_manager.start_capture()
                time.sleep(2)  # Wait for camera to stabilize
                
                frame = camera_manager.get_latest_frame()
                if frame is not None:
                    print(f"✅ Frame captured successfully: {frame.shape}")
                else:
                    print(f"⚠️  No frame captured")
                
                camera_manager.stop_capture()
                camera_manager.release()
                
            else:
                print(f"❌ {name} failed to initialize")
                
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
    
    print(f"\n{'='*60}")
    print("CAMERA SOURCE RECOMMENDATIONS")
    print(f"{'='*60}")
    print("1. Webcam: Works if camera is connected")
    print("2. RTSP: Update IP, username, password in config")
    print("3. HTTP: Update IP and port in config")
    print("4. Video File: Place video file in project directory")
    print("\nTo use NVR camera:")
    print("1. Update app/config.json with your camera details")
    print("2. Test connection with VLC or similar tool")
    print("3. Run the main application and select camera source")

def test_rtsp_connection():
    """Test RTSP connection with user input"""
    print(f"\n{'='*60}")
    print("RTSP CONNECTION TESTER")
    print(f"{'='*60}")
    
    print("Enter your NVR camera details:")
    ip = input("IP Address (e.g., 192.168.1.100): ").strip()
    port = input("Port (default 554): ").strip() or "554"
    username = input("Username (default admin): ").strip() or "admin"
    password = input("Password: ").strip()
    path = input("Stream path (default /stream1): ").strip() or "/stream1"
    
    # Ensure path starts with /
    if not path.startswith('/'):
        path = '/' + path
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}{path}"
    print(f"\nTesting RTSP URL: {rtsp_url}")
    
    try:
        camera_manager = CameraManager(rtsp_url)
        
        if camera_manager.is_initialized:
            print("✅ RTSP connection successful!")
            print("✅ Camera is accessible")
            
            # Update config suggestion
            print(f"\nAdd this to your app/config.json:")
            print(f'"nvr_camera": "{rtsp_url}"')
            
        else:
            print("❌ RTSP connection failed")
            print("Check:")
            print("- IP address and port")
            print("- Username and password")
            print("- Network connectivity")
            print("- Camera is online")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main test function"""
    print("NVR Camera Test Utility")
    print("This script tests different camera sources for the BlazeFace-FRS system")
    
    # Test basic camera sources
    test_camera_sources()
    
    # Interactive RTSP test
    test_rtsp_connection()
    
    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print(f"{'='*60}")
    print("1. Update app/config.json with your camera details")
    print("2. Run: python main.py")
    print("3. Open Attendance Marking")
    print("4. Select your camera source from dropdown")
    print("5. Click Switch to change camera")

if __name__ == "__main__":
    main()
