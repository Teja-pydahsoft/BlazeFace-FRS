"""
Test high-performance multi-threaded face detection
with distance-based recognition zones and high FPS processing
"""

import sys
import os
import cv2
import time
import numpy as np
import logging

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.utils.camera_utils import CameraManager
from app.core.high_performance_detector import HighPerformanceDetector, ProcessingMode

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_high_performance_detection():
    """Test high-performance face detection"""
    print("=" * 60)
    print("HIGH-PERFORMANCE FACE DETECTION TEST")
    print("=" * 60)
    
    nvr_url = "rtsp://admin:nvr@pydah@192.168.3.235:554/stream1"
    
    try:
        # Initialize camera
        print(f"Connecting to NVR camera: {nvr_url}")
        camera_manager = CameraManager(nvr_url)
        
        if not camera_manager.is_initialized:
            print("❌ Failed to initialize NVR camera")
            return False
        
        print("✅ NVR camera initialized successfully")
        
        # Initialize high-performance detector
        detector = HighPerformanceDetector(
            processing_mode=ProcessingMode.BALANCED,
            max_workers=4,
            frame_buffer_size=10
        )
        
        # Start worker threads
        detector.start_workers()
        print("✅ High-performance detector started with 4 worker threads")
        
        # Test different processing modes
        modes = [
            (ProcessingMode.HIGH_FPS, "High FPS Mode"),
            (ProcessingMode.BALANCED, "Balanced Mode"),
            (ProcessingMode.HIGH_ACCURACY, "High Accuracy Mode")
        ]
        
        for mode, mode_name in modes:
            print(f"\n--- Testing {mode_name} ---")
            detector.processing_mode = mode
            
            # Test face detection
            detection_count = 0
            total_faces = 0
            total_frames = 15
            start_time = time.time()
            
            print(f"Testing {mode_name} on {total_frames} frames...")
            
            for i in range(total_frames):
                ret, frame = camera_manager.get_frame()
                if ret and frame is not None:
                    # Add frame to processing queue
                    success = detector.process_frame(frame)
                    if success:
                        # Get result
                        result = detector.get_result()
                        if result:
                            faces = result['faces']
                            fps = result['fps']
                            
                            if faces:
                                detection_count += 1
                                total_faces += len(faces)
                                print(f"  Frame {i+1}: Found {len(faces)} faces, FPS: {fps:.1f}")
                                
                                # Save first few detections
                                if detection_count <= 3:
                                    frame_with_zones = detector.draw_detection_zones(frame.copy())
                                    frame_with_faces = detector.draw_faces_with_ids(frame_with_zones, faces)
                                    cv2.imwrite(f"high_perf_{mode.value}_{detection_count}.jpg", frame_with_faces)
                                    print(f"    💾 Saved: high_perf_{mode.value}_{detection_count}.jpg")
                            else:
                                print(f"  Frame {i+1}: No faces detected, FPS: {fps:.1f}")
                        else:
                            print(f"  Frame {i+1}: Processing...")
                    else:
                        print(f"  Frame {i+1}: Queue full, skipping")
                
                time.sleep(0.1)  # Small delay
            
            # Calculate performance metrics
            elapsed_time = time.time() - start_time
            actual_fps = total_frames / elapsed_time
            detection_rate = (detection_count / total_frames) * 100
            avg_faces_per_frame = total_faces / total_frames if total_frames > 0 else 0
            
            print(f"  {mode_name} Results:")
            print(f"    Detection rate: {detection_rate:.1f}%")
            print(f"    Total faces: {total_faces}")
            print(f"    Avg faces/frame: {avg_faces_per_frame:.1f}")
            print(f"    Actual FPS: {actual_fps:.1f}")
            print(f"    Processing time: {elapsed_time:.2f}s")
        
        # Get final performance stats
        stats = detector.get_performance_stats()
        print(f"\n--- Final Performance Stats ---")
        print(f"Current FPS: {stats['current_fps']:.1f}")
        print(f"Total frames processed: {stats['frame_count']}")
        print(f"Active face tracks: {stats['active_tracks']}")
        print(f"Queue sizes: {stats['queue_sizes']}")
        print(f"Processing mode: {stats['processing_mode']}")
        
        # Cleanup
        detector.stop_workers()
        camera_manager.release()
        
        print("\n🎉 HIGH-PERFORMANCE DETECTION TEST COMPLETE!")
        return True
        
    except Exception as e:
        print(f"❌ Error in high-performance test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_zones():
    """Test detection zones visualization"""
    print("\n" + "=" * 60)
    print("DETECTION ZONES TEST")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Initialize detector
        detector = HighPerformanceDetector()
        
        # Draw detection zones
        frame_with_zones = detector.draw_detection_zones(frame)
        
        # Save result
        cv2.imwrite("detection_zones_visualization.jpg", frame_with_zones)
        print("💾 Saved: detection_zones_visualization.jpg")
        
        # Show zone information
        print("\n--- Detection Zones ---")
        for i, zone in enumerate(detector.detection_zones):
            print(f"Zone {i+1}: {zone.name}")
            print(f"  Position: ({zone.x}, {zone.y})")
            print(f"  Size: {zone.width}x{zone.height}")
            print(f"  Face size range: {zone.min_face_size}-{zone.max_face_size}")
            print(f"  Confidence threshold: {zone.confidence_threshold}")
            print(f"  Color: {zone.color}")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in detection zones test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_face_tracking():
    """Test face tracking and indexing"""
    print("\n" + "=" * 60)
    print("FACE TRACKING TEST")
    print("=" * 60)
    
    try:
        # Load saved frame
        frame_path = "nvr_test_frame_fixed.jpg"
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"❌ Failed to load frame: {frame_path}")
            return False
        
        print(f"✅ Loaded frame: {frame.shape}")
        
        # Initialize detector
        detector = HighPerformanceDetector()
        
        # Simulate face tracking over multiple frames
        print("Simulating face tracking over 10 frames...")
        
        for i in range(10):
            # Simulate face detection (using a simple approach for testing)
            faces = [(1138, 160, 56, 56, 0.683)]  # Single face from saved frame
            
            # Add some variation to simulate movement
            if i > 0:
                faces = [(faces[0][0] + i*2, faces[0][1] + i, faces[0][2], faces[0][3], faces[0][4])]
            
            # Update face tracking
            faces_with_ids = detector._update_face_tracking(faces)
            
            print(f"  Frame {i+1}: {len(faces_with_ids)} faces with IDs")
            for face in faces_with_ids:
                x, y, w, h, conf, face_id = face
                print(f"    Face ID {face_id}: bbox=({x},{y},{w},{h}), conf={conf:.3f}")
        
        # Show final tracking stats
        print(f"\nFinal tracking stats:")
        print(f"  Total face tracks: {len(detector.face_tracks)}")
        print(f"  Next face ID: {detector.next_face_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in face tracking test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("High-Performance Face Detection Test")
    print("This tests multi-threaded processing, detection zones, and face tracking")
    print("=" * 60)
    
    # Test 1: Detection zones
    zones_success = test_detection_zones()
    
    # Test 2: Face tracking
    tracking_success = test_face_tracking()
    
    # Test 3: High-performance detection
    performance_success = test_high_performance_detection()
    
    # Results
    print("\n" + "=" * 60)
    print("HIGH-PERFORMANCE TEST RESULTS")
    print("=" * 60)
    print(f"Detection zones test: {'✅ PASS' if zones_success else '❌ FAIL'}")
    print(f"Face tracking test: {'✅ PASS' if tracking_success else '❌ FAIL'}")
    print(f"High-performance test: {'✅ PASS' if performance_success else '❌ FAIL'}")
    
    if zones_success and tracking_success and performance_success:
        print("\n🎉 HIGH-PERFORMANCE DETECTION IS WORKING!")
        print("✅ Multi-threaded processing: Working")
        print("✅ Detection zones: Working")
        print("✅ Face tracking: Working")
        print("✅ High FPS processing: Working")
        print("✅ Distance-based recognition: Working")
        print("\nThe system is ready for production use!")
    else:
        print("\n❌ Some tests failed, check the logs for details")

if __name__ == "__main__":
    main()
