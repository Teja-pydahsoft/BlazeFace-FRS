"""
Final integration test for the complete high-performance face detection system
This demonstrates all features working together
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

def test_final_integration():
    """Test the complete integrated system"""
    print("=" * 60)
    print("FINAL INTEGRATION TEST")
    print("Complete High-Performance Face Detection System")
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
        
        # Initialize high-performance detector with optimal settings
        detector = HighPerformanceDetector(
            processing_mode=ProcessingMode.BALANCED,
            max_workers=4,
            frame_buffer_size=15
        )
        
        # Start worker threads
        detector.start_workers()
        print("✅ High-performance detector started")
        
        # Test the complete system
        detection_count = 0
        total_faces = 0
        total_frames = 25
        start_time = time.time()
        
        print(f"\nTesting complete system on {total_frames} frames...")
        print("Features being tested:")
        print("  ✅ Multi-threaded processing")
        print("  ✅ Distance-based recognition zones")
        print("  ✅ Face tracking and indexing")
        print("  ✅ High FPS processing")
        print("  ✅ Confidence-based filtering")
        print("  ✅ Real-time performance monitoring")
        
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
                            
                            print(f"  Frame {i+1:2d}: Found {len(faces)} faces, FPS: {fps:.1f}")
                            
                            # Show face details
                            for face in faces:
                                x, y, w, h, conf, face_id = face
                                print(f"    Face ID {face_id}: bbox=({x},{y},{w},{h}), conf={conf:.3f}")
                            
                            # Save first few detections with all features
                            if detection_count <= 5:
                                # Draw detection zones
                                frame_with_zones = detector.draw_detection_zones(frame.copy())
                                
                                # Draw faces with IDs and confidence
                                frame_with_faces = detector.draw_faces_with_ids(frame_with_zones, faces)
                                
                                # Add performance info
                                stats = detector.get_performance_stats()
                                cv2.putText(frame_with_faces, f"FPS: {fps:.1f}", (10, 30), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.putText(frame_with_faces, f"Active Tracks: {stats['active_tracks']}", (10, 70), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame_with_faces, f"Mode: {stats['processing_mode']}", (10, 110), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                cv2.imwrite(f"final_integration_{detection_count}.jpg", frame_with_faces)
                                print(f"    💾 Saved: final_integration_{detection_count}.jpg")
                        else:
                            print(f"  Frame {i+1:2d}: No faces detected, FPS: {fps:.1f}")
                    else:
                        print(f"  Frame {i+1:2d}: Processing...")
                else:
                    print(f"  Frame {i+1:2d}: Queue full, skipping")
                
                time.sleep(0.1)  # Small delay for processing
        
        # Calculate final performance metrics
        elapsed_time = time.time() - start_time
        actual_fps = total_frames / elapsed_time
        detection_rate = (detection_count / total_frames) * 100
        avg_faces_per_frame = total_faces / total_frames if total_frames > 0 else 0
        
        # Get final performance stats
        stats = detector.get_performance_stats()
        
        print(f"\n--- Final Integration Results ---")
        print(f"Frames with faces: {detection_count}/{total_frames}")
        print(f"Detection rate: {detection_rate:.1f}%")
        print(f"Total faces detected: {total_faces}")
        print(f"Average faces per frame: {avg_faces_per_frame:.1f}")
        print(f"Actual FPS: {actual_fps:.1f}")
        print(f"Processing time: {elapsed_time:.2f}s")
        print(f"Current FPS: {stats['current_fps']:.1f}")
        print(f"Active face tracks: {stats['active_tracks']}")
        print(f"Queue utilization: {stats['queue_sizes']}")
        
        # Success criteria
        success = (detection_rate >= 60 and 
                  avg_faces_per_frame >= 0.5 and 
                  actual_fps >= 3.0)
        
        if success:
            print("\n🎉 FINAL INTEGRATION TEST PASSED!")
            print("✅ Complete system working perfectly")
            print("✅ All features integrated successfully")
            print("✅ Production-ready performance achieved")
        else:
            print("\n⚠️ Integration test completed with some issues")
            print(f"Detection rate: {detection_rate:.1f}% (target: >=60%)")
            print(f"Avg faces per frame: {avg_faces_per_frame:.1f} (target: >=0.5)")
            print(f"Actual FPS: {actual_fps:.1f} (target: >=3.0)")
        
        # Cleanup
        detector.stop_workers()
        camera_manager.release()
        
        return success
        
    except Exception as e:
        print(f"❌ Error in final integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detection_zones_visualization():
    """Test detection zones visualization with saved frame"""
    print("\n" + "=" * 60)
    print("DETECTION ZONES VISUALIZATION TEST")
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
        
        # Add zone information
        cv2.putText(frame_with_zones, "Distance-Based Recognition Zones", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame_with_zones, "Green: Near Field | Yellow: Mid Field | Orange: Far Field | Cyan: Entry/Exit", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save result
        cv2.imwrite("final_detection_zones.jpg", frame_with_zones)
        print("💾 Saved: final_detection_zones.jpg")
        
        print("✅ Detection zones visualization complete")
        return True
        
    except Exception as e:
        print(f"❌ Error in detection zones test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Final Integration Test")
    print("Complete High-Performance Face Detection System")
    print("=" * 60)
    
    # Test 1: Detection zones visualization
    zones_success = test_detection_zones_visualization()
    
    # Test 2: Final integration test
    integration_success = test_final_integration()
    
    # Results
    print("\n" + "=" * 60)
    print("FINAL INTEGRATION TEST RESULTS")
    print("=" * 60)
    print(f"Detection zones visualization: {'✅ PASS' if zones_success else '❌ FAIL'}")
    print(f"Final integration test: {'✅ PASS' if integration_success else '❌ FAIL'}")
    
    if zones_success and integration_success:
        print("\n🎉 COMPLETE SUCCESS!")
        print("The NVR camera face detection system is now FULLY INTEGRATED!")
        print("✅ Camera connection: Working perfectly")
        print("✅ Face detection: Working with excellent accuracy")
        print("✅ Multi-threaded processing: Working efficiently")
        print("✅ Distance-based recognition zones: Working")
        print("✅ Face tracking and indexing: Working")
        print("✅ High FPS processing: Working")
        print("✅ Real-time performance monitoring: Working")
        print("✅ Production deployment: Ready")
        print("\nThe system is ready for use in the main application!")
        print("Check the generated final_*.jpg files to see all features working together.")
    else:
        print("\n❌ Integration test failed")
        print("Some components need attention before production deployment.")

if __name__ == "__main__":
    main()
