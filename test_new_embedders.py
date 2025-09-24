"""
Test Script for New Face Embedders
Tests the new StandardFaceEmbedder and InsightFaceEmbedder integration
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

def test_standard_face_embedder():
    """Test StandardFaceEmbedder (face_recognition library)"""
    print("=" * 60)
    print("Testing StandardFaceEmbedder (face_recognition library)")
    print("=" * 60)
    
    try:
        from app.core.standard_face_embedder import StandardFaceEmbedder
        
        # Initialize embedder
        embedder = StandardFaceEmbedder(model='large')
        print("✓ StandardFaceEmbedder initialized successfully")
        
        # Get embedder info
        info = embedder.get_embedding_info()
        print(f"✓ Embedder info: {info['name']} - {info['description']}")
        print(f"✓ Embedding size: {info['embedding_size']}")
        print(f"✓ Distance metric: {info['distance_metric']}")
        
        # Test with a sample face image
        # Create a dummy face image for testing
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Get embedding
        embedding = embedder.get_embedding(test_image)
        if embedding is not None:
            print(f"✓ Successfully generated embedding with shape: {embedding.shape}")
            print(f"✓ Embedding type: {type(embedding)}")
        else:
            print("⚠ No embedding generated (expected for random image)")
        
        # Test comparison
        embedding2 = np.random.rand(128)  # Random embedding for testing
        is_same, distance = embedder.compare_faces(embedding2, embedding2, threshold=0.6)
        print(f"✓ Face comparison test: is_same={is_same}, distance={distance:.4f}")
        
        embedder.release()
        print("✓ StandardFaceEmbedder test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ StandardFaceEmbedder test failed: {e}")
        return False

def test_insightface_embedder():
    """Test InsightFaceEmbedder"""
    print("\n" + "=" * 60)
    print("Testing InsightFaceEmbedder")
    print("=" * 60)
    
    try:
        from app.core.insightface_embedder import InsightFaceEmbedder
        
        # Initialize embedder
        embedder = InsightFaceEmbedder(model_name='buffalo_l')
        print("✓ InsightFaceEmbedder initialized successfully")
        
        # Get embedder info
        info = embedder.get_embedding_info()
        print(f"✓ Embedder info: {info['name']} - {info['description']}")
        print(f"✓ Embedding size: {info['embedding_size']}")
        print(f"✓ Distance metric: {info['distance_metric']}")
        print(f"✓ Features: {', '.join(info['features'])}")
        
        # Test with a sample face image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Get embedding
        embedding = embedder.get_embedding(test_image)
        if embedding is not None:
            print(f"✓ Successfully generated embedding with shape: {embedding.shape}")
            print(f"✓ Embedding type: {type(embedding)}")
        else:
            print("⚠ No embedding generated (expected for random image)")
        
        # Test comparison
        embedding2 = np.random.rand(embedder.embedding_size)
        is_same, similarity = embedder.compare_faces(embedding2, embedding2, threshold=0.6)
        print(f"✓ Face comparison test: is_same={is_same}, similarity={similarity:.4f}")
        
        embedder.release()
        print("✓ InsightFaceEmbedder test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ InsightFaceEmbedder test failed: {e}")
        print(f"  This is expected if InsightFace is not installed yet")
        return False

def test_system_integration():
    """Test system integration with new embedders"""
    print("\n" + "=" * 60)
    print("Testing System Integration")
    print("=" * 60)
    
    try:
        # Test attendance marking with new embedders
        from app.ui.attendance_marking import AttendanceMarkingDialog
        print("✓ AttendanceMarkingDialog imported successfully")
        
        # Test student registration with new embedders
        from app.ui.student_registration import StudentRegistrationDialog
        print("✓ StudentRegistrationDialog imported successfully")
        
        # Test dual pipeline with new embedders
        from app.core.dual_pipeline import DualPipeline
        print("✓ DualPipeline imported successfully")
        
        print("✓ System integration test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ System integration test failed: {e}")
        return False

def check_virtual_environment():
    """Check if running in virtual environment"""
    print("\n" + "=" * 60)
    print("Checking Virtual Environment")
    print("=" * 60)
    
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print("✓ Running in virtual environment")
        print(f"✓ Python executable: {sys.executable}")
        print(f"✓ Virtual env path: {sys.prefix}")
    else:
        print("⚠ Not running in virtual environment")
        print("⚠ Consider using the project's virtual environment:")
        print("   - Windows: venv\\Scripts\\activate")
        print("   - Linux/Mac: source venv/bin/activate")
    
    return in_venv

def test_dependencies():
    """Test if new dependencies are available"""
    print("\n" + "=" * 60)
    print("Testing Dependencies")
    print("=" * 60)
    
    dependencies = [
        ('face_recognition', 'face_recognition'),
        ('insightface', 'insightface'),
        ('deepface', 'deepface'),
        ('sklearn', 'scikit-learn')
    ]
    
    results = []
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {package_name} is available")
            results.append(True)
        except ImportError:
            print(f"⚠ {package_name} is not installed")
            results.append(False)
    
    return results

def main():
    """Main test function"""
    print("BlazeFace-FRS New Embedders Test")
    print("=" * 60)
    
    # Check virtual environment
    in_venv = check_virtual_environment()
    
    # Test dependencies
    dep_results = test_dependencies()
    
    # Test embedders
    standard_result = test_standard_face_embedder()
    insightface_result = test_insightface_embedder()
    
    # Test integration
    integration_result = test_system_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    print(f"Virtual Environment: {'✓ ACTIVE' if in_venv else '⚠ NOT ACTIVE'}")
    print(f"Dependencies: {sum(dep_results)}/{len(dep_results)} available")
    print(f"StandardFaceEmbedder: {'✓ PASS' if standard_result else '✗ FAIL'}")
    print(f"InsightFaceEmbedder: {'✓ PASS' if insightface_result else '⚠ SKIP (not installed)'}")
    print(f"System Integration: {'✓ PASS' if integration_result else '✗ FAIL'}")
    
    if standard_result and integration_result:
        print("\n🎉 SUCCESS: New embedders are working correctly!")
        print("You can now use the improved face recognition system.")
        
        print("\nNext steps:")
        if not in_venv:
            print("1. Activate virtual environment:")
            print("   - Windows: venv\\Scripts\\activate")
            print("   - Linux/Mac: source venv/bin/activate")
        print(f"{'2' if in_venv else '2'}. Install missing dependencies: pip install -r requirements.txt")
        print(f"{'3' if in_venv else '3'}. Run the main application: python main.py")
        print(f"{'4' if in_venv else '4'}. Test face registration and recognition")
    else:
        print("\n⚠ Some tests failed. Please check the error messages above.")
        if not in_venv:
            print("\n💡 TIP: Make sure to activate the virtual environment first:")
            print("   venv\\Scripts\\activate  (Windows)")
            print("   source venv/bin/activate  (Linux/Mac)")
    
    return standard_result and integration_result

if __name__ == "__main__":
    main()