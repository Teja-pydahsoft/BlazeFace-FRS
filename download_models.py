"""
Model Download Script for BlazeFace-FRS
Downloads required model files for face recognition system
"""

import os
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_models_directory():
    """Create models directory if it doesn't exist"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    logger.info(f"Models directory: {models_dir.absolute()}")
    return models_dir

def download_file(url: str, filepath: Path, description: str = ""):
    """Download a file from URL"""
    try:
        if filepath.exists():
            logger.info(f"✓ {description} already exists: {filepath}")
            return True
        
        logger.info(f"Downloading {description}...")
        logger.info(f"URL: {url}")
        logger.info(f"Destination: {filepath}")
        
        # Create parent directories
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                percent = min(100, (downloaded * 100) // total_size)
                print(f"\rProgress: {percent}% ({downloaded}/{total_size} bytes)", end="")
        
        urllib.request.urlretrieve(url, filepath, show_progress)
        print()  # New line after progress
        
        if filepath.exists() and filepath.stat().st_size > 0:
            logger.info(f"✓ Successfully downloaded {description}")
            return True
        else:
            logger.error(f"✗ Failed to download {description} - file is empty or doesn't exist")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error downloading {description}: {e}")
        return False

def download_face_recognition_models():
    """Download face_recognition models (dlib models)"""
    logger.info("=== Downloading face_recognition models ===")
    
    models_dir = create_models_directory()
    
    # face_recognition uses dlib models which are downloaded automatically
    # But we can create a placeholder to indicate they're available
    face_recognition_info = models_dir / "face_recognition_models.txt"
    
    if not face_recognition_info.exists():
        with open(face_recognition_info, 'w') as f:
            f.write("face_recognition models are downloaded automatically by the library\n")
            f.write("Models used:\n")
            f.write("- dlib's ResNet model for face encodings\n")
            f.write("- dlib's HOG face detector\n")
            f.write("- dlib's 68-point facial landmark predictor\n")
        
        logger.info("✓ face_recognition models info created")
    
    return True

def download_insightface_models():
    """Download InsightFace models"""
    logger.info("=== Downloading InsightFace models ===")
    
    models_dir = create_models_directory()
    
    # InsightFace models are downloaded automatically when first used
    # Create info file
    insightface_info = models_dir / "insightface_models.txt"
    
    if not insightface_info.exists():
        with open(insightface_info, 'w') as f:
            f.write("InsightFace models are downloaded automatically by the library\n")
            f.write("Available models:\n")
            f.write("- buffalo_l: Large model (best accuracy)\n")
            f.write("- buffalo_m: Medium model (balanced)\n")
            f.write("- buffalo_s: Small model (fastest)\n")
            f.write("\nModels will be downloaded to ~/.insightface/ on first use\n")
        
        logger.info("✓ InsightFace models info created")
    
    return True

def download_blazeface_model():
    """Download BlazeFace model"""
    logger.info("=== Downloading BlazeFace model ===")
    
    models_dir = create_models_directory()
    blazeface_path = models_dir / "blazeface_640x640.tflite"
    
    # BlazeFace model URL (MediaPipe's BlazeFace model)
    blazeface_url = "https://storage.googleapis.com/mediapipe-models/face_detection/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    
    # Try primary URL first
    success = download_file(blazeface_url, blazeface_path, "BlazeFace model (640x640)")
    
    if not success:
        # Try alternative URL
        blazeface_url_alt = "https://github.com/google/mediapipe/raw/master/mediapipe/modules/face_detection/face_detection_short_range.tflite"
        success = download_file(blazeface_url_alt, blazeface_path, "BlazeFace model (alternative)")
    
    return success

def download_facenet_model():
    """Download FaceNet model"""
    logger.info("=== Downloading FaceNet model ===")
    
    models_dir = create_models_directory()
    facenet_path = models_dir / "facenet_keras.h5"
    
    # Note: This is a placeholder URL - you'll need to find a proper FaceNet model
    # For now, we'll create a note about this
    facenet_note = models_dir / "facenet_model_note.txt"
    
    if not facenet_note.exists():
        with open(facenet_note, 'w') as f:
            f.write("FaceNet Model Note:\n")
            f.write("==================\n\n")
            f.write("The facenet_keras.h5 model needs to be obtained separately.\n")
            f.write("Options:\n")
            f.write("1. Use face_recognition library (recommended) - no separate model needed\n")
            f.write("2. Use InsightFace library (recommended) - no separate model needed\n")
            f.write("3. Download a pre-trained FaceNet model from:\n")
            f.write("   - https://github.com/davidsandberg/facenet\n")
            f.write("   - https://www.kaggle.com/datasets/keras/facenet-keras\n")
            f.write("\nNote: The new StandardFaceEmbedder and InsightFaceEmbedder\n")
            f.write("don't require this model file.\n")
        
        logger.info("✓ FaceNet model note created")
    
    return True

def download_mobilenet_ssd_model():
    """Download MobileNet SSD model for human detection"""
    logger.info("=== Downloading MobileNet SSD model ===")
    
    models_dir = create_models_directory()
    mobilenet_path = models_dir / "mobilenet_ssd.tflite"
    
    # MobileNet SSD model URL (TensorFlow Lite model zoo)
    mobilenet_url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
    
    # Download and extract
    zip_path = models_dir / "mobilenet_ssd.zip"
    
    success = download_file(
        mobilenet_url, 
        zip_path, 
        "MobileNet SSD model (zip)"
    )
    
    if success and zip_path.exists():
        try:
            logger.info("Extracting MobileNet SSD model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract the .tflite file
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.tflite'):
                        zip_ref.extract(file_info, models_dir)
                        extracted_file = models_dir / file_info.filename
                        if extracted_file.exists():
                            # Rename to standard name
                            extracted_file.rename(mobilenet_path)
                            logger.info(f"✓ Extracted and renamed to {mobilenet_path}")
                        break
            
            # Clean up zip file
            zip_path.unlink()
            logger.info("✓ MobileNet SSD model downloaded and extracted")
            return True
            
        except Exception as e:
            logger.error(f"✗ Error extracting MobileNet SSD model: {e}")
            return False
    
    return False

def create_model_info_file():
    """Create a comprehensive model information file"""
    models_dir = create_models_directory()
    info_file = models_dir / "MODEL_INFO.md"
    
    with open(info_file, 'w') as f:
        f.write("# BlazeFace-FRS Model Information\n\n")
        f.write("This directory contains the model files used by the BlazeFace-FRS system.\n\n")
        f.write("## Model Files\n\n")
        f.write("### Face Recognition Models\n")
        f.write("- **face_recognition**: Uses dlib's ResNet model (downloaded automatically)\n")
        f.write("- **insightface**: Uses InsightFace models (downloaded automatically)\n")
        f.write("- **facenet_keras.h5**: Custom FaceNet model (optional, see note)\n\n")
        f.write("### Face Detection Models\n")
        f.write("- **blazeface_640x640.tflite**: BlazeFace face detection model\n")
        f.write("- **mobilenet_ssd.tflite**: MobileNet SSD human detection model\n\n")
        f.write("## Usage\n\n")
        f.write("The system will automatically use the best available models:\n")
        f.write("1. **Primary**: face_recognition library (industry standard)\n")
        f.write("2. **Secondary**: InsightFace library (high accuracy)\n")
        f.write("3. **Fallback**: Custom embedders (if needed)\n\n")
        f.write("## Installation\n\n")
        f.write("Run this script to download required models:\n")
        f.write("```bash\n")
        f.write("python download_models.py\n")
        f.write("```\n")
    
    logger.info("✓ Model information file created")

def main():
    """Main function to download all models"""
    logger.info("Starting model download process...")
    logger.info("=" * 50)
    
    # Create models directory
    models_dir = create_models_directory()
    
    # Download models
    results = []
    
    # Face recognition models (automatic download)
    results.append(("face_recognition", download_face_recognition_models()))
    results.append(("insightface", download_insightface_models()))
    
    # Detection models
    results.append(("blazeface", download_blazeface_model()))
    results.append(("mobilenet_ssd", download_mobilenet_ssd_model()))
    
    # FaceNet model (note only)
    results.append(("facenet", download_facenet_model()))
    
    # Create info file
    create_model_info_file()
    
    # Summary
    logger.info("=" * 50)
    logger.info("Download Summary:")
    logger.info("=" * 50)
    
    for model_name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"{model_name:20} {status}")
    
    # Check what we have
    logger.info("\nFiles in models directory:")
    for file_path in models_dir.iterdir():
        if file_path.is_file():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_path.name:30} ({size_mb:.1f} MB)")
    
    logger.info("\nModel download process completed!")
    logger.info("You can now run the BlazeFace-FRS system with the new embedders.")
    
    # Check if running in virtual environment
    import sys
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        logger.info("\n✓ You are running in a virtual environment")
        logger.info("✓ Ready to install dependencies: pip install -r requirements.txt")
    else:
        logger.info("\n⚠ You are NOT running in a virtual environment")
        logger.info("⚠ Consider activating the project's virtual environment:")
        logger.info("   Windows: venv\\Scripts\\activate")
        logger.info("   Linux/Mac: source venv/bin/activate")

if __name__ == "__main__":
    main()
