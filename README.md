# BlazeFace-FRS System

A high-performance dual detection system combining BlazeFace for face detection and FaceNet for face recognition, with simultaneous human detection capabilities.

## Features

### Core Detection Capabilities
- **BlazeFace Detection**: Ultra-fast face detection using MediaPipe's BlazeFace model
- **FaceNet Embeddings**: Efficient face recognition using FaceNet embeddings
- **Human Detection**: Real-time human detection using MobileNet SSD
- **Dual Pipeline**: Simultaneous face and human detection with separate processing threads

### System Features
- **Real-time Processing**: Optimized for real-time detection and recognition
- **Multi-threaded Architecture**: Separate pipelines for face and human detection
- **Database Integration**: SQLite database for student management and attendance tracking
- **Modern UI**: Tkinter-based graphical interface with live camera feed
- **Configurable Settings**: Adjustable confidence thresholds and detection parameters

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- Windows 10/11 (tested on Windows)

### Setup Instructions

1. **Clone or Download** the project to your desired location

2. **Navigate** to the project directory:
   ```bash
   cd BlazeFace-FRS
   ```

3. **Create Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**:
   ```bash
   python main.py
   ```

## Project Structure

```
BlazeFace-FRS/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── app/
│   ├── __init__.py
│   ├── config.json        # Application configuration
│   ├── constants.py       # System constants
│   ├── core/              # Core detection modules
│   │   ├── blazeface_detector.py    # BlazeFace face detection
│   │   ├── facenet_embedder.py      # FaceNet embeddings
│   │   ├── human_detector.py        # Human detection
│   │   ├── dual_pipeline.py         # Dual pipeline coordinator
│   │   └── database.py              # Database management
│   ├── ui/                # User interface components
│   │   └── main_dashboard.py        # Main application interface
│   └── utils/             # Utility modules
│       └── camera_utils.py         # Camera management
├── database/              # SQLite database files
├── face_data/            # Face encodings and templates
├── models/               # Pre-trained model files
├── assets/               # Application assets
└── logs/                 # Application logs
```

## Usage

### Starting the Application

1. **Launch** the application using `python main.py`
2. **Select Camera** from the control panel if multiple cameras are available
3. **Adjust Settings** such as confidence threshold and detection type
4. **Start Detection** by clicking the "Start Detection" button

### Detection Modes

- **Face Only**: Detects and recognizes faces only
- **Human Only**: Detects human presence only
- **Both**: Simultaneous face and human detection

### Configuration

Edit `app/config.json` to modify:
- Camera settings
- Detection confidence thresholds
- Model paths
- Database settings
- UI preferences

## Technical Details

### BlazeFace Detection
- Uses MediaPipe's BlazeFace model for ultra-fast face detection
- Optimized for mobile and edge devices
- Supports both short-range and full-range detection models

### FaceNet Embeddings
- 128-dimensional face embeddings
- Cosine similarity for face matching
- Configurable similarity thresholds

### Human Detection
- MobileNet SSD-based human detection
- Real-time processing with OpenCV DNN
- Fallback to HOG descriptor if model unavailable

### Dual Pipeline Architecture
- Separate threads for face and human detection
- Queue-based communication between threads
- Synchronized result aggregation
- Configurable processing parameters

## Database Schema

### Students Table
- Student ID, name, email, phone
- Department and year information
- Creation and update timestamps

### Face Encodings Table
- Student ID references
- Binary face encoding data
- Encoding type specification

### Attendance Table
- Student attendance records
- Date, time, and status information
- Detection confidence scores
- Detection type and notes

### Detection Logs Table
- System performance metrics
- Detection statistics
- Processing time measurements

## Performance Optimization

### Threading
- Separate detection pipelines
- Non-blocking UI updates
- Queue-based frame processing

### Memory Management
- Efficient frame handling
- Automatic resource cleanup
- Configurable buffer sizes

### Camera Optimization
- Reduced buffer size for lower latency
- Configurable resolution and FPS
- Automatic camera detection and switching

## Troubleshooting

### Common Issues

1. **Camera Not Detected**
   - Check camera permissions
   - Verify camera is not used by another application
   - Try different camera indices

2. **Detection Not Working**
   - Ensure good lighting conditions
   - Check confidence threshold settings
   - Verify model files are available

3. **Performance Issues**
   - Reduce camera resolution
   - Lower confidence thresholds
   - Close other applications

### Logs
- Check `blazeface_frs.log` for detailed error information
- Enable debug logging in configuration for troubleshooting

## Dependencies

- **OpenCV**: Computer vision operations
- **MediaPipe**: BlazeFace face detection
- **TensorFlow**: FaceNet model support
- **NumPy**: Numerical operations
- **Pillow**: Image processing
- **Tkinter**: GUI framework (included with Python)

## License

This project is developed for educational and research purposes.

## Support

For technical support or questions, please refer to the project documentation or create an issue in the project repository.

## Version History

- **v1.0**: Initial release with BlazeFace and FaceNet integration
- Dual pipeline architecture
- Real-time detection capabilities
- Database integration
- Modern UI interface
