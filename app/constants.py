# Constants for BlazeFace-FRS system

# Detection types
DETECTION_TYPES = {
    'FACE': 'face',
    'HUMAN': 'human',
    'BOTH': 'both'
}

# Pipeline status
PIPELINE_STATUS = {
    'IDLE': 'idle',
    'RUNNING': 'running',
    'STOPPED': 'stopped',
    'ERROR': 'error'
}

# Face detection models
FACE_MODELS = {
    'BLAZEFACE': 'blazeface',
    'OPENCV': 'opencv',
    'MTCNN': 'mtcnn'
}

# Human detection models
HUMAN_MODELS = {
    'MOBILENET_SSD': 'mobilenet_ssd',
    'YOLO': 'yolo',
    'OPENCV_DNN': 'opencv_dnn'
}

# Embedding models
EMBEDDING_MODELS = {
    'FACENET': 'facenet',
    'ARCFACE': 'arcface',
    'VGG_FACE': 'vgg_face'
}

# UI Constants
UI_CONSTANTS = {
    'WINDOW_WIDTH': 1200,
    'WINDOW_HEIGHT': 800,
    'CAMERA_WIDTH': 640,
    'CAMERA_HEIGHT': 480,
    'FACE_BOX_COLOR': (0, 255, 0),
    'HUMAN_BOX_COLOR': (255, 0, 0),
    'TEXT_COLOR': (255, 255, 255),
    'FONT_SCALE': 0.7,
    'THICKNESS': 2
}

# Database tables
DB_TABLES = {
    'STUDENTS': 'students',
    'ATTENDANCE': 'attendance',
    'FACE_ENCODINGS': 'face_encodings',
    'DETECTION_LOGS': 'detection_logs'
}
