# BlazeFace-FRS Face Recognition System Explanation

## **How Face Recognition Works**

### **1. Face Detection Process**
```
Camera Feed → BlazeFace Detector → Face Bounding Boxes → Face Extraction
```

- **BlazeFace (MediaPipe)**: Detects faces in real-time camera feed
- **Confidence Threshold**: Only processes faces with confidence > 0.5
- **Face Extraction**: Crops detected face region from the image

### **2. Face Encoding Process**
```
Face Image → FaceNet Model → 128-Dimensional Vector → Normalization
```

- **FaceNet Model**: Converts face image into a 128-dimensional numerical vector
- **Normalization**: Ensures all vectors have unit length (norm = 1.0)
- **Unique Representation**: Each person's face creates a unique "fingerprint"

### **3. Face Recognition Process**
```
New Face → Encoding → Compare with Database → Cosine Similarity → Match/No Match
```

- **Cosine Similarity**: Measures angle between vectors (0.0 = different, 1.0 = identical)
- **Recognition Threshold**: 0.85 (85% similarity required for match)
- **Gap Requirement**: 0.05 minimum difference between best and second-best match

## **Database Storage**

### **What's Stored**
1. **Face Encodings**: 128-dimensional numerical vectors (BLOB in database)
2. **Face Images**: Actual face photos saved in `face_data/` folder
3. **Student Information**: ID, name, email, phone, department, year
4. **Image Paths**: Links between encodings and saved images

### **Database Schema**
```sql
-- Students table
CREATE TABLE students (
    id INTEGER PRIMARY KEY,
    student_id TEXT UNIQUE,
    name TEXT,
    email TEXT,
    phone TEXT,
    department TEXT,
    year TEXT,
    created_at TIMESTAMP
);

-- Face encodings table
CREATE TABLE face_encodings (
    id INTEGER PRIMARY KEY,
    student_id TEXT,
    encoding BLOB,           -- 128-dimensional vector
    encoding_type TEXT,      -- 'facenet'
    image_path TEXT,         -- Path to saved face image
    created_at TIMESTAMP
);
```

## **Quality Control System**

### **Registration Quality Checks**
1. **Encoding Quality**: Validates numerical properties of face vectors
2. **Cross-Person Similarity**: Prevents registration of faces too similar to existing people
3. **Internal Consistency**: Ensures multiple encodings of same person are reasonably similar
4. **Maximum Limit**: 3 encodings per person to prevent database bloat

### **Recognition Quality Checks**
1. **High Threshold**: 0.85 minimum similarity for recognition
2. **Gap Requirement**: Clear difference between best and second-best matches
3. **Special Pairs**: Handles known problematic face pairs
4. **Consistency Checks**: Prevents rapid switching between similar people

## **File Structure**

```
BlazeFace-FRS/
├── face_data/              # Face images (studentID_timestamp.jpg)
├── database/
│   └── blazeface_frs.db   # SQLite database with encodings
├── models/                 # AI model files
├── app/
│   ├── core/              # Core detection and processing
│   ├── ui/                # User interface components
│   └── utils/             # Utility functions
└── logs/                  # System logs
```

## **Recognition Accuracy**

### **Similarity Scores**
- **0.95-1.00**: Very high similarity (likely same person)
- **0.85-0.94**: High similarity (good match)
- **0.70-0.84**: Moderate similarity (uncertain)
- **0.50-0.69**: Low similarity (likely different person)
- **0.00-0.49**: Very low similarity (different person)

### **System Behavior**
- **Above 0.85**: Recognizes as registered student
- **Below 0.85**: Shows as "Unknown Face"
- **Gap < 0.05**: Rejects match (too ambiguous)

## **Performance Optimizations**

### **Database Optimizations**
- **Limited Encodings**: Maximum 3 per person
- **Quality Selection**: Keeps only best representative encodings
- **Indexed Queries**: Fast student lookups

### **Recognition Optimizations**
- **Frame Skipping**: Processes every 3rd frame for performance
- **Confidence Filtering**: Only processes high-confidence detections
- **Caching**: Stores student names and encodings in memory

## **Troubleshooting**

### **Common Issues**
1. **"No such column: image_path"**: Database needs migration (fixed)
2. **Low Recognition**: Poor lighting, angle, or face positioning
3. **False Matches**: Similar-looking people need re-registration
4. **No Detection**: Camera issues or face not clearly visible

### **Best Practices**
1. **Good Lighting**: Ensure face is well-lit
2. **Clear Positioning**: Face centered and clearly visible
3. **Multiple Angles**: Capture 2-3 different angles per person
4. **Quality Check**: System validates each capture before saving

## **Security Features**

### **Data Protection**
- **Local Storage**: All data stored locally, no cloud uploads
- **Image Encryption**: Face images stored as standard JPEG files
- **Database Security**: SQLite with parameterized queries

### **Privacy Controls**
- **Student Consent**: Clear indication when face is being captured
- **Data Deletion**: Students can be removed from system
- **Access Control**: Only authorized users can access the system

## **System Status**

✅ **Face Detection**: Working with BlazeFace/MediaPipe  
✅ **Face Encoding**: Working with FaceNet model  
✅ **Database Storage**: Working with SQLite  
✅ **Quality Validation**: Active and functional  
✅ **Image Storage**: Now saving to face_data folder  
✅ **Recognition Logic**: Improved with better thresholds  

The system is now fully functional with proper face image storage and enhanced recognition accuracy!
