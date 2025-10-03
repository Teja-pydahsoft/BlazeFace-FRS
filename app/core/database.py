"""
Database management for BlazeFace-FRS system
Handles student data, attendance records, and face encodings
"""

import sqlite3
import json
import numpy as np
import io
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path: str = "database/blazeface_frs.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Students table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS students (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        email TEXT,
                        phone TEXT,
                        department TEXT,
                        year TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Face encodings table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS face_encodings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        encoding BLOB NOT NULL,
                        encoding_type TEXT DEFAULT 'facenet',
                        image_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id)
                    )
                """)
                
                # Attendance table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS attendance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        date DATE NOT NULL,
                        time TIME NOT NULL,
                        status TEXT DEFAULT 'present',
                        confidence REAL,
                        detection_type TEXT,
                        notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id)
                    )
                """)
                
                # Recognition logs table for monitoring and analysis
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS recognition_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT,
                        detected_face_id TEXT,
                        confidence_score REAL NOT NULL,
                        embedder_type TEXT NOT NULL,
                        threshold_used REAL NOT NULL,
                        recognition_result TEXT NOT NULL,
                        image_path TEXT,
                        processing_time_ms REAL,
                        face_bbox_x INTEGER,
                        face_bbox_y INTEGER,
                        face_bbox_w INTEGER,
                        face_bbox_h INTEGER,
                        session_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id)
                    )
                """)
                
                # Performance metrics table for threshold tuning
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        embedder_type TEXT NOT NULL,
                        threshold_value REAL NOT NULL,
                        true_positives INTEGER DEFAULT 0,
                        false_positives INTEGER DEFAULT 0,
                        true_negatives INTEGER DEFAULT 0,
                        false_negatives INTEGER DEFAULT 0,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        test_date DATE NOT NULL,
                        dataset_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Manual overrides table for tracking admin interventions
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS manual_overrides (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT NOT NULL,
                        original_result TEXT,
                        override_action TEXT NOT NULL,
                        override_reason TEXT,
                        admin_user TEXT,
                        attendance_record_id INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (student_id) REFERENCES students (student_id),
                        FOREIGN KEY (attendance_record_id) REFERENCES attendance (id)
                    )
                """)
                
                # Detection logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        detection_type TEXT NOT NULL,
                        faces_detected INTEGER DEFAULT 0,
                        humans_detected INTEGER DEFAULT 0,
                        processing_time REAL,
                        frame_width INTEGER,
                        frame_height INTEGER,
                        notes TEXT
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_face_encodings_student_id ON face_encodings(student_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_logs_timestamp ON detection_logs(timestamp)")
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def add_student(self, student_data: Dict[str, Any]) -> bool:
        """
        Add a new student to the database
        
        Args:
            student_data: Dictionary containing student information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if student already exists
                cursor.execute("SELECT id FROM students WHERE student_id = ?", (student_data.get('student_id'),))
                if cursor.fetchone():
                    self.logger.warning(f"Student already exists: {student_data.get('student_id')}")
                    return False

                cursor.execute("""
                    INSERT INTO students (student_id, name, email, phone, department, year)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    student_data.get('student_id'),
                    student_data.get('name'),
                    student_data.get('email'),
                    student_data.get('phone'),
                    student_data.get('department'),
                    student_data.get('year')
                ))
                
                conn.commit()
                self.logger.info(f"Student added: {student_data.get('student_id')}")
                return True
                
        except sqlite3.IntegrityError:
            self.logger.warning(f"Student already exists: {student_data.get('student_id')}")
            return False
        except Exception as e:
            self.logger.error(f"Error adding student: {str(e)}")
            return False
    
    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """
        Get student information by student ID
        
        Args:
            student_id: Student ID to search for
            
        Returns:
            Student data dictionary or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM students WHERE student_id = ?
                """, (student_id,))
                
                row = cursor.fetchone()
                if row:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, row))
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting student: {str(e)}")
            return None
    
    def get_all_students(self) -> List[Dict[str, Any]]:
        """
        Get all students from the database
        
        Returns:
            List of student data dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT * FROM students ORDER BY name")
                rows = cursor.fetchall()
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting all students: {str(e)}")
            return []

    def add_face_encodings(self, student_id: str, encodings: List[np.ndarray], 
                           encoding_type: str = 'simple', image_paths: List[str] = None) -> bool:
        """
        Add face encodings for a student. Handles both single and multiple encodings.

        Args:
            student_id: Student ID
            encodings: A list of face encoding arrays
            encoding_type: Type of encoding (default: 'simple')
            image_paths: A list of paths to the face images (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                for i, encoding in enumerate(encodings):
                    # Ensure encoding is a numpy array with dtype float32
                    arr = np.asarray(encoding)
                    if arr.dtype != np.float32:
                        arr = arr.astype(np.float32)

                    # Save array to bytes using numpy's binary format for safe roundtrip
                    buf = io.BytesIO()
                    # allow_pickle=False for safety
                    np.save(buf, arr, allow_pickle=False)
                    encoding_bytes = buf.getvalue()

                    image_path = image_paths[i] if image_paths and i < len(image_paths) else None
                    
                    cursor.execute("""
                        INSERT INTO face_encodings (student_id, encoding, encoding_type, image_path)
                        VALUES (?, ?, ?, ?)
                    """, (student_id, encoding_bytes, encoding_type, image_path))
                
                conn.commit()
                self.logger.info(f"{len(encodings)} face encoding(s) added for student: {student_id}")
                return True

        except Exception as e:
            self.logger.error(f"Error adding face encodings: {str(e)}")
            return False

    def add_face_encoding(self, student_id: str, encoding: np.ndarray, 
                         encoding_type: str = 'simple', image_path: str = None) -> bool:
        """
        [DEPRECATED] Add a single face encoding for a student. 
        Use add_face_encodings instead.
        """
        self.logger.warning("The 'add_face_encoding' method is deprecated. Use 'add_face_encodings' instead.")
        return self.add_face_encodings(student_id, [encoding], encoding_type, [image_path])

    def get_face_encodings(self, student_id: str = None) -> List[Tuple[str, np.ndarray, str, str]]:
        """
        Get face encodings from the database
        
        Args:
            student_id: Specific student ID (optional)
            
        Returns:
            List of tuples (student_id, encoding, encoding_type, image_path)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if student_id:
                    cursor.execute("""
                        SELECT student_id, encoding, encoding_type, image_path 
                        FROM face_encodings WHERE student_id = ?
                    """, (student_id,))
                else:
                    cursor.execute("""
                        SELECT student_id, encoding, encoding_type, image_path 
                        FROM face_encodings
                    """)
                
                rows = cursor.fetchall()
                encodings = []
                
                for row in rows:
                    student_id, encoding_bytes, encoding_type, image_path = row
                    try:
                        # Try to load using numpy binary format (new format)
                        buf = io.BytesIO(encoding_bytes)
                        arr = np.load(buf, allow_pickle=False)
                    except Exception:
                        # Fallback: older format may be raw bytes of float64; try to interpret
                        try:
                            arr = np.frombuffer(encoding_bytes, dtype=np.float64)
                            # convert to float32 for consistency
                            arr = arr.astype(np.float32)
                        except Exception as e:
                            self.logger.error(f"Failed to decode encoding for student {student_id}: {e}")
                            continue

                    encodings.append((student_id, arr, encoding_type, image_path))
                
                return encodings

        except Exception as e:
            self.logger.error(f"Error getting face encodings: {str(e)}")
            return []
    
    def get_encoding_types(self) -> List[str]:
        """
        Get all distinct encoding types from the database
        
        Returns:
            List of encoding types used in the database
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT encoding_type FROM face_encodings")
                rows = cursor.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            self.logger.error(f"Error getting encoding types: {str(e)}")
            return []
    
    def add_attendance_record(self, student_id: str, status: str = 'present',
                            confidence: float = None, detection_type: str = None,
                            notes: str = None) -> bool:
        """
        Add attendance record
        
        Args:
            student_id: Student ID
            status: Attendance status ('present', 'absent', 'late')
            confidence: Recognition confidence score (face embedding similarity for face_recognition type)
            detection_type: Type of detection used ('face_recognition', 'manual', etc.)
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                now = datetime.now()
                date_str = now.date().strftime('%Y-%m-%d')
                time_str = now.time().strftime('%H:%M:%S')
                
                cursor.execute("""
                    INSERT INTO attendance (student_id, date, time, status, confidence, detection_type, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (student_id, date_str, time_str, status, confidence, detection_type, notes))
                
                conn.commit()
                self.logger.info(f"Attendance record added for student: {student_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding attendance record: {str(e)}")
            return False
    
    def get_attendance_records(self, student_id: str = None, 
                             date_from: str = None, date_to: str = None) -> List[Dict[str, Any]]:
        """
        Get attendance records
        
        Args:
            student_id: Specific student ID (optional)
            date_from: Start date (YYYY-MM-DD format)
            date_to: End date (YYYY-MM-DD format)
            
        Returns:
            List of attendance record dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT a.*, s.name, s.department 
                    FROM attendance a
                    JOIN students s ON a.student_id = s.student_id
                    WHERE 1=1
                """
                params = []
                
                if student_id:
                    query += " AND a.student_id = ?"
                    params.append(student_id)
                
                if date_from:
                    query += " AND a.date >= ?"
                    params.append(date_from)
                
                if date_to:
                    query += " AND a.date <= ?"
                    params.append(date_to)
                
                query += " ORDER BY a.date DESC, a.time DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting attendance records: {str(e)}")
            return []
    
    def add_detection_log(self, detection_type: str, faces_detected: int = 0,
                         humans_detected: int = 0, processing_time: float = None,
                         frame_width: int = None, frame_height: int = None,
                         notes: str = None) -> bool:
        """
        Add detection log entry
        
        Args:
            detection_type: Type of detection performed
            faces_detected: Number of faces detected
            humans_detected: Number of humans detected
            processing_time: Processing time in seconds
            frame_width: Frame width
            frame_height: Frame height
            notes: Additional notes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO detection_logs (detection_type, faces_detected, humans_detected,
                                              processing_time, frame_width, frame_height, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (detection_type, faces_detected, humans_detected, processing_time,
                     frame_width, frame_height, notes))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding detection log: {str(e)}")
            return False
    
    def get_detection_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent detection logs
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of detection log dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM detection_logs 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting detection logs: {str(e)}")
            return []
    
    def update_student(self, student_id: str, update_data: Dict[str, Any]) -> bool:
        """
        Update student information
        
        Args:
            student_id: Student ID to update
            update_data: Dictionary containing fields to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build update query dynamically
                set_clauses = []
                params = []
                
                for field, value in update_data.items():
                    if field != 'student_id':  # Don't allow updating student_id
                        set_clauses.append(f"{field} = ?")
                        params.append(value)
                
                if not set_clauses:
                    return False
                
                set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                params.append(student_id)
                
                query = f"UPDATE students SET {', '.join(set_clauses)} WHERE student_id = ?"
                
                cursor.execute(query, params)
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Student updated: {student_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating student: {str(e)}")
            return False
    
    def delete_student(self, student_id: str) -> bool:
        """
        Delete student and all related data
        
        Args:
            student_id: Student ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete related records first
                cursor.execute("DELETE FROM face_encodings WHERE student_id = ?", (student_id,))
                cursor.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
                cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
                
                conn.commit()
                
                if cursor.rowcount > 0:
                    self.logger.info(f"Student deleted: {student_id}")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting student: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary containing various statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Student count
                cursor.execute("SELECT COUNT(*) FROM students")
                stats['total_students'] = cursor.fetchone()[0]
                
                # Face encodings count
                cursor.execute("SELECT COUNT(*) FROM face_encodings")
                stats['total_face_encodings'] = cursor.fetchone()[0]
                
                # Attendance records count
                cursor.execute("SELECT COUNT(*) FROM attendance")
                stats['total_attendance_records'] = cursor.fetchone()[0]
                
                # Today's attendance
                cursor.execute("""
                    SELECT COUNT(*) FROM attendance 
                    WHERE date = DATE('now')
                """)
                stats['today_attendance'] = cursor.fetchone()[0]
                
                # Detection logs count
                cursor.execute("SELECT COUNT(*) FROM detection_logs")
                stats['total_detection_logs'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting statistics: {str(e)}")
            return {}
    
    def delete_attendance_records(self, date_from: str = None, date_to: str = None, 
                                 student_id: str = None) -> bool:
        """
        Delete attendance records based on criteria
        
        Args:
            date_from: Start date (YYYY-MM-DD format)
            date_to: End date (YYYY-MM-DD format)
            student_id: Specific student ID (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "DELETE FROM attendance WHERE 1=1"
                params = []
                
                if student_id:
                    query += " AND student_id = ?"
                    params.append(student_id)
                
                if date_from:
                    query += " AND date >= ?"
                    params.append(date_from)
                
                if date_to:
                    query += " AND date <= ?"
                    params.append(date_to)
                
                cursor.execute(query, params)
                deleted_count = cursor.rowcount
                
                conn.commit()
                
                if deleted_count > 0:
                    self.logger.info(f"Deleted {deleted_count} attendance records")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting attendance records: {str(e)}")
            return False
    
    def log_recognition_result(self, student_id: str, detected_face_id: str, 
                             confidence_score: float, embedder_type: str, 
                             threshold_used: float, recognition_result: str,
                             image_path: str = None, processing_time_ms: float = None,
                             face_bbox: tuple = None, session_id: str = None) -> bool:
        """
        Log recognition result for monitoring and analysis
        
        Args:
            student_id: Student ID (None if unknown)
            detected_face_id: Unique identifier for detected face
            confidence_score: Recognition confidence score
            embedder_type: Type of embedder used
            threshold_used: Threshold value used for recognition
            recognition_result: Result ('matched', 'no_match', 'unknown')
            image_path: Path to face image (optional)
            processing_time_ms: Processing time in milliseconds (optional)
            face_bbox: Face bounding box (x, y, w, h) (optional)
            session_id: Session identifier (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Extract bounding box coordinates
                bbox_x = bbox_y = bbox_w = bbox_h = None
                if face_bbox and len(face_bbox) >= 4:
                    bbox_x, bbox_y, bbox_w, bbox_h = face_bbox[:4]
                
                cursor.execute("""
                    INSERT INTO recognition_logs (
                        student_id, detected_face_id, confidence_score, embedder_type,
                        threshold_used, recognition_result, image_path, processing_time_ms,
                        face_bbox_x, face_bbox_y, face_bbox_w, face_bbox_h, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (student_id, detected_face_id, confidence_score, embedder_type,
                      threshold_used, recognition_result, image_path, processing_time_ms,
                      bbox_x, bbox_y, bbox_w, bbox_h, session_id))
                
                conn.commit()
                self.logger.debug(f"Recognition result logged for {student_id or 'unknown'}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging recognition result: {str(e)}")
            return False
    
    def log_manual_override(self, student_id: str, original_result: str,
                           override_action: str, override_reason: str = None,
                           admin_user: str = None, attendance_record_id: int = None) -> bool:
        """
        Log manual override actions for tracking admin interventions
        
        Args:
            student_id: Student ID
            original_result: Original recognition result
            override_action: Action taken (e.g., 'mark_present', 'mark_absent')
            override_reason: Reason for override (optional)
            admin_user: Admin user who made the override (optional)
            attendance_record_id: Related attendance record ID (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO manual_overrides (
                        student_id, original_result, override_action, override_reason,
                        admin_user, attendance_record_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (student_id, original_result, override_action, override_reason,
                      admin_user, attendance_record_id))
                
                conn.commit()
                self.logger.info(f"Manual override logged for student {student_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error logging manual override: {str(e)}")
            return False

    def get_recognition_logs(self, student_id: str = None, embedder_type: str = None,
                           date_from: str = None, date_to: str = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recognition logs for analysis
        
        Args:
            student_id: Filter by student ID (optional)
            embedder_type: Filter by embedder type (optional)
            date_from: Start date (YYYY-MM-DD format)
            date_to: End date (YYYY-MM-DD format)
            limit: Maximum number of records to return
            
        Returns:
            List of recognition log dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = """
                    SELECT * FROM recognition_logs WHERE 1=1
                """
                params = []
                
                if student_id:
                    query += " AND student_id = ?"
                    params.append(student_id)
                
                if embedder_type:
                    query += " AND embedder_type = ?"
                    params.append(embedder_type)
                
                if date_from:
                    query += " AND DATE(created_at) >= ?"
                    params.append(date_from)
                
                if date_to:
                    query += " AND DATE(created_at) <= ?"
                    params.append(date_to)
                
                query += " ORDER BY created_at DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                logs = []
                columns = [description[0] for description in cursor.description]
                
                for row in rows:
                    log_dict = dict(zip(columns, row))
                    logs.append(log_dict)
                
                return logs
                
        except Exception as e:
            self.logger.error(f"Error getting recognition logs: {str(e)}")
            return []
    
    def get_performance_metrics(self, embedder_type: str = None, 
                              date_from: str = None) -> List[Dict[str, Any]]:
        """
        Get performance metrics for threshold tuning
        
        Args:
            embedder_type: Filter by embedder type (optional)
            date_from: Start date for metrics (optional)
            
        Returns:
            List of performance metric dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                
                if embedder_type:
                    query += " AND embedder_type = ?"
                    params.append(embedder_type)
                
                if date_from:
                    query += " AND test_date >= ?"
                    params.append(date_from)
                
                query += " ORDER BY test_date DESC, threshold_value ASC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                metrics = []
                columns = [description[0] for description in cursor.description]
                
                for row in rows:
                    metric_dict = dict(zip(columns, row))
                    metrics.append(metric_dict)
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {str(e)}")
            return []

    def close(self):
        """Close database connection"""
        # SQLite connections are automatically closed when the context manager exits
        pass
