"""
Database management for BlazeFace-FRS system
Handles student data, attendance records, and face encodings
"""

import sqlite3
import json
import numpy as np
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
    
    def add_face_encoding(self, student_id: str, encoding: np.ndarray, 
                         encoding_type: str = 'facenet') -> bool:
        """
        Add face encoding for a student
        
        Args:
            student_id: Student ID
            encoding: Face encoding array
            encoding_type: Type of encoding (default: 'facenet')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert numpy array to bytes
                encoding_bytes = encoding.tobytes()
                
                cursor.execute("""
                    INSERT INTO face_encodings (student_id, encoding, encoding_type)
                    VALUES (?, ?, ?)
                """, (student_id, encoding_bytes, encoding_type))
                
                conn.commit()
                self.logger.info(f"Face encoding added for student: {student_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding face encoding: {str(e)}")
            return False
    
    def get_face_encodings(self, student_id: str = None) -> List[Tuple[str, np.ndarray, str]]:
        """
        Get face encodings from the database
        
        Args:
            student_id: Specific student ID (optional)
            
        Returns:
            List of tuples (student_id, encoding, encoding_type)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if student_id:
                    cursor.execute("""
                        SELECT student_id, encoding, encoding_type 
                        FROM face_encodings WHERE student_id = ?
                    """, (student_id,))
                else:
                    cursor.execute("""
                        SELECT student_id, encoding, encoding_type 
                        FROM face_encodings
                    """)
                
                rows = cursor.fetchall()
                encodings = []
                
                for row in rows:
                    student_id, encoding_bytes, encoding_type = row
                    encoding = np.frombuffer(encoding_bytes, dtype=np.float32)
                    encodings.append((student_id, encoding, encoding_type))
                
                return encodings
                
        except Exception as e:
            self.logger.error(f"Error getting face encodings: {str(e)}")
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
    
    def close(self):
        """Close database connection"""
        # SQLite connections are automatically closed when the context manager exits
        pass
