"""
Test the database fix for attendance records
"""

import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager

def test_database_fix():
    """Test the database fix for attendance records"""
    try:
        print("Testing Database Fix for Attendance Records...")
        
        # Initialize database manager
        db_manager = DatabaseManager("database/test_attendance.db")
        
        # First add a test student
        print("Adding test student...")
        test_student = {
            'student_id': 'TEST123',
            'name': 'Test Student',
            'email': 'test@example.com',
            'phone': '123-456-7890',
            'department': 'Test Department',
            'year': '2024'
        }
        db_manager.add_student(test_student)
        
        # Test adding an attendance record
        print("Testing attendance record insertion...")
        success = db_manager.add_attendance_record(
            student_id="TEST123",
            status='present',
            confidence=0.95,
            detection_type='face_recognition',
            notes="Test attendance record"
        )
        
        if success:
            print("✓ Attendance record added successfully!")
        else:
            print("✗ Failed to add attendance record")
            return
        
        # Test retrieving attendance records
        print("Testing attendance record retrieval...")
        records = db_manager.get_attendance_records()
        
        if records:
            print(f"✓ Retrieved {len(records)} attendance records")
            for record in records:
                print(f"  - Student: {record['student_id']}, Status: {record['status']}, Confidence: {record['confidence']}")
        else:
            print("✗ No attendance records found")
        
        # Test today's attendance
        print("Testing today's attendance...")
        from datetime import datetime
        today = datetime.now().date().strftime('%Y-%m-%d')
        today_records = db_manager.get_attendance_records(date_from=today, date_to=today)
        
        if today_records:
            print(f"✓ Found {len(today_records)} records for today")
        else:
            print("✗ No records found for today")
        
        print("Database fix test completed successfully!")
        
    except Exception as e:
        print(f"Error in database fix test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_database_fix()
