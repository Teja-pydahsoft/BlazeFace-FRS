"""
Check current attendance records in the database
"""

import sys
import os

# Add app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.core.database import DatabaseManager
from datetime import datetime

def check_attendance():
    """Check current attendance records"""
    try:
        print("Checking Attendance Records...")
        
        # Initialize database manager
        db_manager = DatabaseManager("database/blazeface_frs.db")
        
        # Get all students
        students = db_manager.get_all_students()
        print(f"Total students in database: {len(students)}")
        for student in students:
            print(f"  - {student['student_id']}: {student['name']}")
        
        # Get today's attendance
        today = datetime.now().date().strftime('%Y-%m-%d')
        print(f"\nToday's attendance ({today}):")
        today_records = db_manager.get_attendance_records(date_from=today, date_to=today)
        
        if today_records:
            print(f"Found {len(today_records)} records:")
            for record in today_records:
                print(f"  - {record['student_id']} ({record['name']}): {record['status']} at {record['time']} (confidence: {record['confidence']})")
        else:
            print("No attendance records for today")
        
        # Check if student "1234" exists and is marked
        student_1234 = db_manager.get_student("1234")
        if student_1234:
            print(f"\nStudent 1234 found: {student_1234['name']}")
            student_records = db_manager.get_attendance_records(student_id="1234", date_from=today, date_to=today)
            if student_records:
                print(f"Student 1234 is already marked today: {len(student_records)} records")
                for record in student_records:
                    print(f"  - {record['status']} at {record['time']} (confidence: {record['confidence']})")
            else:
                print("Student 1234 is NOT marked today")
        else:
            print("\nStudent 1234 not found in database")
        
        # Get all attendance records
        print(f"\nAll attendance records:")
        all_records = db_manager.get_attendance_records()
        if all_records:
            print(f"Found {len(all_records)} total records:")
            for record in all_records:
                print(f"  - {record['student_id']} ({record['name']}): {record['status']} on {record['date']} at {record['time']}")
        else:
            print("No attendance records found")
        
    except Exception as e:
        print(f"Error checking attendance: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_attendance()
