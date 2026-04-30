import sqlite3

def create_presentation_schedule():
    # Ensure the path to your database is correct
    conn = sqlite3.connect('data/database/attendance.db')
    cursor = conn.cursor()

    # Clearing old schedule data to avoid overlaps during the demo
    cursor.execute("DELETE FROM period_schedule")

    # Schedule Data: 10:00 AM to 1:30 PM (using 24-hour format)
    schedule_data = [
        (1, '10:00', '11:00', 'CS301', 'Morning Demo Session'),  # Starts at 10:00 AM
        (2, '11:08', '12:10', 'CS302', 'Artificial Intelligence'), # 10-minute break included
        (3, '12:14', '13:30', 'CS303', 'Computer Vision Lab')      # Ends at 1:30 PM
    ]

    cursor.executemany('''
        INSERT INTO period_schedule 
        (period_number, start_time, end_time, subject_code, subject_name) 
        VALUES (?, ?, ?, ?, ?)
    ''', schedule_data)

    conn.commit()
    conn.close()
    print("✅ Roadshow Schedule updated for 10:00 AM - 1:30 PM!")

if __name__ == "__main__":
    create_presentation_schedule()