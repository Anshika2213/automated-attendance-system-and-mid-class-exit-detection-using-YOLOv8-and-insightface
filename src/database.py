import sqlite3
import pickle
import os
from datetime import datetime

class Database:
    def __init__(self):
        self.db_path = 'data/database/attendance.db'
        os.makedirs('data/database', exist_ok=True)
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 1. Students table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                roll_no INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT,
                embedding BLOB NOT NULL,
                registered_date TEXT
            )
        ''')
        
        # 2. Attendance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                roll_no INTEGER,
                date TEXT,
                period INTEGER,
                subject_code TEXT,
                entry_time TEXT,
                exit_time TEXT,
                duration_minutes INTEGER,
                status TEXT,
                FOREIGN KEY (roll_no) REFERENCES students(roll_no)
            )
        ''')

        # 3. Period Schedule table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS period_schedule (
                period_number INTEGER PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                subject_code TEXT,
                subject_name TEXT
            )
        ''')

        # 4. ── NEW: Suspicious / face-covered alerts table ─────────────────
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT,
                time        TEXT,
                period      INTEGER,
                subject_code TEXT,
                event_type  TEXT,        -- 'FACE_COVERED_ENTRY' | 'FACE_COVERED_EXIT'
                reid_roll_no INTEGER,    -- roll_no from body Re-ID (NULL if no match)
                reid_name   TEXT,        -- name from body Re-ID  (NULL if no match)
                reid_score  REAL,        -- similarity score (0-1)
                snapshot_path TEXT       -- path to saved frame snapshot
            )
        ''')
        # ──────────────────────────────────────────────────────────────────────

        conn.commit()
        conn.close()

    # ── NEW: Log a suspicious event ───────────────────────────────────────────
    def log_suspicious(self, event_type, period_info,
                       reid_roll_no=None, reid_name=None,
                       reid_score=0.0, snapshot_path=None):
        """
        Log a face-covered crossing event.

        Args:
            event_type    : 'FACE_COVERED_ENTRY' or 'FACE_COVERED_EXIT'
            period_info   : dict with keys 'period', 'subject_code'
            reid_roll_no  : roll_no from body Re-ID (or None)
            reid_name     : name from body Re-ID (or None)
            reid_score    : similarity score from body Re-ID
            snapshot_path : path to the saved frame image
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()
        cursor.execute('''
            INSERT INTO alerts
            (date, time, period, subject_code, event_type,
             reid_roll_no, reid_name, reid_score, snapshot_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            now.date().isoformat(),
            now.strftime('%H:%M:%S'),
            period_info['period']      if period_info else None,
            period_info['subject_code'] if period_info else None,
            event_type,
            reid_roll_no,
            reid_name,
            round(reid_score, 3),
            snapshot_path
        ))

        conn.commit()
        conn.close()

    def get_alerts(self, date=None):
        """Fetch all alerts, optionally filtered by date."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if date:
            cursor.execute('SELECT * FROM alerts WHERE date=? ORDER BY time DESC', (date,))
        else:
            cursor.execute('SELECT * FROM alerts ORDER BY date DESC, time DESC')

        rows = cursor.fetchall()
        conn.close()

        cols = ['id', 'date', 'time', 'period', 'subject_code',
                'event_type', 'reid_roll_no', 'reid_name', 'reid_score', 'snapshot_path']
        return [dict(zip(cols, row)) for row in rows]
    # ──────────────────────────────────────────────────────────────────────────

    def save_student(self, roll_no, name, embedding, email=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        embedding_blob  = pickle.dumps(embedding)
        registered_date = datetime.now().isoformat()
        cursor.execute('''
            INSERT OR REPLACE INTO students 
            (roll_no, name, email, embedding, registered_date)
            VALUES (?, ?, ?, ?, ?)
        ''', (roll_no, name, email, embedding_blob, registered_date))
        conn.commit()
        conn.close()
        return True
    
    def get_student(self, roll_no):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT roll_no, name, email, embedding FROM students WHERE roll_no = ?',
            (roll_no,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'roll_no':   row[0],
                'name':      row[1],
                'email':     row[2],
                'embedding': pickle.loads(row[3])
            }
        return None
    
    def get_all_students(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT roll_no, name, email, embedding FROM students')
        rows = cursor.fetchall()
        conn.close()
        students = []
        for row in rows:
            students.append({
                'roll_no':   row[0],
                'name':      row[1],
                'email':     row[2],
                'embedding': pickle.loads(row[3])
            })
        return students
    
    def delete_student(self, roll_no):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM students WHERE roll_no = ?', (roll_no,))
        conn.commit()
        conn.close()
        return True
    
    def count_students(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM students')
        count = cursor.fetchone()[0]
        conn.close()
        return count


if __name__ == "__main__":
    print("Testing Database Module...")
    db = Database()
    print("✓ Database created (alerts table included)")
    count = db.count_students()
    print(f"✓ Currently registered: {count} students")
    if count > 0:
        print("\nRegistered students:")
        for s in db.get_all_students():
            print(f"  - Roll-{s['roll_no']}: {s['name']}")
    print("\n✓ Database module working!")