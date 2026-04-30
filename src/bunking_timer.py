import threading
import time
import sqlite3
from datetime import datetime, timedelta

class BunkingTimer:
    """
    Monitors students who exit temporarily (TEMP_OUT) and automatically
    marks them as BUNKED if they don't return within 15 minutes.
    
    This is the unique feature of VIGIL-CLASS!
    """
    
    def __init__(self, db_path='data/database/attendance.db'):
        self.db_path = db_path
        self.timeout_minutes = 1 # 15-minute window
        self.check_interval = 30   # Check every 30 seconds
        self.active = False
        self.timer_thread = None
        self.temp_out_students = {}  # {roll_no: exit_timestamp}
        
        print("✓ Bunking Timer initialized (15-minute timeout)")
    
    def start(self):
        """Start the bunking timer in background"""
        if self.active:
            print("⚠️  Timer already running")
            return
        
        self.active = True
        self.timer_thread = threading.Thread(target=self._timer_loop, daemon=True)
        self.timer_thread.start()
        
        print(f"✓ Bunking timer started (checking every {self.check_interval}s)")
    
    def stop(self):
        """Stop the bunking timer"""
        self.active = False
        if self.timer_thread:
            self.timer_thread.join(timeout=2)
        
        print("✓ Bunking timer stopped")
    
    def _timer_loop(self):
        """Main timer loop - runs in background thread"""
        while self.active:
            try:
                self.check_timeouts()
                time.sleep(self.check_interval)
            except Exception as e:
                print(f"⚠️  Timer error: {e}")
                time.sleep(self.check_interval)
    
    def check_timeouts(self):
        """Check all TEMP_OUT students and mark as BUNKED if timeout exceeded"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        current_time = datetime.now()
        
        # Get all TEMP_OUT students today
        cursor.execute('''
            SELECT a.roll_no, s.name, a.period, a.exit_time, ps.subject_name
            FROM attendance a
            JOIN students s ON a.roll_no = s.roll_no
            JOIN period_schedule ps ON a.period = ps.period_number
            WHERE a.date = ? AND a.status = 'TEMP_OUT'
        ''', (today,))
        
        temp_out_students = cursor.fetchall()
        
        for roll_no, name, period, exit_time_str, subject in temp_out_students:
            if not exit_time_str:
                continue
            
            # Parse exit time
            try:
                exit_time = datetime.strptime(exit_time_str, '%H:%M:%S')
                # Combine with today's date
                exit_datetime = datetime.combine(datetime.now().date(), exit_time.time())
                
                # Calculate time elapsed since exit
                time_elapsed = current_time - exit_datetime
                minutes_elapsed = int(time_elapsed.total_seconds() / 60)
                
                # Check if timeout exceeded
                if minutes_elapsed >= self.timeout_minutes:
                    # Mark as BUNKED!
                    cursor.execute('''
                        UPDATE attendance
                        SET status = 'BUNKED'
                        WHERE roll_no = ? AND date = ? AND period = ?
                    ''', (roll_no, today, period))
                    
                    conn.commit()
                    
                    print(f"\n🚨 BUNKING DETECTED!")
                    print(f"   Roll-{roll_no} ({name})")
                    print(f"   Period {period}: {subject}")
                    print(f"   Left at: {exit_time_str}")
                    print(f"   Time elapsed: {minutes_elapsed} minutes")
                    print(f"   Status: TEMP_OUT → BUNKED")
                    print(f"   Action: Marked as bunked (exceeded 15-min limit)")
                
                else:
                    # Still within timeout window
                    remaining = self.timeout_minutes - minutes_elapsed
                    
                    # Only log if this is a new TEMP_OUT (not already tracked)
                    if roll_no not in self.temp_out_students:
                        print(f"\n⏱️  TEMP_OUT: Roll-{roll_no} ({name})")
                        print(f"   Period {period}: {subject}")
                        print(f"   Time remaining: {remaining} minutes")
                        self.temp_out_students[roll_no] = exit_datetime
                    
            except ValueError as e:
                print(f"⚠️  Error parsing time for Roll-{roll_no}: {e}")
                continue
        
        # Clean up returned students from tracking
        current_temp_out = [r[0] for r in temp_out_students]
        returned = [r for r in self.temp_out_students if r not in current_temp_out]
        
        for roll_no in returned:
            print(f"✓ Roll-{roll_no} returned (removed from timeout tracking)")
            del self.temp_out_students[roll_no]
        
        conn.close()
    
    def get_temp_out_count(self):
        """Get count of students currently TEMP_OUT"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        cursor.execute('''
            SELECT COUNT(*) FROM attendance
            WHERE date = ? AND status = 'TEMP_OUT'
        ''', (today,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_bunked_count(self):
        """Get count of students marked as BUNKED today"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        cursor.execute('''
            SELECT COUNT(*) FROM attendance
            WHERE date = ? AND status = 'BUNKED'
        ''', (today,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    def get_timeout_status(self):
        """Get detailed status of all TEMP_OUT students"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        current_time = datetime.now()
        
        cursor.execute('''
            SELECT a.roll_no, s.name, a.period, a.exit_time, ps.subject_name
            FROM attendance a
            JOIN students s ON a.roll_no = s.roll_no
            JOIN period_schedule ps ON a.period = ps.period_number
            WHERE a.date = ? AND a.status = 'TEMP_OUT'
        ''', (today,))
        
        results = []
        
        for roll_no, name, period, exit_time_str, subject in cursor.fetchall():
            if exit_time_str:
                try:
                    exit_time = datetime.strptime(exit_time_str, '%H:%M:%S')
                    exit_datetime = datetime.combine(datetime.now().date(), exit_time.time())
                    
                    time_elapsed = current_time - exit_datetime
                    minutes_elapsed = int(time_elapsed.total_seconds() / 60)
                    minutes_remaining = self.timeout_minutes - minutes_elapsed
                    
                    results.append({
                        'roll_no': roll_no,
                        'name': name,
                        'period': period,
                        'subject': subject,
                        'exit_time': exit_time_str,
                        'minutes_elapsed': minutes_elapsed,
                        'minutes_remaining': max(0, minutes_remaining),
                        'will_be_bunked': minutes_remaining <= 0
                    })
                except:
                    pass
        
        conn.close()
        return results


# Standalone test
if __name__ == "__main__":
    print("\n" + "="*60)
    print("BUNKING TIMER TEST")
    print("="*60)
    
    timer = BunkingTimer()
    timer.start()
    
    print("\nTimer running... Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(10)
            
            # Show status every 10 seconds
            temp_out = timer.get_temp_out_count()
            bunked = timer.get_bunked_count()
            
            print(f"Status: TEMP_OUT={temp_out}, BUNKED={bunked}")
            
    except KeyboardInterrupt:
        print("\n\nStopping timer...")
        timer.stop()
        print("✓ Test complete")