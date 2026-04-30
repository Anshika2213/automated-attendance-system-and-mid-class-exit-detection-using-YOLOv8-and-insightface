import sqlite3
from datetime import datetime

class PeriodManager:
    def __init__(self, db_path='data/database/attendance.db'):
        self.db_path = db_path
        self.current_period = None
        self.current_subject = None
        self.load_schedule()
    
    def load_schedule(self):
        """Load period schedule from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT period_number, start_time, end_time, 
                   subject_code, subject_name
            FROM period_schedule
            ORDER BY period_number
        ''')
        
        self.schedule = {}
        for period, start, end, code, name in cursor.fetchall():
            self.schedule[period] = {
                'start': start,
                'end': end,
                'subject_code': code,
                'subject_name': name
            }
        
        conn.close()
    
    def get_current_period_auto(self):
        """Auto-detect which period based on current time"""
        current_time = datetime.now().strftime('%H:%M')
        
        for period, info in self.schedule.items():
            if info['start'] <= current_time <= info['end']:
                return period, info
        
        return None, None
    
    def start_period(self, period_number):
        """Start a specific period"""
        if period_number not in self.schedule:
            print(f"✗ Invalid period number: {period_number}")
            return False
        
        self.current_period = period_number
        info = self.schedule[period_number]
        self.current_subject = info['subject_code']
        
        print(f"\n{'='*60}")
        print(f"📚 PERIOD {period_number} STARTED")
        print(f"{'='*60}")
        print(f"Subject: {info['subject_name']} ({info['subject_code']})")
        print(f"Time: {info['start']} - {info['end']}")
        print(f"Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}\n")
        
        return True
    
    def end_period(self):
        """End current period and mark absent students"""
        if not self.current_period:
            print("⚠️  No active period to end")
            return
        
        print(f"\n🔔 Ending Period {self.current_period}...")
        
        # Mark absent students
        self.mark_absent_students()
        
        print(f"✓ Period {self.current_period} ended\n")
        
        self.current_period = None
        self.current_subject = None
    
    def mark_absent_students(self):
        """Mark students who didn't attend as ABSENT"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        today = datetime.now().date().isoformat()
        
        # Get all students
        cursor.execute('SELECT roll_no, name FROM students')
        all_students = cursor.fetchall()
        
        # Get students who already have attendance for this period
        cursor.execute('''
            SELECT roll_no FROM attendance
            WHERE date = ? AND period = ?
        ''', (today, self.current_period))
        
        attended = {row[0] for row in cursor.fetchall()}
        
        # Mark absent
        absent_count = 0
        for roll_no, name in all_students:
            if roll_no not in attended:
                cursor.execute('''
                    INSERT INTO attendance
                    (roll_no, date, period, subject_code, status)
                    VALUES (?, ?, ?, ?, ?)
                ''', (roll_no, today, self.current_period, 
                      self.current_subject, 'ABSENT'))
                absent_count += 1
        
        conn.commit()
        conn.close()
        
        print(f"   Marked {absent_count} student(s) as ABSENT")
    
    def get_period_info(self):
        """Get current period details"""
        if not self.current_period:
            return None
        
        return {
            'period': self.current_period,
            'subject_code': self.current_subject,
            'subject_name': self.schedule[self.current_period]['subject_name']
        }
    
    def show_schedule(self):
        """Display period schedule"""
        print("\n" + "="*60)
        print("PERIOD SCHEDULE")
        print("="*60)
        
        for period in sorted(self.schedule.keys()):
            info = self.schedule[period]
            print(f"Period {period}: {info['start']}-{info['end']} | " +
                  f"{info['subject_name']} ({info['subject_code']})")
        
        print("="*60 + "\n")
    def auto_start_monitoring(self):
        """Continuously monitor and auto-start periods"""
        import time
        
        print("\n" + "="*60)
        print("AUTO PERIOD MONITORING - STARTED")
        print("="*60)
        print("System will automatically start/end periods")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")
        
        current_active_period = None
        
        while True:
            try:
                # Check every 30 seconds
                now = datetime.now()
                current_time = now.strftime('%H:%M')
                
                # Find if we're in any period
                active_period = None
                active_info = None
                
                for period, info in self.schedule.items():
                    if info['start'] <= current_time <= info['end']:
                        active_period = period
                        active_info = info
                        break
                
                # Period started
                if active_period and active_period != current_active_period:
                    print(f"\n⏰ TIME: {current_time}")
                    print(f"📚 AUTO-STARTING Period {active_period}")
                    self.start_period(active_period)
                    current_active_period = active_period
                
                # Period ended
                elif not active_period and current_active_period:
                    print(f"\n⏰ TIME: {current_time}")
                    print(f"🔔 AUTO-ENDING Period {current_active_period}")
                    self.end_period()
                    current_active_period = None
                
                # Status update every 5 minutes
                if now.minute % 5 == 0 and now.second < 30:
                    if current_active_period:
                        remaining = self.get_time_remaining(current_active_period)
                        print(f"⏱️  Period {current_active_period} in progress - {remaining} remaining")
                    else:
                        next_period = self.get_next_period()
                        if next_period:
                            print(f"⏸️  Break time - Next: Period {next_period[0]} at {next_period[1]['start']}")
                        else:
                            print(f"⏸️  No more periods today")
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                print("\n\n🛑 Auto-monitoring stopped by user")
                if current_active_period:
                    print(f"Ending Period {current_active_period}...")
                    self.end_period()
                break

    def get_time_remaining(self, period):
        """Calculate time remaining in period"""
        now = datetime.now().strftime('%H:%M')
        end = self.schedule[period]['end']
        
        now_time = datetime.strptime(now, '%H:%M')
        end_time = datetime.strptime(end, '%H:%M')
        
        diff = end_time - now_time
        minutes = int(diff.total_seconds() / 60)
        
        return f"{minutes} minutes"

    def get_next_period(self):
        """Get next upcoming period"""
        now = datetime.now().strftime('%H:%M')
        
        for period in sorted(self.schedule.keys()):
            if self.schedule[period]['start'] > now:
                return period, self.schedule[period]
        
        return None

# Test
if __name__ == "__main__":
    manager = PeriodManager()
    manager.show_schedule()
    
    # Auto-detect
    period, info = manager.get_current_period_auto()
    if period:
        print(f"Current period: {period} ({info['subject_name']})")
    else:
        print("No period currently scheduled (break time)")