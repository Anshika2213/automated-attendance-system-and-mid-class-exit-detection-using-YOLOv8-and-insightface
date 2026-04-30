import threading
import time
import cv2
from entry_exit_tracker import EntryExitTracker
from period_manager import PeriodManager
from bunking_timer import BunkingTimer

class AutoAttendanceSystem:
    def __init__(self):
        print("Initializing Auto Attendance System...")
        self.tracker = EntryExitTracker()
        self.bunking_timer = BunkingTimer()
        self.tracking_active = False
        self.tracking_thread = None
        
    def start_tracking_thread(self):
        """Start tracking with camera window display"""
        self.tracking_active = True
        
        def track_loop():
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("✗ Camera error")
                return
            
            print("Camera tracking started")
            print("Camera window opened - Press 'q' to stop period early")
            
            frame_count = 0
            
            # Create window
            cv2.namedWindow('Auto Attendance - Live', cv2.WINDOW_NORMAL)
            
            while self.tracking_active:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                processed_frame, events = self.tracker.process_frame(frame)
                
                # Display window
                cv2.imshow('Auto Attendance - Live', processed_frame)
                
                # Check for 'q' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✓ Period stopped early by user (pressed 'q')")
                    self.tracking_active = False
                    break
                
                # Save for dashboard (every 0.5 sec)
                if frame_count % 15 == 0:
                    cv2.imwrite('data/live_feed.jpg', processed_frame)
                
                time.sleep(0.001)
            
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Camera tracking stopped")
        
        self.tracking_thread = threading.Thread(target=track_loop, daemon=True)
        self.tracking_thread.start()
    
    def stop_tracking_thread(self):
        """Stop tracking thread"""
        self.tracking_active = False
        if self.tracking_thread:
            self.tracking_thread.join(timeout=2)
        cv2.destroyAllWindows()
    
    def run_auto_system(self):
        """Run complete automated system with bunking detection"""
        print("\n" + "="*60)
        print("AUTOMATED ATTENDANCE SYSTEM")
        print("="*60)
        print("\nFeatures:")
        print("✓ Auto-detect current period based on time")
        print("✓ Auto-start camera tracking when period begins")
        print("✓ Auto-mark attendance (entry/exit)")
        print("✓ 15-minute bunking timer (unique!)")
        print("✓ Auto-end period and mark absents")
        print("✓ Live camera window display")
        print("\nControls:")
        print("- Camera window opens automatically during periods")
        print("- Press 'q' in camera window to end period early")
        print("- Press Ctrl+C in terminal to stop system")
        print("\nTeacher action required: NONE (fully automated)")
        print("="*60 + "\n")
        
        input("Press ENTER to start automated system...")
        
        # Start bunking timer
        self.bunking_timer.start()
        print("✓ 15-minute bunking timer activated")
        
        # Hook into period manager
        original_start = self.tracker.period_manager.start_period
        original_end = self.tracker.period_manager.end_period
        
        def auto_start_period(period_num):
            """Called when period starts"""
            result = original_start(period_num)
            if result:
                # Start camera tracking
                self.start_tracking_thread()
            return result
        
        def auto_end_period():
            """Called when period ends"""
            # Stop camera tracking
            self.stop_tracking_thread()
            
            # End period (marks absents)
            original_end()
        
        # Replace methods with auto versions
        self.tracker.period_manager.start_period = auto_start_period
        self.tracker.period_manager.end_period = auto_end_period
        
        # Run auto monitoring
        try:
            self.tracker.period_manager.auto_start_monitoring()
        except KeyboardInterrupt:
            print("\n\nSystem stopped by user")
            self.stop_tracking_thread()
            self.bunking_timer.stop()
        finally:
            cv2.destroyAllWindows()
            self.bunking_timer.stop()
            print("\n✓ Auto attendance system shutdown complete")



if __name__ == "__main__":
    system = AutoAttendanceSystem()
    system.run_auto_system()