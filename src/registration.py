import cv2
import insightface
import numpy as np
import os
from database import Database

class Registration:
    def __init__(self):
        print("Loading face recognition model...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1)  # CPU mode
        print("✓ Model loaded")
        
        self.db = Database()
    
    def register_from_photo(self, roll_no, name, photo_path, email=None):
        """Register student from a photo file"""
        print(f"\nRegistering: {name} (Roll-{roll_no})")
        print(f"Photo: {photo_path}")
        
        # Read image
        img = cv2.imread(photo_path)
        if img is None:
            return False, f"❌ Could not read image: {photo_path}"
        
        # Detect faces
        faces = self.app.get(img)
        
        if len(faces) == 0:
            return False, "❌ No face detected in photo"
        
        if len(faces) > 1:
            return False, f"❌ Multiple faces detected ({len(faces)}). Use photo with single face."
        
        # Get face embeddingt
        face = faces[0]
        embedding = face.embedding
        confidence = face.det_score
        
        print(f"Face detected with confidence: {confidence:.2f}")
        
        # Normalize embedding (important for comparison)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Save to database
        self.db.save_student(roll_no, name, embedding, email)
        
        return True, f"✓ Successfully registered {name}"
    
    def register_interactive(self):
        """Interactive registration using webcam - No display version"""
        print("\n=== Interactive Registration ===")
        print("Get ready in front of camera...")
        
        # Get student details
        try:
            roll_no = int(input("Enter Roll Number: "))
            name = input("Enter Name: ")
            email = input("Enter Email (optional, press Enter to skip): ").strip()
            if not email:
                email = None
        except:
            return False, "❌ Invalid input"
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return False, "❌ Could not open camera"
        
        print("\nCamera opened.")
        print("Taking photo in 3 seconds...")
        print("3...")
        
        import time
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        
        # Capture multiple frames and pick best one
        best_frame = None
        best_confidence = 0
        
        print("Capturing...")
        
        for i in range(10):  # Try 10 frames
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect face
            faces = self.app.get(frame)
            
            if len(faces) == 1:
                confidence = faces[0].det_score
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_frame = frame.copy()
        
        cap.release()
        
        if best_frame is None:
            return False, "❌ No face detected. Try again with better lighting."
        
        print(f"✓ Photo captured! (confidence: {best_confidence:.2f})")
        
        # Save captured photo
        photo_path = f'data/students/roll_{roll_no}.jpg'
        cv2.imwrite(photo_path, best_frame)
        print(f"✓ Photo saved: {photo_path}")
        
        # Register using captured photo
        return self.register_from_photo(roll_no, name, photo_path, email)
    
    def list_registered(self):
        """List all registered students"""
        students = self.db.get_all_students()
        
        if len(students) == 0:
            print("\nNo students registered yet")
            return
        
        print(f"\n=== Registered Students ({len(students)}) ===")
        for s in students:
            email = s['email'] if s['email'] else 'N/A'
            print(f"Roll-{s['roll_no']}: {s['name']} ({email})")

# Interactive menu
if __name__ == "__main__":
    reg = Registration()
    
    while True:
        print("\n" + "="*50)
        print("STUDENT REGISTRATION MENU")
        print("="*50)
        print("1. Register new student (Interactive - Webcam)")
        print("2. Register from existing photo")
        print("3. List registered students")
        print("4. Exit")
        print("="*50)
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == '1':
            success, msg = reg.register_interactive()
            print(msg)
        
        elif choice == '2':
            try:
                roll_no = int(input("Enter Roll Number: "))
                name = input("Enter Name: ")
                photo_path = input("Enter photo path (e.g., data/students/photo_1.jpg): ")
                email = input("Enter Email (optional): ").strip() or None
                
                success, msg = reg.register_from_photo(roll_no, name, photo_path, email)
                print(msg)
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '3':
            reg.list_registered()
        
        elif choice == '4':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice")