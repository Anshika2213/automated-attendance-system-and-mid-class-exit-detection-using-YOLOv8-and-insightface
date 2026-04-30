from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self):
        print("Loading YOLOv8 person detector...")
        self.model = YOLO('yolov8n.pt')
        print("✓ Detector ready")
    
    def detect_people(self, image):
        """Detect all people in image"""
        # results = self.model(image, verbose=False)  # made changes here
        # Force YOLO to only return humans with > 60% confidence
        results = self.model(image, verbose=False, conf=0.60, iou=0.45)
        people = []
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            
            # Only keep 'person' detections
            if class_name == 'person':
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                people.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
        
        return people
    
    def draw_detections(self, image, people):
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        
        for person in people:
            x1, y1, x2, y2 = person['bbox']
            conf = person['confidence']
            
            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person {conf:.0%}"
            cv2.putText(img_copy, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_copy

# Test
if __name__ == "__main__":
    detector = PersonDetector()
    
    # Test on your photo
    img = cv2.imread('data/students/roll_3.jpg')
    people = detector.detect_people(img)
    
    print(f"Detected {len(people)} person(s)")
    
    for i, person in enumerate(people, 1):
        print(f"  Person {i}: confidence {person['confidence']:.0%}")
    
    # Draw and save
    result_img = detector.draw_detections(img, people)
    cv2.imwrite('data/detection_result.jpg', result_img)
    print("\n✓ Result saved to data/detection_result.jpg")