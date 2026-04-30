import cv2
import numpy as np
import insightface
from database import Database
 
 
class FaceMatcher:
    def __init__(self):
        print("Loading FaceMatcher (InsightFace buffalo_l)...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1)
 
        self.db = Database()
 
        # ── Tunable thresholds ─────────────────────────────────────────────
        # Cosine similarity: 1.0 = identical, 0.0 = unrelated
        # 0.55 works well for real classroom conditions (CPU, 640x480 camera)
        # Raise to 0.65 if you get false positives, lower to 0.50 if misses
        self.MATCH_THRESHOLD   = 0.55
        self.MIN_FACE_SIZE     = 60    # px — faces smaller than this fail silently
 
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("FaceMatcher ready.")
 
    # ── Helpers ───────────────────────────────────────────────────────────────
    def _enhance(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge([self._clahe.apply(l), a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
 
    def _get_embedding(self, img):
        """
        Extract a single face embedding from an image crop.
 
        Strategy:
          1. Run InsightFace detector on the crop as-is.
          2. If no face found, try the enhanced version.
          3. If still nothing, return None.
          4. If multiple faces found, pick the largest (closest to camera).
        """
        for attempt in [img, self._enhance(img)]:
            faces = self.app.get(attempt)
            if not faces:
                continue
 
            # Pick largest face by bounding box area
            face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
 
            x1, y1, x2, y2 = [int(c) for c in face.bbox]
            face_w = x2 - x1
            face_h = y2 - y1
 
            if face_w < self.MIN_FACE_SIZE or face_h < self.MIN_FACE_SIZE:
                continue   # too small — skip
 
            if float(face.det_score) < 0.45:
                continue   # low confidence detection
 
            emb = face.embedding
            return emb / np.linalg.norm(emb)
 
        return None
 
    # ── Main match function ───────────────────────────────────────────────────
    def match_face(self, face_crop):
        """
        Match a face crop against all registered students.
 
        Args:
            face_crop: BGR image (numpy array) — the cropped region from camera
 
        Returns:
            (roll_no, score, name, all_scores)
              roll_no    : matched student's roll number, or None
              score      : cosine similarity (0–1)
              name       : matched student's name, or "Unknown"
              all_scores : list of (roll_no, name, score) for all students — useful for debug
        """
        students = self.db.get_all_students()
        if not students:
            return None, 0.0, "Unknown", []
 
        # Get embedding from the live crop
        query_emb = self._get_embedding(face_crop)
        if query_emb is None:
            return None, 0.0, "Unknown", []
 
        # Compare against all stored embeddings
        all_scores = []
        for s in students:
            stored_emb = s['embedding']   # already normalised at registration
            score      = float(np.dot(query_emb, stored_emb))
            all_scores.append((s['roll_no'], s['name'], score))
 
        # Sort by score descending
        all_scores.sort(key=lambda x: x[2], reverse=True)
 
        best_roll, best_name, best_score = all_scores[0]
 
        # Only return a match if above threshold
        if best_score >= self.MATCH_THRESHOLD:
            return best_roll, best_score, best_name, all_scores
        else:
            return None, best_score, "Unknown", all_scores
 
    # ── Debug helper ──────────────────────────────────────────────────────────
    def debug_match(self, face_crop):
        """
        Print top-3 matches with scores. Useful for tuning threshold.
        Call this instead of match_face() when diagnosing why someone isn't recognised.
        """
        roll_no, score, name, all_scores = self.match_face(face_crop)
 
        print("\n--- FaceMatcher Debug ---")
        print(f"Threshold: {self.MATCH_THRESHOLD}")
        print(f"Result: Roll-{roll_no} ({name}) score={score:.3f}")
        print("Top 3 candidates:")
        for r, n, s in all_scores[:3]:
            flag = " <-- MATCHED" if s >= self.MATCH_THRESHOLD else ""
            print(f"  Roll-{r} ({n}): {s:.3f}{flag}")
        print("-------------------------\n")
 
        return roll_no, score, name, all_scores