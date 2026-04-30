import cv2
import numpy as np


class BodyReID:
    """
    Lightweight body Re-ID using HSV color histograms.

    How it works:
      - When a person is identified (roll_no confirmed), we store a
        color histogram of their body region as their "appearance signature".
      - When a person crosses with face COVERED (roll_no = None), we
        compare their body histogram against all stored signatures.
      - If similarity > threshold → soft-match to that roll_no.

    Why histograms and not a deep Re-ID model?
      - No extra dependencies
      - Fast enough for real-time (< 2ms per comparison)
      - Good enough for classroom scenario (limited people, similar lighting)
    """

    def __init__(self, similarity_threshold=0.70):
        # Cosine similarity threshold for a soft-match to be accepted
        self.similarity_threshold = similarity_threshold

        # {roll_no: histogram}  — updated every time a person is seen clearly
        self.appearance_db = {}

    # ── Histogram helpers ─────────────────────────────────────────────────────

    def _compute_histogram(self, body_crop):
        """
        Compute a normalised HSV histogram for a body crop.
        We use H + S channels only (skip V = brightness) so the
        signature is robust to lighting changes.
        """
        if body_crop is None or body_crop.size == 0:
            return None

        hsv = cv2.cvtColor(body_crop, cv2.COLOR_BGR2HSV)

        # H: 0-180 (18 bins)  S: 0-256 (16 bins)  → 288-dim vector
        hist_h = cv2.calcHist([hsv], [0], None, [18], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])

        hist = np.concatenate([hist_h.flatten(), hist_s.flatten()])
        norm = np.linalg.norm(hist)
        if norm == 0:
            return None
        return hist / norm   # unit vector → cosine sim = dot product

    def _cosine_similarity(self, h1, h2):
        return float(np.dot(h1, h2))   # both already unit vectors

    # ── Public API ────────────────────────────────────────────────────────────

    def update_signature(self, roll_no, frame, bbox):
        """
        Call this every time we CONFIRM a person's identity (roll_no known).
        Stores / refreshes their body histogram.
        """
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        # Use the LOWER 60% of the bounding box as body region
        # (avoids face area, focuses on clothing)
        face_cutoff = y1 + int((y2 - y1) * 0.40)
        x1c = max(0, x1);        x2c = min(w, x2)
        y1c = max(0, face_cutoff); y2c = min(h, y2)

        body_crop = frame[y1c:y2c, x1c:x2c]
        hist = self._compute_histogram(body_crop)

        if hist is not None:
            # Exponential moving average — keeps signature fresh
            if roll_no in self.appearance_db:
                self.appearance_db[roll_no] = (
                    0.7 * self.appearance_db[roll_no] + 0.3 * hist
                )
                # Re-normalise after blending
                norm = np.linalg.norm(self.appearance_db[roll_no])
                if norm > 0:
                    self.appearance_db[roll_no] /= norm
            else:
                self.appearance_db[roll_no] = hist

    def find_match(self, frame, bbox):
        """
        Try to soft-match a FACE-COVERED person against stored signatures.

        Returns:
            (roll_no, similarity)  if match found above threshold
            (None,    0.0)         if no match
        """
        if not self.appearance_db:
            return None, 0.0

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]

        face_cutoff = y1 + int((y2 - y1) * 0.40)
        x1c = max(0, x1);         x2c = min(w, x2)
        y1c = max(0, face_cutoff); y2c = min(h, y2)

        body_crop = frame[y1c:y2c, x1c:x2c]
        query_hist = self._compute_histogram(body_crop)

        if query_hist is None:
            return None, 0.0

        best_roll, best_sim = None, 0.0
        for roll_no, stored_hist in self.appearance_db.items():
            sim = self._cosine_similarity(query_hist, stored_hist)
            if sim > best_sim:
                best_sim  = sim
                best_roll = roll_no

        if best_sim >= self.similarity_threshold:
            return best_roll, best_sim
        return None, 0.0