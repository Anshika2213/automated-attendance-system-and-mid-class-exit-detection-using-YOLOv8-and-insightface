"""
entry_exit_tracker.py  —  PERFORMANCE-OPTIMISED VERSION
========================================================

Root causes of lag (and what we fix):
  1. InsightFace runs ON the main thread → blocks every frame
     FIX: face ID runs in a background thread via a queue.
          Main thread NEVER waits for it.

  2. InsightFace runs on EVERY frame (30/s)
     FIX: ID requests are throttled to 1 per person per 0.4 s,
          and we skip frames with motion blur (useless for face ID anyway).

  3. Full-res frames sent to YOLO AND InsightFace
     FIX: Detection on a 480p copy; display at original res.

  4. light_enhance.fastNlMeansDenoising is slow (~40ms per frame)
     FIX: Removed from the hot path.  CLAHE only (< 1ms).

  5. identify_attempts capped at 20 → stops trying after 20 frames
     FIX: Keep retrying until roll_no is confirmed OR period ends.
"""
import cv2
import time
import sqlite3
import numpy as np
import os
import threading
import queue
from datetime import datetime
from person_detector import PersonDetector
from face_matcher import FaceMatcher
from database import Database
from period_manager import PeriodManager
from body_reid import BodyReID


def is_blurry(crop, threshold=60.0):
    """Laplacian variance blur check. Blurry frames are useless for face ID."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


# ── IoU Multi-Person Tracker ──────────────────────────────────────────────────
class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_lost_frames=8):
        self.iou_threshold   = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.tracks  = {}
        self.next_id = 1

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        if inter == 0: return 0.0
        a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        return inter / (a1 + a2 - inter)

    def update(self, detections):
        if not detections:
            for tid in list(self.tracks):
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] > self.max_lost_frames:
                    del self.tracks[tid]
            return []

        results      = []
        matched_tids = set()
        matched_dets = set()

        for tid, track in list(self.tracks.items()):
            best_iou, best_idx = self.iou_threshold, -1
            for i, det in enumerate(detections):
                if i in matched_dets: continue
                iou = self._iou(track['bbox'], det)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0:
                self.tracks[tid]['bbox'] = detections[best_idx]
                self.tracks[tid]['lost'] = 0
                matched_tids.add(tid)
                matched_dets.add(best_idx)
                results.append(detections[best_idx] + [tid])

        for tid in self.tracks:
            if tid not in matched_tids:
                self.tracks[tid]['lost'] += 1
                if self.tracks[tid]['lost'] <= self.max_lost_frames:
                    results.append(self.tracks[tid]['bbox'] + [tid])

        dead = [t for t in self.tracks if self.tracks[t]['lost'] > self.max_lost_frames]
        for tid in dead:
            del self.tracks[tid]

        for i, det in enumerate(detections):
            if i not in matched_dets:
                self.tracks[self.next_id] = {'bbox': det, 'lost': 0}
                results.append(det + [self.next_id])
                self.next_id += 1

        return results


# ── Background Face-ID Worker ─────────────────────────────────────────────────
class FaceIDWorker:
    """
    Runs InsightFace in a SEPARATE THREAD so the main loop never blocks.

    Main thread submits face crops → worker thread processes them →
    results sit in output queue → main thread picks them up for free.
    """

    def __init__(self, face_matcher):
        self.face_matcher = face_matcher
        self._in_q   = queue.Queue(maxsize=4)   # drop old jobs if full
        self._out_q  = queue.Queue()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()

    def stop(self):
        self._running = False

    def submit(self, track_id, face_crop):
        """Non-blocking. Silently drops job if queue is full (backpressure)."""
        try:
            self._in_q.put_nowait((track_id, face_crop))
        except queue.Full:
            pass

    def get_results(self):
        """Drain all finished results instantly. Never blocks."""
        results = []
        while True:
            try:
                results.append(self._out_q.get_nowait())
            except queue.Empty:
                break
        return results

    def _loop(self):
        while self._running:
            try:
                track_id, face_crop = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                roll_no, score, name, _ = self.face_matcher.match_face(face_crop)
                self._out_q.put((track_id, roll_no, score, name))
            except Exception:
                self._out_q.put((track_id, None, 0.0, "Unknown"))


# ── Main Tracker ──────────────────────────────────────────────────────────────
class EntryExitTracker:
    def __init__(self):
        print("Initializing Entry/Exit Tracker...")

        self.person_detector = PersonDetector()
        self.face_matcher    = FaceMatcher()
        self.db              = Database()
        self.period_manager  = PeriodManager()
        self.body_reid       = BodyReID(similarity_threshold=0.72)

        # Start background face-ID thread
        self.face_worker = FaceIDWorker(self.face_matcher)
        self.face_worker.start()
        print("✓ Background face-ID thread started")

        self.REVERSE_DIRECTION = True
        self.tracker   = IoUTracker(iou_threshold=0.3, max_lost_frames=8)
        self.door_x    = None
        self.door_zone = 80

        self.track_identities = {}
        self.last_event_time  = {}
        self.cooldown_seconds = 8

        # How often to submit a face-ID request per track (seconds)
        self.id_interval = 0.4

        os.makedirs('data/alerts', exist_ok=True)

        # Pre-build CLAHE object (reused every frame, saves ~0.2ms)
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print("✓ Tracker ready!")
        print(f"  Door zone : +-{self.door_zone}px around centre")
        print(f"  Direction : {'RIGHT to LEFT = ENTRY' if self.REVERSE_DIRECTION else 'LEFT to RIGHT = ENTRY'}")
        print(f"  Face ID   : threaded, interval={self.id_interval}s, blur-filtered")

    # ── Fast CLAHE enhance ────────────────────────────────────────────────────
    def _enhance(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge([self._clahe.apply(l), a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ── Zone helper ───────────────────────────────────────────────────────────
    def get_zone(self, center_x):
        if center_x < self.door_x - self.door_zone:
            return 'LEFT'
        elif center_x > self.door_x + self.door_zone:
            return 'RIGHT'
        else:
            return 'CROSSING'

    # ── Database helpers ──────────────────────────────────────────────────────
    def mark_entry(self, roll_no, name, confidence):
        period_info = self.period_manager.get_period_info()
        if not period_info:
            print(f"\n  No active period")
            return "No active period"

        today        = datetime.now().date().isoformat()
        current_time = datetime.now().strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT status FROM attendance WHERE roll_no=? AND date=? AND period=?',
            (roll_no, today, period_info['period'])
        )
        existing = cursor.fetchone()

        if existing:
            status = existing[0]
            if status == 'PRESENT':
                print(f"\n  Roll-{roll_no} ({name}) already PRESENT")
            elif status == 'TEMP_OUT':
                cursor.execute('SELECT entry_time FROM attendance WHERE roll_no=? AND date=? AND period=?',
                               (roll_no, today, period_info['period']))
                entry_str = cursor.fetchone()[0]
                duration  = int((datetime.strptime(current_time, '%H:%M:%S') -
                                 datetime.strptime(entry_str, '%H:%M:%S')).total_seconds() / 60)
                cursor.execute(
                    "UPDATE attendance SET status='PRESENT', exit_time=NULL, duration_minutes=? "
                    "WHERE roll_no=? AND date=? AND period=?",
                    (duration, roll_no, today, period_info['period'])
                )
                conn.commit()
                print(f"\n RE-ENTRY: Roll-{roll_no} ({name}) | {current_time}")
            elif status in ('ABSENT', 'BUNKED'):
                cursor.execute(
                    "UPDATE attendance SET status='PRESENT', entry_time=? "
                    "WHERE roll_no=? AND date=? AND period=?",
                    (current_time, roll_no, today, period_info['period'])
                )
                conn.commit()
                print(f"\n LATE ENTRY: Roll-{roll_no} ({name}) | {current_time}")
            conn.close()
            return "Updated"

        cursor.execute(
            "INSERT INTO attendance (roll_no, date, period, subject_code, entry_time, status) "
            "VALUES (?,?,?,?,?,'PRESENT')",
            (roll_no, today, period_info['period'], period_info['subject_code'], current_time)
        )
        conn.commit()
        conn.close()
        print(f"\n ENTRY MARKED: Roll-{roll_no} ({name}) | {current_time} | {confidence:.0%}")
        return "Marked PRESENT"

    def mark_exit(self, roll_no, name, confidence):
        period_info = self.period_manager.get_period_info()
        if not period_info:
            return "No active period"

        today        = datetime.now().date().isoformat()
        current_time = datetime.now().strftime("%H:%M:%S")

        conn   = sqlite3.connect(self.db.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE attendance SET status='TEMP_OUT', exit_time=? "
            "WHERE roll_no=? AND date=? AND period=? AND status='PRESENT'",
            (current_time, roll_no, today, period_info['period'])
        )
        if cursor.rowcount > 0:
            cursor.execute('SELECT entry_time FROM attendance WHERE roll_no=? AND date=? AND period=?',
                           (roll_no, today, period_info['period']))
            entry_str = cursor.fetchone()[0]
            duration  = int((datetime.strptime(current_time, '%H:%M:%S') -
                             datetime.strptime(entry_str, '%H:%M:%S')).total_seconds() / 60)
            cursor.execute(
                'UPDATE attendance SET duration_minutes=? WHERE roll_no=? AND date=? AND period=?',
                (duration, roll_no, today, period_info['period'])
            )
            conn.commit()
            print(f"\n EXIT MARKED: Roll-{roll_no} ({name}) | {current_time} | {duration} min")
            result = "Marked TEMP_OUT"
        else:
            result = "Not present / already TEMP_OUT"
        conn.close()
        return result

    # ── Face-covered crossing handler ─────────────────────────────────────────
    def handle_face_covered_crossing(self, frame, bbox, is_entry,
                                     reid_roll_no, reid_name, reid_score, track_id):
        period_info   = self.period_manager.get_period_info()
        event_type    = 'FACE_COVERED_ENTRY' if is_entry else 'FACE_COVERED_EXIT'
        ts            = datetime.now().strftime('%Y%m%d_%H%M%S')
        snapshot_path = f'data/alerts/{event_type}_track{track_id}_{ts}.jpg'
        x1, y1, x2, y2 = bbox

        snap = frame.copy()
        cv2.rectangle(snap, (x1,y1),(x2,y2),(0,0,255),3)
        cv2.putText(snap, "FACE COVERED", (x1,y1-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imwrite(snapshot_path, snap)

        if reid_roll_no:
            print(f"\n FACE COVERED -- body Re-ID -> Roll-{reid_roll_no} ({reid_name}) [sim={reid_score:.2f}]")
            if is_entry:
                self.mark_entry(reid_roll_no, f"{reid_name}[SOFT]", reid_score)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),3)
                cv2.putText(frame, f"ENTRY (body): {reid_name}",
                            (x1,y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,165,255),2)
            else:
                self.mark_exit(reid_roll_no, f"{reid_name}[SOFT]", reid_score)
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),3)
                cv2.putText(frame, f"EXIT (body): {reid_name}",
                            (x1,y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,165,255),2)
            self.db.log_suspicious(event_type, period_info,
                                   reid_roll_no=reid_roll_no, reid_name=reid_name,
                                   reid_score=reid_score, snapshot_path=snapshot_path)
        else:
            print(f"\n SUSPICIOUS {event_type}: unknown person, face covered!")
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),4)
            cv2.putText(frame, "SUSPICIOUS: FACE COVERED",
                        (x1,y1-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
            cv2.rectangle(frame,(0,0),(frame.shape[1],45),(0,0,200),-1)
            cv2.putText(frame, "TEACHER ALERT: UNKNOWN PERSON - FACE COVERED",
                        (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,255,255),2)
            self.db.log_suspicious(event_type, period_info, snapshot_path=snapshot_path)

    # ── Frame processing ──────────────────────────────────────────────────────
    def process_frame(self, frame):
        height, width = frame.shape[:2]

        if self.door_x is None:
            self.door_x = width // 2

        left_zone  = self.door_x - self.door_zone
        right_zone = self.door_x + self.door_zone

        # ── Step 1: pick up finished face-ID results (instant, no waiting) ──
        for (track_id, roll_no, score, name) in self.face_worker.get_results():
            if track_id in self.track_identities:
                state = self.track_identities[track_id]
                state['pending_id'] = False
                if roll_no and score > 0.68:
                    state['roll_no'] = roll_no
                    state['name']    = name
                    state['face_hidden_frames'] = 0
                    state['face_covered']        = False
                    self.body_reid.update_signature(
                        roll_no, frame, state.get('last_bbox', [0,0,1,1]))
                    print(f"  [BG] Track#{track_id} -> Roll-{roll_no} ({name})") # [{score:.2f}] removed this from print statements
                else:
                    state['face_hidden_frames'] = state.get('face_hidden_frames', 0) + 1

        # ── Step 2: enhance (CLAHE only, ~0.5ms) ────────────────────────────
        enhanced = self._enhance(frame)

        # ── Step 3: detect on 480p copy (2x faster YOLO) ────────────────────
        det_scale = min(1.0, 480 / height)
        det_frame = cv2.resize(enhanced, (int(width * det_scale), 480)) if det_scale < 1.0 else enhanced

        people_raw = self.person_detector.detect_people(det_frame)

        # Scale bboxes back to original resolution
        people = []
        for p in people_raw:
            if p.get('confidence', 1.0) < 0.50:
                continue
            x1, y1, x2, y2 = p['bbox']
            if det_scale < 1.0:
                x1 = int(x1/det_scale); y1 = int(y1/det_scale)
                x2 = int(x2/det_scale); y2 = int(y2/det_scale)
            people.append([x1, y1, x2, y2])

        # ── Step 4: NMS ─────────────────────────────────────────────────────
        clean = []
        for b1 in people:
            dup = False
            for b2 in clean:
                xl = max(b1[0],b2[0]); yt = max(b1[1],b2[1])
                xr = min(b1[2],b2[2]); yb = min(b1[3],b2[3])
                if xr > xl and yb > yt:
                    inter = (xr-xl)*(yb-yt)
                    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
                    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
                    if inter/float(a1+a2-inter) > 0.30:
                        dup = True; break
            if not dup:
                clean.append(b1)

        tracked = self.tracker.update(clean)
        events  = []
        now     = time.time()

        # ── Step 5: draw door zone ───────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (left_zone,0),(right_zone,height),(0,0,200),-1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.line(frame, (self.door_x,0),(self.door_x,height),(0,0,255),2)
        cv2.putText(frame, "DOOR", (self.door_x-28,22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
        cv2.putText(frame, f"d/f=move door  door_x={self.door_x}",
                    (10,height-12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200,200,200),1)

        period_info = self.period_manager.get_period_info()
        if period_info:
            cv2.putText(frame, f"P{period_info['period']}: {period_info['subject_name']}",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,255,0),2)
        else:
            cv2.putText(frame, "No Active Period",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,(0,0,255),2)

        # ── Step 6: per-track logic ──────────────────────────────────────────
        for td in tracked:
            x1, y1, x2, y2, track_id = [int(v) for v in td]
            center_x = (x1 + x2) // 2

            if track_id not in self.track_identities:
                self.track_identities[track_id] = {
                    'roll_no':            None,
                    'name':               None,
                    'last_id_request':    0,
                    'pending_id':         False,
                    'zone':               None,
                    'event_fired':        False,
                    'face_hidden_frames': 0,
                    'face_covered':       False,
                    'last_bbox':          [x1, y1, x2, y2],
                }

            state = self.track_identities[track_id]
            state['last_bbox'] = [x1, y1, x2, y2]

            # Submit face-ID to background thread (non-blocking)
            already_confirmed = state['roll_no'] is not None
            time_ok    = (now - state['last_id_request']) > self.id_interval
            not_pending = not state['pending_id']

            if not_pending and time_ok and not already_confirmed:
                # Upper 55% of bounding box = face region
                fy1 = max(0, y1)
                fy2 = min(height, y1 + int((y2-y1) * 0.55))
                fx1, fx2 = max(0, x1), min(width, x2)

                if fx2 > fx1 and fy2 > fy1:
                    face_crop = enhanced[fy1:fy2, fx1:fx2]
                    if not is_blurry(face_crop):
                        # YUV equalise (keeps InsightFace happy in low light)
                        yuv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YUV)
                        yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                        face_crop = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                        self.face_worker.submit(track_id, face_crop)
                        state['pending_id']      = True
                        state['last_id_request'] = now
                    else:
                        state['face_hidden_frames'] += 1

            # Decide face_covered flag
            if state['face_hidden_frames'] >= 10 and not state['roll_no']:
                state['face_covered'] = True
            else:
                state['face_covered'] = False

            roll_no = state['roll_no']
            name    = state['name'] or "Unknown"

            # Zone crossing
            current_zone = self.get_zone(center_x)
            if current_zone != 'CROSSING':
                prev_zone = state['zone']
                if prev_zone is not None and prev_zone != current_zone:
                    if not state['event_fired']:
                        cooldown_key = roll_no if roll_no else f"track_{track_id}"
                        can_trigger  = True
                        if cooldown_key in self.last_event_time:
                            if now - self.last_event_time[cooldown_key] < self.cooldown_seconds:
                                can_trigger = False

                        if can_trigger:
                            if self.REVERSE_DIRECTION:
                                is_entry = (prev_zone=='RIGHT' and current_zone=='LEFT')
                                is_exit  = (prev_zone=='LEFT'  and current_zone=='RIGHT')
                            else:
                                is_entry = (prev_zone=='LEFT'  and current_zone=='RIGHT')
                                is_exit  = (prev_zone=='RIGHT' and current_zone=='LEFT')

                            print(f"\n  ZONE SWITCH Track#{track_id} ({name}): "
                                  f"{prev_zone}->{current_zone} | "
                                  f"entry={is_entry} face_covered={state['face_covered']}")

                            if not state['face_covered'] and roll_no:
                                if is_entry:
                                    self.mark_entry(roll_no, name, 0.9)
                                    events.append(f"ENTRY: {name}")
                                    self.last_event_time[cooldown_key] = now
                                    state['event_fired'] = True
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),4)
                                    cv2.putText(frame, f"ENTRY: {name}",
                                                (x1,y1-12),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
                                elif is_exit:
                                    self.mark_exit(roll_no, name, 0.9)
                                    events.append(f"EXIT: {name}")
                                    self.last_event_time[cooldown_key] = now
                                    state['event_fired'] = True
                                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,255),4)
                                    cv2.putText(frame, f"EXIT: {name}",
                                                (x1,y1-12),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
                            else:
                                reid_roll, reid_score = self.body_reid.find_match(
                                    frame, [x1,y1,x2,y2])
                                reid_name = None
                                if reid_roll:
                                    s = self.db.get_student(reid_roll)
                                    reid_name = s['name'] if s else f"Roll-{reid_roll}"
                                if is_entry or is_exit:
                                    self.handle_face_covered_crossing(
                                        frame, [x1,y1,x2,y2],
                                        is_entry=is_entry,
                                        reid_roll_no=reid_roll, reid_name=reid_name,
                                        reid_score=reid_score, track_id=track_id
                                    )
                                    events.append(
                                        f"{'ENTRY' if is_entry else 'EXIT'}: "
                                        f"{'SOFT:'+str(reid_name) if reid_roll else 'SUSPICIOUS'}"
                                    )
                                    self.last_event_time[cooldown_key] = now
                                    state['event_fired'] = True

                state['zone'] = current_zone
                if state['event_fired'] and prev_zone == current_zone:
                    state['event_fired'] = False

            # Draw bounding box
            if state['face_covered']:
                color = (0, 0, 255)
            elif current_zone == 'CROSSING':
                color = (0, 165, 255)
            elif roll_no:
                color = (0, 255, 0)
            else:
                color = (128, 128, 128)

            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            zone_char   = 'X' if current_zone=='CROSSING' else ('<' if current_zone=='LEFT' else '>')
            fc_tag      = " [FACE?]" if state['face_covered'] else ""
            pending_tag = " ..." if state['pending_id'] else ""
            label = f"#{track_id}:{name}{fc_tag}{pending_tag} {zone_char}"
            cv2.putText(frame, label, (x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            if not roll_no and not state['face_covered']:
                cv2.putText(frame, "identifying...",
                            (x1,y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(200,200,0),1)
            elif state['face_covered']:
                cv2.putText(frame, "Show face to camera",
                            (x1,y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(0,0,255),1)

        cv2.putText(frame, f"People: {len(tracked)}", (width-140,28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,(255,255,255),2)

        return frame, events

    # ── Run loop ──────────────────────────────────────────────────────────────
    def run(self, duration=3600, camera_index=1):
        period_info = self.period_manager.get_period_info()
        if not period_info:
            print("\n  No active period! Start a period first.")
            return

        print(f"\n Tracking Period {period_info['period']}: {period_info['subject_name']}")
        print(f"  Camera : {camera_index}")
        print(f"  Keys   : q=quit  d=door right  f=door left\n")

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"  Camera {camera_index} failed, trying 0...")
            cap = cv2.VideoCapture(1)
            if not cap.isOpened():
                print("  No camera found"); return

        # Force lower resolution for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # always get the freshest frame

        start_time = time.time()
        cv2.namedWindow('Entry/Exit Tracking', cv2.WINDOW_NORMAL)

        fps_t = time.time(); fps_c = 0; fps_v = 0.0

        try:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.02); continue

                processed_frame, events = self.process_frame(frame)

                # FPS counter overlay
                fps_c += 1
                if time.time() - fps_t >= 1.0:
                    fps_v = fps_c / (time.time() - fps_t)
                    fps_c = 0; fps_t = time.time()
                cv2.putText(processed_frame, f"FPS:{fps_v:.0f}",
                            (processed_frame.shape[1]-70, processed_frame.shape[0]-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1)

                cv2.imshow('Entry/Exit Tracking', processed_frame)
                cv2.imwrite('data/live_feed.jpg', processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n Stopped"); break
                elif key == ord('d'):
                    self.door_x = min(self.door_x+10, frame.shape[1]-10)
                    print(f"  door_x -> {self.door_x}")
                elif key == ord('f'):
                    self.door_x = max(self.door_x-10, 10)
                    print(f"  door_x -> {self.door_x}")

        except KeyboardInterrupt:
            print("\n Stopped (Ctrl+C)")
        finally:
            self.face_worker.stop()
            cap.release()
            cv2.destroyAllWindows()
            print("\n Tracking stopped")


if __name__ == "__main__":
    tracker = EntryExitTracker()
    tracker.period_manager.show_schedule()
    tracker.period_manager.start_period(1)
    tracker.run(duration=3600, camera_index=1)