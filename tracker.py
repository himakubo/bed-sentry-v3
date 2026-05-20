"""
tracker.py — Bed-Sentry v3
Simple state engine for single patient monitoring.
No IDs, no tracking. Just: person detected → check zone → state.
States: ON BED | AT EDGE | GETTING OUT | OFF BED
"""

import time
import threading
from datetime import datetime
from collections import deque
import numpy as np
from zone_manager import point_in_polygon as pip

GRACE_SAFE             =  8.0  # seconds outside near zone → OFF BED from ON BED  # increase to 30s for night deployment
GRACE_AT_RISK          =  5.0  # seconds when last state was AT EDGE or GETTING OUT
DEBOUNCE = {                   # frames candidate must hold before committing (per target state)
    "ON BED":      10,
    "AT EDGE":      4,
    "GETTING OUT":  3,
    "OFF BED":      3,
}
LYING_FLAT_RATIO       = 1.6   # bbox w/h above this → person lying flat → suppress AT EDGE
MOTION_HOLDBACK_THRESH = 200   # if motion_score < this and ON BED, skip grace timer

print(f"[Tracker] GRACE_SAFE={GRACE_SAFE}s  GRACE_AT_RISK={GRACE_AT_RISK}s  HOLDBACK_THRESH={MOTION_HOLDBACK_THRESH}")


def bbox_overlap_ratio(bbox, polygon) -> float:
    """Fraction of the bbox area that overlaps with the polygon's axis-aligned bounding box."""
    if not polygon or len(polygon) < 3:
        return 0.0
    x1, y1, x2, y2 = bbox
    pts = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.int32)
    bx  = int(pts[:, 0].min())
    by  = int(pts[:, 1].min())
    bw  = int(pts[:, 0].max()) - bx
    bh  = int(pts[:, 1].max()) - by
    ix1 = max(x1, bx);  iy1 = max(y1, by)
    ix2 = min(x2, bx + bw); iy2 = min(y2, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    bbox_area    = max((x2 - x1) * (y2 - y1), 1)
    return intersection / bbox_area


class StateEngine:
    """
    Single patient state machine.
    Call update() every frame with detection result.
    """

    STATES = ["ON BED", "AT EDGE", "GETTING OUT", "OFF BED"]

    ALERT_LEVELS = {
        "ON BED":      0,
        "AT EDGE":     2,
        "GETTING OUT": 3,
        "OFF BED":     4,
    }

    STATE_COLORS = {
        "ON BED":      (0, 200,  60),
        "AT EDGE":     (0, 140, 255),
        "GETTING OUT": (0,  60, 255),
        "OFF BED":     (0,   0, 220),
    }

    def __init__(self):
        self._lock       = threading.Lock()
        self.state       = "ON BED"
        self.state_since = time.time()
        self.grace_start = None
        self._pending    = None            # (candidate, count)
        self.log_entries = deque(maxlen=500)
        self.alerts      = deque(maxlen=200)
        self._debug      = {}

    def _grace_needed(self) -> float:
        return GRACE_SAFE if self.state == "ON BED" else GRACE_AT_RISK

    def update(self, person_detected: bool, bbox, zones: list, now: float,
               motion_score: float = 0.0, motion_edge_score: float = 0.0):
        """Call every frame. bbox = (x1,y1,x2,y2) or None."""
        with self._lock:
            debug = {"motion_score": round(motion_score),
                     "motion_edge_score": round(motion_edge_score)}

            if not person_detected or bbox is None:
                # Edge motion holdback: patient still moving at edge → reset grace
                if (self.state == "AT EDGE"
                        and motion_edge_score > MOTION_HOLDBACK_THRESH):
                    self.grace_start = None
                    self._debug = debug
                    return

                # Bed motion holdback: suppress grace timer only when patient was visibly in bed last frame
                if (self.state == "ON BED"
                        and motion_score < MOTION_HOLDBACK_THRESH
                        and self._debug.get("torso_in_bed", False)):
                    self._debug = debug
                    return

                if self.grace_start is None:
                    self.grace_start = now
                grace_elapsed = now - self.grace_start
                # Fast grace: both zones quiet when already at risk → 3 s timeout
                if (self.state in ("AT EDGE", "GETTING OUT")
                        and motion_score      < MOTION_HOLDBACK_THRESH
                        and motion_edge_score < MOTION_HOLDBACK_THRESH):
                    grace_needed = 3.0
                else:
                    grace_needed = self._grace_needed()
                debug["grace_elapsed"] = round(grace_elapsed, 1)
                debug["grace_needed"]  = grace_needed
                self._debug = debug
                if grace_elapsed >= grace_needed:
                    self._commit("OFF BED", now)
                return

            # Person detected — cancel grace
            self.grace_start = None

            bed_zone = next((z for z in zones if z.get("type") == "bed"), None)
            if not bed_zone:
                self._debug = debug
                return

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w  = x2 - x1
            h  = y2 - y1
            ar = w / max(h, 1)
            lying_flat = ar >= LYING_FLAT_RATIO

            bed_poly  = bed_zone.get("polygon",      [])
            edge_poly = bed_zone.get("edge_polygon",  [])
            near_poly = bed_zone.get("near_polygon",  [])

            torso_in_bed  = pip(cx, cy, bed_poly)
            head_in_bed   = pip(cx, y1, bed_poly)
            foot_in_bed   = pip(cx, y2, bed_poly)
            torso_in_edge = pip(cx, cy, edge_poly)
            head_in_edge  = pip(cx, y1, edge_poly)
            torso_in_near = pip(cx, cy, near_poly)
            head_in_near  = pip(cx, y1, near_poly)

            overlap = bbox_overlap_ratio(bbox, bed_poly)

            debug.update({
                "ar":            round(ar, 2),
                "lying_flat":    lying_flat,
                "torso_in_bed":  torso_in_bed,
                "head_in_bed":   head_in_bed,
                "foot_in_bed":   foot_in_bed,
                "torso_in_edge": torso_in_edge,
                "head_in_edge":  head_in_edge,
                "torso_in_near": torso_in_near,
                "head_in_near":  head_in_near,
                "bbox_overlap":  round(overlap, 2),
            })

            # Fast-path: person detected but outside all zones → OFF BED immediately (3-frame debounce)
            if (not torso_in_near and not head_in_near
                    and not torso_in_bed and not head_in_bed
                    and overlap < 0.40):
                if self._pending and self._pending[0] == "OFF BED":
                    count = self._pending[1] + 1
                    if count >= 3:
                        self._commit("OFF BED", now)
                    else:
                        self._pending = ("OFF BED", count)
                else:
                    self._pending = ("OFF BED", 1)
                debug["candidate"] = "OFF BED"
                self._debug = debug
                return

            # Candidate state logic
            if torso_in_bed or overlap >= 0.30:
                candidate = "ON BED"

            elif lying_flat and (torso_in_edge or head_in_edge):
                candidate = "AT EDGE"

            elif not torso_in_bed and not lying_flat:
                if torso_in_edge or head_in_edge:
                    candidate = "GETTING OUT" if not foot_in_bed else "AT EDGE"
                elif torso_in_near or head_in_near:
                    candidate = "AT EDGE"
                else:
                    candidate = "OFF BED"

            else:
                candidate = self.state

            debug["candidate"] = candidate
            self._debug = debug

            if candidate == self.state:
                self._pending = None
                return

            # Force state chain: ON BED → AT EDGE → GETTING OUT (no skipping)
            if self.state == "ON BED" and candidate == "GETTING OUT":
                candidate = "AT EDGE"

            # Asymmetric debounce: harder to recover to ON BED from a risky state
            if candidate == "ON BED" and self.state in ("AT EDGE", "GETTING OUT"):
                required = 15
            else:
                required = DEBOUNCE.get(candidate, 10)

            if self._pending and self._pending[0] == candidate:
                count = self._pending[1] + 1
                if count >= required:
                    self._commit(candidate, now)
                else:
                    self._pending = (candidate, count)
            else:
                self._pending = (candidate, 1)

    def _commit(self, new_state: str, now: float):
        if new_state == self.state:
            return
        prev = self.state
        self.state       = new_state
        self.state_since = now
        self._pending    = None
        self.grace_start = None

        ts    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = self.ALERT_LEVELS.get(new_state, 0)
        msg   = new_state

        self.alerts.appendleft({
            "level": level, "level_name": ["NONE","INFO","WARNING","ALERT","CRITICAL"][level],
            "message": msg, "time": ts, "acknowledged": False,
        })
        self.log_entries.appendleft({
            "time": ts, "prev_state": prev, "new_state": new_state,
            "message": msg, "level": level,
            "level_name": ["NONE","INFO","WARNING","ALERT","CRITICAL"][level],
            "zone": "", "role": "patient",
        })
        print(f"[{ts}] {prev} → {new_state}")

    def acknowledge_alert(self):
        with self._lock:
            for a in list(self.alerts)[:5]:
                a["acknowledged"] = True

    def get_snapshot(self) -> dict:
        with self._lock:
            level = self.ALERT_LEVELS.get(self.state, 0)
            return {
                "patient_state":    self.state,
                "state_duration_s": round(time.time() - self.state_since, 1),
                "alert_level":      level,
                "alert_name":       ["NONE","INFO","WARNING","ALERT","CRITICAL"][level],
                "nurse_nearby":     False,
                "tracks":           [],
                "patient_id":       1,
                "log":              list(self.log_entries)[:100],
                "alerts":           list(self.alerts)[:20],
                "debug":            dict(self._debug),
            }
