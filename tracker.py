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
from zone_manager import point_in_polygon as pip

OFF_BED_GRACE  = 5.0   # seconds outside near zone → OFF BED
DEBOUNCE       = 10    # frames candidate must hold before committing


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
        self.state       = "ON BED"        # assume on bed at start
        self.state_since = time.time()
        self.grace_start = None
        self._pending    = None            # (candidate, count)
        self.log_entries = deque(maxlen=500)
        self.alerts      = deque(maxlen=200)

    def update(self, person_detected: bool, bbox, zones: list, now: float):
        """Call every frame. bbox = (x1,y1,x2,y2) or None."""
        with self._lock:
            if not person_detected or bbox is None:
                # No person — start grace period
                if self.grace_start is None:
                    self.grace_start = now
                if now - self.grace_start >= OFF_BED_GRACE:
                    self._commit("OFF BED", now)
                return

            # Person detected — reset grace
            self.grace_start = None

            bed_zone = next((z for z in zones if z.get("type") == "bed"), None)
            if not bed_zone:
                return

            x1,y1,x2,y2 = bbox
            cx = (x1+x2)//2
            cy = (y1+y2)//2

            bed_poly  = bed_zone.get("polygon",     [])
            edge_poly = bed_zone.get("edge_polygon", [])
            near_poly = bed_zone.get("near_polygon", [])

            # Use multiple points for robust detection
            # Head = top of box, torso = centre, foot = bottom
            torso_in_bed  = pip(cx, cy, bed_poly)
            head_in_bed   = pip(cx, y1, bed_poly)
            torso_in_edge = pip(cx, cy, edge_poly)
            head_in_edge  = pip(cx, y1, edge_poly)
            torso_in_near = pip(cx, cy, near_poly)
            head_in_near  = pip(cx, y1, near_poly)

            # ON BED: torso OR head inside bed polygon
            # AT EDGE: torso left bed but head still in edge zone
            # GETTING OUT: both torso and head outside bed
            # OFF BED: completely outside near zone

            # Determine candidate state
            if torso_in_bed:
                candidate = "ON BED"
            elif head_in_bed and not torso_in_bed:
                # Leaning — torso moving out but head still over bed
                candidate = "AT EDGE"
            elif torso_in_edge or head_in_edge:
                candidate = "AT EDGE"
            elif not torso_in_near and not head_in_near:
                candidate = "OFF BED"
            else:
                candidate = self.state  # hold in near zone

            # Debounce
            if candidate == self.state:
                self._pending = None
                return

            if self._pending and self._pending[0] == candidate:
                count = self._pending[1] + 1
                if count >= DEBOUNCE:
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
                "patient_state": self.state,
                "alert_level":   level,
                "alert_name":    ["NONE","INFO","WARNING","ALERT","CRITICAL"][level],
                "nurse_nearby":  False,
                "tracks":        [],
                "patient_id":    1,
                "log":           list(self.log_entries)[:100],
                "alerts":        list(self.alerts)[:20],
            }
