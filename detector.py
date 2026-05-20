"""
detector.py — Bed-Sentry v3
Simple YOLO detection, no tracking, no IDs.
Uses StateEngine from tracker.py.
"""

import cv2
import threading
import time
import numpy as np
from collections import deque
from ultralytics import YOLO
from zone_manager import load_zones, get_zone_color_bgr
from tracker import StateEngine

PERSON_CLASS     = 0
CONF_THRESHOLD   = 0.15
RECONNECT_DELAY  = 3
MAX_FAIL_FRAMES  = 10
EDGE_COLOR       = (0, 140, 255)
NEAR_COLOR       = (80,  40,   0)
LYING_FLAT_RATIO = 1.8


class DetectorState:
    def __init__(self):
        self._lock        = threading.Lock()
        self._zone_lock   = threading.Lock()
        self.frame_jpg    = None
        self.running      = False
        self.fps          = 0.0
        self.cam_status   = "CONNECTING"
        self._zones       = load_zones()
        self.engine       = StateEngine()
        self.motion_score = 0.0
        self.test_mode    = False
        self.test_log     = deque(maxlen=200)
        self.last_state_text       = ""
        self.last_state_time       = 0.0
        self._prev_displayed_state = "ON BED"

    def get_zones(self):
        with self._zone_lock:
            return list(self._zones)

    def reload_zones(self):
        zones = load_zones()
        with self._zone_lock:
            self._zones = zones
        print(f"[Detector] Zones reloaded: {[z.get('label') for z in zones]}")

    def set_frame(self, jpg):
        with self._lock:
            self.frame_jpg = jpg

    def get_frame(self):
        with self._lock:
            return self.frame_jpg

    def get_snapshot(self):
        snap  = self.engine.get_snapshot()
        zones = self.get_zones()
        with self._lock:
            snap["fps"]          = round(self.fps, 1)
            snap["cam_status"]   = self.cam_status
            snap["motion_score"] = round(self.motion_score)
        snap["zones"] = [
            {"id": z.get("id"), "label": z.get("label"),
             "color": z.get("color"), "type": z.get("type", "bed"),
             "polygon": z.get("polygon", []),
             "edge_polygon": z.get("edge_polygon", []),
             "edge_manual": z.get("edge_manual", False)}
            for z in zones
        ]
        snap["zone_count"] = len(zones)
        return snap


def compute_motion_score(frame, prev_frame, bed_polygon) -> float:
    """Pixel-count of changed areas within the bed polygon (frame differencing)."""
    if prev_frame is None or frame.shape != prev_frame.shape:
        return 0.0
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    if bed_polygon and len(bed_polygon) >= 3:
        pts = np.array([[p["x"], p["y"]] for p in bed_polygon], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    else:
        mask[:] = 255
    gray_cur  = cv2.cvtColor(frame,      cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_cur, gray_prev)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    return float(np.sum(diff > 25))


def draw_zones(frame, zones):
    for zone in zones:
        if zone.get("type") != "bed":
            continue
        color = tuple(get_zone_color_bgr(zone.get("color", "green")))
        label = zone.get("label", "Zone")

        near = zone.get("near_polygon", [])
        if len(near) >= 3:
            pts = np.array([[p["x"], p["y"]] for p in near], dtype=np.int32)
            cv2.polylines(frame, [pts], True, NEAR_COLOR, 1, cv2.LINE_AA)

        edge = zone.get("edge_polygon", [])
        if len(edge) >= 3:
            pts = np.array([[p["x"], p["y"]] for p in edge], dtype=np.int32)
            cv2.polylines(frame, [pts], True, EDGE_COLOR, 2, cv2.LINE_AA)

        bed = zone.get("polygon", [])
        if len(bed) >= 3:
            pts = np.array([[p["x"], p["y"]] for p in bed], dtype=np.int32)
            ov  = frame.copy()
            cv2.fillPoly(ov, [pts], color)
            cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [pts], True, color, 2)
            cv2.putText(frame, label,
                        (bed[0]["x"]+6, bed[0]["y"]+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def open_camera(index=0, width=1280, height=400):
    cap = cv2.VideoCapture(f"/dev/video{index}", cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def usb_reset(index=0):
    import glob
    paths = glob.glob(f"/sys/class/video4linux/video{index}/device/../../../")
    if paths:
        bus_id = paths[0].rstrip('/').split('/')[-1]
        try:
            with open('/sys/bus/usb/drivers/usb/unbind','w') as f: f.write(bus_id)
            time.sleep(2)
            with open('/sys/bus/usb/drivers/usb/bind','w') as f: f.write(bus_id)
            time.sleep(2)
        except: pass


def run_detector(state: DetectorState, model_name="yolov8n",
                 camera_index=0, frame_skip=2, width=1280, height=400):

    print(f"[Detector] Loading model ...")
    try:
        model = YOLO("yolov8s.pt")
        print("[Detector] Using yolov8s model")
    except Exception:
        model = YOLO("yolov8n.pt")
        print("[Detector] Fallback to yolov8n model")
    print("[Detector] Model ready.")

    state.running   = True
    fail_count      = 0
    reconnect_count = 0
    cap             = None
    best_bbox          = None
    best_conf          = 0.0
    person_detected    = False
    prev_frame         = None
    motion_score       = 0.0
    motion_edge_score  = 0.0
    last_snap_time     = 0.0

    def open_cap():
        nonlocal cap
        if cap: cap.release()
        import subprocess
        subprocess.run(['sudo','chmod','666',f'/dev/video{camera_index}'],
                       capture_output=True)
        cap = open_camera(camera_index, width, height)
        return cap.isOpened()

    while state.running and not open_cap():
        state.cam_status = "CONNECTING"
        time.sleep(RECONNECT_DELAY)

    print(f"[Detector] Camera {width}x{height} OK")
    state.cam_status = "OK"
    prev_time   = time.time()
    frame_count = 0

    try:
        while state.running:
            ret, frame = cap.read()

            if not ret:
                fail_count += 1
                if fail_count >= MAX_FAIL_FRAMES:
                    reconnect_count += 1
                    state.cam_status = "RECONNECTING"
                    cap.release()
                    if reconnect_count % 3 == 0:
                        usb_reset(camera_index)
                    time.sleep(RECONNECT_DELAY)
                    import subprocess
                    subprocess.run(['sudo','chmod','666',f'/dev/video{camera_index}'],
                                   capture_output=True)
                    cap = open_camera(camera_index, width, height)
                    if cap.isOpened():
                        state.cam_status = "OK"
                        fail_count = 0
                    else:
                        fail_count = MAX_FAIL_FRAMES
                continue
            else:
                fail_count = 0
                state.cam_status = "OK"

            frame_count += 1
            now       = time.time()
            state.fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            zones     = state.get_zones()

            # Motion scores — bed zone and edge zone, computed every frame
            bed_zone          = next((z for z in zones if z.get("type") == "bed"), None)
            bed_poly          = bed_zone.get("polygon",      []) if bed_zone else []
            edge_poly         = bed_zone.get("edge_polygon", []) if bed_zone else []
            motion_score      = compute_motion_score(frame, prev_frame, bed_poly)
            motion_edge_score = compute_motion_score(frame, prev_frame, edge_poly)
            prev_frame        = frame.copy()
            state.motion_score = motion_score

            # YOLO inference every N frames
            if frame_count % frame_skip == 0:
                results = model(frame, classes=[PERSON_CLASS],
                                conf=CONF_THRESHOLD, verbose=False)
                if results and results[0].boxes:
                    best = max(results[0].boxes, key=lambda b: float(b.conf[0]))
                    best_bbox = tuple(map(int, best.xyxy[0]))
                    best_conf = float(best.conf[0])
                    person_detected = True
                else:
                    person_detected = False
                    best_bbox = None
                    best_conf = 0.0
                    # Fallback: lower conf on downscaled frame
                    small = cv2.resize(frame, (640, 200))
                    results2 = model(small, classes=[PERSON_CLASS],
                                     conf=0.10, verbose=False)
                    if results2 and results2[0].boxes:
                        best2 = max(results2[0].boxes, key=lambda b: float(b.conf[0]))
                        scaleX = frame.shape[1] / 640
                        scaleY = frame.shape[0] / 200
                        x1, y1, x2, y2 = map(int, best2.xyxy[0])
                        best_bbox = (int(x1*scaleX), int(y1*scaleY),
                                     int(x2*scaleX), int(y2*scaleY))
                        best_conf = float(best2.conf[0])
                        person_detected = True
                    # Third attempt: larger input size for lying/unusual poses
                    if not person_detected:
                        large = cv2.resize(frame, (960, 300))
                        results3 = model(large, classes=[PERSON_CLASS],
                                         conf=0.08, verbose=False, imgsz=960)
                        if results3 and results3[0].boxes:
                            best3  = max(results3[0].boxes, key=lambda b: float(b.conf[0]))
                            scaleX = frame.shape[1] / 960
                            scaleY = frame.shape[0] / 300
                            x1, y1, x2, y2 = map(int, best3.xyxy[0])
                            best_bbox = (int(x1*scaleX), int(y1*scaleY),
                                         int(x2*scaleX), int(y2*scaleY))
                            best_conf = float(best3.conf[0])
                            person_detected = True

            # Update state engine with motion scores
            state.engine.update(person_detected, best_bbox, zones, now,
                                motion_score=motion_score,
                                motion_edge_score=motion_edge_score)

            # State transition flash overlay tracking
            current_state = state.engine.state
            if current_state != state._prev_displayed_state:
                state.last_state_text = (
                    f"{state._prev_displayed_state} -> {current_state}"
                )
                state.last_state_time       = now
                state._prev_displayed_state = current_state

            # Test mode: snapshot every 2 s
            if state.test_mode and (now - last_snap_time) >= 2.0:
                last_snap_time = now
                state.test_log.append({
                    "t":      time.strftime("%H:%M:%S"),
                    "state":  state.engine.state,
                    "motion": round(motion_score),
                    "debug":  dict(state.engine._debug),
                })

            # Draw zones
            draw_zones(frame, zones)

            # Draw person detection
            if person_detected and best_bbox:
                x1, y1, x2, y2 = best_bbox
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                w  = x2 - x1
                h  = y2 - y1
                ar = w / max(h, 1)
                s  = state.engine.state
                COLS = {
                    "ON BED":      (0, 200,  60),
                    "AT EDGE":     (0, 140, 255),
                    "GETTING OUT": (0,  60, 255),
                    "OFF BED":     (0,   0, 220),
                }
                col = COLS.get(s, (0, 200, 60))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3)
                cv2.putText(frame, s, (x1, y1-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                # Posture + conf labels
                posture  = "FLAT" if ar >= 1.8 else "SIT" if ar >= 0.8 else "STAND"
                cv2.putText(frame, f"AR:{ar:.1f} {posture}", (x1, y2 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                cv2.putText(frame, f"CONF:{best_conf:.2f}", (x1, y2 + 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
                ov_val = state.engine._debug.get("bbox_overlap", 0.0)
                cv2.putText(frame, f"OV:{ov_val:.0%}", (x1, y2 + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                # 3 detection dots: head (yellow), torso (white), foot (cyan)
                cv2.circle(frame, (cx, y1), 6, (255, 200,   0), -1)
                cv2.circle(frame, (cx, cy), 6, (255, 255, 255), -1)
                cv2.circle(frame, (cx, y2), 6, (  0, 230, 230), -1)

            # State transition flash overlay (shown for 2 s after each transition)
            if state.last_state_text and (now - state.last_state_time) < 2.0:
                fh, fw = frame.shape[:2]
                txt  = state.last_state_text
                ts   = cv2.FONT_HERSHEY_DUPLEX
                (tw, th), _ = cv2.getTextSize(txt, ts, 1.0, 2)
                tx   = (fw - tw) // 2
                ty   = fh // 2
                cv2.rectangle(frame, (tx - 10, ty - th - 10),
                              (tx + tw + 10, ty + 10), (0, 0, 0), -1)
                cv2.putText(frame, txt, (tx, ty),
                            ts, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

            # Banner
            fh, fw = frame.shape[:2]
            s  = state.engine.state
            lv = state.engine.ALERT_LEVELS.get(s, 0)
            BCOLS = {0:(0,200,60), 2:(0,140,255), 3:(0,60,255), 4:(0,0,220)}
            bc = BCOLS.get(lv, (0,200,60))
            cv2.rectangle(frame, (0,0), (fw,50), (12,12,18), -1)
            cv2.putText(frame, f"  {s}", (4, 34),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, bc, 2, cv2.LINE_AA)
            mot_k  = int(motion_score / 1000)
            me_k   = int(motion_edge_score / 1000)
            cv2.putText(frame, f"MOT:{mot_k}k", (fw-360, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1, cv2.LINE_AA)
            cv2.putText(frame, f"ME:{me_k}k", (fw-440, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120,120,120), 1, cv2.LINE_AA)
            cv2.putText(frame, f"FPS:{state.fps:.0f}", (fw-90, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160,160,160), 1, cv2.LINE_AA)
            cc = (0,200,60) if state.cam_status=="OK" else (0,80,220)
            cv2.putText(frame, f"CAM:{state.cam_status}", (fw-230, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cc, 1, cv2.LINE_AA)

            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            state.set_frame(jpg.tobytes())

    except Exception as e:
        print(f"[Detector] Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        if cap: cap.release()
        state.running = False
        print("[Detector] Stopped.")
