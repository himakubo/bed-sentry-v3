# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run locally (testing)
python app.py --width 1280 --height 400 --skip 3
# Dashboard: http://localhost:8080

# Deploy to Pi and restart service
bash deploy.sh

# Check service on Pi
sudo systemctl status bed-sentry-v3
sudo journalctl -u bed-sentry-v3 -f
```

## Architecture

Single-patient bed occupancy monitor deployed on Raspberry Pi 4. No person tracking IDs — the system monitors one patient only, always picking the highest-confidence YOLO detection.

**Data flow:** Camera → YOLO inference (every N frames) → StateEngine → Flask API → browser dashboard

**Module responsibilities:**

- `app.py` — Flask entry point. Starts `run_detector()` in a daemon thread, owns the `DetectorState` singleton, exposes API routes. The `state` object is the bridge between detector thread and HTTP handlers.
- `detector.py` — Camera loop. `DetectorState` is a thread-safe container (frame buffer + zone list + `StateEngine` ref). `run_detector()` handles camera reconnection (USB reset on every 3rd failure), YOLO inference with frame skipping, and overlay drawing. Calls `state.engine.update()` every frame.
- `tracker.py` — `StateEngine`: 4-state machine (`ON BED → AT EDGE → GETTING OUT → OFF BED`). Uses debounce (10 frames before committing a new state) and a 5-second grace timer before declaring OFF BED. Zone membership is checked at three points: head `(cx, y1)`, torso `(cx, cy)`, foot `(cx, y2)`.
- `zone_manager.py` — Persists zones to `config.json`. When a bed polygon is saved, auto-generates `edge_polygon` (+80px) and `near_polygon` (+160px) via OpenCV morphological dilation.
- `templates/index.html` — Single-page dashboard. Polls `/api/status` every second; renders MJPEG stream from `/stream`. All zone-drawing UI (click-to-place polygon) is client-side JavaScript.

**Zone hierarchy** (all derived from the single drawn bed polygon):
```
bed polygon  (ON BED)       — green fill
edge polygon (AT EDGE)      — +EDGE_OFFSET px outward, orange outline
near polygon (OFF BED grace) — +NEAR_OFFSET px outward, dark blue outline
```

## Key Tunables

| Constant | Location | Default | Effect |
|---|---|---|---|
| `CONF_THRESHOLD` | `detector.py` | 0.20 | Lower → detect more (more false positives) |
| `DEBOUNCE` | `tracker.py` | 10 frames | Higher → slower but stabler state changes |
| `OFF_BED_GRACE` | `tracker.py` | 5.0 s | Time outside near zone before OFF BED triggers |
| `EDGE_OFFSET` | `zone_manager.py` | 80 px | Width of edge zone around bed |
| `NEAR_OFFSET` | `zone_manager.py` | 160 px | Width of near zone around bed |
| `--skip` | CLI arg | 3 | Run YOLO every N frames; higher = faster video, less accurate |

## Known Issues / Roadmap

- **False AT EDGE** when patient sits at top of bed due to camera angle — state logic needs tuning
- **Nurse proximity detection** (T2 feature) — not yet implemented
- **MQTT alerts** to KuboCare backend — not yet implemented

## Deployment Target

Pi hostname `kcpi2508-l-027`, Tailscale. Deploy script uses `scp` + `systemctl restart`. Full hardware details in `CONTEXT.md`.
