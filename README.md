# Bed-Sentry v3 🛏️

Bed occupancy monitoring system for care homes using Raspberry Pi 4, YOLOv8, and a USB camera. Detects patient states in real time and raises alerts when a patient is at risk of falling out of bed.

---

## How It Works

```
Camera → YOLOv8 detects person → check torso position against zones → state machine → alerts
```

No tracking IDs. No ByteTrack. Just simple, reliable single-patient monitoring.

### States

| State | Colour | Alert | Description |
|-------|--------|-------|-------------|
| ON BED | 🟢 Green | None | Patient torso inside bed zone |
| AT EDGE | 🟠 Orange | Warning | Patient torso in edge zone |
| GETTING OUT | 🔴 Red | Alert | Patient torso in edge, foot outside |
| OFF BED | 🔴 Dark Red | Critical | Patient outside near zone for 5s |

### Zone System

When you draw a bed zone, three zones are auto-generated:

```
BED polygon     → green fill  → ON BED
EDGE polygon    → orange line → AT EDGE / GETTING OUT  (+60px outward)
NEAR polygon    → blue line   → OFF BED grace area     (+160px outward)
```

---

## Hardware Requirements

- Raspberry Pi 4 (4GB RAM recommended)
- USB camera (tested with Jieli MJPEG camera at 1280x400)
- Camera mounted above/beside bed at 45° angle
- Network connection (WiFi or Ethernet)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/himakubo/bed-sentry-v3.git
cd bed-sentry-v3
git checkout v3-state-engine
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Install dependencies

```bash
pip install ultralytics flask opencv-python-headless numpy
```

### 4. Run manually (for testing)

```bash
source venv/bin/activate
python app.py --width 1280 --height 400 --skip 3
```

Open browser: `http://<pi-ip>:8080`

### 5. Run as systemd service (auto-start on boot)

```bash
sudo cp bed-sentry-v3.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bed-sentry-v3
sudo systemctl start bed-sentry-v3
```

Check status:
```bash
sudo systemctl status bed-sentry-v3
sudo journalctl -u bed-sentry-v3 -f
```

---

## First Time Use

### Step 1 — Draw Bed Zone
1. Open dashboard in browser
2. Click **✏️ Draw Zone**
3. Click 4 corners of the bed mattress on the video
4. Click point 1 again to close the polygon
5. Click **Save Zone**
6. Green bed zone appears with orange edge and blue near zones

### Step 2 — Watch States
| Action | Expected State |
|--------|---------------|
| Sit/lie on mattress | ON BED |
| Move to bed edge | AT EDGE |
| Stand up, lean on bed | GETTING OUT |
| Walk away 5 seconds | OFF BED |
| Return to bed | ON BED |

### Step 3 — Acknowledge Alerts
- Click **✓ Acknowledge** on red popup to silence alarm
- Event log records all state changes with timestamps

---

## Configuration

Edit `zone_manager.py` to adjust zone sizes:

```python
EDGE_OFFSET = 60    # pixels outward from bed polygon for edge zone
NEAR_OFFSET = 160   # pixels outward from bed polygon for near zone
```

Edit `tracker.py` to adjust timing:

```python
OFF_BED_GRACE  = 5.0    # seconds outside near zone before OFF BED alert
DEBOUNCE       = 10     # frames new state must hold before committing
```

Edit service file to adjust camera settings:

```
--width 1280    # camera width
--height 400    # camera height
--skip 3        # run YOLO every N frames (higher = faster video, less accurate)
```

---

## File Structure

```
bed-sentry-v3/
├── app.py              # Flask server, API routes
├── detector.py         # Camera loop, YOLO inference, drawing
├── tracker.py          # State engine (ON BED → AT EDGE → OFF BED)
├── zone_manager.py     # Zone save/load, polygon expansion
├── templates/
│   └── index.html      # Dashboard UI
├── bed-sentry-v3.service  # systemd service file
└── requirements.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard |
| `/stream` | GET | MJPEG video stream |
| `/api/status` | GET | Current state snapshot |
| `/api/zones` | POST | Save zones |
| `/api/zones/<id>` | DELETE | Delete a zone |
| `/api/acknowledge` | POST | Acknowledge alert |

---

## Troubleshooting

**Video frozen / low FPS**
- Increase `--skip` value in service file (try `--skip 4` or `--skip 5`)
- Check camera with `v4l2-ctl --list-devices`

**Zone not showing after save**
- Hard refresh browser `Ctrl+Shift+R`
- Check `config.json` exists in project folder

**Person not detected**
- Lower confidence: change `CONF_THRESHOLD = 0.20` in `detector.py`
- Ensure full/partial body visible in frame
- Check lighting conditions

**State not changing to AT EDGE**
- Redraw zone smaller — only cover mattress surface, not bed frame
- Move torso clearly into orange zone area
- Adjust `EDGE_OFFSET` larger in `zone_manager.py`

---

## Built With

- [YOLOv8](https://github.com/ultralytics/ultralytics) — person detection
- [OpenCV](https://opencv.org/) — video capture and drawing
- [Flask](https://flask.palletsprojects.com/) — web dashboard
- Raspberry Pi 4 — edge deployment

---

*Built for KuboCare — senior care technology*
