# Bed-Sentry v3 Context

## Hardware
- Pi: pi@kcpi2508-l-027, Tailscale IP: 100.83.213.24
- Camera: Jieli USB, /dev/video0, 1280x400 MJPEG
- Service: bed-sentry-v3.service, port 8080
- Venv: /home/pi/bed-sentry/venv/

## Architecture
- detector.py: YOLO detection, no tracking, picks highest confidence box
- tracker.py: StateEngine, 4 states: ON BED, AT EDGE, GETTING OUT, OFF BED
- zone_manager.py: draws 3 zones from one polygon (bed/edge/near)
- app.py: Flask server, API routes
- templates/index.html: dashboard UI

## Current Issues
- False AT EDGE alerts when patient sits at top of bed (camera angle)

## Next Tasks
- Fix false AT EDGE alerts
- Nurse proximity detection (T2)
- MQTT alerts to KuboCare backend
