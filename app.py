"""
app.py — Bed-Sentry v3 Flask Server
"""
import argparse, threading, time
from flask import Flask, Response, jsonify, request, render_template
from detector import DetectorState, run_detector
from zone_manager import save_zones, validate_zone, ZONE_COLORS

app   = Flask(__name__)
state = DetectorState()

def generate_stream():
    while True:
        frame = state.get_frame()
        if frame:
            yield (b'--FRAME\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.033)

@app.route('/stream')
def video_stream():
    return Response(generate_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=FRAME')

@app.route('/api/status')
def api_status():
    return jsonify(state.get_snapshot())

@app.route('/api/zones', methods=['POST'])
def api_save_zones():
    data  = request.get_json(force=True)
    zones = data.get("zones", [])
    if not zones:
        return jsonify({"ok": False, "error": "No zones"}), 400
    for z in zones:
        if not validate_zone(z):
            return jsonify({"ok": False, "error": f"Invalid zone"}), 400
    save_zones(zones)
    state.reload_zones()
    return jsonify({"ok": True})

@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def api_delete_zone(zone_id):
    current = state.get_zones()
    updated = [z for z in current if z.get("id") != zone_id]
    save_zones(updated)
    state.reload_zones()
    return jsonify({"ok": True})

@app.route('/api/acknowledge', methods=['POST'])
def api_acknowledge():
    state.engine.acknowledge_alert()
    return jsonify({"ok": True})

@app.route('/')
def index():
    return render_template('index.html')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--model',  type=str, default='yolov8n')
    parser.add_argument('--skip',   type=int, default=3)
    parser.add_argument('--width',  type=int, default=1280)
    parser.add_argument('--height', type=int, default=400)
    parser.add_argument('--port',   type=int, default=8080)
    args = parser.parse_args()

    t = threading.Thread(
        target=run_detector,
        kwargs=dict(state=state, model_name=args.model,
                    camera_index=args.camera, frame_skip=args.skip,
                    width=args.width, height=args.height),
        daemon=True
    )
    t.start()
    print(f"\n=== Bed-Sentry v3 === http://0.0.0.0:{args.port}\n")
    app.run(host='0.0.0.0', port=args.port, threaded=True, debug=False)

if __name__ == '__main__':
    main()
