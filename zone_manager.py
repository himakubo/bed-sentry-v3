"""
zone_manager.py — Zone Manager for Bed-Sentry v3
Handles bed zone storage and auto-generates edge + near zones.
"""

import json
import os
import numpy as np
import cv2

CONFIG_FILE = "config.json"

# Pixel offsets for auto-generated zones
EDGE_OFFSET = 50   # pixels outward from bed polygon
NEAR_OFFSET = 120  # pixels outward from bed polygon


def expand_polygon(polygon: list, offset: int) -> list:
    """
    Expand a polygon outward by offset pixels using morphological dilation.
    Returns list of {x, y} dicts.
    """
    if len(polygon) < 3:
        return polygon

    pts = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.int32)

    # Create a mask, dilate it, find new contour
    # Work in a canvas sized to the bounding box + padding
    xs = [p["x"] for p in polygon]
    ys = [p["y"] for p in polygon]
    pad = offset + 10
    x0, y0 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    w = max(xs) - x0 + pad * 2
    h = max(ys) - y0 + pad * 2

    mask = np.zeros((h, w), dtype=np.uint8)
    shifted = pts - np.array([x0, y0])
    cv2.fillPoly(mask, [shifted], 255)

    # Dilate
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (offset*2+1, offset*2+1))
    dilated = cv2.dilate(mask, kernel)

    # Find contour
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return polygon

    # Simplify contour
    contour = contours[0]
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx  = cv2.approxPolyDP(contour, epsilon, True)

    # Shift back
    result = [{"x": int(p[0][0] + x0), "y": int(p[0][1] + y0)} for p in approx]
    return result


def load_config() -> dict:
    """Load full config from config.json."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"[Zone] Load error: {e}")
    return {}


def save_config(config: dict) -> bool:
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"[Zone] Save error: {e}")
        return False


def load_zones() -> list:
    """
    Load zones from config. Returns list of zone dicts each with:
    {id, label, color, polygon, edge_polygon, near_polygon, type}
    """
    config = load_config()
    zones  = config.get("zones", [])

    # Auto-generate near_polygon only if missing; never touch edge_polygon
    updated = False
    for z in zones:
        if z.get("type") == "bed" and z.get("polygon"):
            if not z.get("near_polygon"):
                z["near_polygon"] = expand_polygon(z["polygon"], NEAR_OFFSET)
                updated = True

    if updated:
        config["zones"] = zones
        save_config(config)

    if zones:
        print(f"[Zone] Loaded {len(zones)} zone(s)")
    return zones


def save_zones(zones: list) -> bool:
    """Save zones. Only auto-generates near_polygon; edge_polygon is never touched."""
    for z in zones:
        if z.get("type") == "bed" and z.get("polygon"):
            z["near_polygon"] = expand_polygon(z["polygon"], NEAR_OFFSET)

    config = load_config()
    config["zones"] = zones
    print(f"[Zone] Saved {len(zones)} zone(s)")
    return save_config(config)


def validate_zone(zone: dict) -> bool:
    try:
        if not isinstance(zone.get("id"), str): return False
        if not isinstance(zone.get("label"), str): return False

        def _valid_poly(poly):
            if not poly or len(poly) < 3: return False
            return all(isinstance(pt.get("x"), (int, float)) and
                       isinstance(pt.get("y"), (int, float)) for pt in poly)

        # Accept zone if it has a valid polygon OR a valid edge_polygon
        return _valid_poly(zone.get("polygon")) or _valid_poly(zone.get("edge_polygon"))
    except Exception:
        return False


def update_offsets(edge: int, near: int) -> bool:
    """Update NEAR_OFFSET and regenerate near polygons only."""
    global EDGE_OFFSET, NEAR_OFFSET
    EDGE_OFFSET = edge
    NEAR_OFFSET = near
    config = load_config()
    zones  = config.get("zones", [])
    for z in zones:
        if z.get("type") == "bed" and z.get("polygon"):
            z["near_polygon"] = expand_polygon(z["polygon"], NEAR_OFFSET)
    config["zones"] = zones
    print(f"[Zone] Offsets updated: near={near}px")
    return save_config(config)


def point_in_polygon(px: int, py: int, polygon: list) -> bool:
    if not polygon or len(polygon) < 3:
        return False
    pts = np.array([[p["x"], p["y"]] for p in polygon], dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0


ZONE_COLORS = [
    {"name": "green",  "bgr": [0, 200, 60],  "hex": "#00e676"},
    {"name": "blue",   "bgr": [220, 100, 0], "hex": "#4fc3f7"},
    {"name": "orange", "bgr": [0, 140, 255], "hex": "#ff9800"},
    {"name": "purple", "bgr": [200, 0, 200], "hex": "#ce93d8"},
    {"name": "yellow", "bgr": [0, 220, 220], "hex": "#ffeb3b"},
    {"name": "red",    "bgr": [0, 60, 220],  "hex": "#ff4757"},
]

def get_zone_color_bgr(color_name: str) -> list:
    for c in ZONE_COLORS:
        if c["name"] == color_name:
            return c["bgr"]
    return ZONE_COLORS[0]["bgr"]
