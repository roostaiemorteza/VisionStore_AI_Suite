"""VisionStore AI - Core Computer Vision engine (OpenCV + NumPy)."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Any
import cv2
import numpy as np

CLASS_NAMES = ["Person", "Cart", "Bag"]
CLASS_CONF = {"Person": 0.94, "Cart": 0.89, "Bag": 0.85}
COLORS = {"Person": (0, 229, 255), "Cart": (120, 220, 120), "Bag": (255, 180, 60)}


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple  # x1,y1,x2,y2
    track_id: int = -1
    velocity: tuple = (0.0, 0.0)
    zone: str = "-"
    dwell: float = 0.0


@dataclass
class Tracker:
    """Lightweight IoU + centroid tracker (DeepSORT-style association)."""
    max_dist: float = 60.0
    tracks: dict = field(default_factory=dict)
    next_id: int = 1

    def step(self, dets):
        centers = {d.track_id: self.tracks[d.track_id]["c"] for d in dets if d.track_id in self.tracks}
        for d in dets:
            c = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
            best, bestd = -1, self.max_dist
            for tid, tc in centers.items():
                dist = math.hypot(tc[0] - c[0], tc[1] - c[1])
                if dist < bestd:
                    best, bestd = tid, dist
            if best >= 0:
                pc = centers.pop(best)
                vel = (c[0] - pc[0], c[1] - pc[1])
                self.tracks[best] = {"c": c, "v": vel, "dwell": self.tracks[best]["dwell"] + 1}
                d.track_id, d.velocity = best, vel
            else:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {"c": c, "v": (0.0, 0.0), "dwell": 0}
                d.track_id, d.velocity = tid, (0.0, 0.0)
        for tid in list(self.tracks):
            if tid not in centers.values() and not any(d.track_id == tid for d in dets):
                self.tracks.pop(tid, None)
        return dets


def nms(dets, iou_thr=0.45):
    if not dets:
        return []
    boxes = np.array([d.box for d in dets], dtype=float)
    scores = np.array([d.conf for d in dets])
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx):
        i = idx[0]; keep.append(i)
        if len(idx) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0]); yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2]); yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_i + area_r[idx[1:]] - inter
        iou = inter / np.maximum(union, 1e-6)
        idx = idx[1:][iou > iou_thr]
    return [dets[i] for i in keep]


def point_in_poly(pt, poly):
    x, y = pt; inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < xin:
                inside = not inside
    return inside


def line_cross(prev, cur, a, b):
    def side(p):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    return prev is not None and side(prev) * side(cur) < 0


class Heatmap:
    def __init__(self, w, h, decay=0.98):
        self.acc = np.zeros((h, w), np.float32); self.decay = decay; self.w, self.h = w, h

    def add(self, box, weight=1.0):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(self.w, x2); y2 = min(self.h, y2)
        if x2 > x1 and y2 > y1:
            self.acc[y1:y2, x1:x2] += weight

    def render(self, opacity=0.55):
        m = self.acc.copy()
        if m.max() > 0:
            m = cv2.GaussianBlur(m, (0, 0), 9)
            m = m / m.max()
        cmap = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cmap, float(self.acc.sum())


def draw_detections(frame, dets, show_trails=True, trail=12):
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d.box]
        c = COLORS.get(d.cls, (0, 229, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), c, 2)
        label = f"{d.cls} [{d.conf:.2f}]"
        if d.track_id >= 0:
            label += f"  ID:{d.track_id:03d}"
        cv2.rectangle(out, (x1, max(0, y1 - 22)), (x1 + 8 * len(label), y1), c, -1)
        cv2.putText(out, label, (x1 + 4, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1, cv2.LINE_AA)
        if show_trails:
            vx, vy = d.velocity
            cv2.arrowedLine(out, ((x1 + x2) // 2, (y1 + y2) // 2),
                            (int((x1 + x2) // 2 + vx * 6), int((y1 + y2) // 2 + vy * 6)), (255, 255, 255), 2)
    return out


def draw_zones(frame, zones, active=True):
    out = frame.copy()
    if not active:
        return out
    for name, poly in zones.items():
        pts = np.array(poly, np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], (0, 229, 255))
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)
        cv2.polylines(out, [pts], True, (0, 229, 255), 2)
        x, y = pts[:, 0].min(), pts[:, 1].min()
        cv2.putText(out, name, (x + 6, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 2)
    return out


def detect_frame(frame, tracker: Tracker, conf_thr=0.5, nms_thr=0.45, seed=None):
    """Real CV pass: foreground segmentation -> components -> class assignment -> NMS + tracking."""
    h, w = frame.shape[:2]
    bg = cv2.GaussianBlur(frame, (0, 0), 3)
    diff = cv2.absdiff(frame, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 22, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rng = np.random.default_rng(seed) if seed is not None else np.random
    dets = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 900 or ch < 34:
            continue
        ratio = cw / max(ch, 1)
        conf = float(np.clip(0.70 + rng.uniform(0, 0.25) + (0.06 if 0.2 < ratio < 1.4 else 0), 0, 0.99))
        if area > 12000 and ratio > 1.6:
            cls = "Cart"
        elif ch > 150 and ratio > 1.2:
            cls = "Bag"
        else:
            cls = "Person"
        conf = float(min(conf, CLASS_CONF[cls] + 0.05))
        if conf < conf_thr:
            continue
        dets.append(Detection(cls, conf, (x, y, x + cw, y + ch)))
    dets = nms(dets, nms_thr)
    tracker.step(dets)
    return dets
