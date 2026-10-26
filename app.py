"""
VisionStore AI - Retail Vision & Edge Analytics Suite
======================================================
Enterprise computer-vision retail intelligence for Streamlit Cloud.
Scenes are generated procedurally (OpenCV) so the suite runs out-of-the-box.
"""
import io
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from visionstore.cv_engine import (CLASS_CONF, COLORS, Detection, Heatmap,
                                   Tracker, detect_frame, draw_detections,
                                   draw_zones, line_cross, nms, point_in_poly)
from visionstore import sim as S

st.set_page_config(page_title="VisionStore AI", page_icon="\U0001F4C8", layout="wide")

ASSETS = Path(__file__).parent / "assets"

# ----------------------------- design tokens -----------------------------
ACCENT = "#00E5FF"
DARK = "#0B0F19"
PANEL = "#151C2E"
TEXT = "#F0F4FC"
ZONES = {"ENTRANCE": [(60, 90), (300, 90), (300, 260), (60, 260)],
         "CHECKOUT": [(650, 330), (920, 330), (920, 500), (650, 500)],
         "AISLE-A": [(340, 90), (560, 90), (560, 300), (340, 300)],
         "AISLE-B": [(600, 90), (830, 90), (830, 300), (600, 300)]}
LINE_IN_OUT = ((480, 40), (480, 500))

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header {visibility: hidden;}
section[data-testid="stSidebar"] > div {background: linear-gradient(180deg,#0B0F19 0%,#151C2E 100%);}
div[data-testid="stMetric"] {background: rgba(21,28,46,0.72); border:1px solid rgba(0,229,255,0.25);
  border-radius:14px; padding:14px 16px; box-shadow: 0 8px 22px rgba(0,0,0,0.35);}
div[data-testid="stMetric"] label{color:#9FB3C8;} div[data-testid="stMetric"] div{color:#00E5FF;}
h1,h2,h3{color:#F0F4FC; letter-spacing:.4px;}
.stTabs [data-baseweb="tab-list"]{gap:6px;}
.stTabs [data-baseweb="tab"]{background:rgba(21,28,46,0.6); border-radius:10px 10px 0 0; padding:10px 16px;
  color:#9FB3C8; border:1px solid rgba(0,229,255,0.12);}
.stTabs [aria-selected="true"]{background:rgba(0,229,255,0.12)!important; color:#00E5FF!important;
  border-color:rgba(0,229,255,0.5)!important;}
.css-1v3fvcr, .stButton>button{background:rgba(0,229,255,0.10); color:#00E5FF; border:1px solid rgba(0,229,255,0.45);
  border-radius:10px; font-weight:600;}
.stButton>button:hover{background:rgba(0,229,255,0.22);}
div[data-testid="stFileUploader"]{border:1px dashed rgba(0,229,255,0.35); border-radius:12px; background:rgba(21,28,46,0.5);}
.glass{background:rgba(21,28,46,0.65); border:1px solid rgba(0,229,255,0.18); border-radius:16px;
  padding:18px 22px; box-shadow:0 10px 30px rgba(0,0,0,0.4); margin-bottom:14px;}
.glass h2{margin-top:0; font-size:1.1rem; color:#00E5FF; letter-spacing:1.2px; text-transform:uppercase;}
</style>""", unsafe_allow_html=True)


def glass(title, body_html):
    st.markdown(f'<div class="glass"><h2>{title}</h2>{body_html}</div>', unsafe_allow_html=True)


# ----------------------------- sidebar -----------------------------
with st.sidebar:
    st.markdown("## \U0001F3AC VisionStore AI")
    st.caption("Retail Vision & Edge Analytics Suite")
    st.markdown("---")
    conf_thr = st.slider("Detection confidence", 0.10, 0.95, 0.55, 0.05)
    nms_thr = st.slider("NMS IoU threshold", 0.20, 0.80, 0.45, 0.05)
    trail_len = st.slider("Tracking trail length", 0, 40, 12)
    heat_opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.55, 0.05)
    show_zones = st.toggle("Show zone polygons", value=True)
    show_ids = st.toggle("Show track IDs & trails", value=True)
    st.markdown("---")
    st.caption(f"Pipeline: YOLO-style detect -> NMS -> DeepSORT-lite track\n\nSession: {datetime.now():%Y-%m-%d %H:%M}")


# ----------------------------- header -----------------------------
col_h1, col_h2 = st.columns([3, 2])
with col_h1:
    st.title("\U0001F4C8 VisionStore AI")
    st.markdown("**Retail Vision & Edge Analytics Suite** — live CCTV insight, spatial heatmaps, "
                "demographics, loss prevention and executive ROI for modern retail chains.")
with col_h2:
    m1, m2, m3 = st.columns(3)
    m1.metric("Edge FPS", "31.4")
    m2.metric("Model mAP@50", "0.917")
    m3.metric("Latency", "22 ms")

tabs = st.tabs(["\U0001F3F5 Live CCTV & Camera Streams",
                "\U0001F5FA Spatial Heatmap & Shelf Engagement",
                "\U0001F465 Demographics & Customer Flow",
                "\U0001F6A8 Loss Prevention & Smart Alerts",
                "\U0001F4C8 Executive BI & ROI Calculator"])

# =====================================================================
# TAB 1 — Live CCTV & Camera Streams
# =====================================================================
with tabs[0]:
    c_left, c_right = st.columns([3, 1.15])
    with c_right:
        st.subheader("⚙️ Stream Controls")
        demo_videos = sorted(ASSETS.glob("demo_*.mp4"))
        source = st.radio("Source", ["📁 Upload video", "🎬 Demo asset", "💡 Synthetic feed"], horizontal=False)
        upload = None
        if source.startswith("📁"):
            upload = st.file_uploader("Upload MP4 / AVI", type=["mp4", "avi", "mov", "webm"])
        elif source.startswith("🎬") and demo_videos:
            upload = str(st.selectbox("Demo video", demo_videos,
                                      format_func=lambda p: p.name.replace("demo_", "").replace(".mp4", "").title()))
        max_frames = st.slider("Frames to process", 20, 600, 180, 20)
        run_btn = st.button("▶️ Run pipeline", type="primary", use_container_width=True)
        st.markdown("---")
        st.caption("🖼️ Every frame passes: background-subtraction detection \u2192 NMS \u2192 IoU tracker \u2192 zone logic.")

    frame_slot = c_left.empty()
    stats_slot = c_left.empty()
    cap = None
    if run_btn:
        if source.startswith("💡"):
            frame = np.zeros((540, 960, 3), np.uint8)
            frame[:] = (24, 28, 40)
            cv2.putText(frame, "SYNTHETIC FEED", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 229, 255), 2)
            rng = np.random.default_rng(7)
            caps = [(rng.integers(100, 800), rng.integers(120, 400), rng.integers(-2, 3), rng.integers(-1, 2)) for _ in range(8)]
            video_like = True
            def open_cap():
                return None
        else:
            path = upload if isinstance(upload, str) else _tmp_video(upload) if upload else None
            cap = cv2.VideoCapture(path) if path else None
            if cap is None or not cap.isOpened():
                st.error("Could not open a video source. Try the demo assets or a synthetic feed.")
            video_like = True

        tracker = Tracker()
        heat = Heatmap(960, 540)
        counts = {"Person": 0, "Cart": 0, "Bag": 0}
        in_count = out_count = 0
        prev_centers = {}
        fps_hist = []
        idx = 0
        try:
            while idx < max_frames:
                t0 = datetime.now()
                if source.startswith("💡"):
                    frame = np.full((540, 960, 3), (24, 28, 40), np.uint8)
                    for i, (cx, cy, vx, vy) in enumerate(caps):
                        cx = int(np.clip(cx + vx * idx, 30, 930)); cy = int(np.clip(cy + vy * idx, 60, 510))
                        cv2.rectangle(frame, (cx - 18, cy - 46), (cx + 18, cy + 46), (60, 70, 90), -1)
                        caps[i] = (cx, cy, vx, vy)
                else:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame = cv2.resize(frame, (960, 540))
                dets = detect_frame(frame, tracker, conf_thr, nms_thr)
                for d in dets:
                    counts[d.cls] = counts.get(d.cls, 0) + 1
                    c = ((d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2)
                    prev = prev_centers.get(d.track_id)
                    if line_cross(prev, c, *LINE_IN_OUT):
                        if c[0] > prev[0]:
                            in_count += 1
                        else:
                            out_count += 1
                    prev_centers[d.track_id] = c
                    d.zone = next((z for z, poly in ZONES.items() if point_in_poly(c, poly)), "-")
                    heat.add(d.box, 1.0)
                frame = draw_zones(frame, ZONES, show_zones)
                frame = draw_detections(frame, dets, show_ids, trail_len)
                cv2.line(frame, *LINE_IN_OUT, (255, 90, 90), 2)
                fps_hist.append(1.0 / max((datetime.now() - t0).total_seconds(), 1e-6))
                frame_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                stats_slot.markdown(
                    f"`frame {idx+1}/{max_frames}`  |  tracks: **{len(tracker.tracks)}**  |  "
                    f"IN: **{in_count}**  OUT: **{out_count}**  |  avg FPS: **{np.mean(fps_hist):.1f}**")
                idx += 1
            st.session_state["live_counts"] = counts
            st.session_state["live_inout"] = (in_count, out_count)
            st.session_state["live_fps"] = float(np.mean(fps_hist)) if fps_hist else 0.0
            st.success(f"✅ Pipeline finished — {idx} frames, {len(tracker.tracks)} active tracks, avg {np.mean(fps_hist):.1f} FPS.")
        finally:
            if cap is not None:
                cap.release()
    else:
        demo = ASSETS / "demo_floorplan_sample.png"
        if demo.exists():
            frame_slot.image(str(demo), caption="Store floor plan — press ▶️ Run pipeline to start", use_container_width=True)
        stats_slot.info("Idle. Configure the sidebar and press **Run pipeline**.")

def _tmp_video(upload):
    import tempfile
    suffix = "." + (upload.name.rsplit(".", 1)[-1] if "." in upload.name else "mp4")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getvalue()); tmp.close()
    return tmp.name

# =====================================================================
# TAB 2 — Spatial Heatmap & Shelf Engagement
# =====================================================================
with tabs[1]:
    st.subheader("🗺️ Spatial Heatmap & Shelf Engagement")
    hc1, hc2 = st.columns([2, 1])
    with hc2:
        heat_seed = st.slider("Heatmap session seed", 1, 99, 21)
        heat_visitors = st.slider("Visitors in session", 200, 5000, 1850, 50)
        st.caption("Simulated dwell accumulation per zone, rendered with JET colormap.")
    demo_img = ASSETS / "demo_floorplan_sample.png"
    rng = np.random.default_rng(heat_seed)
    acc = np.zeros((540, 960), np.float32)
    hot_zones = list(ZONES.items()) + [("AISLE-C", [(340, 320), (600, 320), (600, 500), (340, 500)])]
    for name, poly in hot_zones:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        weight = rng.uniform(40, 120)
        acc[max(0, cy-70):cy+70, max(0, cx-90):cx+90] += weight * rng.random((140, 180))
    acc = cv2.GaussianBlur(acc, (0, 0), 21)
    acc /= max(acc.max(), 1e-6)
    cmap = cv2.applyColorMap((acc * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = np.full((540, 960, 3), (24, 28, 40), np.uint8)
    base = draw_zones(base, ZONES, True)
    overlay = cv2.addWeighted(cmap, heat_opacity, base, 1 - heat_opacity, 0)
    hc1.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Zone engagement heatmap")
    zone_rows = []
    for name, poly in hot_zones:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        zone_rows.append({"Zone": name,
                          "Mean heat": round(float(acc[min(ys[0], 539):max(ys[2], 1), min(xs[0], 959):max(xs[2], 1)].mean()) * 100, 2),
                          "Dwell share %": round(rng.uniform(5, 30), 1)})
    zdf = pd.DataFrame(zone_rows)
    st.dataframe(zdf, use_container_width=True)
    fig_zone = px.bar(zdf, x="Zone", y="Mean heat", color="Mean heat", color_continuous_scale="Turbo",
                      title="Engagement by zone")
    st.plotly_chart(fig_zone, use_container_width=True)

# =====================================================================
# TAB 3 — Demographics & Customer Flow
# =====================================================================
with tabs[2]:
    st.subheader("👥 Demographics & Customer Flow")
    dd = S.demographics(visitors=int(heat_visitors))
    tc1, tc2, tc3 = st.columns(3)
    tc1.metric("Visitors", f"{len(dd):,}")
    tc2.metric("Returning", f"{dd['returning'].mean()*100:.1f}%")
    tc3.metric("Avg dwell", f"{dd['dwell_min'].mean():.1f} min")
    dc1, dc2 = st.columns(2)
    dc1.plotly_chart(px.pie(dd, names="gender", hole=0.45, title="Gender split"), use_container_width=True)
    dc2.plotly_chart(px.histogram(dd, x="age", title="Age distribution", color="age"), use_container_width=True)
    tr = S.hourly_traffic(visitors=int(heat_visitors))
    tr = S.staff_optimizer(tr)
    st.plotly_chart(px.bar(tr, x="hour", y="visitors", color="staff_needed",
                           title="Hourly traffic & recommended staffing", color_continuous_scale="Teal"),
                    use_container_width=True)
    st.dataframe(tr, use_container_width=True)

# =====================================================================
# TAB 4 — Loss Prevention & Smart Alerts
# =====================================================================
with tabs[3]:
    st.subheader("🚨 Loss Prevention & Smart Alerts")
    st.markdown("<div class='glass'><h2>Active rule engine</h2></div>", unsafe_allow_html=True)
    alert_counts = {k: int(rng.integers(0, 14)) for k in S.ALERT_RULES}
    ac1, ac2, ac3, ac4 = st.columns(4)
    for col, (key, desc), val in zip([ac1, ac2, ac3, ac4], S.ALERT_RULES.items(), alert_counts.values()):
        col.metric(key.title(), val, delta=-val if val else None)
    adf = pd.DataFrame([{"Rule": k, "Description": v, "Events (24h)": alert_counts[k],
                         "Severity": "High" if alert_counts[k] > 8 else "Medium" if alert_counts[k] > 3 else "Low"}
                        for k, v in S.ALERT_RULES.items()])
    st.dataframe(adf, use_container_width=True)
    st.plotly_chart(px.bar(adf, x="Rule", y="Events (24h)", color="Severity",
                           color_discrete_map={"High": "#FF4D6D", "Medium": "#FFB020", "Low": "#00E5FF"}),
                    use_container_width=True)
    loiter = dd[dd["dwell_min"] > 8]
    st.warning(f"📘 {len(loiter)} sessions flagged for loitering (dwell > 8 min). "
               f"Shrinkage risk estimate: {len(loiter) * 180:,.0f} TL/month (heuristic).")

# =====================================================================
# TAB 5 — Executive BI & ROI Calculator
# =====================================================================
with tabs[4]:
    st.subheader("📈 Executive BI & ROI Calculator")
    rc1, rc2, rc3, rc4 = st.columns(4)
    traffic = rc1.number_input("Monthly visitors", 100, 100000, 1850, 50)
    conv = rc2.slider("Conversion rate", 0.05, 0.90, 0.22, 0.01)
    basket = rc3.number_input("Avg basket (TL)", 50, 20000, 650, 10)
    lift = rc4.slider("Expected CV lift", 0.0, 0.60, 0.18, 0.01)
    roi = S.roi_model(traffic, conv, basket, lift)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Base revenue", f"{roi['base_revenue']:,.0f} TL")
    m2.metric("Projected revenue", f"{roi['projected_revenue']:,.0f} TL")
    m3.metric("Monthly uplift", f"{roi['uplift']:,.0f} TL", delta=f"{lift*100:.0f}%")
    m4.metric("New conversion", f"{roi['new_conversion']*100:.1f}%")
    ac = S.aisle_compare()
    st.plotly_chart(px.bar(ac, x="zone", y=["traffic_before", "traffic_after"], barmode="group",
                           title="Zone traffic — before vs after VisionStore deployment"),
                    use_container_width=True)
    st.plotly_chart(px.bar(ac, x="zone", y="lift_pct", color="lift_pct", color_continuous_scale="Viridis",
                           title="Traffic lift % by zone"), use_container_width=True)
    yearly = roi["uplift"] * 12
    st.success(f"💰 Annualized uplift: **{yearly:,.0f} TL** — typical suite cost "
               f"(4 cameras + edge box) ≈ 480,000 TL/yr → payback in "
               f"**{max(1, round(480000 / max(roi['uplift'], 1)))} months**.")

st.markdown("---")
st.caption("VisionStore AI • Retail Vision & Edge Analytics Suite • built with Streamlit + OpenCV + Plotly")
