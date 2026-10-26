# VisionStore AI — Retail Vision & Edge Analytics Suite
# ویژن‌استور AI — مجموعه آنالیتیکس بینایی ماشین و مانیتورینگ زنده خرده‌فروشی

Enterprise computer-vision retail intelligence built with **Streamlit + OpenCV + Plotly**.
مجموعه‌ای سازمانی از هوش بینایی ماشین برای خرده‌فروشی، ساخته‌شده با **Streamlit + OpenCV + Plotly**.

All scenes are generated procedurally (OpenCV background subtraction) — the suite runs out-of-the-box with **no GPU and no external model downloads**.
همه صحنه‌ها به‌صورت رویه‌ای تولید می‌شوند (تفریق پس‌زمینه در OpenCV) — بدون GPU و بدون دانلود مدل خارجی اجرا می‌شود.

---

## ✨ Features | قابلیت‌ها

| Tab | English | فارسی |
|---|---|---|
| 🏕 **1. Live CCTV & Camera Streams** | Upload a video, pick a demo asset, or run a synthetic feed. Full pipeline: detect → NMS → IoU tracking → zones → in/out line counting, live FPS & track stats. | آپلود ویدیو، انتخاب ویدیوی نمونه یا فید سینتتیک. پایپ‌لاین کامل: تشخیص → NMS → ردیابی IoU → ناحیه‌ها → شمارش ورود/خروج، آمار FPS و تراک زنده. |
| 🗺 **2. Spatial Heatmap & Shelf Engagement** | JET-colormap engagement heatmap over the floor plan + per-zone mean-heat and dwell-share table & charts. | نقشه حرارتی تعامل با کلر‌مپ JET روی پلان فروشگاه + جدول و نمودار میانگین گرمای هر ناحیه و سهم توقف. |
| 👥 **3. Demographics & Customer Flow** | Seeded demographics (gender, age, dwell, returning), hourly traffic curve and automatic staffing recommendation. | جمعیت‌شناسی سید‌دار (جنسیت، سن، توقف، مشتری بازگشتی)، ترافیک ساعتی و پیشنهاد خودکار نیروی انسانی. |
| 🚨 **4. Loss Prevention & Smart Alerts** | Rule engine (loitering, crowd, queue, off-hours intrusion), severity scoring and shrinkage-risk estimate. | موتور قواعد (وقت‌تلف‌کردن، ازدحام، صف، نفوذ خارج از ساعت کاری)، سطح‌بندی خطر و برآورد ریسک کسری کالا. |
| 📈 **5. Executive BI & ROI Calculator** | Interactive ROI model (traffic, conversion, basket, lift), zone before/after comparison and payback-period estimate. | مدل ROI تعاملی (ترافیک، نرخ تبدیل، سبد خرید، لیفت)، مقایسه قبل/بعد ناحیه‌ها و برآورد دوره بازگشت سرمایه. |

---

## 🚀 Quick Start | شروع سریع

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501` — then press **▶️ Run pipeline** in Tab 1.
سپس `http://localhost:8501` را باز کنید و در تب ۱ دکمه **Run pipeline** را بزنید.

---

## 📁 Project Structure | ساختار پروژه

```
visionstore_ai/
├── app.py                     # Streamlit UI (Tabs 1–5)
├── requirements.txt
├── .streamlit/config.toml     # Dark neon theme
├── assets/                    # Demo videos + floor plan
│   ├── demo_clothing_store.mp4
│   ├── demo_entrance_corridor.mp4
│   ├── demo_supermarket_aisle.mp4
│   └── demo_floorplan_sample.png
└── visionstore/               # Core library
    ├── __init__.py
    ├── cv_engine.py           # Detection, NMS, IoU tracker, heatmap, zone drawing
    └── sim.py                 # Seeded retail/BI simulation (traffic, demographics, ROI, alerts)
```

### `visionstore.cv_engine`
- `detect_frame()` — background subtraction → contours → class assignment (Person / Cart / Bag) → confidence filtering.
- `nms()` — vectorized Non-Maximum Suppression.
- `Tracker` — IoU + centroid tracker (DeepSORT-lite style) with dwell & velocity.
- `Heatmap` — decaying accumulation buffer with JET rendering.
- `draw_detections`, `draw_zones`, `line_cross`, `point_in_poly` — overlay & geometry utilities.

### `visionstore.sim`
- `hourly_traffic`, `demographics`, `aisle_compare` — deterministic (seeded) synthetic retail data.
- `roi_model`, `staff_optimizer`, `ALERT_RULES` — BI & alert logic.

---

## ⚙️ Sidebar Controls | کنترل‌های نوار کناری

| Control | Description | توضیح |
|---|---|---|
| Detection confidence | Minimum detection score (0.10–0.95) | حداقل امتیاز تشخیص |
| NMS IoU threshold | Overlap threshold for suppression | آستانه حذف باکس‌های هم‌پوشان |
| Tracking trail length | Motion arrow length on frames | طول فلش حرکت روی فریم‌ها |
| Heatmap opacity | Heat overlay strength (Tab 2) | شدت لایه حرارتی |
| Show zones / Show IDs | Toggle overlays | نمایش/عدم نمایش ناحیه‌ها و شناسه‌ها |

---

## 🧪 Testing | تست

```bash
python -m py_compile app.py visionstore/cv_engine.py visionstore/sim.py   # syntax
python -c "import app"                                                    # import / mock run (headless)
```

---

## 📦 Deployment | استقرار

Works on **Streamlit Community Cloud** with zero config — just push this folder to GitHub and set the main file to `app.py`. `opencv-python-headless` keeps the image small.
روی **Streamlit Community Cloud** بدون تنظیمات اضافی اجرا می‌شود — کافی است پوشه را به GitHub بفرستید و فایل اصلی را `app.py` تنظیم کنید.

---

## 📄 License | مجوز
MIT — free for commercial and educational use.
آزاد برای استفاده تجاری و آموزشی.
