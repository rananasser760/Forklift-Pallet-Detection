import csv
import datetime as dt
import html
import io
import json
import re
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE = Path(__file__).parent
MODELS_DIR, RESULTS = BASE / "models", BASE / "results"
CSV_PATH = RESULTS / "summary.csv"
IMGSZ, NMS_IOU, DEFAULT_CONF = 768, 0.7, 0.35  # same values used in the notebook
CLASSES = ["forklift", "pallet"]
MODELS_DIR.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)


def available():
    return sorted(p.name for p in MODELS_DIR.iterdir() if p.suffix in (".pt", ".onnx"))


if not available():
    raise SystemExit("Put best.pt and/or best.onnx inside the models/ folder.")
DEFAULT = "best.pt" if "best.pt" in available() else available()[0]
_cache = {}


def get_model(name):
    if name not in available():  # also blocks path tricks
        raise HTTPException(404, f"Model not found: {name}")
    if name not in _cache:
        m = YOLO(str(MODELS_DIR / name), task="detect")
        m.predict(Image.new("RGB", (IMGSZ, IMGSZ)), imgsz=IMGSZ, device="cpu", verbose=False)  # warm-up
        _cache[name] = m
    return _cache[name]


def rows():
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


REPORT_HEAD = """<!DOCTYPE html><html><head><meta charset="UTF-8"><title>PetroChoice - Detection Report</title><style>
body{font-family:system-ui,Arial,sans-serif;background:#eef4ff;color:#0f2a4a;margin:0;padding:32px}
h1{margin:0 0 4px;color:#1d4ed8}.sub{color:#667085;margin-bottom:20px}
.kpi{display:flex;gap:12px;margin-bottom:20px}.kpi div{background:#fff;border-radius:10px;padding:14px 22px;box-shadow:0 1px 3px #0001}
.kpi b{display:block;font-size:1.6rem}table{width:100%;border-collapse:collapse;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 1px 3px #0001}
th,td{padding:10px 12px;text-align:left;border-bottom:1px solid #eceff3;font-size:.9rem}th{background:#1d4ed8;color:#fff}
img{height:70px;border-radius:6px}</style></head><body>"""


def write_report():
    rs = rows()
    tf, tp = sum(int(r["forklifts"]) for r in rs), sum(int(r["pallets"]) for r in rs)
    body = "".join(
        f"<tr><td><a href='{r['folder']}/annotated.jpg'><img src='{r['folder']}/annotated.jpg'></a></td>"
        f"<td>{html.escape(r['image'])}</td><td>{r['model']}</td><td>{r['conf']}</td><td>{r['forklifts']}</td>"
        f"<td>{r['pallets']}</td><td>{r['inference_ms']} ms</td><td>{r['timestamp']}</td></tr>"
        for r in reversed(rs)
    )
    page = (
        REPORT_HEAD
        + f"<h1>Forklift &amp; Pallet Detection Report</h1><div class='sub'>YOLO11s | {IMGSZ} px | NMS IoU {NMS_IOU} | CPU</div>"
        + f"<div class='kpi'><div><b>{len(rs)}</b>Images</div><div><b>{tf}</b>Forklifts</div><div><b>{tp}</b>Pallets</div></div>"
        + "<table><tr><th>Preview</th><th>Image</th><th>Model</th><th>Conf</th><th>Forklifts</th><th>Pallets</th>"
        + f"<th>Time</th><th>Date</th></tr>{body}</table></body></html>"
    )
    (RESULTS / "report.html").write_text(page, encoding="utf-8")


def save_run(img, r, filename, model_name, conf, dets, ms):
    now = dt.datetime.now()
    stem = re.sub(r"[^\w-]", "_", Path(filename or "image").stem)[:40]
    folder = now.strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_" + stem
    d = RESULTS / folder
    d.mkdir(parents=True)
    img.save(d / "original.jpg", quality=95)
    Image.fromarray(r.plot(line_width=3)[..., ::-1]).save(d / "annotated.jpg", quality=95)
    summary = {
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "image": filename, "model": model_name, "conf": conf,
        "forklifts": sum(x["label"] == "forklift" for x in dets), "pallets": sum(x["label"] == "pallet" for x in dets),
        "total": len(dets), "inference_ms": ms, "folder": folder,
    }
    (d / "detections.json").write_text(json.dumps({**summary, "detections": dets}, indent=2), encoding="utf-8")
    new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary))
        if new:
            w.writeheader()
        w.writerow(summary)
    write_report()
    return folder


get_model(DEFAULT)
write_report()
app = FastAPI(title="PetroChoice Forklift & Pallet Detector")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/results", StaticFiles(directory=RESULTS), name="results")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models": available(), "default": DEFAULT, "imgsz": IMGSZ, "device": "cpu"}


@app.get("/models")
def models():
    return {"models": available(), "default": DEFAULT}


@app.get("/history")
def history(limit: int = 8):
    return list(reversed(rows()))[:limit]


@app.post("/predict")
def predict(file: UploadFile = File(...), conf: float = Form(DEFAULT_CONF), model: str = Form(DEFAULT),
            save: bool = Form(False)):
    m = get_model(model)
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(file.file.read()))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file")
    t0 = time.perf_counter()
    r = m.predict(img, conf=conf, iou=NMS_IOU, imgsz=IMGSZ, device="cpu", verbose=False)[0]
    ms = round((time.perf_counter() - t0) * 1000, 1)
    dets = [
        {"label": CLASSES[int(c)], "conf": round(float(s), 3), "box": [round(v, 1) for v in b]}
        for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())
    ]
    saved = save_run(img, r, file.filename, model, conf, dets, ms) if save else None
    return {"width": img.width, "height": img.height, "ms": ms, "model": model, "detections": dets, "saved": saved}