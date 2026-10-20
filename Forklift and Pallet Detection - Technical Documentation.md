# Forklift and Pallet Detection: Technical Documentation

Petrochoice AI Engineer, Round 2 | Rana Nasser

## 1. Overview

This project is an object detector that finds **forklifts** and **pallets** (empty or loaded) in warehouse images. It uses a single-stage **YOLO11s** model fine-tuned from COCO weights at 768 px. The trained model is served by a **FastAPI** application with a browser UI, so images can be uploaded and the detections inspected live. Every saved run is written to a `results/` folder as an annotated image, a JSON file, a CSV log and an HTML report.

| Item | Value |
| --- | --- |
| Task | Object detection, 2 classes (forklift, pallet) |
| Model | Ultralytics YOLO11s, COCO-pretrained, 9.4 M parameters |
| Input size | 768 px |
| Data | 1,660 train / 213 validation / 187 test images |
| Test mAP@0.5 / mAP@0.5:0.95 | 0.863 / 0.540 |
| Test precision / recall | 0.826 / 0.793 |
| CPU speed (ONNX, batch 1) | 210 ms per image (4.8 FPS) |
| Serving | FastAPI + HTML/CSS/JS, CPU inference |

## 2. System architecture

```
TRAINING (Kaggle notebook, Tesla T4)
  Roboflow + Kaggle datasets -> filter to 2 classes -> splits -> YOLO11n / YOLO11s
  -> selection on validation -> best.pt -> ONNX export

SERVING (local machine, CPU)
  Browser (static/index.html)
     |  POST /predict  (image, conf, model, save)
     v
  FastAPI (app.py) -> Ultralytics YOLO (best.pt or best.onnx, 768 px, NMS IoU 0.7)
     |-> JSON: boxes, labels, confidences, inference time -> drawn on a canvas
     |-> results/<run>/ : original.jpg, annotated.jpg, detections.json
     |-> results/summary.csv, results/report.html
```

## 3. Repository structure

```
app.py               # FastAPI backend
static/index.html    # UI (HTML, CSS and JS in one file)
models/              # best.pt and/or best.onnx
results/             # created automatically, one folder per saved run
notebooks/final_petrochoice.ipynb   # data preparation, training, evaluation, export
requirements.txt
README.md
```

The weights come from the notebook export (`yolo11s_best.pt` and `yolo11s_best.onnx` inside `forklift_pallet_weights.zip`). Copy them into `models/` as `best.pt` and `best.onnx`.

## 4. Data

### 4.1 Sources

| Source | Images | Classes | Notes |
| --- | --- | --- | --- |
| Warehouse (Roboflow Universe) | 1,075 (944 train, 131 valid) | forklift, pallet, pallet\_truck, small\_load\_carrier, stillage | 11,776 pallet boxes but only 5 forklift boxes in train |
| Forklift (Kaggle) | 1,011 (885 train, 84 valid, 42 test) | forklift, person | 1,000 forklift boxes in train, no pallets |

Both sources are merged, with a source prefix on every file name so the origin of each image stays traceable.

### 4.2 Filtering rules

- Class names are matched exactly after normalising case and spaces, so `pallet_truck` is never mistaken for `pallet`.
- Boxes of every class other than forklift and pallet are removed and become background. This affects 11,836 boxes (person 985, stillage 1,926, pallet\_truck 571, small\_load\_carrier 8,354), so the model learns that these objects are not forklifts or pallets.
- Images that contain only removed classes are kept as hard negatives, using a capped sample of 80 images.
- Images with an empty or missing label file, or an unreadable image, are dropped (3 images), because the background label cannot be trusted.
- Polygons are converted to boxes, and boxes are validated, clipped to the image, and de-duplicated.
- The finished dataset is re-read from disk and checked independently of the code that wrote it. Audit sheets list 174 kept images where a removed-class box overlaps a kept box, and the 80 negatives, for visual review.

### 4.3 Final splits and statistics

| Split | Images | Forklift boxes | Pallet boxes |
| --- | --- | --- | --- |
| train | 1,660 | 905 | 10,347 |
| validation | 213 | 96 | 1,480 |
| test | 187 | 100 | 1,118 |

A difference-hash near-duplicate check removed 0 images in the final run. Across all boxes, **70.4% cover less than 1% of the image**, so small objects dominate.

### 4.4 Class imbalance and missing labels

- After filtering the ratio is about 12:1 (1,101 forklift boxes against 13,275 pallet boxes). Train images that contain the rarest class (forklift) are **repeated up to 3 times** (train becomes 3,324 images with 2,715 forklift boxes), and a **class-weighted classification loss** is used (`cls_pw = 0.5`). AP is always reported per class.
- Each source annotates only one class: the Kaggle forklift images have no pallet labels, and the warehouse images have only 5 forklift labels. Visible but unlabeled objects count as background. This is measured per source in the error analysis. Pseudo-labelling is implemented in the notebook but switched off in the final run.

## 5. Model and training

### 5.1 Why YOLO11

A single-stage, anchor-free detector gives real-time speed, mature tooling, COCO weights for transfer learning and a simple ONNX export. Faster R-CNN is slower and heavier to deploy, and RT-DETR needs more data and compute. YOLO11n is the lightweight baseline and YOLO11s is compared against it.

### 5.2 Training settings

| Setting | Value |
| --- | --- |
| Weights | `yolo11n.pt`, `yolo11s.pt` (COCO) |
| Image size | 768 px |
| Epochs / patience | 60 / 20 (early stopping) |
| Batch size | automatic (`-1`) |
| Learning rate schedule | cosine |
| Augmentation | mosaic (off for the last 10 epochs), horizontal flip 0.5, no vertical flip |
| Class-weighted loss power | 0.5 |
| Oversampling | rarest class repeated up to 3 times |
| Seed | 42 |
| Hardware | Tesla T4. YOLO11n 39 min, YOLO11s 58 min |

The notebook has a second preset, `accurate` (960 px, 100 epochs), because most boxes are small and the validation curve was still rising slowly at 60 epochs.

### 5.3 Selection protocol

- The model is chosen on **validation** mAP@0.5:0.95, and **YOLO11s** wins (0.491 against 0.476).
- The NMS IoU threshold is tuned on validation only: 0.5 gives 0.476, 0.6 gives 0.486, **0.7 gives 0.491**, 0.8 gives 0.488.
- The operating confidence (**0.35**) is the value that maximises F1 on validation, from a sweep over 0.05 to 0.95.
- The test set is used once, for the final numbers.

## 6. Evaluation

| Model | Split | mAP50 | mAP50-95 | Precision | Recall |
| --- | --- | --- | --- | --- | --- |
| YOLO11n | val | 0.830 | 0.476 | 0.855 | 0.773 |
| YOLO11n | test | 0.875 | 0.548 | 0.869 | 0.806 |
| **YOLO11s** | val | 0.838 | 0.491 | 0.867 | 0.778 |
| **YOLO11s** | test | 0.863 | 0.540 | 0.826 | 0.793 |

Per class on the test set (YOLO11s):

| Class | AP50 | mAP50-95 |
| --- | --- | --- |
| forklift | 0.902 | 0.631 |
| pallet | 0.823 | 0.450 |

Test-time augmentation raised validation mAP50-95 from 0.491 to 0.507 but more than doubled inference time (about 26 ms against 11 ms on GPU), so it is not used.

## 7. Error analysis

At the fixed confidence of 0.35, IoU 0.5, on the test set: precision 0.775, recall 0.774, F1 0.775, with 943 true positives, 273 false alarms on background, 275 missed objects and **0 forklift/pallet mix-ups**.

| Breakdown | Result |
| --- | --- |
| Forklifts found | 80.0% (median IoU of found boxes 0.865) |
| Pallets found | 77.2% (median IoU 0.791) |
| Pallets found, small (<0.2% of image, 376 objects) | 62.5% |
| Pallets found, medium (0.2-1%, 467 objects) | 83.3% |
| Pallets found, large (>1%, 275 objects) | 86.9% |
| Warehouse source, pallet precision / recall | 0.77 / 0.77 |
| Forklift source, forklift precision / recall | 0.89 / 0.80 |

Missed objects are small: the median relative area is 0.19% of the image, against 0.42% for all boxes. Some false alarms are probably real pallets that were never labeled, which is the incomplete-annotation effect described in section 4.4.

## 8. Export and CPU performance

The model is exported with `YOLO(best.pt).export(format="onnx", imgsz=768, simplify=True)` (opset 18, slimmed with onnxslim). Sizes: `best.pt` 19.2 MB and `best.onnx` 36.2 MB.

| Runtime | Mean | p95 | FPS | Test mAP50-95 |
| --- | --- | --- | --- | --- |
| PyTorch (CPU) | 254.2 ms | 265.4 ms | 3.9 | 0.540 |
| ONNX Runtime (CPU) | 210.1 ms | 219.6 ms | 4.8 | 0.532 |

Measured on a Kaggle Intel Xeon 2.0 GHz CPU, batch 1, 768 px. On the Tesla T4 inference takes about 13 ms per image. Speed depends on the machine: on a laptop CPU the demo needs roughly half a second per image.

## 9. Demo application

### 9.1 Key features

- Upload one or many images with drag and drop. Each image gets a thumbnail, and clicking a thumbnail shows its result without reloading the page.
- Model selector that lists every `.pt` and `.onnx` file in `models/`.
- Confidence slider (default 0.35). Moving it re-runs the current image as a **preview** that is not saved.
- Per-class counts (forklifts, pallets) and inference time for the current image.
- New uploads and the **Run & save to results** button write the run to `results/`.
- Recent-results panel and a link to the full HTML report.
- Light theme by default with a dark-mode toggle, remembered in the browser.

### 9.2 API reference

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/` | Serves the UI |
| GET | `/health` | Status, available models, default model, image size, device |
| GET | `/models` | `{models: [...], default: "best.pt"}` |
| GET | `/history?limit=8` | Most recent saved runs, newest first |
| POST | `/predict` | Runs detection on one image (multipart form) |
| GET | `/docs` | Interactive OpenAPI documentation |
| GET | `/results/...` | Static access to saved files and `report.html` |

`POST /predict` form fields: `file` (image, required), `conf` (float, default 0.35), `model` (file name from `/models`, default `best.pt` if present), `save` (bool, default false).

Example response:

```json
{
  "width": 1280, "height": 720, "ms": 492.5, "model": "best.onnx",
  "detections": [
    {"label": "forklift", "conf": 0.9, "box": [152.0, 71.3, 255.8, 266.0]}
  ],
  "saved": "20261020_043327_150_image"
}
```

Boxes are `[x1, y1, x2, y2]` in original image pixels. `saved` is the run folder name, or `null` for a preview. Errors: `400` for an invalid image and `404` for an unknown model name (only names found in `models/` are accepted, which also prevents path tricks).

### 9.3 Inference configuration

The constants mirror the notebook: image size 768, NMS IoU 0.7, default confidence 0.35, device CPU. Images are corrected for EXIF rotation and converted to RGB. Each model is loaded lazily on first use, warmed up with a dummy image and cached in memory. The endpoint is a plain function, so FastAPI runs it in a worker thread and the server stays responsive.

### 9.4 Results folder

```
results/
  summary.csv      # one row per saved image
  report.html      # totals and thumbnails, regenerated after every save
  20261020_043327_150_image/
    original.jpg
    annotated.jpg
    detections.json   # every box: label, confidence, coordinates
```

`summary.csv` columns: `timestamp, image, model, conf, forklifts, pallets, total, inference_ms, folder`. The report shows image, model, confidence, counts per class, inference time and date for every run, plus total images, forklifts and pallets.

## 10. Setup and run

```
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open http://127.0.0.1:8000. `requirements.txt` lists `fastapi`, `uvicorn[standard]`, `python-multipart`, `ultralytics`, `onnxruntime` and `pillow`. Run the server from the terminal where the virtual environment is active, and select that environment as the interpreter in VS Code.

## 11. Limitations and future work

- **Domain shift:** training images are public datasets, so scores may not transfer to Petrochoice cameras. Next step: label real frames and test on them.
- **Small and occluded pallets** are the weakest case (62.5% found when small). Options: the 960 px preset, more data with floor-level, wrapped and angled pallets, and a larger model.
- **Incomplete annotations:** each source labels one class, so some objects count as background. Options: review pseudo-labels and active learning.
- **Small test set:** scores are noisy until confirmed on a representative test set.
- **Video:** add ByteTrack for stable IDs so one pallet is not counted twice.
- **Target hardware:** optimise for the deployment device (for example OpenVINO on Intel CPUs), because speed was measured on Kaggle and a laptop, not on the target.

## 12. Reproducibility and licensing

Training used Ultralytics 8.4.182, PyTorch 2.11.0 and ONNX Runtime 1.31.0 on Kaggle, with seed 42 and the settings in section 5.2. Ultralytics is released under AGPL-3.0, and the licenses of both datasets should be checked before any commercial use.
