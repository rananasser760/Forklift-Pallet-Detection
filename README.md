# Forklift and Pallet Detection

A computer vision project for detecting forklifts and pallets in warehouse environments using YOLO11 and transfer learning.

## Overview

The goal of this project is to build an object detection pipeline that identifies forklifts and pallets in warehouse images. The model predicts object classes, bounding boxes, and confidence scores.

The project focuses on dataset preparation, model fine-tuning, evaluation, visual error analysis, and practical deployment considerations.

## Objectives

- Detect forklifts and pallets in warehouse images.
- Fine-tune a pretrained object detection model on a custom dataset.
- Evaluate detection performance using standard object detection metrics.
- Analyze false positives, false negatives, and challenging visual conditions.
- Identify limitations and potential improvements for real-world use.

## Model and Tools

- **Model:** YOLO11n (pretrained object detection model)
- **Framework:** Ultralytics
- **Language:** Python
- **Deep Learning:** PyTorch
- **Environment:** Kaggle Notebooks
- **Visualization and Analysis:** Matplotlib and the evaluation tools provided by Ultralytics

## Dataset

The project uses a warehouse object detection dataset obtained from Roboflow Universe.

The dataset contains the following annotated classes:

- `forklift`
- `pallet`
- `pallet_truck`
- `small_load_carrier`
- `stillage`

**Dataset split:**

| Split | Number of images |
|---|---:|
| Training | 944 |
| Validation | 131 |
| Test | 0 |
| Total | 1,075 |

The dataset does not currently include a separate test split. Validation metrics will therefore be reported as validation results, not as independent test performance.

Dataset source: [Roboflow Universe — Warehouse Dataset](https://universe.roboflow.com/warehouse-demo/warehouse-o0yuy)

Dataset version, license, and annotation quality should be verified before reuse or redistribution.

## Methodology

1. Inspect the dataset structure, class definitions, and bounding-box annotations.
2. Load pretrained YOLO11n weights.
3. Fine-tune the model on the training split.
4. Evaluate the trained model on the validation split.
5. Visualize predictions and inspect detection errors.
6. Document limitations and potential improvements.

## Evaluation

The model will be evaluated using:

- **Precision:** Measures the proportion of predicted detections that are correct.
- **Recall:** Measures the proportion of ground-truth objects successfully detected.
- **mAP@0.50:** Mean average precision at an IoU threshold of 0.50.
- **mAP@0.50:0.95:** Mean average precision averaged across multiple IoU thresholds.

Final metric values and per-class results will be added after training and evaluation.

## Limitations

Potential challenges include:

- Occluded or partially visible objects.
- Small or distant objects.
- Poor lighting and unusual camera angles.
- Similar-looking warehouse equipment.
- Limited dataset size and class imbalance.
- The absence of an independent test split.

These limitations will be investigated through visual inspection and error analysis.

## Future Improvements

- Evaluate on a separate, representative test set.
- Improve dataset coverage and annotation quality.
- Compare model configurations and image sizes.
- Measure inference latency and throughput.
- Explore deployment through an API if required.

## Project Status

In progress — dataset inspection, model training, and evaluation are being developed.

## Disclaimer

This repository documents an experimental object detection solution. Reported performance will reflect actual completed experiments and the evaluation data used.
