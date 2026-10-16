"""Phase 3B — Real Downstream Object Detection Evaluation.

Evaluates YOLOv8n detection accuracy across 4 pipeline configurations
using the VisDrone2019-DET validation set (real annotated UAV imagery).

Pipelines:
  A: Raw high-resolution baseline (original → YOLO)
  B: Raw reduced-resolution baseline (original → resize 640×360 → YOLO)
  C: Reduced-resolution enhanced (original → resize 640×360 → Zero-DCE → YOLO)
  D: High-resolution enhanced (original → resize 720p → Zero-DCE → YOLO)
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# CUDA / ONNX Runtime setup (must come before ort import)
# ---------------------------------------------------------------------------
cudnn_dir = r"C:\Users\azhar\CODES\ai\.venv\Lib\site-packages\torch\lib"
cuda_dir = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"
if os.path.exists(cudnn_dir):
    os.add_dll_directory(cudnn_dir)
if os.path.exists(cuda_dir):
    os.add_dll_directory(cuda_dir)

local_libs = os.path.abspath(".venv_libs")
if os.path.exists(local_libs) and local_libs not in sys.path:
    sys.path.insert(0, local_libs)

import onnxruntime as ort

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# VisDrone → COCO class mapping
# NOTE: This is a cross-dataset mapping and is imperfect.
# VisDrone "van" maps to COCO "car" (semantic mismatch possible).
# VisDrone "people" (non-upright) maps to COCO "person".
VISDRONE_TO_COCO = {
    1: 0,   # pedestrian → person
    2: 0,   # people → person
    3: 1,   # bicycle → bicycle
    4: 2,   # car → car
    5: 2,   # van → car
    6: 7,   # truck → truck
    9: 5,   # bus → bus
    10: 3,  # motor → motorcycle
}

# Reverse: COCO class IDs we care about → human-readable names
EVAL_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

EVAL_CLASS_IDS = sorted(EVAL_CLASSES.keys())

# YOLO settings (fixed across all configs)
YOLO_CONF = 0.25
YOLO_IOU = 0.45
YOLO_IMGSZ = 640


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_visdrone_annotation(ann_path: str) -> List[Dict]:
    """Parse a single VisDrone annotation file.

    Format per line: bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion
    Returns list of dicts with keys: bbox_xyxy, class_id (COCO), visdrone_cat
    """
    boxes = []
    with open(ann_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            score = int(parts[4])
            cat = int(parts[5])

            # Skip ignored regions, score=0, and unmapped categories
            if score == 0 or cat not in VISDRONE_TO_COCO:
                continue
            if w <= 0 or h <= 0:
                continue

            coco_id = VISDRONE_TO_COCO[cat]
            boxes.append({
                "bbox_xyxy": [x, y, x + w, y + h],
                "class_id": coco_id,
                "class_name": EVAL_CLASSES[coco_id],
                "visdrone_cat": cat,
                "width": w,
                "height": h,
            })
    return boxes


def scale_gt_boxes(gt_boxes: List[Dict], orig_hw: Tuple[int, int],
                   target_hw: Tuple[int, int]) -> List[Dict]:
    """Scale ground-truth boxes from original image size to target size."""
    oh, ow = orig_hw
    th, tw = target_hw
    sx = tw / ow
    sy = th / oh
    scaled = []
    for b in gt_boxes:
        x1, y1, x2, y2 = b["bbox_xyxy"]
        scaled.append({
            **b,
            "bbox_xyxy": [x1 * sx, y1 * sy, x2 * sx, y2 * sy],
        })
    return scaled


# ---------------------------------------------------------------------------
# IoU / mAP computation
# ---------------------------------------------------------------------------

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions_to_gt(
    preds: List[Dict], gt: List[Dict], iou_threshold: float
) -> Tuple[List[bool], List[bool]]:
    """Match predictions to ground truth for a single image at a given IoU threshold.

    Returns:
        pred_tp: list of bool, same length as preds (True if matched to a GT)
        gt_matched: list of bool, same length as gt (True if matched by a pred)
    """
    pred_tp = [False] * len(preds)
    gt_matched = [False] * len(gt)

    # Sort predictions by confidence (highest first)
    sorted_indices = sorted(range(len(preds)), key=lambda i: preds[i]["confidence"], reverse=True)

    for pi in sorted_indices:
        best_iou = 0.0
        best_gi = -1
        for gi, g in enumerate(gt):
            if gt_matched[gi]:
                continue
            if preds[pi]["class_id"] != g["class_id"]:
                continue
            iou = compute_iou(preds[pi]["bbox_xyxy"], g["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_threshold and best_gi >= 0:
            pred_tp[pi] = True
            gt_matched[best_gi] = True

    return pred_tp, gt_matched


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """Compute AP using the 101-point interpolation (COCO style)."""
    if not precisions or not recalls:
        return 0.0
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        prec_at_t = 0.0
        for i in range(len(mrec)):
            if mrec[i] >= t:
                prec_at_t = max(prec_at_t, mpre[i])
                break
        ap += prec_at_t
    return ap / 101.0


def compute_metrics(
    all_preds: List[List[Dict]],
    all_gt: List[List[Dict]],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Compute detection metrics across all images.

    Returns dict with precision, recall, mAP@0.5, mAP@0.5:0.95, per-class AP, etc.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.5 to 0.95

    # Flatten all predictions and GT with image indices
    all_pred_flat = []
    all_gt_flat = []
    for img_i, (preds, gt) in enumerate(zip(all_preds, all_gt)):
        for p in preds:
            all_pred_flat.append({**p, "img_idx": img_i})
        for g in gt:
            all_gt_flat.append({**g, "img_idx": img_i})

    total_gt = len(all_gt_flat)
    total_preds = len(all_pred_flat)

    results = {
        "total_gt": total_gt,
        "total_preds": total_preds,
        "per_class": {},
        "confidence_distribution": {},
    }

    # Confidence distribution
    if all_pred_flat:
        confs = [p["confidence"] for p in all_pred_flat]
        results["confidence_distribution"] = {
            "min": float(np.min(confs)),
            "max": float(np.max(confs)),
            "mean": float(np.mean(confs)),
            "median": float(np.median(confs)),
            "p25": float(np.percentile(confs, 25)),
            "p75": float(np.percentile(confs, 75)),
        }

    # Per-class, per-IoU-threshold AP
    class_aps_all_iou = defaultdict(list)  # class_id → [AP at each IoU threshold]

    for iou_thr in iou_thresholds:
        for cls_id in EVAL_CLASS_IDS:
            # Collect class-specific predictions and GT per image
            cls_preds_by_img = defaultdict(list)
            cls_gt_by_img = defaultdict(list)

            for p in all_pred_flat:
                if p["class_id"] == cls_id:
                    cls_preds_by_img[p["img_idx"]].append(p)
            for g in all_gt_flat:
                if g["class_id"] == cls_id:
                    cls_gt_by_img[g["img_idx"]].append(g)

            # Match across all images
            all_tp = []
            all_conf = []
            total_cls_gt = sum(len(v) for v in cls_gt_by_img.values())

            all_img_indices = set(list(cls_preds_by_img.keys()) + list(cls_gt_by_img.keys()))
            for img_i in all_img_indices:
                preds_i = cls_preds_by_img.get(img_i, [])
                gt_i = cls_gt_by_img.get(img_i, [])
                pred_tp, _ = match_predictions_to_gt(preds_i, gt_i, iou_thr)
                for pi, p in enumerate(preds_i):
                    all_tp.append(pred_tp[pi])
                    all_conf.append(p["confidence"])

            if total_cls_gt == 0:
                class_aps_all_iou[cls_id].append(0.0)
                continue

            # Sort by confidence
            sorted_idx = sorted(range(len(all_conf)), key=lambda i: all_conf[i], reverse=True)
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []

            for idx in sorted_idx:
                if all_tp[idx]:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                precisions.append(tp_cumsum / (tp_cumsum + fp_cumsum))
                recalls.append(tp_cumsum / total_cls_gt)

            ap = compute_ap(precisions, recalls)
            class_aps_all_iou[cls_id].append(ap)

    # Compute per-class summary
    total_tp_50 = 0
    total_fp_50 = 0
    total_fn_50 = 0

    for cls_id in EVAL_CLASS_IDS:
        cls_name = EVAL_CLASSES[cls_id]
        aps = class_aps_all_iou[cls_id]
        ap50 = aps[0] if aps else 0.0
        ap50_95 = float(np.mean(aps)) if aps else 0.0

        # Count TP/FP/FN at IoU=0.5 for this class
        cls_tp = 0
        cls_fp = 0
        cls_gt_total = 0
        for img_i in range(len(all_preds)):
            preds_cls = [p for p in all_preds[img_i] if p["class_id"] == cls_id]
            gt_cls = [g for g in all_gt[img_i] if g["class_id"] == cls_id]
            cls_gt_total += len(gt_cls)
            if preds_cls and gt_cls:
                tp_flags, gt_flags = match_predictions_to_gt(preds_cls, gt_cls, 0.5)
                cls_tp += sum(tp_flags)
                cls_fp += sum(1 for f in tp_flags if not f)
            elif preds_cls:
                cls_fp += len(preds_cls)

        cls_fn = cls_gt_total - cls_tp
        cls_prec = cls_tp / (cls_tp + cls_fp) if (cls_tp + cls_fp) > 0 else 0.0
        cls_rec = cls_tp / cls_gt_total if cls_gt_total > 0 else 0.0

        results["per_class"][cls_name] = {
            "AP50": round(ap50, 4),
            "AP50_95": round(ap50_95, 4),
            "precision": round(cls_prec, 4),
            "recall": round(cls_rec, 4),
            "TP": cls_tp,
            "FP": cls_fp,
            "FN": cls_fn,
            "GT_count": cls_gt_total,
        }
        total_tp_50 += cls_tp
        total_fp_50 += cls_fp
        total_fn_50 += cls_fn

    # Overall metrics
    all_ap50 = [class_aps_all_iou[c][0] for c in EVAL_CLASS_IDS if class_aps_all_iou[c]]
    all_ap50_95 = [float(np.mean(class_aps_all_iou[c])) for c in EVAL_CLASS_IDS if class_aps_all_iou[c]]

    results["mAP50"] = round(float(np.mean(all_ap50)), 4) if all_ap50 else 0.0
    results["mAP50_95"] = round(float(np.mean(all_ap50_95)), 4) if all_ap50_95 else 0.0
    results["precision"] = round(total_tp_50 / (total_tp_50 + total_fp_50), 4) if (total_tp_50 + total_fp_50) > 0 else 0.0
    results["recall"] = round(total_tp_50 / (total_tp_50 + total_fn_50), 4) if (total_tp_50 + total_fn_50) > 0 else 0.0
    results["total_TP"] = total_tp_50
    results["total_FP"] = total_fp_50
    results["total_FN"] = total_fn_50

    return results


# ---------------------------------------------------------------------------
# Zero-DCE ONNX Enhancement
# ---------------------------------------------------------------------------

class ZeroDCEEnhancer:
    """Zero-DCE ONNX CUDA enhancer for evaluation."""

    def __init__(self, onnx_path: str, fp16: bool = True):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CUDAExecutionProvider"]
        )
        active = self.session.get_providers()
        if "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"CUDAExecutionProvider NOT active! Got: {active}. "
                "Cannot proceed — would silently fall back to CPU."
            )
        self.fp16 = fp16
        self.provider = active[0]
        print(f"  Zero-DCE ONNX session: provider={self.provider}", flush=True)

    def enhance(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhance a BGR image. Returns (enhanced_bgr, latency_ms)."""
        t0 = time.perf_counter()
        rgb = img_bgr[:, :, ::-1].astype(np.float16 if self.fp16 else np.float32) / 255.0
        tensor_in = np.ascontiguousarray(
            rgb.transpose(2, 0, 1)[np.newaxis, :, :, :]
        )
        ort_out = self.session.run(None, {"input": tensor_in})[0]
        out = np.clip(ort_out[0].transpose(1, 2, 0), 0, 1)
        enhanced_bgr = (out * 255).astype(np.uint8)[:, :, ::-1]
        t1 = time.perf_counter()
        return np.ascontiguousarray(enhanced_bgr), (t1 - t0) * 1000


# ---------------------------------------------------------------------------
# YOLO Detector Wrapper
# ---------------------------------------------------------------------------

class YOLODetector:
    """Wrapper around Ultralytics YOLOv8n for consistent evaluation."""

    def __init__(self, weights: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.model_name = weights
        print(f"  YOLO loaded: {weights}", flush=True)

    def detect(self, img_bgr: np.ndarray) -> Tuple[List[Dict], float]:
        """Run detection on a BGR image.

        Returns (predictions, latency_ms).
        Each prediction: {bbox_xyxy, class_id, class_name, confidence}
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            img_bgr,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            imgsz=YOLO_IMGSZ,
            verbose=False,
            device="cpu",
        )
        t1 = time.perf_counter()

        preds = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in EVAL_CLASSES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                preds.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": EVAL_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                })
        return preds, (t1 - t0) * 1000


# ---------------------------------------------------------------------------
# Pipeline Runners
# ---------------------------------------------------------------------------

def run_pipeline_a(img_bgr: np.ndarray, gt_boxes: List[Dict],
                   orig_hw: Tuple[int, int], detector: YOLODetector
                   ) -> Dict:
    """Pipeline A: Raw high-resolution baseline.
    
    Uses original image as-is. YOLO internally handles letterboxing to imgsz=640.
    GT boxes remain in original image coordinates.
    """
    h, w = img_bgr.shape[:2]
    t_pre_start = time.perf_counter()
    # No resize — use original resolution
    t_pre_end = time.perf_counter()

    # GT boxes are already in original coordinates, no scaling needed
    preds, yolo_ms = detector.detect(img_bgr)

    return {
        "preds": preds,
        "gt": gt_boxes,  # original coordinates
        "image": img_bgr,
        "timing": {
            "preprocess_ms": (t_pre_end - t_pre_start) * 1000,
            "enhance_ms": 0.0,
            "yolo_ms": yolo_ms,
            "total_ms": (t_pre_end - t_pre_start) * 1000 + yolo_ms,
        },
        "resolution": f"{w}x{h}",
    }


def run_pipeline_b(img_bgr: np.ndarray, gt_boxes: List[Dict],
                   orig_hw: Tuple[int, int], detector: YOLODetector
                   ) -> Dict:
    """Pipeline B: Raw reduced-resolution baseline (640×360)."""
    t_pre_start = time.perf_counter()
    img_resized = cv2.resize(img_bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
    t_pre_end = time.perf_counter()

    target_hw = (360, 640)
    scaled_gt = scale_gt_boxes(gt_boxes, orig_hw, target_hw)

    preds, yolo_ms = detector.detect(img_resized)

    return {
        "preds": preds,
        "gt": scaled_gt,
        "image": img_resized,
        "timing": {
            "preprocess_ms": (t_pre_end - t_pre_start) * 1000,
            "enhance_ms": 0.0,
            "yolo_ms": yolo_ms,
            "total_ms": (t_pre_end - t_pre_start) * 1000 + yolo_ms,
        },
        "resolution": "640x360",
    }


def run_pipeline_c(img_bgr: np.ndarray, gt_boxes: List[Dict],
                   orig_hw: Tuple[int, int], detector: YOLODetector,
                   enhancer: ZeroDCEEnhancer) -> Dict:
    """Pipeline C: Reduced-resolution enhanced (640×360 + Zero-DCE)."""
    t_pre_start = time.perf_counter()
    img_resized = cv2.resize(img_bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
    t_pre_end = time.perf_counter()

    enhanced, enhance_ms = enhancer.enhance(img_resized)

    target_hw = (360, 640)
    scaled_gt = scale_gt_boxes(gt_boxes, orig_hw, target_hw)

    preds, yolo_ms = detector.detect(enhanced)

    return {
        "preds": preds,
        "gt": scaled_gt,
        "image": enhanced,
        "timing": {
            "preprocess_ms": (t_pre_end - t_pre_start) * 1000,
            "enhance_ms": enhance_ms,
            "yolo_ms": yolo_ms,
            "total_ms": (t_pre_end - t_pre_start) * 1000 + enhance_ms + yolo_ms,
        },
        "resolution": "640x360",
    }


def run_pipeline_d(img_bgr: np.ndarray, gt_boxes: List[Dict],
                   orig_hw: Tuple[int, int], detector: YOLODetector,
                   enhancer_720: ZeroDCEEnhancer) -> Dict:
    """Pipeline D: High-resolution enhanced (720p + Zero-DCE).
    
    All images are resized to exactly 1280x720 to match the static ONNX input shape.
    This is a quality-reference configuration, NOT a claimed real-time pipeline.
    """
    t_pre_start = time.perf_counter()
    # Always resize to 1280x720 for the static ONNX model
    img_resized = cv2.resize(img_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)
    t_pre_end = time.perf_counter()

    target_hw = (720, 1280)
    scaled_gt = scale_gt_boxes(gt_boxes, orig_hw, target_hw)

    enhanced, enhance_ms = enhancer_720.enhance(img_resized)
    preds, yolo_ms = detector.detect(enhanced)

    return {
        "preds": preds,
        "gt": scaled_gt,
        "image": enhanced,
        "timing": {
            "preprocess_ms": (t_pre_end - t_pre_start) * 1000,
            "enhance_ms": enhance_ms,
            "yolo_ms": yolo_ms,
            "total_ms": (t_pre_end - t_pre_start) * 1000 + enhance_ms + yolo_ms,
        },
        "resolution": "1280x720",
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_detections(img: np.ndarray, preds: List[Dict], gt: List[Dict],
                    iou_threshold: float = 0.5) -> np.ndarray:
    """Draw GT and predictions on image with TP/FP/FN annotations."""
    vis = img.copy()
    pred_tp, gt_matched = match_predictions_to_gt(preds, gt, iou_threshold)

    # Draw GT boxes
    for gi, g in enumerate(gt):
        x1, y1, x2, y2 = [int(v) for v in g["bbox_xyxy"]]
        if gt_matched[gi]:
            # Matched GT: green solid
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
        else:
            # Missed GT (FN): yellow dashed
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            label = f"MISS:{g['class_name']}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

    # Draw predictions
    for pi, p in enumerate(preds):
        x1, y1, x2, y2 = [int(v) for v in p["bbox_xyxy"]]
        if pred_tp[pi]:
            # True positive: blue
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 150, 0), 2)
            label = f"TP:{p['class_name']} {p['confidence']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 150, 0), 1)
        else:
            # False positive: red
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"FP:{p['class_name']} {p['confidence']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    return vis


def save_error_analysis(
    all_results: Dict[str, List[Dict]],
    image_names: List[str],
    output_dir: str,
    max_examples: int = 8,
):
    """Save representative visual error analysis examples."""
    os.makedirs(output_dir, exist_ok=True)

    # Find interesting cases by comparing pipelines
    cases = {
        "enhancement_helps": [],       # missed by raw_360 but detected by enhanced_360
        "enhancement_hurts": [],       # detected by raw_360 but missed by enhanced_360
        "enhancement_fp": [],          # false positive only in enhanced_360
        "resolution_impact": [],       # detected at 720p but missed at 360p
        "small_object_comparison": [], # images with small objects
    }

    for i, name in enumerate(image_names):
        res_a = all_results["A"][i]
        res_b = all_results["B"][i]
        res_c = all_results["C"][i]

        gt_b = res_b["gt"]
        gt_c = res_c["gt"]

        # Match at IoU=0.5
        _, gt_matched_b = match_predictions_to_gt(res_b["preds"], gt_b, 0.5)
        _, gt_matched_c = match_predictions_to_gt(res_c["preds"], gt_c, 0.5)
        pred_tp_b, _ = match_predictions_to_gt(res_b["preds"], gt_b, 0.5)
        pred_tp_c, _ = match_predictions_to_gt(res_c["preds"], gt_c, 0.5)

        # Enhancement helps: GT matched in C but not B
        missed_b = sum(1 for m in gt_matched_b if not m)
        missed_c = sum(1 for m in gt_matched_c if not m)
        if missed_b > missed_c:
            cases["enhancement_helps"].append((i, missed_b - missed_c))

        # Enhancement hurts: GT matched in B but not C
        if missed_c > missed_b:
            cases["enhancement_hurts"].append((i, missed_c - missed_b))

        # Enhancement FP: FP in C but not B
        fp_b = sum(1 for t in pred_tp_b if not t)
        fp_c = sum(1 for t in pred_tp_c if not t)
        if fp_c > fp_b:
            cases["enhancement_fp"].append((i, fp_c - fp_b))

        # Resolution impact: detected at 720p, missed at 360p
        gt_a = res_a["gt"]
        _, gt_matched_a = match_predictions_to_gt(res_a["preds"], gt_a, 0.5)
        missed_a = sum(1 for m in gt_matched_a if not m)
        if missed_b > missed_a:
            cases["resolution_impact"].append((i, missed_b - missed_a))

        # Small objects: images with many small GT boxes
        small_count = sum(1 for g in res_a["gt"] if
                         (g["bbox_xyxy"][2] - g["bbox_xyxy"][0]) *
                         (g["bbox_xyxy"][3] - g["bbox_xyxy"][1]) < 1024)
        if small_count >= 3:
            cases["small_object_comparison"].append((i, small_count))

    # Save top examples for each case
    saved_count = 0
    for case_name, case_list in cases.items():
        case_list.sort(key=lambda x: x[1], reverse=True)
        for rank, (img_i, _) in enumerate(case_list[:max_examples]):
            # Save side-by-side comparison for relevant pipelines
            imgs = {}
            for pipe_name in ["A", "B", "C"]:
                r = all_results[pipe_name][img_i]
                vis = draw_detections(r["image"], r["preds"], r["gt"])
                imgs[pipe_name] = vis

            # Create side-by-side (resize all to same height for concat)
            target_h = 360
            resized = []
            labels = ["Raw 720p", "Raw 360p", "Enhanced 360p"]
            for pipe_name, label in zip(["A", "B", "C"], labels):
                im = imgs[pipe_name]
                scale = target_h / im.shape[0]
                im_r = cv2.resize(im, (int(im.shape[1] * scale), target_h))
                # Add label bar
                bar = np.zeros((30, im_r.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                im_labeled = np.vstack([bar, im_r])
                resized.append(im_labeled)

            # Pad to same width if needed
            max_w = max(im.shape[1] for im in resized)
            padded = []
            for im in resized:
                if im.shape[1] < max_w:
                    pad = np.zeros((im.shape[0], max_w - im.shape[1], 3), dtype=np.uint8)
                    im = np.hstack([im, pad])
                padded.append(im)

            combined = np.hstack(padded)
            fname = f"{case_name}_{rank+1}_{image_names[img_i]}"
            cv2.imwrite(os.path.join(output_dir, fname), combined)
            saved_count += 1

    print(f"  Saved {saved_count} visual analysis examples to {output_dir}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3B Detection Evaluation")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (10 images)")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0=all)")
    args = parser.parse_args()

    print("=" * 70, flush=True)
    print("AURA Phase 3B — Real Downstream Object Detection Evaluation", flush=True)
    print("Dataset: VisDrone2019-DET validation set (UAV imagery benchmark)", flush=True)
    print("=" * 70, flush=True)

    # Paths
    images_dir = os.path.join("data", "visdrone", "VisDrone2019-DET-val", "images")
    annots_dir = os.path.join("data", "visdrone", "VisDrone2019-DET-val", "annotations")

    if not os.path.isdir(images_dir) or not os.path.isdir(annots_dir):
        print(f"ERROR: VisDrone dataset not found at {images_dir}", flush=True)
        print("Run: python -m scripts.download_visdrone", flush=True)
        sys.exit(1)

    # Collect image list
    all_images = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(".jpg")])
    print(f"\nFound {len(all_images)} images in VisDrone val set", flush=True)

    if args.smoke:
        # Select 10 diverse images (evenly spaced)
        step = max(1, len(all_images) // 10)
        all_images = all_images[::step][:10]
        print(f"SMOKE TEST: Using {len(all_images)} images", flush=True)
    elif args.max_images > 0:
        all_images = all_images[:args.max_images]
        print(f"Limited to {len(all_images)} images", flush=True)

    # Initialize components
    print("\nInitializing components...", flush=True)
    detector = YOLODetector("yolov8n.pt")

    # Zero-DCE ONNX models (FP16, 4 curve iterations — consistent with Phase 3A.3)
    onnx_360 = "outputs/models/zerodce_fp16_c4_640x360.onnx"
    onnx_720 = "outputs/models/zerodce_fp16_c4_1280x720.onnx"

    # Fallbacks
    if not os.path.exists(onnx_360):
        onnx_360 = "outputs/models/zerodce_fp16_c4.onnx"
    if not os.path.exists(onnx_720):
        onnx_720 = "outputs/models/zerodce_fp16.onnx"  # 8-curve fallback

    print(f"  Zero-DCE 360p ONNX: {onnx_360} (exists={os.path.exists(onnx_360)})", flush=True)
    print(f"  Zero-DCE 720p ONNX: {onnx_720} (exists={os.path.exists(onnx_720)})", flush=True)

    enhancer_360 = ZeroDCEEnhancer(onnx_360, fp16=True)

    # Pipeline D: all images resized to exactly 1280x720 to match ONNX input
    enhancer_720 = None
    try:
        enhancer_720 = ZeroDCEEnhancer(onnx_720, fp16=True)
    except Exception as e:
        print(f"  WARNING: Could not load 720p enhancer: {e}", flush=True)
        print("  Pipeline D will be skipped.", flush=True)

    # Process all images through all pipelines
    print(f"\n{'='*70}", flush=True)
    print(f"Processing {len(all_images)} images through all pipelines...", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = {"A": [], "B": [], "C": []}
    if enhancer_720:
        all_results["D"] = []

    for idx, img_name in enumerate(all_images):
        img_path = os.path.join(images_dir, img_name)
        ann_name = os.path.splitext(img_name)[0] + ".txt"
        ann_path = os.path.join(annots_dir, ann_name)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"  SKIP: Could not read {img_name}", flush=True)
            continue

        orig_h, orig_w = img_bgr.shape[:2]
        orig_hw = (orig_h, orig_w)

        # Parse ground truth
        gt_boxes = parse_visdrone_annotation(ann_path) if os.path.exists(ann_path) else []

        # Pipeline A: Raw high-res
        res_a = run_pipeline_a(img_bgr, gt_boxes, orig_hw, detector)
        all_results["A"].append(res_a)

        # Pipeline B: Raw 360p
        res_b = run_pipeline_b(img_bgr, gt_boxes, orig_hw, detector)
        all_results["B"].append(res_b)

        # Pipeline C: Enhanced 360p
        res_c = run_pipeline_c(img_bgr, gt_boxes, orig_hw, detector, enhancer_360)
        all_results["C"].append(res_c)

        # Pipeline D: Enhanced 720p (if available)
        if enhancer_720:
            try:
                res_d = run_pipeline_d(img_bgr, gt_boxes, orig_hw, detector, enhancer_720)
                all_results["D"].append(res_d)
            except Exception as e:
                # Some images may have different aspect ratios
                print(f"  Pipeline D error on {img_name}: {e}", flush=True)
                all_results["D"].append(None)

        if (idx + 1) % 10 == 0 or idx == len(all_images) - 1:
            print(f"  [{idx+1}/{len(all_images)}] {img_name} | "
                  f"A:{res_a['timing']['total_ms']:.0f}ms "
                  f"B:{res_b['timing']['total_ms']:.0f}ms "
                  f"C:{res_c['timing']['total_ms']:.0f}ms",
                  flush=True)

    # Compute metrics for each pipeline
    print(f"\n{'='*70}", flush=True)
    print("Computing detection metrics...", flush=True)
    print(f"{'='*70}", flush=True)

    pipeline_labels = {
        "A": "Raw baseline (720p)",
        "B": "Low-res baseline (640x360)",
        "C": "AURA candidate (640x360 + Zero-DCE FP16 4C)",
    }
    if "D" in all_results:
        pipeline_labels["D"] = "Quality reference (720p + Zero-DCE FP16)"

    report = {
        "experiment": "AURA Phase 3B — Downstream Object Detection Evaluation",
        "dataset": "VisDrone2019-DET validation set (real annotated UAV imagery)",
        "dataset_note": "NOT project FPV footage. Cross-dataset class mapping applied.",
        "num_images": len(all_images),
        "detector": "YOLOv8n (COCO pretrained, v8.4.51)",
        "detector_settings": {
            "confidence_threshold": YOLO_CONF,
            "iou_threshold": YOLO_IOU,
            "input_size": YOLO_IMGSZ,
            "device": "CPU (PyTorch 2.12.0+cpu)",
        },
        "class_mapping": {
            "note": "Cross-dataset mapping (VisDrone→COCO). Imperfect: van→car, people→person.",
            "classes_evaluated": list(EVAL_CLASSES.values()),
        },
        "pipelines": {},
    }

    for pipe_key, pipe_label in pipeline_labels.items():
        results_list = all_results[pipe_key]
        # Filter out None entries (pipeline D failures)
        valid = [r for r in results_list if r is not None]
        if not valid:
            continue

        all_preds = [r["preds"] for r in valid]
        all_gt = [r["gt"] for r in valid]

        metrics = compute_metrics(all_preds, all_gt)

        # Timing statistics
        timings = [r["timing"] for r in valid]
        preprocess_ms = [t["preprocess_ms"] for t in timings]
        enhance_ms = [t["enhance_ms"] for t in timings]
        yolo_ms = [t["yolo_ms"] for t in timings]
        total_ms = [t["total_ms"] for t in timings]

        timing_stats = {
            "preprocess": {
                "mean": round(float(np.mean(preprocess_ms)), 2),
                "median": round(float(np.median(preprocess_ms)), 2),
                "p95": round(float(np.percentile(preprocess_ms, 95)), 2),
            },
            "enhancement": {
                "mean": round(float(np.mean(enhance_ms)), 2),
                "median": round(float(np.median(enhance_ms)), 2),
                "p95": round(float(np.percentile(enhance_ms, 95)), 2),
            },
            "yolo": {
                "mean": round(float(np.mean(yolo_ms)), 2),
                "median": round(float(np.median(yolo_ms)), 2),
                "p95": round(float(np.percentile(yolo_ms, 95)), 2),
            },
            "total": {
                "mean": round(float(np.mean(total_ms)), 2),
                "median": round(float(np.median(total_ms)), 2),
                "p95": round(float(np.percentile(total_ms, 95)), 2),
            },
            "effective_fps": round(1000.0 / float(np.median(total_ms)), 2),
        }

        report["pipelines"][pipe_key] = {
            "label": pipe_label,
            "resolution": valid[0]["resolution"],
            "metrics": metrics,
            "timing": timing_stats,
            "images_evaluated": len(valid),
        }

        # Print summary
        print(f"\n  {pipe_label}:", flush=True)
        print(f"    Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}", flush=True)
        print(f"    mAP@0.5:  {metrics['mAP50']:.4f} | mAP@0.5:0.95: {metrics['mAP50_95']:.4f}", flush=True)
        print(f"    TP: {metrics['total_TP']} | FP: {metrics['total_FP']} | FN: {metrics['total_FN']}", flush=True)
        print(f"    Median total latency: {timing_stats['total']['median']:.1f} ms "
              f"({timing_stats['effective_fps']:.1f} FPS)", flush=True)

    # Save reports
    os.makedirs("outputs", exist_ok=True)
    json_path = "outputs/phase3b_detection_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved JSON report: {json_path}", flush=True)

    # CSV summary
    csv_path = "outputs/phase3b_detection_report.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Pipeline", "Resolution", "Enhancement", "Precision", "Recall",
            "mAP50", "mAP50_95", "Median_Total_ms", "FPS",
            "TP", "FP", "FN",
        ])
        for pipe_key in ["A", "B", "C", "D"]:
            if pipe_key not in report["pipelines"]:
                continue
            p = report["pipelines"][pipe_key]
            m = p["metrics"]
            t = p["timing"]
            enh = "None" if pipe_key in ("A", "B") else "Zero-DCE FP16 4C"
            writer.writerow([
                p["label"], p["resolution"], enh,
                m["precision"], m["recall"], m["mAP50"], m["mAP50_95"],
                t["total"]["median"], t["effective_fps"],
                m["total_TP"], m["total_FP"], m["total_FN"],
            ])
    print(f"Saved CSV report: {csv_path}", flush=True)

    # Visual error analysis
    print("\nGenerating visual error analysis...", flush=True)
    vis_dir = "outputs/phase3b_visual_analysis"
    save_error_analysis(all_results, all_images, vis_dir)

    # Print comparison table
    print(f"\n{'='*100}", flush=True)
    print(f"{'Pipeline':<40} | {'Res':<9} | {'Prec':>5} | {'Rec':>5} | {'mAP50':>6} | {'mAP50-95':>8} | {'Med ms':>7} | {'FPS':>5}", flush=True)
    print(f"{'='*100}", flush=True)
    for pipe_key in ["A", "B", "C", "D"]:
        if pipe_key not in report["pipelines"]:
            continue
        p = report["pipelines"][pipe_key]
        m = p["metrics"]
        t = p["timing"]
        print(f"{p['label']:<40} | {p['resolution']:<9} | "
              f"{m['precision']:5.3f} | {m['recall']:5.3f} | "
              f"{m['mAP50']:6.4f} | {m['mAP50_95']:8.4f} | "
              f"{t['total']['median']:7.1f} | {t['effective_fps']:5.1f}",
              flush=True)
    print(f"{'='*100}", flush=True)

    print("\nPhase 3B evaluation complete!", flush=True)
    print("NOTE: YOLO CPU latency is NOT representative of final GPU deployment.", flush=True)
    print("      Detection accuracy comparison is the primary result.", flush=True)


if __name__ == "__main__":
    main()
