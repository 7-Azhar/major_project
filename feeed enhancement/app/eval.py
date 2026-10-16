"""Downstream AI Machine Perception Evaluation Architecture for AURA Engine.

NOTE & DISCLAIMER:
The evaluation interfaces in this module (including YoloEvaluator) currently serve as an
EVALUATION INTERFACE / SCAFFOLD to verify pipeline evaluation integration and metric contracts.
They use synthetic contour feature detection heuristics and must NOT be presented as a
real trained YOLO deep neural network experiment until actual deep neural network weights (e.g. YOLOv8 ONNX)
are loaded and evaluated in a future acceleration phase.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import cv2
import numpy as np


@dataclass
class BoundingBox:
    """Represents object detection bounding box annotation."""

    class_id: int
    class_name: str
    confidence: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass
class DetectionEvaluationResult:
    """Holds summary downstream machine perception performance metrics."""

    evaluator_name: str
    processor_name: str
    total_frames_evaluated: int = 0
    precision: float = 0.0
    recall: float = 0.0
    map_50: float = 0.0
    avg_confidence_raw: float = 0.0
    avg_confidence_enhanced: float = 0.0
    confidence_delta_pct: float = 0.0
    false_positives_raw: int = 0
    false_positives_enhanced: int = 0
    missed_detections_raw: int = 0
    missed_detections_enhanced: int = 0
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator_name": self.evaluator_name,
            "processor_name": self.processor_name,
            "total_frames_evaluated": self.total_frames_evaluated,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "map_50": round(self.map_50, 4),
            "avg_confidence_raw": round(self.avg_confidence_raw, 4),
            "avg_confidence_enhanced": round(self.avg_confidence_enhanced, 4),
            "confidence_delta_pct": round(self.confidence_delta_pct, 2),
            "false_positives_raw": self.false_positives_raw,
            "false_positives_enhanced": self.false_positives_enhanced,
            "missed_detections_raw": self.missed_detections_raw,
            "missed_detections_enhanced": self.missed_detections_enhanced,
            "notes": self.notes,
        }


class BaseDetectionEvaluator(abc.ABC):
    """Abstract interface contract for downstream object detector evaluators."""

    @abc.abstractmethod
    def evaluate_stream(
        self,
        raw_frames: List[np.ndarray],
        enhanced_frames: List[np.ndarray],
        processor_name: str = "Enhanced",
        ground_truth_annotations: Optional[List[List[BoundingBox]]] = None,
    ) -> DetectionEvaluationResult:
        """Runs downstream detector on raw vs enhanced frames and compares detection metrics."""
        pass


class YoloEvaluator(BaseDetectionEvaluator):
    """
    YOLO Downstream Machine Perception Evaluator.

    Evaluates object detection performance across Raw FPV feeds vs Enhanced feeds.
    Detects target objects, computes IoU matching against ground truth boxes, and
    measures Precision, Recall, mAP@0.5, Confidence Shifts, and False Positive drops.
    """

    def __init__(self, iou_threshold: float = 0.5, conf_threshold: float = 0.25):
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold

    @staticmethod
    def compute_iou(box1: BoundingBox, box2: BoundingBox) -> float:
        """Compute Intersection over Union (IoU) between two bounding boxes."""
        xi1 = max(box1.xmin, box2.xmin)
        yi1 = max(box1.ymin, box2.ymin)
        xi2 = min(box1.xmax, box2.xmax)
        yi2 = min(box1.ymax, box2.ymax)

        inter_width = max(0.0, xi2 - xi1)
        inter_height = max(0.0, yi2 - yi1)
        inter_area = inter_width * inter_height

        box1_area = max(0.0, box1.xmax - box1.xmin) * max(0.0, box1.ymax - box1.ymin)
        box2_area = max(0.0, box2.xmax - box2.xmin) * max(0.0, box2.ymax - box2.ymin)

        union_area = box1_area + box2_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _detect_synthetic_features(self, frame: np.ndarray) -> List[BoundingBox]:
        """
        Lightweight color & gradient region detector for synthetic/test frames
        when standalone YOLO weights are absent.
        """
        boxes: List[BoundingBox] = []
        if frame is None or frame.size == 0:
            return boxes

        # Convert to HSV to find prominent target shapes (e.g., cyan/yellow/magenta objects)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect yellow moving circle in synthetic stream
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

        contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area > 300:
                x, y, w, h = cv2.boundingRect(c)
                # Compute detection confidence based on contour circularity / contrast
                perimeter = cv2.arcLength(c, True)
                circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
                conf = min(0.99, max(0.30, float(circularity * 0.95)))
                boxes.append(BoundingBox(0, "target_drone", conf, float(x), float(y), float(x + w), float(y + h)))

        # Detect magenta rotating box in synthetic stream
        lower_magenta = np.array([140, 100, 100])
        upper_magenta = np.array([170, 255, 255])
        mask_magenta = cv2.inRange(hsv, lower_magenta, upper_magenta)
        contours_mag, _ = cv2.findContours(mask_magenta, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_mag:
            area = cv2.contourArea(c)
            if area > 200:
                x, y, w, h = cv2.boundingRect(c)
                conf = min(0.98, max(0.35, float(area / (w * h + 1e-5))))
                boxes.append(BoundingBox(1, "target_vehicle", conf, float(x), float(y), float(x + w), float(y + h)))

        return boxes

    def evaluate_stream(
        self,
        raw_frames: List[np.ndarray],
        enhanced_frames: List[np.ndarray],
        processor_name: str = "Enhanced",
        ground_truth_annotations: Optional[List[List[BoundingBox]]] = None,
    ) -> DetectionEvaluationResult:
        """Evaluates downstream detection metrics between raw and enhanced streams."""
        total_frames = min(len(raw_frames), len(enhanced_frames))
        if total_frames == 0:
            return DetectionEvaluationResult(
                evaluator_name="YoloEvaluator",
                processor_name=processor_name,
                notes="No frames provided for evaluation.",
            )

        conf_sum_raw = 0.0
        conf_sum_enh = 0.0
        count_raw_dets = 0
        count_enh_dets = 0

        tp_raw, fp_raw, fn_raw = 0, 0, 0
        tp_enh, fp_enh, fn_enh = 0, 0, 0

        import cv2

        for i in range(total_frames):
            rf = raw_frames[i]
            ef = enhanced_frames[i]

            dets_raw = self._detect_synthetic_features(rf)
            dets_enh = self._detect_synthetic_features(ef)

            for d in dets_raw:
                conf_sum_raw += d.confidence
                count_raw_dets += 1

            for d in dets_enh:
                conf_sum_enh += d.confidence
                count_enh_dets += 1

            # Ground truth matching if provided, else use raw consensus
            gt_boxes = (
                ground_truth_annotations[i]
                if ground_truth_annotations and i < len(ground_truth_annotations)
                else dets_raw
            )

            # Match Raw Detections
            matched_gt_raw = set()
            for d in dets_raw:
                matched = False
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx not in matched_gt_raw and self.compute_iou(d, gt) >= self.iou_threshold:
                        matched_gt_raw.add(gt_idx)
                        matched = True
                        break
                if matched:
                    tp_raw += 1
                else:
                    fp_raw += 1
            fn_raw += len(gt_boxes) - len(matched_gt_raw)

            # Match Enhanced Detections
            matched_gt_enh = set()
            for d in dets_enh:
                matched = False
                for gt_idx, gt in enumerate(gt_boxes):
                    if gt_idx not in matched_gt_enh and self.compute_iou(d, gt) >= self.iou_threshold:
                        matched_gt_enh.add(gt_idx)
                        matched = True
                        break
                if matched:
                    tp_enh += 1
                else:
                    fp_enh += 1
            fn_enh += len(gt_boxes) - len(matched_gt_enh)

        prec_enh = tp_enh / (tp_enh + fp_enh + 1e-6)
        rec_enh = tp_enh / (tp_enh + fn_enh + 1e-6)
        map50_enh = (prec_enh + rec_enh) / 2.0  # Proxy mAP@0.5

        avg_conf_raw = conf_sum_raw / max(1, count_raw_dets)
        avg_conf_enh = conf_sum_enh / max(1, count_enh_dets)
        conf_delta_pct = (
            ((avg_conf_enh - avg_conf_raw) / max(1e-5, avg_conf_raw)) * 100.0
            if avg_conf_raw > 0
            else 0.0
        )

        return DetectionEvaluationResult(
            evaluator_name="YoloEvaluator",
            processor_name=processor_name,
            total_frames_evaluated=total_frames,
            precision=float(prec_enh),
            recall=float(rec_enh),
            map_50=float(map50_enh),
            avg_confidence_raw=float(avg_conf_raw),
            avg_confidence_enhanced=float(avg_conf_enh),
            confidence_delta_pct=float(conf_delta_pct),
            false_positives_raw=fp_raw,
            false_positives_enhanced=fp_enh,
            missed_detections_raw=fn_raw,
            missed_detections_enhanced=fn_enh,
            notes="Evaluated downstream object detection performance on target features.",
        )
