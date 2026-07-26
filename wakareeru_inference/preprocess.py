from dataclasses import dataclass
from typing import Any

from PIL import Image

from wakareeru_inference.config import CropConfig
from wakareeru_inference.crop import CropCandidate, make_crop_candidates
from wakareeru_inference.detector import Detection, GroundingDinoDetector
from wakareeru_inference.image_io import load_image_from_event


@dataclass(frozen=True)
class PreprocessResult:
    image: Image.Image
    detections: list[Detection]
    crop_candidates: list[CropCandidate]


def preprocess_event(
    *,
    event: dict[str, Any],
    detector: GroundingDinoDetector,
    crop_config: CropConfig,
    detection_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
) -> PreprocessResult:
    image = load_image_from_event(event)
    return preprocess_image(
        image=image,
        detector=detector,
        crop_config=crop_config,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )


def preprocess_image(
    *,
    image: Image.Image,
    detector: GroundingDinoDetector,
    crop_config: CropConfig,
    detection_threshold: float | None = None,
    nms_iou_threshold: float | None = None,
) -> PreprocessResult:
    detections = detector.detect(
        image,
        detection_threshold=detection_threshold,
        nms_iou_threshold=nms_iou_threshold,
    )
    crop_candidates = make_crop_candidates(
        image=image,
        detections=detections,
        config=crop_config,
    )
    return PreprocessResult(
        image=image,
        detections=detections,
        crop_candidates=crop_candidates,
    )
