from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.ops import batched_nms
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.utils.generic import ModelOutput

from wakareeru_inference.config import DetectorConfig


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    score: float
    label: str
    source_index: int

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def get_torch_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, ModelOutput):
        return value.__class__(**{key: detach_to_cpu(item) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    return value


def resolve_detection_thresholds(
    config: DetectorConfig,
    detection_threshold: float | None,
) -> tuple[float, float]:
    if detection_threshold is None:
        return float(config.box_threshold), float(config.text_threshold)
    threshold = float(detection_threshold)
    return threshold, threshold


def resolve_nms_iou_threshold(
    config: DetectorConfig,
    nms_iou_threshold: float | None,
) -> float:
    if nms_iou_threshold is None:
        return float(config.nms_iou_threshold)
    return float(nms_iou_threshold)


class GroundingDinoDetector:
    def __init__(self, *, config: DetectorConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            config.model_path,
            local_files_only=config.local_files_only,
        )
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            config.model_path,
            local_files_only=config.local_files_only,
        ).to(device)
        self.model.eval()

    @torch.inference_mode()
    def detect(
        self,
        image: Image.Image,
        *,
        detection_threshold: float | None = None,
        nms_iou_threshold: float | None = None,
    ) -> list[Detection]:
        box_threshold, text_threshold = resolve_detection_thresholds(
            self.config,
            detection_threshold,
        )
        labels = [[self.config.text_prompt]]
        target_sizes = [image.size[::-1]]
        inputs = self.processor(
            images=[image],
            text=labels,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        outputs = detach_to_cpu(self.model(**inputs))
        token_ids = inputs["input_ids"].detach().cpu()
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            token_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )
        return self._postprocess_results(
            results[0],
            nms_iou_threshold=nms_iou_threshold,
        )

    def _postprocess_results(
        self,
        result: dict,
        *,
        nms_iou_threshold: float | None = None,
    ) -> list[Detection]:
        boxes = result["boxes"]
        scores = result["scores"]
        text_labels = result["text_labels"]
        if len(boxes) == 0:
            return []

        label_to_id = {label: idx for idx, label in enumerate(dict.fromkeys(text_labels))}
        label_ids = torch.tensor([label_to_id[label] for label in text_labels], dtype=torch.long)
        keep = batched_nms(
            boxes.float(),
            scores.float(),
            label_ids,
            resolve_nms_iou_threshold(self.config, nms_iou_threshold),
        )
        detections = []
        for output_index in keep.tolist():
            score = float(scores[output_index].item())
            if score < float(self.config.min_box_score):
                continue
            detections.append(
                Detection(
                    bbox=tuple(float(v) for v in boxes[output_index].tolist()),
                    score=score,
                    label=str(text_labels[output_index]),
                    source_index=int(output_index),
                )
            )
        detections.sort(key=lambda item: item.score, reverse=True)
        return detections[: int(self.config.max_detections)]
