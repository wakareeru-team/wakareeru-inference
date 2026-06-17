from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DetectorConfig(BaseModel):
    model_path: str
    local_files_only: bool = True
    text_prompt: str = "a train"
    box_threshold: float = 0.2
    text_threshold: float = 0.2
    nms_iou_threshold: float = 0.5
    min_box_score: float = 0.0
    max_detections: int = 3


class CropConfig(BaseModel):
    padding_ratio: float = 0.04
    select_policy: Literal["highest_score", "largest_area", "all"] = "highest_score"
    fallback_policy: Literal["whole_image", "error"] = "whole_image"


class ClassifierConfig(BaseModel):
    model_dir: str
    local_files_only: bool = True
    top_k: int = 5


class ConfusionGroup(BaseModel):
    id: str
    labels: list[str] = Field(default_factory=list)


class PostprocessConfig(BaseModel):
    min_classification_probability: float = Field(ge=0.0, le=1.0)
    confusion_groups_enabled: bool = False
    confusion_groups: list[ConfusionGroup] = Field(default_factory=list)


class ServiceConfig(BaseModel):
    device: str = "auto"
    detector: DetectorConfig
    crop: CropConfig
    classifier: ClassifierConfig
    postprocess: PostprocessConfig


def load_service_config(path: str | Path) -> ServiceConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    payload = resolve_model_paths(payload=payload, config_path=path)
    return ServiceConfig.model_validate(payload)


def resolve_model_paths(*, payload: dict, config_path: Path) -> dict:
    base_dir = config_path.resolve().parent.parent
    detector_path = Path(payload["detector"]["model_path"]).expanduser()
    classifier_path = Path(payload["classifier"]["model_dir"]).expanduser()
    if not detector_path.is_absolute():
        payload["detector"]["model_path"] = str(base_dir / detector_path)
    if not classifier_path.is_absolute():
        payload["classifier"]["model_dir"] = str(base_dir / classifier_path)
    return payload
