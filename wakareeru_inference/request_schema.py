from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class InferenceOptions(BaseModel):
    """Optional, request-scoped inference configuration overrides."""

    model_config = ConfigDict(extra="forbid")

    detection_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        strict=True,
    )
    nms_iou_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        strict=True,
    )
    fallback_to_whole_image: bool | None = Field(default=None, strict=True)


def parse_inference_options(payload: dict[str, Any]) -> InferenceOptions:
    raw_options = payload.get("inference_options")
    if raw_options is None:
        return InferenceOptions()
    return InferenceOptions.model_validate(raw_options)
