import pytest
from pydantic import ValidationError

from wakareeru_inference.config import CropConfig, DetectorConfig
from wakareeru_inference.detector import resolve_detection_thresholds
from wakareeru_inference.request_schema import InferenceOptions, parse_inference_options
from wakareeru_inference.service import crop_config_with_overrides


def test_parse_inference_options_accepts_request_overrides() -> None:
    options = parse_inference_options(
        {
            "inference_options": {
                "detection_threshold": 0.3,
                "fallback_to_whole_image": False,
            }
        }
    )

    assert options == InferenceOptions(
        detection_threshold=0.3,
        fallback_to_whole_image=False,
    )


@pytest.mark.parametrize("threshold", [-0.01, 1.01, "0.3"])
def test_parse_inference_options_rejects_invalid_threshold(threshold) -> None:
    with pytest.raises(ValidationError):
        parse_inference_options(
            {"inference_options": {"detection_threshold": threshold}}
        )


def test_parse_inference_options_rejects_non_boolean_fallback() -> None:
    with pytest.raises(ValidationError):
        parse_inference_options(
            {"inference_options": {"fallback_to_whole_image": "false"}}
        )


def test_crop_fallback_override_does_not_mutate_service_config() -> None:
    default = CropConfig(
        padding_ratio=0.04,
        select_policy="all",
        fallback_policy="whole_image",
    )

    overridden = crop_config_with_overrides(
        default,
        InferenceOptions(fallback_to_whole_image=False),
    )

    assert default.fallback_policy == "whole_image"
    assert overridden.fallback_policy == "error"


def test_detection_threshold_overrides_both_gdino_thresholds() -> None:
    config = DetectorConfig(
        model_path="models/grounding-dino",
        box_threshold=0.2,
        text_threshold=0.4,
    )

    assert resolve_detection_thresholds(config, None) == (0.2, 0.4)
    assert resolve_detection_thresholds(config, 0.3) == (0.3, 0.3)
