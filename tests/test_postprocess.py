from PIL import Image

from wakareeru_inference.config import PostprocessConfig
from wakareeru_inference.config import VersionConfig
from wakareeru_inference.crop import CropCandidate
from wakareeru_inference.detector import Detection
from wakareeru_inference.localization import LocalizedLabel, LocalizationIndex
from wakareeru_inference.postprocess import (
    attach_postprocess,
    build_response,
    build_subject_payload,
)
from wakareeru_inference.response_schema import ClassificationStatus, DetectionStatus


LOCALIZATION = LocalizationIndex(
    by_label_id={
        0: LocalizedLabel(
            label={"ja": "101系", "en": "101 series", "zh": "101系"},
            operators={
                "ja": ["国鉄"],
                "en": ["Japanese National Railways"],
                "zh": ["日本国有铁道"],
            },
            wiki_title_ja="国鉄101系電車",
        ),
        68: LocalizedLabel(
            label={"ja": "D51形", "en": "Class D51", "zh": "D51型"},
            operators={
                "ja": ["国鉄"],
                "en": ["Japanese National Railways"],
                "zh": ["日本国有铁道"],
            },
            wiki_title_ja="国鉄D51形蒸気機関車",
        ),
    }
)


def test_low_confidence_keeps_prediction_payload() -> None:
    predictions = [
        {
            "label_id": 68,
            "label": "D51形",
            "probability": 0.13,
        }
    ]

    payload = attach_postprocess(
        predictions=predictions,
        config=PostprocessConfig(min_classification_probability=0.5),
        localization=LOCALIZATION,
    )

    assert payload["status"] == ClassificationStatus.LOW_CONFIDENCE.value
    assert payload["top_prediction"] == {
        "label_id": 68,
        "label": {"ja": "D51形", "en": "Class D51", "zh": "D51型"},
        "operator": {
            "ja": ["国鉄"],
            "en": ["Japanese National Railways"],
            "zh": ["日本国有铁道"],
        },
        "wiki_title_ja": "国鉄D51形蒸気機関車",
        "probability": 0.13,
    }
    assert payload["top_k"] == [payload["top_prediction"]]


def test_classified_status_when_confidence_meets_threshold() -> None:
    predictions = [
        {
            "label_id": 68,
            "label": "D51形",
            "probability": 0.91,
        }
    ]

    payload = attach_postprocess(
        predictions=predictions,
        config=PostprocessConfig(min_classification_probability=0.5),
        localization=LOCALIZATION,
    )

    assert payload["status"] == ClassificationStatus.CLASSIFIED.value


def test_subject_payload_keeps_detection_before_classification() -> None:
    candidate = CropCandidate(
        image=Image.new("RGB", (10, 10)),
        detection=Detection(
            bbox=(1.0, 2.0, 8.0, 9.0),
            score=0.75,
            label="a train",
            source_index=0,
        ),
        bbox=(1, 2, 8, 9),
        status=DetectionStatus.DETECTED,
    )

    payload = build_subject_payload(
        index=0,
        candidate=candidate,
        predictions=[],
        postprocess_config=PostprocessConfig(min_classification_probability=0.5),
        localization=LOCALIZATION,
    )

    assert list(payload.keys()) == ["index", "detection", "classification"]
    assert list(payload["detection"].keys()) == ["bbox", "status", "label", "score"]


def test_response_includes_runtime_metadata() -> None:
    payload = build_response(
        subject_predictions=[],
        postprocess_config=PostprocessConfig(min_classification_probability=0.5),
        version_config=VersionConfig(
            inference="0.1.0",
            detector="grounding-dino",
            classifier="wakareeru-0.1.0-alpha.1",
        ),
        localization=LOCALIZATION,
    )

    assert list(payload.keys()) == ["status", "metadata", "subjects"]
    assert payload["metadata"] == {
        "inference_version": "0.1.0",
        "detector_version": "grounding-dino",
        "classifier_version": "wakareeru-0.1.0-alpha.1",
    }
