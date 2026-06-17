from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from wakareeru_inference.config import PostprocessConfig
from wakareeru_inference.crop import CropCandidate
from wakareeru_inference.response_schema import ClassificationStatus, ResponseStatus

if TYPE_CHECKING:
    from wakareeru_inference.predict import SubjectPrediction


@dataclass(frozen=True)
class ConfusionGroupIndex:
    label_to_group: dict[str, str]


def build_confusion_group_index(config: PostprocessConfig) -> ConfusionGroupIndex:
    label_to_group = {}
    for group in config.confusion_groups:
        for label in group.labels:
            label_to_group[label] = group.id
    return ConfusionGroupIndex(label_to_group=label_to_group)


def attach_postprocess(
    *,
    predictions: list[dict[str, Any]],
    config: PostprocessConfig,
) -> dict[str, Any]:
    classification_payload = build_classification_payload(
        predictions,
        min_probability=config.min_classification_probability,
    )
    if config.confusion_groups_enabled:
        apply_confusion_groups(
            classification_payload=classification_payload,
            confusion_groups=build_confusion_group_index(config),
        )
    return classification_payload


def build_classification_payload(
    predictions: list[dict[str, Any]],
    *,
    min_probability: float,
) -> dict[str, Any]:
    if not predictions:
        return {
            "status": ClassificationStatus.NO_PREDICTION.value,
            "top_prediction": None,
            "top_k": [],
            "confusion_group": None,
            "group_candidates": [],
        }
    top_prediction = predictions[0]
    status = ClassificationStatus.CLASSIFIED
    if float(top_prediction["probability"]) < min_probability:
        status = ClassificationStatus.LOW_CONFIDENCE
    return {
        "status": status.value,
        "top_prediction": top_prediction,
        "top_k": predictions,
        "confusion_group": None,
        "group_candidates": [],
    }


def apply_confusion_groups(
    *,
    classification_payload: dict[str, Any],
    confusion_groups: ConfusionGroupIndex,
) -> None:
    top_prediction = classification_payload["top_prediction"]
    if top_prediction is None:
        return

    group_id = confusion_groups.label_to_group.get(str(top_prediction["label"]))
    group_candidates = []
    if group_id is not None:
        group_candidates = [
            prediction
            for prediction in classification_payload["top_k"]
            if confusion_groups.label_to_group.get(str(prediction["label"])) == group_id
        ]
    classification_payload["confusion_group"] = group_id
    classification_payload["group_candidates"] = group_candidates


def build_response(
    *,
    subject_predictions: list["SubjectPrediction"],
    postprocess_config: PostprocessConfig,
) -> dict[str, Any]:
    if not subject_predictions:
        return {
            "status": ResponseStatus.NO_DETECTION.value,
            "subjects": [],
        }

    subjects = [
        build_subject_payload(
            index=index,
            candidate=subject_prediction.candidate,
            predictions=subject_prediction.predictions,
            postprocess_config=postprocess_config,
        )
        for index, subject_prediction in enumerate(subject_predictions)
    ]
    return {
        "status": ResponseStatus.OK.value,
        "subject_count": len(subjects),
        "subjects": subjects,
    }


def build_subject_payload(
    *,
    index: int,
    candidate: CropCandidate,
    predictions: list[dict[str, Any]],
    postprocess_config: PostprocessConfig,
) -> dict[str, Any]:
    detection_payload = {
        "status": candidate.status.value,
        "bbox": list(candidate.bbox) if candidate.bbox else None,
        "score": candidate.detection.score if candidate.detection else None,
        "label": candidate.detection.label if candidate.detection else None,
    }
    return {
        "index": index,
        "detection": detection_payload,
        "classification": attach_postprocess(
            predictions=predictions,
            config=postprocess_config,
        ),
    }
