import logging
import time
from typing import Any

from PIL import Image

from model_core.loader import LoadedClassifier, load_classifier
from wakareeru_inference.config import ServiceConfig
from wakareeru_inference.detector import GroundingDinoDetector, get_torch_device
from wakareeru_inference.postprocess import build_response
from wakareeru_inference.predict import predict_subjects
from wakareeru_inference.preprocess import preprocess_event, preprocess_image

logger = logging.getLogger(__name__)


class WakareeruService:
    def __init__(self, config: ServiceConfig) -> None:
        start_time = time.perf_counter()
        self.config = config
        self.device = get_torch_device(config.device)
        logger.info("Initializing Wakareeru service on device=%s", self.device)
        try:
            logger.info("Loading detector model from %s", config.detector.model_path)
            self.detector = GroundingDinoDetector(
                config=config.detector,
                device=self.device,
            )
            logger.info("Detector model loaded successfully")

            logger.info("Loading classifier artifact from %s", config.classifier.model_dir)
            self.classifier: LoadedClassifier = load_classifier(
                config.classifier.model_dir,
                device=self.device,
                local_files_only=config.classifier.local_files_only,
            )
            logger.info("Classifier artifact loaded successfully")
        except Exception:
            logger.exception("Failed to initialize Wakareeru service")
            raise
        logger.info(
            "Wakareeru service initialized in %.3fs",
            time.perf_counter() - start_time,
        )

    def predict_event(self, event: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()
        payload = event.get("input", event)
        top_k = int(payload.get("top_k", self.config.classifier.top_k))
        logger.info(
            "Starting event inference: payload_keys=%s top_k=%s",
            sorted(str(key) for key in payload.keys()),
            top_k,
        )

        preprocess_result = preprocess_event(
            event=event,
            detector=self.detector,
            crop_config=self.config.crop,
        )
        logger.info(
            "Preprocess complete: image_size=%s detections=%s crop_candidates=%s",
            preprocess_result.image.size,
            len(preprocess_result.detections),
            len(preprocess_result.crop_candidates),
        )

        subject_predictions = predict_subjects(
            classifier=self.classifier,
            crop_candidates=preprocess_result.crop_candidates,
            top_k=top_k,
            device=self.device,
        )

        response = build_response(
            subject_predictions=subject_predictions,
            postprocess_config=self.config.postprocess,
        )
        logger.info(
            "Inference result: %s elapsed=%.3fs",
            summarize_response(response),
            time.perf_counter() - start_time,
        )
        return response

    def predict_image(self, *, image: Image.Image, top_k: int | None = None) -> dict[str, Any]:
        start_time = time.perf_counter()
        if top_k is None:
            top_k = int(self.config.classifier.top_k)
        logger.info("Starting image inference: image_size=%s top_k=%s", image.size, top_k)

        preprocess_result = preprocess_image(
            image=image,
            detector=self.detector,
            crop_config=self.config.crop,
        )
        logger.info(
            "Preprocess complete: image_size=%s detections=%s crop_candidates=%s",
            preprocess_result.image.size,
            len(preprocess_result.detections),
            len(preprocess_result.crop_candidates),
        )

        subject_predictions = predict_subjects(
            classifier=self.classifier,
            crop_candidates=preprocess_result.crop_candidates,
            top_k=top_k,
            device=self.device,
        )

        response = build_response(
            subject_predictions=subject_predictions,
            postprocess_config=self.config.postprocess,
        )
        logger.info(
            "Inference result: %s elapsed=%.3fs",
            summarize_response(response),
            time.perf_counter() - start_time,
        )
        return response


def summarize_response(response: dict[str, Any]) -> dict[str, Any]:
    subjects = response.get("subjects", [])
    return {
        "status": response.get("status"),
        "subject_count": response.get("subject_count", len(subjects)),
        "subjects": [
            summarize_subject(subject)
            for subject in subjects
        ],
    }


def summarize_subject(subject: dict[str, Any]) -> dict[str, Any]:
    classification = subject.get("classification", {})
    top_prediction = classification.get("top_prediction") or {}
    return {
        "index": subject.get("index"),
        "detection_status": (subject.get("detection") or {}).get("status"),
        "classification_status": classification.get("status"),
        "top_label": top_prediction.get("label"),
        "top_probability": top_prediction.get("probability"),
        "confusion_group": classification.get("confusion_group"),
    }
