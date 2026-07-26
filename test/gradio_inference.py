"""Interactive Gradio client for testing the Wakareeru inference Docker container.

Install the UI dependency when needed:

    pip install "gradio>=5,<7"

Then run:

    python test/gradio_inference.py --inbrowser
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont

DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_LOCAL_DOCKER_URL = "http://127.0.0.1:8000/runsync"
BOX_COLORS = (
    "#ff3b30",
    "#007aff",
    "#34c759",
    "#ff9500",
    "#af52de",
    "#00a7a7",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch an interactive Gradio client for the Wakareeru inference backend.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=os.getenv(
            "WAKAREERU_INFERENCE_URL",
            os.getenv("INFERENCE_ENDPOINT_URL", DEFAULT_LOCAL_DOCKER_URL),
        ),
        help="Local Docker inference URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(
            "RUNPOD_API_KEY",
            os.getenv("INFERENCE_API_KEY", ""),
        ),
        help="Optional bearer token. Prefer the RUNPOD_API_KEY environment variable.",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--inbrowser", action="store_true")
    return parser.parse_args()


def encode_image(image: Image.Image) -> tuple[str, int]:
    normalized = image.convert("RGB")
    buffer = io.BytesIO()
    normalized.save(buffer, format="JPEG", quality=95)
    content = buffer.getvalue()
    return base64.b64encode(content).decode("ascii"), len(content)


def build_request_payload(
    image: Image.Image,
    *,
    top_k: int,
    detection_threshold: float,
    nms_iou_threshold: float,
    fallback_to_whole_image: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_base64, image_bytes = encode_image(image)
    inference_options = {
        "detection_threshold": float(detection_threshold),
        "nms_iou_threshold": float(nms_iou_threshold),
        "fallback_to_whole_image": bool(fallback_to_whole_image),
    }
    payload = {
        "input": {
            "image_base64": image_base64,
            "top_k": int(top_k),
            "inference_options": inference_options,
        }
    }
    preview = {
        "input": {
            "image_base64": f"<omitted: {image_bytes} JPEG bytes>",
            "top_k": int(top_k),
            "inference_options": inference_options,
        }
    }
    return payload, preview


def unwrap_inference_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("The endpoint response must be a JSON object.")

    if "output" not in payload:
        return payload

    output = payload["output"]
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as error:
            raise ValueError("RunPod output is not valid JSON.") from error
    if not isinstance(output, dict):
        runpod_status = payload.get("status", "unknown")
        raise ValueError(f"RunPod returned status={runpod_status!r} without an object output.")
    return output


def localized_label(prediction: Mapping[str, Any], language: str) -> str | None:
    label = prediction.get("label")
    if isinstance(label, Mapping):
        for key in (language, "en", "ja", "zh"):
            value = label.get(key)
            if isinstance(value, str) and value:
                return value
        return None
    if label is None:
        return None
    return str(label)


def subject_caption(subject: Mapping[str, Any], language: str) -> str:
    detection = subject.get("detection")
    classification = subject.get("classification")
    detection = detection if isinstance(detection, Mapping) else {}
    classification = classification if isinstance(classification, Mapping) else {}

    parts = [f"#{subject.get('index', '?')}"]
    detection_label = detection.get("label")
    if detection_label:
        parts.append(str(detection_label))
    detection_score = detection.get("score")
    if isinstance(detection_score, (int, float)):
        parts.append(f"det={float(detection_score):.3f}")

    top_prediction = classification.get("top_prediction")
    if isinstance(top_prediction, Mapping):
        label = localized_label(top_prediction, language)
        if label:
            parts.append(label)
        probability = top_prediction.get("probability")
        if isinstance(probability, (int, float)):
            parts.append(f"cls={float(probability):.3f}")
    return " | ".join(parts)


def load_font(image: Image.Image) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    font_size = max(14, min(30, max(image.size) // 45))
    candidates = (
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), font_size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_caption(
    draw: ImageDraw.ImageDraw,
    *,
    position: tuple[int, int],
    caption: str,
    color: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    image_size: tuple[int, int],
) -> None:
    try:
        bounds = draw.textbbox((0, 0), caption, font=font)
    except UnicodeEncodeError:
        caption = caption.encode("ascii", errors="replace").decode("ascii")
        bounds = draw.textbbox((0, 0), caption, font=font)

    padding = 4
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    x = min(max(0, position[0]), max(0, image_size[0] - text_width - padding * 2))
    y = min(max(0, position[1]), max(0, image_size[1] - text_height - padding * 2))
    background = (x, y, x + text_width + padding * 2, y + text_height + padding * 2)
    draw.rectangle(background, fill=color)
    draw.text(
        (x + padding, y + padding - bounds[1]),
        caption,
        fill="white",
        font=font,
    )


def annotate_response(
    image: Image.Image,
    inference_response: Mapping[str, Any],
    *,
    language: str,
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = load_font(annotated)
    line_width = max(2, round(max(annotated.size) / 350))
    subjects = inference_response.get("subjects")
    if not isinstance(subjects, list):
        return annotated

    for position, subject in enumerate(subjects):
        if not isinstance(subject, Mapping):
            continue
        detection = subject.get("detection")
        detection = detection if isinstance(detection, Mapping) else {}
        bbox = detection.get("bbox")
        color = BOX_COLORS[position % len(BOX_COLORS)]
        caption = subject_caption(subject, language)

        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x1, y1, x2, y2 = (round(float(value)) for value in bbox)
            x1 = min(max(0, x1), annotated.width)
            y1 = min(max(0, y1), annotated.height)
            x2 = min(max(0, x2), annotated.width)
            y2 = min(max(0, y2), annotated.height)
            if x2 > x1 and y2 > y1:
                draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
                caption_y = y1 - max(24, line_width * 6)
                draw_caption(
                    draw,
                    position=(x1, caption_y),
                    caption=caption,
                    color=color,
                    font=font,
                    image_size=annotated.size,
                )
                continue

        draw_caption(
            draw,
            position=(8, 8 + position * 36),
            caption=caption,
            color=color,
            font=font,
            image_size=annotated.size,
        )
    return annotated


def error_result(
    image: Image.Image | None,
    message: str,
    *,
    request_preview: dict[str, Any] | None = None,
    upstream_response: Any = None,
) -> tuple[Image.Image | None, str, dict[str, Any], Any, dict[str, Any]]:
    return (
        image,
        f"❌ {message}",
        {"error": message},
        upstream_response,
        request_preview or {},
    )


def call_inference(
    endpoint_url: str,
    api_key: str,
    image: Image.Image | None,
    detection_threshold: float,
    nms_iou_threshold: float,
    fallback_to_whole_image: bool,
    top_k: int,
    timeout_seconds: float,
    language: str,
) -> tuple[Image.Image | None, str, dict[str, Any], Any, dict[str, Any]]:
    endpoint_url = endpoint_url.strip()
    if not endpoint_url:
        return error_result(image, "请输入完整的推理 URL。")
    if image is None:
        return error_result(None, "请选择一张图片。")

    payload, request_preview = build_request_payload(
        image,
        top_k=int(top_k),
        detection_threshold=float(detection_threshold),
        nms_iou_threshold=float(nms_iou_threshold),
        fallback_to_whole_image=bool(fallback_to_whole_image),
    )
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
    }
    if api_key.strip():
        headers["authorization"] = f"Bearer {api_key.strip()}"

    try:
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=float(timeout_seconds),
        )
    except requests.RequestException as error:
        return error_result(
            image,
            f"请求失败：{error}",
            request_preview=request_preview,
        )

    try:
        upstream_response: Any = response.json()
    except ValueError:
        upstream_response = {"body": response.text}

    if not response.ok:
        return error_result(
            image,
            f"HTTP {response.status_code}：推理后端返回错误。",
            request_preview=request_preview,
            upstream_response=upstream_response,
        )

    try:
        inference_response = unwrap_inference_response(upstream_response)
    except ValueError as error:
        return error_result(
            image,
            str(error),
            request_preview=request_preview,
            upstream_response=upstream_response,
        )

    annotated = annotate_response(
        image,
        inference_response,
        language=language,
    )
    status = inference_response.get("status", "unknown")
    subjects = inference_response.get("subjects")
    subject_count = len(subjects) if isinstance(subjects, list) else 0
    summary = (
        f"✅ HTTP {response.status_code} · inference status: `{status}` · "
        f"subjects: `{subject_count}`"
    )
    return annotated, summary, inference_response, upstream_response, request_preview


def build_demo(*, default_endpoint_url: str, default_api_key: str):
    try:
        import gradio as gr
    except ModuleNotFoundError as error:
        raise SystemExit(
            'Gradio is not installed. Run: pip install "gradio>=5,<7"',
        ) from error

    with gr.Blocks(title="Wakareeru Inference Tester") as demo:
        gr.Markdown(
            """
            # Wakareeru Inference Tester

            上传图片并调用本地 Docker 中的推理服务，默认地址为
            `http://127.0.0.1:8000/runsync`。API Key 仅用于需要鉴权的自定义地址；
            本地测试容器不需要填写。
            """
        )
        with gr.Row():
            endpoint_url = gr.Textbox(
                value=default_endpoint_url,
                label="Inference URL",
                placeholder=DEFAULT_LOCAL_DOCKER_URL,
                scale=3,
            )
            api_key = gr.Textbox(
                value=default_api_key,
                label="API Key",
                type="password",
                scale=2,
            )

        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="测试图片",
                height=420,
            )
            annotated_output = gr.Image(
                type="pil",
                label="检测框可视化",
                height=420,
                interactive=False,
            )

        with gr.Row():
            detection_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.1,
                step=0.01,
                label="Detection threshold",
            )
            nms_iou_threshold = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.35,
                step=0.01,
                label="NMS IoU threshold",
            )
            top_k = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Top K",
            )

        with gr.Row():
            fallback_to_whole_image = gr.Checkbox(
                value=False,
                label="无检测时使用整图",
            )
            language = gr.Dropdown(
                choices=["zh", "ja", "en"],
                value="zh",
                label="可视化标签语言",
            )
            timeout_seconds = gr.Slider(
                minimum=5,
                maximum=600,
                value=DEFAULT_TIMEOUT_SECONDS,
                step=5,
                label="请求超时（秒）",
            )

        run_button = gr.Button("运行推理", variant="primary")
        status_output = gr.Markdown()
        with gr.Tab("Inference response"):
            inference_output = gr.JSON()
        with gr.Tab("Raw upstream response"):
            upstream_output = gr.JSON()
        with gr.Tab("Request preview"):
            request_output = gr.JSON()

        run_button.click(
            fn=call_inference,
            inputs=[
                endpoint_url,
                api_key,
                image_input,
                detection_threshold,
                nms_iou_threshold,
                fallback_to_whole_image,
                top_k,
                timeout_seconds,
                language,
            ],
            outputs=[
                annotated_output,
                status_output,
                inference_output,
                upstream_output,
                request_output,
            ],
        )
    return demo


def main() -> None:
    args = parse_args()
    demo = build_demo(
        default_endpoint_url=args.endpoint_url,
        default_api_key=args.api_key,
    )
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        inbrowser=args.inbrowser,
        show_error=True,
    )


if __name__ == "__main__":
    main()
