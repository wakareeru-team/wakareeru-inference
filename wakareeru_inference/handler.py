import logging
import os
import sys
from pathlib import Path
from typing import Any

from wakareeru_inference.response_schema import ResponseStatus
from wakareeru_inference.config import load_service_config
from wakareeru_inference.service import WakareeruService

from dotenv import load_dotenv
import runpod
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
runpod.api_key = os.getenv("RUNPOD_API_KEY")
DEFAULT_CONFIG_PATH = Path("configs/service_config.yaml")


class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno <= self.max_level


def configure_logging() -> None:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(os.getenv("WAKAREERU_LOG_LEVEL", "INFO").upper())
    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)


configure_logging()
logger = logging.getLogger(__name__)


def build_service() -> WakareeruService:
    config_path = Path(os.getenv("WAKAREERU_SERVICE_CONFIG", DEFAULT_CONFIG_PATH))
    logger.info("Loading service config from %s", config_path)
    return WakareeruService(load_service_config(config_path))


SERVICE = build_service()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    try:
        return SERVICE.predict_event(event)
    except Exception as exc:
        logger.exception("Handler failed while processing event")
        return {
            "status": ResponseStatus.ERROR.value,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        
# Start the Serverless function when the script is run
if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
