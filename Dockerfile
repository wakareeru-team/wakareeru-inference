ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM --platform=linux/amd64 ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WAKAREERU_SERVICE_CONFIG=configs/service_config.yaml \
    WAKAREERU_MODEL_ROOT=/app/models \
    R2_BUCKET=models \
    RCLONE_CONFIG_R2_TYPE=s3 \
    RCLONE_CONFIG_R2_PROVIDER=Cloudflare \
    RCLONE_CONFIG_R2_REGION=auto

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git rclone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-image.txt ./
RUN pip install -r requirements-image.txt

COPY . ./
RUN pip install --no-deps -e .

CMD ["sh", "-c", "set -eu; : \"${WAKAREERU_CLASSIFIER_VERSION:?WAKAREERU_CLASSIFIER_VERSION is required}\"; : \"${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID is required}\"; : \"${R2_SECRET_ACCESS_ID:?R2_SECRET_ACCESS_ID is required}\"; : \"${R2_ENDPOINT:?R2_ENDPOINT is required}\"; case \"${WAKAREERU_CLASSIFIER_VERSION}\" in *[!A-Za-z0-9._-]*|.|..) echo 'WAKAREERU_CLASSIFIER_VERSION contains unsupported characters' >&2; exit 2;; esac; export RCLONE_CONFIG_R2_ACCESS_KEY_ID=\"${R2_ACCESS_KEY_ID}\" RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=\"${R2_SECRET_ACCESS_ID}\" RCLONE_CONFIG_R2_ENDPOINT=\"${R2_ENDPOINT}\"; model_dir=\"${WAKAREERU_MODEL_ROOT}/${WAKAREERU_CLASSIFIER_VERSION}\"; mkdir -p \"${model_dir}\"; rclone copy \"r2:${R2_BUCKET}/${WAKAREERU_CLASSIFIER_VERSION}\" \"${model_dir}\" --immutable; test -d \"${model_dir}/backbone\"; test -d \"${model_dir}/processor\"; test -f \"${model_dir}/classifier.safetensors\"; test -f \"${model_dir}/model_config.json\"; test -f \"${model_dir}/labels.json\"; test -f \"${model_dir}/manifest.json\"; export WAKAREERU_CLASSIFIER_MODEL_DIR=\"${model_dir}\"; exec python -m wakareeru_inference.handler"]
