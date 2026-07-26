#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  WAKAREERU_CLASSIFIER_VERSION=<version> ./test/start_local_docker.sh [--build]

Options:
  --build  Build the local Docker image before starting the container.
  -h, --help

Optional environment variables:
  WAKAREERU_LOCAL_IMAGE      Docker image tag (default: wakareeru-inference:local)
  WAKAREERU_LOCAL_PORT       Host port bound to 127.0.0.1 (default: 8000)
  WAKAREERU_DOCKER_PLATFORM  Docker platform (default: linux/amd64)
EOF
}

build_image=false
while (($# > 0)); do
  case "$1" in
    --build)
      build_image=true
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

classifier_version="${WAKAREERU_CLASSIFIER_VERSION:-}"
if [[ -z "$classifier_version" ]]; then
  echo "WAKAREERU_CLASSIFIER_VERSION is required." >&2
  exit 2
fi
if [[ "$classifier_version" == "." || "$classifier_version" == ".." ]] \
  || [[ ! "$classifier_version" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "WAKAREERU_CLASSIFIER_VERSION contains unsupported characters." >&2
  exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
docker_image="${WAKAREERU_LOCAL_IMAGE:-wakareeru-inference:local}"
host_port="${WAKAREERU_LOCAL_PORT:-8000}"
docker_platform="${WAKAREERU_DOCKER_PLATFORM:-linux/amd64}"
models_dir="$repo_dir/models"
classifier_dir="$models_dir/$classifier_version"

if [[ ! "$host_port" =~ ^[0-9]+$ ]] || ((host_port < 1 || host_port > 65535)); then
  echo "WAKAREERU_LOCAL_PORT must be an integer from 1 to 65535." >&2
  exit 2
fi
if [[ ! -d "$models_dir/grounding-dino" ]]; then
  echo "Missing detector model directory: $models_dir/grounding-dino" >&2
  exit 1
fi
if [[ ! -d "$classifier_dir" ]]; then
  echo "Missing classifier artifact directory: $classifier_dir" >&2
  exit 1
fi

if [[ "$build_image" == true ]]; then
  docker build \
    --platform "$docker_platform" \
    -t "$docker_image" \
    "$repo_dir"
elif ! docker image inspect "$docker_image" >/dev/null 2>&1; then
  echo "Docker image $docker_image does not exist; rerun with --build." >&2
  exit 1
fi

exec docker run --rm \
  --platform "$docker_platform" \
  -p "127.0.0.1:${host_port}:8000" \
  -v "$models_dir:/app/models:ro" \
  -e "WAKAREERU_CLASSIFIER_VERSION=$classifier_version" \
  -e "WAKAREERU_CLASSIFIER_MODEL_DIR=/app/models/$classifier_version" \
  "$docker_image" \
  python -m wakareeru_inference.handler \
  --rp_serve_api \
  --rp_api_host=0.0.0.0 \
  --rp_api_port=8000 \
  --rp_api_concurrency=1
