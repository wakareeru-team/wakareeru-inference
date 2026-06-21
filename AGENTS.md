# wakareeru-inference

`wakareeru-inference` 是 `wakareeru` 项目的 serverless 推理后端。它负责加载本地检测模型与分类模型，接收平台传入的 event，完成图片读取、主体检测、裁切、分类和响应组装。HTTP endpoint、鉴权、队列和外部流量入口通常由 RunPod / Azure 等 serverless 平台提供。

## 给 Agent 的工作原则

- 在没有明确要求实现的情形下请不要直接触碰代码；用户询问技术细节或请求纠错时，主动说明用到的 library、配置项和入口边界。
- 只实现用户明确要求的任务。这个仓库是推理 worker，不要主动扩大范围到训练管线、数据集构建、标签体系调整或平台供应商深度接入。
- 优先遵循现有模块、配置和 API schema；不要为了“更完整”而新增不必要抽象、自动修复逻辑、HTTP 框架或平台绑定。
- 工具函数保持职责单一，不要隐含函数名不提到的细节。例如图片读取只负责解码与基础 RGB/EXIF 处理，不应偷偷加入模型特定 resize、normalize 或 crop 策略。
- pre-processing、prediction、post-processing 保持解耦：输入解析和图片解码在 `image_io.py`，检测与裁切在 `detector.py` / `crop.py` / `preprocess.py`，分类调用在 `predict.py`，响应和业务后处理在 `postprocess.py`。
- 不要让 post-processing 反向依赖模型内部实现细节；如果 API 输出需要稳定字段，优先通过明确的数据结构传递，而不是让内部状态字符串随意泄漏成外部契约。
- 配置项应写入 `configs/service_config.yaml` 和 `wakareeru_inference/config.py`。推理正式路径不要在代码里用 `.get(..., 默认值)` 静默兜底新配置；缺失关键配置时应尽早报错。
- 模型 artifact 不提交进仓库。默认从本地 `models/` 读取；与开发伙伴同步时使用私有对象存储或明确版本目录，避免依赖可变的本地状态。
- `wakareeru` 主仓库通过 Git dependency 提供 `model_core`。发布或部署前优先固定 tag 或 commit SHA，不要让生产镜像漂在 `main` 上。
- Wakareeru 分类模型 artifact 由主仓库 `python -m trainer.export_inference_model` 导出。artifact schema、`backbone.path` / `processor` 路径解析、缺失文件报错和禁止回退到 Hugging Face cache 的规则，应在主仓库的 `model_core.loader` 中实现；本仓库只把 `classifier.model_dir` 交给 `load_classifier`，不要在推理侧重复拼装或解析分类模型内部路径。
- 推理侧应要求 `classifier.model_dir` 指向完整本地 artifact，包含 `backbone/`、`processor/`、`classifier.safetensors`、`model_config.json`、`labels.json` 和 `manifest.json`。检测模型是否允许下载由 detector 配置单独决定，不应和分类模型 artifact 规则混在一起。
- 分类模型的 resize/crop 输入尺寸以 artifact 内 `model_config.json` 的 `image_size` 为准；主仓库导出时会同步 `processor/preprocessor_config.json` 的默认 `size` / `crop_size`。不要在本仓库另行硬编码分类输入尺寸。
- GPU/serverless 镜像通常由基础镜像提供 CUDA 版 `torch` 和 `torchvision`。镜像构建时优先用 `requirements-image.txt` 安装非 PyTorch 运行依赖，再用 `pip install --no-deps -e .` 安装本仓库，避免覆盖平台提供的 PyTorch。
- 如果 handler、event schema、响应 schema、模型目录结构、配置项或入口命令发生非显然变化，同步更新 README 和本文件。
- 不要把一次性测试结果、当前模型分数、临时实验结论、具体样本数或易过期部署状态写进本文件；这类信息应放在 docs、issue、PR 或实验记录里。
- `models/` 下通常是本地模型副本。除非用户明确要求，不要清理、重建、覆盖或移动这些文件。

## 稳定边界

- 本仓库不负责训练模型；训练、导出和数据集构建在 `wakareeru` 主仓库。
- Docker 容器启动时根据 `WAKAREERU_CLASSIFIER_VERSION` 从私有 R2 `models` bucket 根目录同步完整分类 artifact；R2 凭据由平台运行时 secret 注入。检测模型仍由镜像或部署层提供，不在推理代码中自动下载。
- 分类模型不依赖 Hugging Face 本地 cache；`classifier.model_dir` 必须指向完整的 Wakareeru artifact 目录。`model_core.load_classifier` 负责从该目录加载本地 backbone 与 processor，本仓库不为分类模型另设 backbone 配置项或本地 loader 分叉。
- 本仓库目前不提供 FastAPI / Flask 等自建 HTTP API；平台应调用 `wakareeru_inference.handler.handler`。
- `handler` 接收 serverless event，默认读取 `event["input"]`，核心图片字段是 `input.image_base64`。
- 模型路径相对仓库根目录解析，默认配置在 `configs/service_config.yaml`。

## 仓库结构

```text
wakareeru_inference/
  handler.py        # serverless handler 入口和 worker 生命周期内 service 复用
  service.py        # 推理服务编排：preprocess -> predict -> postprocess
  image_io.py       # event 图片读取、base64/data URL 解码、RGB/EXIF 处理
  detector.py       # Grounding-DINO 加载、检测与 NMS
  crop.py           # 检测框选择、padding 与 crop candidate 构造
  preprocess.py     # event/image 到 detection/crop candidates
  predict.py        # 调用 model_core 对 crop 做分类
  postprocess.py    # 响应 payload 与 confusion group 等业务后处理
  config.py         # service_config.yaml 的 pydantic schema 与路径解析
configs/
  service_config.yaml
requirements-image.txt
scripts/
  smoke_test_handler.py
tests/
  test_image_io.py
models/
  ...               # 本地模型副本，不应提交
```

## 开发检查

```bash
ruff check .
python -m py_compile wakareeru_inference/*.py scripts/smoke_test_handler.py
pytest
```

如果当前环境没有安装 dev 依赖，可先安装：

```bash
pip install -e ".[dev]"
```

## 模型同步约定

模型可以继续使用主仓库已有的 Cloudflare R2 / rclone 工作流，但推理代码不应直接依赖某个开发者本机路径。推荐使用不可变版本目录，例如：

```text
r2:<bucket>/models/
  detector/grounding-dino/<version>/
  classifier/wakareeru-0.1.0-alpha.1/
```

本地或容器内同步到：

```text
models/
  grounding-dino/
  wakareeru-0.1.0-alpha.1/
    backbone/
    processor/
    classifier.safetensors
    model_config.json
    labels.json
    manifest.json
```

Wakareeru 分类 artifact 由主仓库导出，不在本仓库内重新组装。`model_config.json` 记录分类架构、`backbone.path`、label 数和训练输入 `image_size`；`manifest.json` 记录 artifact 版本、checkpoint 来源、backbone / processor 子目录与训练指标摘要。artifact 内部字段由 `model_core` 读取并用于分类预处理，本仓库不重复解析。

共享给开发伙伴时使用只读 R2 token，并限制到模型 prefix。生产配置应指向明确版本，不要只依赖 `latest` 这类可变目录。
