# 知物 · 家具视觉检索与问答

上传室内照片并描述目标，系统自动定位家具、检索商品资料，并给出附来源的回答。基于 GLA-CLIP / DINO、CLIP 商品检索与 Qwen 多模态重排构建。

**Built with Qwen.** 这是研究与演示项目。上游 GLA-CLIP 和本项目使用的 Qwen2.5-VL **3B** 模型包含非商业许可限制；本仓库不统一授权为 MIT。发布、使用前请阅读 [NOTICE](NOTICE) 和 [LICENSES](LICENSES)。ABO 数据另外适用其数据许可。

## 功能

- 根据问题自动定位目标；疑似粘连时调用 Grounding DINO + SAM 2 修复。
- GLA-CLIP 语义分割支持重叠窗口 KV 压缩，提供 location / sim08 等设置。
- 商品类别召回与全库补漏合并，使用 Qwen 适配器重排候选。
- 支持“明显不同 / 可能同款 / 证据不足”的核验；候选商品的材质不会自动当成用户物体的材质。
- 本地网页展示目标裁剪、商品资料、来源和运行记录；无需外部大模型 API。

## 1. 运行要求

推荐 Ubuntu 22.04 或 Windows 的 WSL2 Ubuntu，Python **3.10**、NVIDIA GPU 和可用的 CUDA 驱动。当前推理使用 BF16，需 GPU 支持。原环境为 RTX 5070 Ti Laptop GPU（12 GB）、PyTorch 2.14.0+cu130；12 GB 是已用硬件配置，不是对所有图片的显存保证。

默认安装脚本匹配该环境。其他 GPU/驱动可在 `.env` 中调整 Torch、Torchvision 和下载源，但更换版本需要重新验证，尤其是 MMCV。

磁盘建议预留 **40–60 GB**（预算）：包含 Python 环境、基础权重、ABO 小图归档及解压数据。源码包仅包含代码、小型适配器及校准资料，不含基础模型、商品图片、私人照片、索引或运行日志。网页部署不需要 VOC/ADE20K 验证集。

## 2. 安装环境

解压源码包，或克隆你自己上传的仓库，然后进入包含本 README 的目录。在 **Ubuntu 终端**执行：

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-dev build-essential git curl
cp .env.example .env
bash tools/install.sh
```

以上系统包命令针对 Ubuntu 22.04。其他发行版请先准备 Python 3.10，再用 `PYTHON=/绝对路径/python bash tools/install.sh`。不要在 Windows Anaconda Prompt 中直接安装这套 Linux 环境。

脚本创建三个环境：

| 环境 | 用途 |
| --- | --- |
| `.venv-gla` | CLIP、DINO、MMSegmentation、网页及索引 |
| `.venv-rag` | 新版 Transformers，目标修复与 Qwen 核验 |
| `.venv-posttrain` | PEFT 适配器重排及后训练工具 |

MMCV 2.1.0 会编译 CPU 扩展以匹配当前 Torch；本项目分割推理路径不使用其 CUDA 自定义算子。编译需时间和内存，内存不足可在 `.env` 设置 `MAX_JOBS=1`。不要混装 `mmcv-lite`，也不要直接使用上游未固定版本的 requirements。

**复用现有环境：** 不运行 install.sh，在 `.env` 填写 `GLA_PYTHON`、`RAG_PYTHON`、`DPO_PYTHON` 三个解释器的绝对路径；可同时填写已有 `ABO_ROOT`、`DINO_WEIGHTS`、`HF_HUB_CACHE`。示例路径见 `.env.example`。环境之间通过路径复用依赖，创建后请勿移动环境目录。

## 3. 下载基础模型

```bash
bash tools/download_models.sh
```

脚本下载 CLIP ViT-B/16、DINO ViT-B/8、Qwen2.5-VL-3B-Instruct、Grounding DINO tiny、SAM 2.1 tiny。模型版本固定在 [models/base_manifest.json](models/base_manifest.json)，CLIP/DINO 校验 SHA256。需要能访问官方模型下载站和 Hugging Face。

默认 CLIP 位于 `~/.cache/clip`，Hugging Face 模型位于 `~/.cache/huggingface/hub`，DINO 位于 `models/base`。已有缓存可复用；若脚本报告 HF revision mismatch，请为 `.env` 设置一个新的 `HF_HUB_CACHE` 目录再下载，以免覆盖其他项目的缓存版本。

源码包自带 `models/clean_e` 和 `models/dpo` 两套小型 LoRA 适配器。**默认是清洗样本后的 E 组排序模型**，并非未经修改的原始 DPO。设置 `RAG_RERANK_MODEL=dpo` 可使用 DPO 对照。适配器必须配合对应基础模型和校准文件使用。

## 4. 准备 ABO 商品知识库

从官方存储下载商品元数据和小尺寸图片：

```bash
bash tools/download_abo.sh
bash tools/prepare_abo_rag.sh
```

默认数据目录为 `data/ABO`。如果已有解压数据，跳过下载，直接运行：

```bash
bash tools/prepare_abo_rag.sh /你的/ABO目录
```

目录应含：

```text
ABO/
├── listings/metadata/listings_*.json.gz
└── images/
    ├── metadata/images.csv.gz
    └── small/...
```

脚本筛选家具、整理商品 ID/图片/属性/来源，并建立本地 CLIP 向量索引。索引存于 `rag_data/index`，图片仍引用 ABO 目录，移动数据后需要重建索引。数量取决于元数据及可读图片，不强制固定为演示环境的 8533 条。

也可使用自己的 CSV/JSONL 商品资料，通过网页“知识库”导入。结构参考 [examples/catalog.example.jsonl](examples/catalog.example.jsonl)；它是字段模板，需替换为真实图片路径和来源，未提供的属性应留空。

ABO 官方入口：[Amazon Berkeley Objects](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)。资料中缺失的材质/尺寸无法靠下载或索引补齐。

## 5. 检查并启动

```bash
bash tools/check_install.sh --runtime
bash tools/run_region_rag.sh
```

打开 **http://localhost:7860/**。上传照片、输入问题，例如“找左边椅子的相似商品，比较资料中记载的材质”，然后点击自动定位与检索。首次模型加载和建立索引较慢。

默认仅监听本机。远程服务器可用 `ssh -L 7860:127.0.0.1:7860 用户名@服务器` 后访问本地地址。网页没有多用户鉴权，不应直接作为公网服务开放。

`.env` 设置 `PORT=7861` 可更换端口。运行记录在 `rag_data/jobs`；不应上传这些记录，其中可能包含用户照片与问题。停止服务按 Ctrl+C。

## 架构概览

```text
图片 + 问题 → 目标解析 → GLA-CLIP 分割 → 必要时实例修复
        → 目标裁剪 → CLIP 类别召回 + 全库补漏
        → Qwen 适配器重排 → 双图/单图核验 → 三态判断
        → 读取候选商品资料 → 有来源回答 / 说明证据不足
```

离线：ABO 元数据与图片 → 商品记录 → CLIP 向量索引。
详细模块、评测边界见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 常见问题

- **CUDA unavailable / BF16 不支持：** 检查 `nvidia-smi`、WSL GPU 驱动和 Torch CUDA 版本；当前流程没有等价的 CPU 部署承诺。
- **mmseg / mmcv._ext 导入失败：** 检查实际使用的 GLA_PYTHON；确认 MMCV 编译成功，不能用 mmcv-lite 替代。
- **离线找不到 Qwen/SAM 模型：** 先完成模型下载，确认启动和下载使用同一个 HF_HUB_CACHE；运行阶段默认读取本地缓存。
- **Evaluated E implementation changed：** E 模型校验四个已评测文件的字节哈希；不要改变其换行或代码后继续使用旧校准。仓库 `.gitattributes` 保持 LF。
- **知识库为空：** 先完成 prepare_abo_rag；JSONL 中的图片路径必须在 Ubuntu 中可读。
- **找不到目标 / 没有匹配 / 属性未知：** 分别对应定位、商品匹配和资料缺失，不代表同一种错误。相似外观不能证明同款身份。
- **K 改为 30/50：** 可在网页实验设置调整，但现有阈值按 K=12 校准；更改 K 后应重新评估拒绝率，不能沿用原 FAR 保证。

## 测试与交付范围

```bash
source tools/env.sh
"$GLA_PYTHON" -m unittest discover -s region_rag_tests
```

交付时的实际检查记录见 [docs/DELIVERY_CHECKS.md](docs/DELIVERY_CHECKS.md)。本包保留研究脚本，但不含历史训练集、全部中间结果和完整实验目录；不要将推理部署包理解为所有训练实验的一键复现归档。原项目文档见 [docs/UPSTREAM_README.md](docs/UPSTREAM_README.md)。

目前确认的是现有 GPU 环境下的源码/加载检查，尚未在空白机器上完整重装所有依赖；安装脚本不等同于经过所有系统版本验证的安装器。

上传 GitHub 时上传解压目录内容即可，`.gitignore` 已排除环境、缓存和数据。不要提交 `.env`、API 密钥、私人照片或生成日志。保留 NOTICE、LICENSES、vendor 许可及模型说明。
