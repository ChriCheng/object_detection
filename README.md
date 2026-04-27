# 基于 MindSpore 的 YOLOv8n 目标检测实验报告

项目基于 **MindSpore** 深度学习框架和 **MindYOLO** 目标检测工具箱，选用 **YOLOv8n** 模型，在 **MS COCO 2017 val2017 验证集或验证集子集** 上完成目标检测测试、推理可视化和指标评估。

本项目并不是从零重新实现 YOLOv8n 的全部网络细节，而是在 MindSpore 生态下搭建一个完整、可复现、可提交的目标检测实验流程，包括环境配置、数据下载、数据格式转换、模型权重下载、图片推理、COCO 指标评估、日志记录和实验报告整理。

---

## 1. 实验目标

本实验主要目标如下：

1. 使用 MindSpore 框架完成一个深度学习目标检测实验；
2. 使用 MindYOLO 中的 YOLOv8n 模型进行目标检测测试；
3. 在 COCO val2017 数据集或其子集上完成检测效果评估；
4. 输出检测可视化结果，便于观察模型实际检测效果；
5. 记录 Precision、Recall、mAP@0.5、mAP@0.5:0.95、FPS 等指标；
6. 整理代码、脚本、日志、结果和实验报告，使项目能够复现和提交。

---

## 2. 算法原理与网络结构

YOLO 系列算法属于单阶段目标检测算法。与 Faster R-CNN 等两阶段检测方法不同，YOLO 将候选区域生成、目标分类和边界框回归统一在一个端到端网络中完成，因此具有较快的推理速度和较好的工程部署能力。

YOLOv8n 中的 `n` 表示 nano 版本，是 YOLOv8 系列中规模较小、参数量较低、推理速度较快的模型。它适合课程实验、快速验证和资源受限环境下的目标检测任务。

YOLOv8n 网络整体可以划分为三部分：

- **Backbone**：负责从输入图像中提取多层次语义特征；
- **Neck**：负责融合不同尺度的特征，使模型能够同时处理大目标、中目标和小目标；
- **Detection Head**：负责输出目标类别、边界框位置和置信度信息，并经过置信度筛选与 NMS 得到最终检测结果。

在本实验中，模型输入尺寸设置为 `640×640`，使用预训练权重 `weights/yolov8n.ckpt` 进行推理和评估。

---

## 3. 数据集与评价指标

### 3.1 数据集

本项目使用 **MS COCO 2017 val2017** 作为测试数据集。COCO 是目标检测领域常用的公开数据集，包含 80 个常见目标类别，例如 person、car、bus、dog、cat、bicycle 等。

完整的 `val2017` 验证集包含 5000 张图像。考虑到课程实验和普通单卡环境的运行时间，本项目默认支持从完整验证集中抽取前若干张图像构造子集，例如 500 张图像子集。

项目中的数据目录结构如下：

```text
data/
├── raw/              # 原始 COCO 数据，例如 val2017 图像和 annotations
├── coco_mindyolo/    # 转换后供 MindYOLO 使用的数据目录
└── predict_images/   # 用于单独推理和可视化的图片目录
```

其中，`data/predict_images/` 用于存放需要可视化推理的图片。推理结果会输出到：

```text
results/detect_results/
```

### 3.2 评价指标

本实验主要使用以下指标评价检测效果：

- **Precision**：精确率，用于衡量预测为正样本的目标中有多少是真实目标；
- **Recall**：召回率，用于衡量真实目标中有多少被模型成功检测出；
- **mAP@0.5**：IoU 阈值为 0.5 时的平均精度；
- **mAP@0.5:0.95**：COCO 标准指标，在多个 IoU 阈值下计算平均精度，更加严格；
- **FPS**：每秒处理图像数量，用于衡量模型推理速度。

---

## 4. 项目结构

当前项目结构如下：

```text
|-- README.md                         
|-- configs                           # 项目配置目录
|   `-- project.yaml                  # 记录数据路径、权重路径、模型参数等项目配置

|-- data                              # 数据目录
|   |-- coco_mindyolo                 # 转换后的 MindYOLO 格式 COCO 数据
|   |-- predict_images                # 用于单张或多张图片推理可视化的输入图片
|   `-- raw                           # 原始 COCO 数据集文件，如 val2017 图像和 annotations

|-- logs                              # 日志目录
|   |-- eval_yolov8n.log              # YOLOv8n 在 COCO 数据集上的评估日志
|   `-- predict_yolov8n.log           # YOLOv8n 图片推理日志

|-- results                           # 实验结果输出目录
|   |-- detect_results                # 推理生成的检测可视化图片结果
|   |-- metrics                       # 从评估日志中解析或保存的指标结果
|   |-- runs_infer                    # MindYOLO 推理过程生成的原始输出目录
|   `-- runs_test                     # MindYOLO 测试评估过程生成的原始输出目录

|-- scripts                           # Shell 脚本目录，用于自动化完成下载、准备、推理和评估
|   |-- download_coco_val2017.sh      # 下载 COCO val2017 图像和标注文件
|   |-- download_yolov8n_weight.sh    # 下载 YOLOv8n 预训练权重
|   |-- env_mindspore_gpu.sh          # 配置 MindSpore GPU 运行环境变量
|   |-- prepare_coco_subset.sh        # 构造 COCO val2017 子集并转换为 MindYOLO 所需格式
|   |-- run_all.sh                    # 一键执行数据准备、推理和评估流程
|   |-- run_eval_yolov8n.sh           # 运行 YOLOv8n 模型评估
|   |-- run_predict_yolov8n.sh        # 运行 YOLOv8n 图片推理并保存可视化结果
|   `-- setup_mindyolo.sh             # 下载或初始化 MindYOLO 第三方代码

|-- src                               # Python 辅助代码目录
|   |-- check_project.py              # 检查项目关键文件和目录是否完整
|   |-- parse_mindyolo_eval_log.py    # 解析 MindYOLO 评估日志并提取指标
|   |-- prepare_coco_subset.py        # 生成 COCO 子集并完成标签格式转换
|   `-- summarize_project.py          # 汇总项目结构、日志和实验结果

|-- third_party                       # 第三方依赖代码目录
|   `-- mindyolo                      

`-- weights                           # 模型权重目录
    `-- yolov8n.ckpt                  
```

主要目录说明如下：

| 路径                      | 说明                                               |
| ------------------------- | -------------------------------------------------- |
| `configs/project.yaml`    | 项目配置文件，记录数据路径、权重路径和实验参数     |
| `data/raw/`               | 原始 COCO 数据集目录                               |
| `data/coco_mindyolo/`     | 转换后的 MindYOLO 格式数据                         |
| `data/predict_images/`    | 用于可视化预测的输入图片                           |
| `scripts/`                | 项目运行脚本，包括环境配置、数据准备、推理、评估等 |
| `src/`                    | 辅助 Python 脚本，包括数据准备、日志解析和项目检查 |
| `third_party/mindyolo/`   | MindYOLO 第三方代码                                |
| `weights/`                | 模型权重目录                                       |
| `logs/`                   | 推理和评估日志                                     |
| `results/detect_results/` | 推理后的可视化检测结果                             |
| `results/metrics/`        | 整理后的评估指标结果                               |
| `docs/`                   | 实验报告、检查记录和结果模板                       |

---

## 5. 环境配置



| 项目           | 配置      |
| -------------- | --------- |
| Python         | 3.9       |
| 深度学习框架   | MindSpore |
| 目标检测工具箱 | MindYOLO  |
| 模型           | YOLOv8n   |
| 输入尺寸       | 640×640   |
| 推荐运行设备   | GPU       |

创建并进入 Conda 环境：

```bash
conda env create -f environment.yml
conda activate sr
```

安装依赖：

```bash
pip install -r requirements.txt
```

初始化 MindYOLO：

```bash
bash scripts/setup_mindyolo.sh
```

如果使用 GPU，运行脚本前建议加载 GPU 环境变量：

```bash
source scripts/env_mindspore_gpu.sh
```

该脚本用于设置 `DEVICE_TARGET=GPU`、`CUDA_HOME`、`CUDA_VISIBLE_DEVICES` 和 `LD_LIBRARY_PATH` 等环境变量。

---

## 6. 数据准备

下载 COCO val2017 数据集：

```bash
bash scripts/download_coco_val2017.sh
```

构造 COCO 子集，例如抽取 500 张验证图像：

```bash
bash scripts/prepare_coco_subset.sh 500
```

该步骤会将 COCO 标注转换为 MindYOLO 需要的 YOLO txt 标签格式，并生成类似如下的数据索引文件：

```text
data/coco_mindyolo/coco/val2017.txt
```

如果需要进行单张或多张图片的可视化检测，可以将图片放入：

```text
data/predict_images/
```

例如：

```bash
mkdir -p data/predict_images
cp data/raw/val2017/000000000139.jpg data/predict_images/
```

---

## 7. 权重下载

下载 YOLOv8n 预训练权重：

```bash
bash scripts/download_yolov8n_weight.sh
```

下载完成后，权重文件保存为：

```text
weights/yolov8n.ckpt
```

由于模型权重文件通常较大，不建议上传到 GitHub。可以在 `.gitignore` 中忽略 `weights/*.ckpt`，只保留目录占位文件，例如 `.gitkeep`。

---

## 8. 推理与可视化

运行 YOLOv8n 图片推理：

```bash
bash scripts/run_predict_yolov8n.sh
```

推理输入目录为：

```text
data/predict_images/
```

推理日志保存到：

```text
logs/predict_yolov8n.log
```

检测结果图片保存到：

```text
results/detect_results/
```


---

## 9. 测试评估

运行 COCO 子集或完整验证集评估：

```bash
bash scripts/run_eval_yolov8n.sh
```

评估日志保存到：

```text
logs/eval_yolov8n.log
```

评估结果和解析后的指标文件保存到：

```text
results/metrics/
```

也可以一键执行完整流程：

```bash
bash scripts/run_all.sh 500
```

其中 `500` 表示使用前 500 张 COCO val2017 图像构造测试子集。如果希望使用完整验证集，可以根据脚本设置改为 5000 或直接使用完整 `val2017`。

---

## 10. 实验结果记录

本项目的评估结果以 `logs/eval_yolov8n.log` 和 `results/metrics/` 中的文件为准。下面给出实验结果记录模板：

| 模型    | 数据集            | 测试图像数 | 输入尺寸 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |    FPS |
| ------- | ----------------- | ---------: | -------: | --------: | -----: | ------: | -----------: | -----: |
| YOLOv8n | COCO val2017 子集 |        500 |  640×640 |    见日志 | 见日志 |  见日志 |       见日志 | 见日志 |

MindYOLO 官方 benchmark 中，YOLOv8n 在 COCO 2017 上的 Box mAP 约为 37%。由于本项目可能使用的是 COCO 子集而不是完整 5000 张验证集，因此实际结果会受样本数量、样本分布、硬件环境和推理配置影响。当使用完整 val2017、相同输入尺寸和相同权重时，评估结果应更接近官方参考结果。

---

## 11. 实验结果与分析

本实验使用 YOLOv8n 预训练模型在 COCO val2017 子集上进行目标检测评估。测试阶段共读取 500 张验证图像，其中 500 张图像被成功识别，0 张缺失，0 张损坏，说明数据集路径、图像文件和标签文件均能被 MindYOLO 正确加载。实验使用 GPU 作为推理设备，输入图像尺寸为 640×640，batch size 设置为 16。模型加载的权重文件为 `weights/yolov8n.ckpt`，该权重为 YOLOv8n 在 COCO 数据集上的预训练权重。

从整体检测精度来看，YOLOv8n 在本实验 500 张 COCO val2017 子集上取得了 mAP@0.5:0.95 = 0.409，mAP@0.5 = 0.568，mAP@0.75 = 0.445。其中，mAP@0.5:0.95 是 COCO 目标检测任务中更严格、更综合的指标，它在 IoU 从 0.50 到 0.95 的多个阈值上计算平均精度，因此能够同时反映模型的目标分类能力和边界框定位精度。mAP@0.5 的数值较高，达到 0.568，说明在较宽松的 IoU 阈值下，模型能够较好地检测出图像中的目标；而 mAP@0.75 为 0.445，低于 mAP@0.5，说明当评价标准对定位精度要求更高时，模型的边界框预测仍存在一定误差。

从不同尺度目标的检测结果来看，YOLOv8n 对大目标和中等目标的检测效果明显优于小目标。实验中小目标 AP 为 0.209，中等目标 AP 为 0.462，大目标 AP 为 0.547。可以看出，随着目标尺寸增大，模型检测精度逐渐提升。这一现象符合目标检测任务的一般规律：小目标在图像中占据的像素区域较少，经过多层卷积和下采样后特征信息更容易丢失，同时小目标也更容易受到背景干扰、遮挡和密集分布的影响，因此检测难度更高；而中等目标和大目标具有更完整的纹理、轮廓和语义信息，模型更容易提取有效特征并完成定位。

从召回率结果来看，AR@1 为 0.347，AR@10 为 0.573，AR@100 为 0.624。随着每张图像允许保留的候选检测框数量增加，平均召回率明显提升，说明模型能够产生较多有效候选框，但如果只保留极少数量的预测框，会遗漏一部分真实目标。不同尺度下，大目标 AR 为 0.772，中等目标 AR 为 0.689，小目标 AR 为 0.400，也进一步说明模型对大目标和中等目标的召回能力更强，而小目标仍是主要难点。

从推理速度来看，本实验中模型在 640×640 输入尺寸下的推理耗时约为 20.9 ms，NMS 后处理耗时约为 33.3 ms，总耗时约为 54.2 ms/image，对应整体速度约为 18.45 FPS。如果只考虑网络前向推理部分，则约为 47.85 FPS。可以看出，YOLOv8n 作为 nano 版本模型，网络本身推理速度较快，但本实验中 NMS 后处理耗时较高，对整体速度造成了一定影响。这可能与 batch size、CPU/GPU 数据同步、后处理实现方式以及当前环境中 fast_coco_eval 未启用有关。

综合来看，YOLOv8n 在本实验中取得了较好的检测效果。其 mAP@0.5:0.95 达到 40.9%，mAP@0.5 达到 56.8%，说明模型能够较稳定地完成 COCO 常见类别目标检测任务。与 MindYOLO 官方 benchmark 中 YOLOv8n 在完整 MS COCO 2017 上约 37.2% 的 Box mAP 相比，本实验在 500 张子集上得到的 40.9% 略高，这可能是由于子集样本分布与完整验证集不同造成的，不能简单认为模型性能超过官方结果。

### 评估结果表

| 指标 | 数值 | 说明 |
|---|---:|---|
| mAP@0.5:0.95 | 0.409 | COCO 主指标，综合评价多 IoU 阈值下的检测精度 |
| mAP@0.5 | 0.568 | IoU=0.50 时的平均精度，评价标准相对宽松 |
| mAP@0.75 | 0.445 | IoU=0.75 时的平均精度，更强调定位准确性 |
| AP small | 0.209 | 小目标检测精度 |
| AP medium | 0.462 | 中等目标检测精度 |
| AP large | 0.547 | 大目标检测精度 |
| AR@1 | 0.347 | 每张图最多保留 1 个检测结果时的平均召回率 |
| AR@10 | 0.573 | 每张图最多保留 10 个检测结果时的平均召回率 |
| AR@100 | 0.624 | 每张图最多保留 100 个检测结果时的平均召回率 |
| 推理耗时 | 20.9 ms/image | 网络前向推理耗时 |
| NMS 耗时 | 33.3 ms/image | 非极大值抑制后处理耗时 |
| 总耗时 | 54.2 ms/image | 推理与后处理总耗时 |
| FPS | 18.45 | 按总耗时计算得到的整体处理速度 |
---

## 12. 项目检查

项目提供了检查脚本：

```bash
python src/check_project.py
```

该脚本可用于检查项目关键路径是否存在，例如数据目录、权重文件、日志目录、结果目录和第三方 MindYOLO 代码目录。

也可以生成项目摘要：

```bash
python src/summarize_project.py
```

如果需要从 MindYOLO 评估日志中提取指标，可以运行：

```bash
python src/parse_mindyolo_eval_log.py
```




## 13. 总结

本项目基于 MindSpore 和 MindYOLO 完成了 YOLOv8n 目标检测实验。项目实现了 COCO 数据准备、YOLO 格式标签转换、预训练权重下载、图片推理、检测结果可视化、COCO 指标评估、日志记录和实验报告整理等完整流程。

实验表明，YOLOv8n 能够在较低计算成本下完成常见目标类别检测，具有较好的速度优势和工程实践价值。本项目可作为课程作业提交项目，也可作为后续自定义数据集训练、模型微调和目标检测部署实验的基础。

---
## 14. 其他
本项目已在github上开源：https://github.com/ChriCheng/object_detection


## 15. 参考资料

1. MindYOLO GitHub: https://github.com/mindspore-lab/mindyolo
2. MindYOLO Benchmark: https://github.com/mindspore-lab/mindyolo/blob/master/benchmark_results.md
3. MindYOLO Getting Started: https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md
4. COCO Dataset: https://cocodataset.org/
5. YOLOv8: https://github.com/ultralytics/ultralytics