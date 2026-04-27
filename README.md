# 基于 MindSpore 的 YOLOv8n 目标检测实验报告

## 1. 实验概述

本项目完成“作业三：实现任意一种基于深度学习的目标检测算法在某一目标检测数据集上的测试，并自选指标对检测效果进行评估”的要求。实验采用 MindSpore 生态中的 MindYOLO 工具箱，选择 YOLOv8n 作为目标检测模型，在 MS COCO 2017 验证集或验证集子集上进行检测测试，并使用 Precision、Recall、mAP@0.5、mAP@0.5:0.95、FPS 等指标评估检测效果。

本项目不是从零重写 YOLOv8n 的底层模块，而是在 MindSpore 框架下完成一个完整、可复现、可提交的目标检测实验流程，包括数据准备、模型权重下载、推理可视化、测试评估、日志整理和实验报告撰写。

## 2. 算法与网络结构

YOLO 系列属于单阶段目标检测算法。与 Faster R-CNN 等两阶段方法相比，YOLO 将目标类别预测和边界框回归统一在一个端到端网络中完成，因此具有较高的推理速度。YOLOv8n 中的 n 表示 nano 版本，模型参数量小、速度快，适合课程实验和有限算力环境。

YOLOv8n 网络由 Backbone、Neck 和 Detection Head 三部分组成。Backbone 负责提取图像多层语义特征；Neck 负责融合不同尺度特征，使模型兼顾大目标、中目标和小目标；Detection Head 输出类别和边界框预测，随后通过置信度过滤和 NMS 得到最终检测结果。

## 3. 数据集与指标

实验使用 MS COCO 2017 验证集。COCO 是目标检测领域常用数据集，包含 80 个常见目标类别。完整 val2017 有 5000 张图像，普通 CPU 或单卡环境测试耗时较长，因此本项目默认支持抽取前 500 张图像构造子集。

评价指标包括：Precision 用于衡量误检情况；Recall 用于衡量漏检情况；mAP@0.5 表示 IoU 阈值为 0.5 时的平均精度；mAP@0.5:0.95 是 COCO 标准下更严格的平均精度；FPS 衡量推理速度。

## 4. 项目结构

```text
yolov8n_object_detection/
├── README.md
├── requirements.txt
├── environment.yml
├── configs/project.yaml
├── scripts/
│   ├── setup_mindyolo.sh
│   ├── download_coco_val2017.sh
│   ├── download_yolov8n_weight.sh
│   ├── prepare_coco_subset.sh
│   ├── run_predict_yolov8n.sh
│   ├── run_eval_yolov8n.sh
│   └── run_all.sh
├── src/
│   ├── prepare_coco_subset.py
│   ├── parse_mindyolo_eval_log.py
│   ├── summarize_project.py
│   └── check_project.py
├── data/
├── weights/
├── logs/
├── results/
├── docs/
└── third_party/
```

该结构参考 sentiment_classification 中“根目录放运行脚本、核心逻辑放 src、日志和结果单独保存”的方式，并针对目标检测任务增加 data、weights、results 和 docs 目录。

## 5. 环境配置

推荐环境：Python 3.9、MindSpore 2.5.0、MindYOLO 0.5.0、YOLOv8n、输入尺寸 640×640。MindSpore 请根据 CPU、GPU 或 Ascend 环境单独安装。

```bash
conda env create -f environment.yml
conda activate yolov8n_ms
pip install -r requirements.txt
bash scripts/setup_mindyolo.sh
```

## 6. 数据准备

```bash
bash scripts/download_coco_val2017.sh
bash scripts/prepare_coco_subset.sh 500
```

脚本会将 COCO 标注转换为 MindYOLO 需要的 YOLO txt 标签，并生成 `data/coco_mindyolo/coco/val2017.txt`。

## 7. 权重下载

```bash
bash scripts/download_yolov8n_weight.sh
```

权重保存到 `weights/yolov8n.ckpt`。权重文件较大，已写入 `.gitignore`，不建议上传 GitHub。

## 8. 推理与评估

```bash
DEVICE_TARGET=CPU bash scripts/run_predict_yolov8n.sh
DEVICE_TARGET=CPU bash scripts/run_eval_yolov8n.sh
```

也可以一键执行：

```bash
DEVICE_TARGET=CPU bash scripts/run_all.sh 500
```

## 9. 实验结果记录

| 模型 | 数据集 | 测试图像数 | 输入尺寸 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n | COCO val2017 子集 | 500 | 640×640 | 以日志为准 | 以日志为准 | 以日志为准 | 以日志为准 | 以日志为准 |

MindYOLO 官方 benchmark 中，YOLOv8n 在 MS COCO 2017 上的 Box mAP 约为 37.2% / 37.3%，参数量约 3.2M。使用完整 val2017 且环境一致时，结果应与官方结果接近；使用 500 张子集时，指标会因样本分布不同而波动。

## 10. 结果分析

YOLOv8n 对尺寸较大、边界清晰、遮挡较少的目标通常检测较好，例如 person、car、bus、dog 等类别。对于小目标、密集目标和严重遮挡目标，模型可能出现漏检；对于外观相似类别，如 cup 与 bowl、truck 与 bus 等，可能出现误检。

YOLOv8n 的优势是轻量、速度快、部署成本低。相比 YOLOv8s、YOLOv8m 等更大模型，YOLOv8n 精度略低，但更适合课程实验和资源受限场景。本实验通过预训练模型测试方式完成目标检测流程，能够体现深度学习目标检测算法的基本工作过程。

## 11. 总结

本项目基于 MindSpore 和 MindYOLO 实现了 YOLOv8n 目标检测模型的测试与评估流程。实验完成了 COCO 数据集准备、YOLO 格式标签转换、预训练模型推理、检测结果可视化、指标评估和日志整理，可作为课程作业提交项目，也可作为后续自定义数据集微调的基础。

## 12. 参考资料

1. MindYOLO GitHub: https://github.com/mindspore-lab/mindyolo
2. MindYOLO Benchmark: https://github.com/mindspore-lab/mindyolo/blob/master/benchmark_results.md
3. MindYOLO Getting Started: https://github.com/mindspore-lab/mindyolo/blob/master/GETTING_STARTED.md
4. COCO Dataset: https://cocodataset.org/
