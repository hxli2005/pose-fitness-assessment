# Pose-based Fitness Action Assessment

Multi-Model 2D Pose Estimation for Fitness Action Assessment —— 研究生《计算机视觉》课程期末项目。

## 研究问题

姿态估计模型的精度-速度权衡，如何影响下游健身动作识别与评分的最终质量？

## 核心工作

1. **E1 模型对比**：在 COCO val2017 上横比 RTMPose-M / YOLO11n-Pose / ViTPose-Base 的 OKS-mAP、推理速度、参数量
2. **E2 下游分类**：三模型输出的关键点分别作为特征，训练 MLP 做深蹲/卷腹/俯卧撑 3 类动作分类
3. **E3 平滑消融**：对比无平滑 / 滑动平均 / 1€ filter 对计数 MAE 的影响
4. **E4 鲁棒性**：输入分辨率、自遮挡对 PCK 和计数误差的影响
5. **动作评分**：基于关节角度规则，输出每次 rep 的标准/不标准 + 错误类型（膝内扣、塌腰、半程）

## 目录结构

```
pose-fitness-assessment/
├── configs/       # mmpose/ultralytics 实验 config
├── scripts/       # 数据准备、评估、可视化脚本
├── notebooks/     # 探索性分析、画图
├── data/          # 数据集（gitignored）
├── outputs/       # 实验输出（gitignored）
└── docs/          # 报告草稿、demo 视频
```

## 使用的数据集

| 用途 | 数据集 | 获取方式 |
|---|---|---|
| 姿态精度（上游） | COCO val2017 keypoints | cocodataset.org |
| 健身动作（下游） | Fit3D | https://fit3d.is.tue.mpg.de/ |
| 动作计数 | Countix / RepCount | github.com/PulkitKandel/RepCount |

## 使用的模型

| 模型 | 仓库 | Checkpoint |
|---|---|---|
| RTMPose-M | open-mmlab/mmpose | rtmpose_m_8xb256-420e_coco-256x192 |
| YOLO11n-Pose | ultralytics/ultralytics | yolo11n-pose.pt |
| ViTPose-Base | open-mmlab/mmpose | vitpose-base_coco-256x192 |

## 环境安装

```bash
# 准备环境（建议 Python 3.10 + CUDA 11.8）
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# mmpose 生态
pip install -U openmim
mim install mmengine "mmcv>=2.0.0" "mmdet>=3.0.0" "mmpose>=1.3.0"

# ultralytics
pip install ultralytics

# 工具
pip install pycocotools one-euro-filter seaborn matplotlib pandas
```

## 复现路线

详见 `docs/plan.md`（来自 `~/.claude/plans/bright-mixing-harbor.md` 的副本）。

- W1：环境 + 三模型 COCO 推理 + E1
- W2：Fit3D/Countix 数据准备 + pipeline
- W3：E2 + E3 + 评分/计数模块
- W4：E4 + Failure case + 报告 + demo 视频
