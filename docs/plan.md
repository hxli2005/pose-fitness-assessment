# 计算机视觉期末报告选题方案

## Context

用户是硕士研究生，需要在研究生"计算机视觉"课程中提交一份期末报告。报告形式为**应用型小项目**（代码实现 + 实验 + 可视化分析），技术深度偏向**模型对比与消融实验**，兴趣方向是经典视觉任务，算力充足（多卡/云 GPU）。

经过多轮澄清，最终选题锁定为：**基于多模型对比的健身动作姿态估计与动作评分系统**。这个选题的价值在于——它不是单纯"调个 YOLO 出个 demo"，而是通过系统性地对比不同姿态估计模型（轻量/单阶段/高精度）在下游健身场景中的表现，回答一个有研究价值的问题：**姿态估计模型的精度-速度权衡如何影响下游动作识别与评分的最终质量？** 同时交付物既有论文级指标（OKS-mAP / PCK），又有应用价值（实时计数 + 评分 demo），适合课程评审场合。

---

## 推荐选题：多模型对比的健身动作姿态估计与评分系统

**英文题目建议**：*Multi-Model 2D Pose Estimation for Fitness Action Assessment: A Comparative Study*

### 核心贡献点（报告里要讲清楚的三件事）

1. 在 COCO val 上横向对比 **RTMPose-M / YOLO11-Pose / ViTPose-Base** 三种 paradigm 的姿态估计器（OKS-mAP / 推理速度 / 参数量 Pareto 图）
2. 把各模型产出的关键点接入下游**动作分类 + 计数**任务，分析上游精度如何传导到下游指标（关键点精度 ≠ 下游效果）
3. 设计基于关节角度规则的**深蹲/卷腹/俯卧撑评分模块**，可输出错误类型（膝内扣、塌腰、半程等）

---

## Pipeline

```
输入视频 (30fps)
    ↓
人体检测 (YOLOX/YOLOv8 detector)
    ↓
姿态估计 (RTMPose-M  |  YOLO11-Pose  |  ViTPose-Base)  ← 三模型切换
    ↓
关键点时序 (过滤低置信 + 1€ filter / 滑动平均)
    ↓
关节角度计算 (膝/髋/肘/肩)
    ↓
┌──────────┬──────────┬──────────┐
计数        评分        动作分类
(峰值+滞回) (规则阈值)   (MLP / ST-GCN)
    ↓
可视化：骨架叠加 + 角度曲线 + 评分卡片
```

---

## 模型对比矩阵（E1 实验）

| 模型 | 仓库 | Checkpoint | COCO val AP (ref) | 参数量 | 推理速度 (参考) |
|---|---|---|---|---|---|
| RTMPose-M | open-mmlab/mmpose | `rtmpose_m_8xb256-420e_coco-256x192` | ~74.7 | ~10M | ~150 fps |
| YOLO11n-Pose | ultralytics/ultralytics | `yolo11n-pose.pt` | ~71.5 | ~2.8M | ~200 fps |
| ViTPose-Base | open-mmlab/mmpose | `vitpose-base_coco-256x192` | ~77.3 | ~86M | ~25 fps |
| RTMPose-S *(加分)* | open-mmlab/mmpose | `rtmpose_s_...` | ~69.8 | ~4.7M | ~280 fps |

---

## 数据集方案

**上游评估（姿态精度，必做）**
- **COCO val2017 Keypoints**：~5000 图 person 子集，用 `pycocotools.COCOeval` 输出 OKS-mAP / AR
- 下载：`http://images.cocodataset.org/zips/val2017.zip` + `annotations_trainval2017.zip`

**下游评估（健身任务，必做）**
- **Fit3D**（https://fit3d.is.tue.mpg.de/）：~8300 条运动序列，含 7 个健身动作（squats/lunges/crunches/push-ups 等）、多视角 RGB + 3D 关节标注。用 perspective projection 得到 2D GT
- **Countix / RepCount**（https://github.com/PulkitKandel/RepCount）：~2500 条 YouTube 视频，用于计数评测（每段 rep count 已标注）

**自建数据（加分）**
- YouTube 下载 10-20 段健身教学视频，ffmpeg 抽帧 + Label Studio/CVAT 手工标注 500 帧（深蹲/卷腹/俯卧撑各 ~150 帧）

---

## 消融实验（至少 4 组，全部必做）

| 实验 | 自变量 | 因变量 | 数据集 |
|---|---|---|---|
| **E1** 模型对比 | RTMPose-M / YOLO11n-Pose / ViTPose-Base | OKS-mAP, AR, fps, 参数量 | COCO val |
| **E2** 下游分类 | 三模型输出的关键点序列（作为特征） | 动作分类 Accuracy (3 类) | Fit3D 子集 |
| **E3** 平滑消融 | 无平滑 / 滑动平均 (w=3,5,7) / 1€ filter (β=0.5/1/2) | 计数 MAE | Countix exercise |
| **E4** 分辨率 & 遮挡鲁棒性 | 256×192 / 320×240 / 640×480；随机遮挡 5-20% | PCK@0.05, 计数 MAE | Fit3D |
| E5 *(加分)* | bbox 扩展比 1.0×/1.25×/1.5× | wrist/ankle PCK | Fit3D |

---

## 动作评分 / 计数模块

**计数**：基于关节角度峰值检测 + hysteresis（深蹲用膝角 hip-knee-ankle，卷腹用 hip-shoulder-knee，俯卧撑用 shoulder-elbow-wrist）。进入阈值 <100°，复位阈值 >140°，记一次 rep。

**评分规则**（阈值参考 ACE/NSCA 训练指南）：

| 动作 | 评分维度 | 标准区间 | 错误类型 |
|---|---|---|---|
| 深蹲 | 膝关节最小角度 | 70°-90° | 半程（膝角未到 90°） |
| 深蹲 | 膝盖外展对称性 | 左右膝角差 <10° | 膝内扣 (valgus) |
| 卷腹 | 肩抬升角 | 30°-45° | 幅度不足 |
| 俯卧撑 | 肘最低角 | 70°-90° | 半程 |
| 俯卧撑 | 肩-髋-踝直线度 | 共线偏差 <15° | 塌腰 |

输出：合格 rep 比率 + 错误类型标签列表。

---

## 可视化与报告亮点

- 骨架叠加视频（mmpose `demo/topdown_pose_estimation_demo.py` 模板）
- 实时角度曲线图 + rep 峰值标注（matplotlib 动画/GIF）
- E2 动作分类混淆矩阵
- **Pareto 图**：x=fps, y=mAP, 气泡大小=参数量
- **Failure case 剖析**（至少 3 类）：遮挡、侧身视角、宽松服装

---

## 关键仓库与文件

**要 clone / 参考的仓库**
- `open-mmlab/mmpose`（RTMPose + ViTPose + 评估脚本 `tools/test.py`）
- `ultralytics/ultralytics`（YOLO11-Pose，一行 `yolo pose val/predict`）
- `open-mmlab/mmaction2`（可选，做 ST-GCN 动作分类）
- `ViTAE-Transformer/ViTPose`（配置参考）

**需要重点查看的上游文件**
- `mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose_m_8xb256-420e_coco-256x192.py` — 修改 test pipeline 指向 Fit3D
- `mmpose/mmpose/datasets/datasets/coco/coco_dataset.py` — 了解关键点标注格式 `[x,y,v]*17`，用于 Fit3D 投影后的适配
- `ultralytics/ultralytics/cfg/models/11/yolo11-pose.yaml` — 确认 17-keypoint 输出与 COCO/mmpose 对齐
- RepCount repo 的 `data/*.json` — rep count 标注对齐脚本的输入

**工具依赖**
- `pip install mmpose mmdet mmengine pycocotools ultralytics one-euro-filter label-studio seaborn`

---

## 4 周时间安排（建议）

| 周次 | 任务 | 产出 |
|---|---|---|
| W1 | 环境 + 三模型 COCO 推理跑通，完成 **E1** | COCO val AP 表 + Pareto 图 v1 |
| W2 | Fit3D/Countix 数据准备 + 抽帧/投影脚本 + 单段视频 pipeline | 第一个骨架叠加 demo |
| W3 | **E2 + E3** + 评分/计数模块 | 混淆矩阵 + 计数 MAE 表 + 角度曲线 |
| W4 | **E4** + Failure case + 报告撰写 + demo 视频剪辑 | 最终报告 + 代码 repo + demo.mp4 |

---

## 交付物清单

**必做**
- 课程报告 PDF（含 4 组实验表、Pareto 图、混淆矩阵、角度曲线、failure case 剖析）
- GitHub repo（`configs/`, `scripts/`, `notebooks/` + 一键复现 README）
- 2-3 分钟 demo 视频（三个动作的骨架叠加 + 评分卡片）

**加分项**
- Streamlit/Gradio 交互 demo（上传视频自动评分）
- ST-GCN 替换 MLP 分类头
- 自建 500 帧标注数据公开

---

## 验证方式（end-to-end 跑通的标志）

1. `python tools/test.py configs/.../rtmpose_m_coco.py <ckpt>` 在 COCO val 上输出 OKS-mAP ≈ 74.7 ± 1
2. `yolo pose val model=yolo11n-pose.pt data=coco-pose.yaml` 输出 AP ≈ 71 ± 1
3. 输入一段 10 秒深蹲视频，pipeline 能输出：骨架叠加视频 + 正确的 rep 计数（人工对比 < ±1 误差）+ 评分结果
4. E2 的 MLP 分类器在 3 类动作上 Acc > 85%（是否能显著区分姿态模型质量）
5. Pareto 图清晰显示 ViTPose 高精度低速度、YOLO11n 低精度高速度、RTMPose 居中的 trade-off
