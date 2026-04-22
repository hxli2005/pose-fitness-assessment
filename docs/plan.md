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

---

## 任务拆解（10 个子任务，按依赖顺序执行）

将 4 周方案按功能模块拆成 10 个可独立完成的子任务。每个任务附带：输入、产出文件、完成标志。依赖顺序如箭头所示。

```
T1 环境 ──► T2 COCO 数据 ──► T3 三模型推理封装 ──► T4 E1 评估 + Pareto
                                   │
                                   ▼
                              T5 Fit3D/Countix 数据
                                   │
                                   ├──► T6 关键点时序 + 角度 + 计数/评分
                                   │        │
                                   │        ├──► T7 可视化（骨架+角度曲线+评分卡片）
                                   │        │
                                   │        └──► T9 E3 平滑消融
                                   │
                                   └──► T8 E2 下游分类
                                   
                              T10 E4 分辨率/遮挡鲁棒性 ──► T11 报告 + demo 视频
```

### T1 — 环境与依赖（0.5 天）
- **输入**：空的 `~/code/pose-fitness-assessment/` 仓库
- **做什么**：新建 `.venv`，安装 PyTorch (CUDA 11.8)、mmpose 生态（via `openmim`）、ultralytics、pycocotools、one-euro-filter、seaborn 等；写一个 `scripts/setup_env.sh` 把所有命令固化
- **产出文件**：`scripts/setup_env.sh`、`requirements.txt`（pip freeze 快照）
- **完成标志**：`python -c "import mmpose, ultralytics, pycocotools"` 无报错；`nvidia-smi` 能看到显卡

### T2 — COCO val2017 数据准备（0.5 天）
- **输入**：T1 环境
- **做什么**：下载 `val2017.zip` + `annotations_trainval2017.zip`，解压到 `data/coco/`；抽取 `person_keypoints_val2017.json` 中的 person 子集（约 2693 图）；写一个 sanity-check notebook 可视化 3 张图+标注
- **产出文件**：`scripts/download_coco.sh`、`data/coco/{val2017/, annotations/}`、`notebooks/01_coco_sanity.ipynb`
- **完成标志**：notebook 能画出关键点叠加图；标注 json 能被 `pycocotools.COCO` 正确加载

### T3 — 三模型推理封装（1.5 天）
- **输入**：T1 环境
- **做什么**：实现一个统一的 `PoseEstimator` 接口类，内部分别调用 RTMPose-M（mmpose inferencer API）、YOLO11n-Pose（ultralytics API）、ViTPose-Base（mmpose inferencer）。接口签名：`estimator.infer(image) -> List[Keypoints17]`，输出格式统一到 COCO 17 点 `[x, y, conf]`
- **产出文件**：`scripts/pose_estimators.py`（统一接口 + 三子类）、`configs/models.yaml`（记录模型→checkpoint 映射）、`notebooks/02_model_smoke_test.ipynb`（三模型在同一张图上跑出骨架叠加）
- **完成标志**：同一张 COCO 图片，三模型均能输出 17 个关键点且坐标合理；骨架叠加图可视化通过肉眼检查

### T4 — E1 实验：COCO val OKS-mAP + Pareto 图（1 天）
- **输入**：T2 数据、T3 推理接口
- **做什么**：写 `scripts/eval_coco.py`，遍历 COCO val，调 `PoseEstimator` + pycocotools `COCOeval`，对三模型分别产出 OKS-mAP/AR；同时用 `time.perf_counter` 在 100 张固定图上测平均推理速度、用 `torch.numel` 统计参数量；最后在 notebook 里画 Pareto 气泡图
- **产出文件**：`scripts/eval_coco.py`、`outputs/e1/results.csv`（模型×指标表）、`notebooks/03_e1_pareto.ipynb`、`outputs/e1/pareto.png`
- **完成标志**：三模型 OKS-mAP 与论文参考值 ±1 以内；Pareto 图清晰显示三点 trade-off（见总方案验证点 1/2/5）

### T5 — Fit3D / Countix 数据准备（1.5 天）
- **输入**：T1 环境
- **做什么**：
  1. 下载 Fit3D 授权版（需注册），取 squat/crunch/push-up 三个动作的子集；写脚本把 3D 关节投影到 2D（利用 Fit3D 提供的相机内参）得到 2D GT 关键点
  2. 下载 Countix / RepCount annotations + YouTube 视频（用 `yt-dlp`），抽出 exercise 类（jumping jacks / squats / sit-ups）
  3. 统一用 ffmpeg 抽帧到 30fps，存为 `data/fit3d/{action}/{clip_id}/frame_%05d.jpg` 结构
- **产出文件**：`scripts/prepare_fit3d.py`（下载+投影）、`scripts/prepare_countix.py`（下载+抽帧+对齐 rep count 标注）、`data/fit3d/`、`data/countix/`、一份数据清单 `docs/data_inventory.md`
- **完成标志**：Fit3D 投影出的 2D 关键点叠在原图上肉眼看着对；Countix 每段视频的 rep 起止时间戳能映射到正确帧号

### T6 — 时序后处理 + 关节角度 + 计数/评分（1.5 天）
- **输入**：T3 推理接口、T5 数据
- **做什么**：
  1. `scripts/temporal_filter.py`：封装低置信过滤、滑动平均、1€ filter 三种平滑
  2. `scripts/joint_angles.py`：实现膝/髋/肘/肩角度的向量法计算函数
  3. `scripts/rep_counter.py`：基于关节角度峰值 + hysteresis 的计数器（进入 <100°，复位 >140°）
  4. `scripts/fitness_scorer.py`：按总方案第 6 节的规则表，对每次 rep 输出 `{standard: bool, errors: [半程/膝内扣/...]}`
- **产出文件**：上述 4 个脚本、`notebooks/04_counter_demo.ipynb`（对一段自录深蹲视频跑通计数+评分）
- **完成标志**：对一段 10 秒、10 次深蹲的视频，计数结果在 ±1 误差内；评分模块能正确标出人为做错的 2-3 次

### T7 — 可视化：骨架叠加 + 角度曲线 + 评分卡片（1 天）
- **输入**：T6 pipeline
- **做什么**：写 `scripts/render_demo.py`，给定一段视频，输出：（a）叠加骨架 + 关节角度数值的 MP4；（b）用 matplotlib 画膝角时序曲线 + rep 峰值标记的 GIF；（c）右上角贴"第 X 次 / 标准|膝内扣"评分卡片
- **产出文件**：`scripts/render_demo.py`、`outputs/demos/squat_rtmpose.mp4`（一段示例）
- **完成标志**：一个外行人看视频能立刻明白"正在数第几次，这次做对没"

### T8 — E2 实验：下游动作分类（1 天）
- **输入**：T3 推理接口、T5 Fit3D 数据
- **做什么**：对 Fit3D 的三类（squat/crunch/push-up）片段按 30 帧滑窗切样本，用三个姿态模型分别抽取 17×2×30 的关键点特征；训练一个 2 层 MLP（128→3）做分类；记录三组 Accuracy + 混淆矩阵
- **产出文件**：`scripts/extract_keypoints_fit3d.py`、`scripts/train_action_mlp.py`、`outputs/e2/results.csv`、`outputs/e2/confusion_matrix_*.png`（三张，每模型一张）
- **完成标志**：至少一个姿态模型的 MLP Acc > 85%；三模型 Acc 差距能被报告里解释为"上游精度对下游的影响"

### T9 — E3 实验：平滑消融（0.5 天）
- **输入**：T6 计数模块、T5 Countix 数据
- **做什么**：在 Countix exercise 子集上，用 RTMPose-M 产生关键点序列，扫描 `{无平滑, MA-3, MA-5, MA-7, 1€-β0.5, 1€-β1.0, 1€-β2.0}` 7 组设置，记录计数 MAE
- **产出文件**：`scripts/eval_e3_smoothing.py`、`outputs/e3/results.csv`、`notebooks/05_e3_plot.ipynb`（柱状图）
- **完成标志**：能看出平滑方法间明显差异；最优配置的 MAE 比无平滑至少降低 30%

### T10 — E4 实验：分辨率 + 遮挡鲁棒性（1 天）
- **输入**：T3 推理接口、T5 Fit3D 数据
- **做什么**：
  1. 分辨率：把 Fit3D 子集图像 resize 到 256×192 / 320×240 / 640×480，三模型 × 三分辨率统计 PCK@0.05
  2. 遮挡：对每帧随机画 5% / 10% / 20% 面积的黑矩形，统计 PCK 和计数 MAE
- **产出文件**：`scripts/eval_e4_robustness.py`、`outputs/e4/results.csv`、`notebooks/06_e4_curves.ipynb`（两张折线图）
- **完成标志**：ViTPose 在高遮挡下应显著优于 YOLO11n；有清晰的鲁棒性曲线

### T11 — 报告 + demo 视频 + README 打磨（1.5 天）
- **输入**：T4/T7/T8/T9/T10 的所有产出
- **做什么**：
  1. 写课程报告 LaTeX/Word（Intro / Related Work / Method / Experiments / Discussion / Conclusion）
  2. 剪一段 2-3 分钟 demo 视频（三类动作 × 带评分卡片）
  3. 打磨 README，让第三方能一键复现 E1-E4
- **产出文件**：`docs/report.pdf`、`docs/demo.mp4`、最终版 `README.md`
- **完成标志**：报告能讲清三个核心贡献点（总方案第 11 行）；demo 视频对外人无需解释即可看懂；`scripts/setup_env.sh && scripts/download_coco.sh && scripts/eval_coco.py` 能跑通

---

### 任务总工作量（粗估）

| 阶段 | 任务 | 工作日 |
|---|---|---|
| W1 | T1 + T2 + T3 + T4 | 3.5 |
| W2 | T5 + T6 | 3 |
| W3 | T7 + T8 + T9 | 2.5 |
| W4 | T10 + T11 | 2.5 |
| **合计** |  | **11.5 工作日** |

适合每晚 2-3 小时、持续 4 周完成。遇到卡点随时调整 T5（Fit3D 授权/下载）或 T8（自建数据替代）。
