# REC: AerialVLN-S 视频动作监督

这里整理了最终配对研究使用的代码：单目视频恢复相机运动、构建动作监督、训练 AeroAct、运行官方导航评测和配对统计。核心实验脚本保留原始文件名和字节哈希，未为本次提交重写算法。

**最终主流程：RGB → ViPE（MoGe2-L 深度先验）→ 原始尺度相机轨迹 → 相机／机体外参转换 → 离散动作 → AeroAct。**

主训练标签调用 `predict(poses, calibrate=False)`，不使用 GT 拟合尺度。已知动作原语的尺度优化是独立消融；完整 GT Sim(3) 只作诊断。GT 仍参与路线筛选、配对边界设计、GT 对照和评分。

## 代码入口

| 位置 | 内容 |
| --- | --- |
| `recovery/infer_v1.py` | 冻结的 RGB-only ViPE 推理入口，固定内参、seed 0 |
| `source/primitive_calibration.py` | 原始尺度动作量化，以及可选原语尺度消融 |
| `source/build_metric_cohort_v2.py` | 从原始 ViPE 位姿重新生成不读取 GT 动作值的 REC 标签 |
| `source/build_optical_screened_cohort_v7.py`、`source/build_target_valid_cohort_v1.py` | 几何代理筛选与监督完整性筛选，形成最终 C4 子集 |
| `source/formal_training_entry.py`、`source/formal_training_bootstrap_v6.py` | 最终训练、稳定 RoPE 修复和分布式记录 |
| `source/final_policy_worker_v2.py`、`source/final_policy_controller_v2.py` | 最终 checkpoint 的官方导航评测 |
| `source/analyze_final_paired_uncertainty_v3.py` | 三个训练种子的配对路线聚类 bootstrap |
| `third_party/vipe/`、`third_party/aeroact/` | 实验实际使用的依赖代码，保留上游许可 |
| `configs/training/`、`configs/evaluation/` | 六个已完成训练和对应评测的原始配置 |
| `provenance/` | 数据版本、源文件哈希、上游差异和已记录的实验摘要 |

## CPU 快速检查

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.txt
.venv/bin/python scripts/verify_release.py
.venv/bin/python -m pytest -q
```

上述 CPU 检查已在原 Linux / Python 3.10 环境执行；macOS 本次未通过 SciPy 二进制依赖加载，未声称完成 Mac 运行验证。从已有位姿文件生成动作：

```bash
.venv/bin/python scripts/pose_to_actions.py \
  --poses /path/to/predicted_camera_c2w.npz \
  --output /path/to/actions.json \
  --complete-demo
```

输入为 `data` 键下的 `N×4×4` camera-to-world 光学坐标位姿。默认保留原始尺度；只有明确给出 `--complete-demo` 才附加已知示范终点的 Stop。`--scale-mode primitive` 只用于尺度消融。

GPU 推理、训练、评测和原始实验目录要求见 [复现说明](docs/REPRODUCE.md)。本仓库包含代码与配置，不包含 RGB 数据、模型权重、仿真场景、CUDA 编译产物或完整运行收据；克隆仓库不等于已具备完整训练环境。

## 最终实验版本

- 数据：`complete-optical-target-valid-v4`，2264 条路线、5915 条指令、253714 帧共享 RGB。
- 主配对组：`gt-joint-boundary` 与 `rec-joint-boundary`，各 351983 个样本。
- 训练种子：10、11、12；每组 global batch 256、1 epoch、1374 次优化更新。
- 官方导航：每模型 Seen 333 条、Unseen 531 条；采样种子 10、temperature 0.2、最多 500 次动作。
- 筛选属于 GT 和仿真辅助的研究子集；表面代理通过不代表物理无碰撞认证。

本次代码整理只运行静态检查和 CPU 合成几何检查，没有重新训练或运行完整导航评测。
