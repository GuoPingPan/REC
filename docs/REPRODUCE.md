# 复现入口与环境

## 1. 版本和目录约定

`source/` 中的实验脚本保留了原始文件名、路径约定、哈希校验和收据检查。带版本号的早期依赖只有在最终代码仍显式引用它们时才保留，例如几何筛选恢复时会核验之前 worker 的哈希。它们不代表新增对照结果。

原始实验目录：

```text
/mnt/data/users/kindel/uav/
  experiments/
    aeroact/20261018-012530-nvila8b-aerialvln-s-official/
    aerialvln-s/20261020-104043-vipe-gt-action-rgb-recovery/
    aerialvln-s/20261022-002103-vipe-paper-repair/
```

部分训练配置采用容器挂载别名 `/mnt/data/uav/`；同一共享数据也曾通过 `/mnt/data_20t/users/kindel/uav/` 访问。这些是历史环境路径，不是仓库内目录。重新部署时需要恢复数据布局和挂载映射；修改路径或脚本后必须为新实验重新生成配置与哈希，不能把新运行冒充原实验。

## 2. RGB → 相机轨迹

推理使用 Linux、NVIDIA CUDA 和单独的 ViPE Python 环境。依赖与 CUDA 扩展安装见 `third_party/vipe/README.md`、`third_party/vipe/pyproject.toml` 和 `third_party/vipe/setup.py`。设置模型缓存并准备实验使用的 ViPE 权重和 MoGe2-L 权重，不能只安装 CPU 依赖后直接运行。

输入 JSON 结构：

```json
{
  "video": "/absolute/path/to/lossless_video.mkv",
  "video_sha256": "<完整视频文件的SHA256>",
  "frame_count": 62,
  "frames": [{"sha256_rgb": "<第0帧解码后RGB字节的SHA256>"}],
  "K": [[112.0, 0.0, 112.0], [0.0, 112.0, 112.0], [0.0, 0.0, 1.0]]
}
```

`frames` 应包含所有帧，按解码顺序排列；上面只示意第一项。输入必须是 224×224，图像哈希在 BGR→RGB 后计算。原实验每个导航观测一帧；播放帧率不是连续飞行的物理采样率。原 ViPE 批处理要求至少 8 帧，不通过插值或补帧改变输入。

```bash
export VIPE_MOGE2_CHECKPOINT=/absolute/path/to/moge-2-vitl/model.pt
export PYTHONPATH="$PWD/third_party/vipe:${PYTHONPATH:-}"
python recovery/infer_v1.py \
  --input /absolute/path/to/input.json \
  --source "$PWD/third_party/vipe" \
  --output /absolute/path/to/prediction
```

输出 `predicted_camera_c2w.npz` 和 `inference_receipt.json`。推理入口不读取 GT 位姿、GT 动作或仿真深度标签。所导出的 ViPE 代码与最终冻结代码目录中相同文件逐一比较，329 个文件无差异；仓库裁掉了与运行无关的文档、测试和工作流文件，具体打包文件见 `provenance/source_manifest.json`。

## 3. 相机轨迹 → REC 标签

主标签的算法入口是：

```python
from primitive_calibration import predict
prediction = predict(camera_c2w, calibrate=False)
```

已知条件为光学坐标到机体 FRD 坐标转换、相机前向 0.5 米外参、七种运动原语和每个相邻观测间隔对应一个原语的时钟。Stop 来自完整示范终点，未从图像独立估计。

`scripts/pose_to_actions.py` 是本次整理添加的便捷命令行，直接调用同一冻结模块，不另实现量化器。它不导入原始批处理的 GT 对齐标签。

最终数据构建顺序：

1. `build_controlled_cohort.py` 整理完整且满足代理条件的路线，并构建配对输入。
2. **`build_metric_cohort_v2.py` 从原始 ViPE 位姿重新生成 REC 动作，移除早期 GT Sim(3) 标签。**
3. 光学几何检查和 `build_optical_screened_cohort_v7.py` 形成筛选子集。
4. `build_target_valid_cohort_v1.py` 对称排除丢失 Stop 监督的五条指令，形成最终 C4。

这些构建器依赖原始视频、位姿缓存、官方标注、几何检查记录和父版本收据。它们不是面向任意新数据的一键导入器。`provenance/C4/validation_report.json` 记录最终计数和标注哈希；完整 RGB、路线标签、原始标注不随代码上传。

`recovery/common.py` 的对齐函数和 `provenance/recovery_capture_protocol.json` 保留了早期采集／诊断来源，不能用其中的 Sim(3) 处理替代最终主标签逻辑。

## 4. 训练

实际完成的六份配置在 `configs/training/`，全部为 4 节点、32 GPU、每 GPU batch 4、梯度累积 2、global batch 256。保留配置里的历史状态字段；完成状态另由收据核验，并非修改候选配置来表示成功。

训练使用 Python 3.10、PyTorch 2.3.0、Transformers 4.37.2、DeepSpeed 0.9.5。`configs/aeroact-runtime.freeze.txt` 是记录环境的包版本清单，不应当作跨平台一键安装文件；上游安装入口见 `third_party/aeroact/environment_setup.sh`。

`source/formal_training_entry.py` 在进入上游 `llava/train/train_mem.py` 前安装 `rotary_compat.py` 的稳定位置编码修复，记录实际数据、每个 rank 的进度并检查训练预算。不要绕过该入口直接使用上游 shell 脚本来声称复现本实验。

原分布式容器启动入口：

```bash
python source/formal_training_bootstrap_v6.py --config /absolute/path/to/training_config.json
```

bootstrap 需要原始实验根目录中的数据完整性收据、冻结源码清单、模型 READY 收据、运行环境归档、8 GPU/节点及 `NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`。这些外部资产不在 Git 内，缺失时它会拒绝启动。此命令不是提交云任务的命令；本仓库整理和推送没有创建训练任务。

`third_party/aeroact/` 保留实际使用的代码和相对别名链接。上游是 `return-sleep/AeroAct` 的 `0417d39d4a739ed985ef33b9cb9be4327543640e`；具体兼容性差异见 `provenance/upstream_source_difference_audit.json` 和 `patches/aeroact-upstream-0417.patch`。本仓库不含上游示例数据、媒体或旧云启动器。

## 5. 评测和统计

`configs/evaluation/` 保留六份实际评测计划。`prepare_final_policy_evaluation_v2.py`、`final_policy_controller_v2.py`、`final_policy_worker_v2.py` 会核验最终 checkpoint 与训练完成收据，使用官方 `eval.py` 和 Seen／Unseen 标注。它们依赖 Linux AirSim 场景、相应 GPU、模型权重、原始数据和实验收据；仓库不附带这些资产。

`merge_paired_final_evaluations_v1.py` 将独立 GT／REC 结果组成同种子的配对评测。`analyze_final_paired_uncertainty_v3.py` 仅接受三组已完成的配对结果：

```bash
python source/analyze_final_paired_uncertainty_v3.py \
  --root /absolute/path/to/restored-experiment \
  --evaluation-root /absolute/path/to/restored-experiment/eval/final_C4_paired_seed10_001 \
  --evaluation-root /absolute/path/to/restored-experiment/eval/final_C4_paired_seed11_001 \
  --evaluation-root /absolute/path/to/restored-experiment/eval/final_C4_paired_seed12_001 \
  --official-directory /absolute/path/to/official_annotations
```

统计保持 GT／REC 配对，并将同一物理路线的全部指令一起重采样。默认 10000 次、分析种子 20261022。该命令保护已有输出，不能用它覆盖已完成统计。

## 6. 代码核验边界

已完成的 CPU 检查使用原 Linux / Python 3.10 环境、NumPy 1.26.0、SciPy 1.15.3。固定 SciPy 版本是为了保留原实验使用的旋转 API 行为；不以修改冻结几何代码的方式适配新版本。macOS 本次遇到 SciPy 1.15.3 wheel 的动态库加载错误，未完成 Mac 运行验证。

`scripts/verify_release.py` 检查复制文件哈希、六份训练配置、六份评测计划、AeroAct 代码与原始冻结清单的一致性。CPU 测试覆盖独立合成机体动作、相机杠杆臂、全局坐标变换、尺度消融和命令行默认原始尺度。

这些检查不代表本机重新完成 CUDA 推理、分布式训练或官方仿真评测。历史完整实验摘要位于 `provenance/`，原始逐回合和权重收据留在实验存储中。
