# GSRecon

GSRecon 是一个基于 `gsplat` 的 3D Gaussian Splatting 重建工程，当前版本聚焦于：

- 以 COLMAP 稀疏重建结果作为输入初始化高斯点云。
- 使用 `gsplat` 完成可微渲染。
- 支持两类密度控制策略：`absgrad` 和 `mcmc`。
- 支持按视角学习相机位姿增量的 `pose refine`。
- 提供训练脚本和离线评估脚本两类对外接口。

当前实现只支持 `renderer_type="gsplat"`，数据侧只支持 COLMAP 的 `PINHOLE` 和 `SIMPLE_PINHOLE` 相机模型。

## 1. 项目结构

核心代码入口如下：

- [train.py](/home/wyc/Workspace/GSRecon/train.py)：训练入口。
- [metrics.py](/home/wyc/Workspace/GSRecon/metrics.py)：离线评估入口。
- [configs/gsplat/truck.yaml](/home/wyc/Workspace/GSRecon/configs/gsplat/truck.yaml)：训练配置示例。
- [gaussian_splatting/gsplat/gaussian_model.py](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gaussian_model.py)：高斯参数初始化、优化器和 densification。
- [gaussian_splatting/gsplat/gsplat_renderer.py](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gsplat_renderer.py)：渲染器与 pose refine 模块。
- [datasets/colmap_loader.py](/home/wyc/Workspace/GSRecon/datasets/colmap_loader.py)：COLMAP 数据读取与数据集封装。
- [utils/eval_utils.py](/home/wyc/Workspace/GSRecon/utils/eval_utils.py)：PSNR / SSIM / LPIPS 评估。

## 2. 整体算法流程

### 2.1 数据准备

输入数据目录应为 COLMAP 风格：

```text
data/<scene_name>/
├── images/
└── sparse/0/
    ├── cameras.bin or cameras.txt
    ├── images.bin or images.txt
    └── points3D.bin / points3D.txt / points3D.ply
```

训练时首先通过 [COLMAPSceneInfo](/home/wyc/Workspace/GSRecon/datasets/colmap_loader.py) 读取：

- 图像路径
- 相机内参
- 世界到相机外参 `extrinsic`
- 稀疏点云文件路径

随后 [COLMAPDataset](/home/wyc/Workspace/GSRecon/datasets/colmap_loader.py) 完成：

- 图像缩放
- 内参同步缩放
- 平移按 `scene_scale` 缩放
- 可选的 CPU / GPU 预加载
- 按 `eval_split_interval` 划分 train / val

### 2.2 高斯初始化

训练开始后，脚本会从 COLMAP 稀疏点云初始化高斯参数：

1. 读取 `points3D` 得到初始 3D 点和颜色。
2. 使用 `simple_knn.distCUDA2` 估计邻域尺度。
3. 过滤尺度过大的离群点。
4. 构造高斯参数：
   - `xyz`
   - `opacity`
   - `scaling`
   - `rotation`
   - SH 颜色系数

对应实现位于 [GaussianModel.create_from_pcd](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gaussian_model.py)。

### 2.3 渲染与损失

每次训练迭代流程如下：

1. 从 train split 随机取一个视角。
2. 读取该视角的原始 `extrinsic` 和 `intrinsic`。
3. 如果开启 `pose_optimize`，则对当前视角外参做 pose refine。
4. 调用 `gsplat.rendering.rasterization` 生成渲染图像。
5. 计算损失：
   - `L1`
   - `SSIM`
   - 若使用 `mcmc`，还会加入 `opacity_reg` 与 `scale_reg`
6. 反向传播并更新高斯参数和相机位姿参数。

渲染入口是 [GSplatRenderer.render](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gsplat_renderer.py)，训练主循环位于 [optimize](/home/wyc/Workspace/GSRecon/train.py)。

### 2.4 Pose Refine

相机位姿优化不是直接覆盖原始外参，而是对每个视角学习一个相对增量：

- 每个 `view_id` 对应一行 `embedding`。
- 每行参数是 9 维：3 维平移增量 + 6 维旋转表示。
- 训练时使用 `原始外参 + 当前视角增量` 作为渲染外参。

更具体地说：

1. 将输入的 `w2c` 外参转换为 `c2w`。
2. 按 `view_id` 取出该视角的增量变换。
3. 在 `c2w` 上右乘增量变换。
4. 再转换回 `w2c` 用于渲染。

对应实现见 [GSplatCameraOptModule](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gsplat_renderer.py) 与 [GSplatPoseRefiner](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gsplat_renderer.py)。

### 2.5 Densification

当前支持两种密度控制策略：

#### `absgrad`

- 基于 2D 投影位置的绝对梯度累计统计。
- 周期性执行 clone / split / prune。
- 适合更接近标准 3DGS 的训练流程。

#### `mcmc`

- 基于 opacity 分布重定位低质量高斯。
- 按概率新增高斯。
- 每步额外注入 SGLD 噪声。

这两套逻辑都实现在 [gaussian_model.py](/home/wyc/Workspace/GSRecon/gaussian_splatting/gsplat/gaussian_model.py) 中。

### 2.6 训练中评估

如果配置中 `training_params.eval=True`，训练过程会定期计算三类指标：

- `Metrics/train_psnr`
- `Metrics/train_ssim`
- `Metrics/train_lpips`
- `Metrics/eval_psnr`
- `Metrics/eval_ssim`
- `Metrics/eval_lpips`

其中：

- train 指标使用 refine 后的相机外参。
- val 指标继续使用原始外参。

对应逻辑在 [train.py](/home/wyc/Workspace/GSRecon/train.py) 和 [utils/eval_utils.py](/home/wyc/Workspace/GSRecon/utils/eval_utils.py)。

### 2.7 训练输出

训练结束后会输出：

- 最终高斯模型 `gaussians/iteration_<iter>.ply`
- 训练耗时 `logs/timing.json`
- 配置快照 `config.yaml`
- 若开启 pose refine，则输出 `pose_refined/sparse/0/` 下的 COLMAP 文本模型

典型结果目录如下：

```text
results/gsplat/truck/exp_006/
├── config.yaml
├── gaussians/
│   └── iteration_30000.ply
└── logs/
    └── timing.json
```

如果导出了位姿优化结果，还会有：

```text
pose_refined/sparse/0/
├── cameras.txt
├── images.txt
└── points3D.txt
```

## 3. 对外接口

### 3.1 训练接口

训练入口是 [train.py](/home/wyc/Workspace/GSRecon/train.py)。

调用方式：

```bash
python3 train.py --config configs/gsplat/truck.yaml
```

配置文件中最重要的字段包括：

#### `scene`

- `data_path`：COLMAP 数据目录。
- `output_path`：实验结果根目录。
- `exp_name`：当前实验名。

#### `training_params`

- `iterations`：训练轮数。
- `preload` / `preload_device` / `preload_dtype`：数据预加载策略。
- 当 `preload_device="cpu"` 且 `preload_dtype="fp16"` 时，图像和 mask 会以 half 存储在 RAM 中；训练和评估在使用点会转回 `float32` 计算损失和指标。
- `lambda_dssim`：L1 与 SSIM 的权重平衡。
- `optimizer_type`：高斯参数优化器。
- `eval` / `eval_interval`：训练期评估开关与频率。
- `pose_optimize`：是否开启 pose refine。
- `densification_params`：densification 策略和参数。

#### `renderer_params`

- `renderer_type`：当前仅支持 `"gsplat"`。

#### `model_params`

- `image_scale`：图像缩放比例。
- `scene_scale`：场景尺度缩放。
- `spherical_harmonics` / `sh_degree`：SH 颜色建模配置。
- `max_num_gaussians`：高斯数量上限。

配置示例见 [configs/gsplat/truck.yaml](/home/wyc/Workspace/GSRecon/configs/gsplat/truck.yaml)。

### 3.2 离线评估接口

离线评估入口是 [metrics.py](/home/wyc/Workspace/GSRecon/metrics.py)。

调用方式：

```bash
python3 metrics.py --exp_dir results/gsplat/truck/exp_006 --split train val
```

常用参数：

- `--exp_dir`：实验目录，脚本会读取其中的 `config.yaml` 和 `gaussians/`。
- `--device`：评估设备，例如 `cuda:0`。
- `--ply`：可选，手动指定某个 `.ply` 文件。
- `--split`：指定评估 `train`、`val` 或两者。

离线评估会输出：

- `metrics/train.json`
- `metrics/val.json`

每个 JSON 中包含：

- 平均 `psnr`
- 平均 `ssim`
- 平均 `lpips`
- 每张图像的逐帧指标

如果实验目录下存在 `pose_refined/sparse/0/images.txt`，那么 train split 的离线评估会优先使用 refine 后的相机位姿。

### 3.3 Python 级接口

如果需要从 Python 代码中调用，当前最常用的公开接口有：

```python
from gaussian_splatting import create_renderer, create_pose_refiner
from utils.eval_utils import evaluate_gaussian_photometric
from utils.config_utils import load_config
```

用途分别是：

- `load_config(path)`：读取并递归合并 YAML 配置。
- `create_renderer(renderer_type, **kwargs)`：创建渲染器。
- `create_pose_refiner(...)`：创建 pose refine 模块。
- `evaluate_gaussian_photometric(...)`：对数据集计算 PSNR / SSIM / LPIPS。

## 4. 推荐使用流程

### 4.1 训练

1. 准备 COLMAP 数据目录。
2. 修改配置文件中的 `scene.data_path` 和实验名。
3. 运行训练：

```bash
python3 train.py --config configs/gsplat/truck.yaml
```

### 4.2 查看结果

训练结束后优先关注：

- `gaussians/iteration_*.ply`
- `logs/timing.json`
- `pose_refined/sparse/0/images.txt`

### 4.3 离线评估

```bash
python3 metrics.py --exp_dir results/gsplat/truck/exp_001 --split train val
```

评估结果会写入 `metrics/train.json` 和 `metrics/val.json`。

## 5. 运行依赖

项目当前代码依赖以下 Python 包或扩展：

- `torch`
- `gsplat`
- `simple_knn`
- `fused_ssim`
- `lpips`
- `opencv-python`
- `plyfile`
- `PyYAML`
- `munch`
- `tensorboard`
- `tqdm`
- `numpy`

其中 `gsplat`、`simple_knn`、`fused_ssim` 为运行训练和渲染的关键依赖。

## 6. 当前版本限制

- 仅支持 COLMAP 稀疏重建作为输入。
- 仅支持 `PINHOLE` / `SIMPLE_PINHOLE` 相机模型。
- 仅实现 `gsplat` 渲染后端。
- train split 的训练期指标使用 refine 后位姿，val split 仍使用原始位姿；离线评估则会在存在导出位姿时优先使用 refine 后的 train 位姿。
