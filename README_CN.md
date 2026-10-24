# BCGAN: 模糊形态学约束 CycleGAN (Blurred-Morphology-Constrained CycleGAN)

论文 **"Motion-blur-tolerant imaging flow cytometry extends the throughput–sensitivity frontier"** 官方代码库。  
模型全称：**BCGAN (Blurred-Morphology-Constrained CycleGAN)**。

[English](README.md) | [中文](README_CN.md)

---

## 1. 环境配置

```bash
conda create -n bcgan python=3.10 -y
conda activate bcgan
pip install -r requirements.txt
```

---

## 2. 数据集格式

数据存放在 `./datasets/[数据集名称]` 下，结构如下：

```text
datasets/jurkat/
├── trainA/          # 模糊细胞图 (输入域 A)
├── trainB/          # 清晰细胞图 (目标域 B，非配对)
├── masks_visible/   # trainA 对应的单细胞轮廓掩膜
└── testA/           # 测试集模糊细胞图
```

> **说明**：明场单通道灰度图训练时保持 `--input_nc 1 --output_nc 1`；微流控流动具有单向性，训练默认禁用水平翻转 (`--no_flip`)。

---

## 3. 模型训练

### 方式 A：离线 HTML 看板（推荐，无需后台服务）

```powershell
python train.py `
  --dataroot "./datasets/jurkat_mini_sample" `
  --mask_dataroot "./datasets/jurkat_mini_sample/masks_visible" `
  --name jurkat_bcgan `
  --model cycle_gan `
  --display_id 1 `
  --batch_size 1 `
  --input_nc 1 `
  --output_nc 1 `
  --lr 0.0002 `
  --lambda_physics 1.5 `
  --physics_kernels 15,20,23,27 `
  --weight_dice 1.0 `
  --weight_proj 1.5 `
  --phys_warmup_start 0 `
  --phys_warmup_end 3 `
  --phys_decay_start 100 `
  --phys_decay_end 200 `
  --n_epochs 100 `
  --n_epochs_decay 100 `
  --gpu_ids 0
```
训练结果与对比图自动输出至 `./checkpoints/jurkat_bcgan/web/index.html`。

### 方式 B：Visdom 实时监控

1. 新终端启动：`python -m visdom.server -p 8097`
2. 将训练命令中 `--display_id 0` 改为 `--display_id 1`，访问 `http://localhost:8097`。

---

## 4. 模型测试 / 推理

```powershell
python test.py `
  --dataroot "./datasets/jurkat" `
  --name jurkat_bcgan `
  --model cycle_gan `
  --phase test `
  --input_nc 1 `
  --output_nc 1 `
  --results_dir "./results/" `
  --epoch latest
```
去模糊结果保存在 `./results/jurkat_bcgan/test_latest/images/`。

---

## 5. 核心参数调节

| 参数 | 默认值 | 说明 |
| :--- | :---: | :--- |
| `--lambda_physics` | `1.5` | 物理约束损失总权重 |
| `--physics_kernels` | `20,15,23,27` | 模拟流动拖尾核尺度（对应不同流速/曝光） |
| `--weight_dice` | `1.0` | 面积重合约束权重 |
| `--weight_proj` | `1.5` | Y 轴垂直边界投影约束权重 |
| `--phys_decay_start/end` | `100 / 200` | 物理约束权重衰减起始与归零轮次 |

---

## 6. Citation

```bibtex
@article{fan2026motion,
  title={Motion-blur-tolerant imaging flow cytometry extends the throughput--sensitivity frontier},
  author={Fan, Hangtian and Chen, Jingjing and Zhang, Hongbo and Mao, Yuxin and Wang, Zi and Xin, Chen and Huang, Ziwei and Zhang, Yue and Wang, Xiaojie and Liu, Guangli and others},
  journal={bioRxiv},
  year={2026}
}
```
