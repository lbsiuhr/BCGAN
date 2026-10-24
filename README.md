# BCGAN: Blurred-Morphology-Constrained CycleGAN

Official PyTorch implementation of **BCGAN (Blurred-Morphology-Constrained CycleGAN)** for paper:  
**"Motion-blur-tolerant imaging flow cytometry extends the throughput–sensitivity frontier"**

[English](README.md) | [中文](README_CN.md)

---

## 1. Environment Setup

```bash
conda create -n bcgan python=3.10 -y
conda activate bcgan
pip install -r requirements.txt
```

---

## 2. Dataset Structure

Organize dataset directories under `./datasets/[dataset_name]`:

```text
datasets/jurkat/
├── trainA/          # Blurred single-cell images (Domain A)
├── trainB/          # In-focus cell images (Domain B, unpaired)
├── masks_visible/   # Corresponding segmentation masks for trainA
└── testA/           # Blurred cell images for inference
```

> **Note**: Brightfield images use `--input_nc 1 --output_nc 1`. Horizontal flip is disabled by default (`--no_flip`) due to directional fluidic flow.

---

## 3. Training

### Option A: Offline HTML Logging (Recommended)

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
Outputs and comparative visuals are automatically rendered to `./checkpoints/jurkat_bcgan/web/index.html`.

### Option B: Real-Time Visdom Dashboard

1. Start server in a separate terminal: `python -m visdom.server -p 8097`
2. Change `--display_id 0` to `--display_id 1` in the training command, and open `http://localhost:8097`.

---

## 4. Testing / Inference

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
Restored images are saved to `./results/jurkat_bcgan/test_latest/images/`.

---

## 5. Key Options

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--lambda_physics` | `1.5` | Overall weight of the blurred-morphology constraint |
| `--physics_kernels`| `20,15,23,27` | Kernels simulating velocity blur along the flow axis |
| `--weight_dice` | `1.0` | Weight for Dice mask overlap constraint |
| `--weight_proj` | `1.5` | Weight for direct Y-axis projection alignment |
| `--phys_decay_start/end` | `100 / 200` | Start and end epochs for linear decay of the physics constraint |

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

---

## 7. Acknowledgments

This repository is built upon [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
