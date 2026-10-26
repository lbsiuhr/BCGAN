# BCGAN: Blurred-Morphology-Constrained CycleGAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **BCGAN (Blurred-Morphology-Constrained CycleGAN)** for the paper:  
**"Motion-blur-tolerant imaging flow cytometry extends the throughput–sensitivity frontier"**

[English](README.md) | [中文说明](README_CN.md)

---

## 1. System Requirements

### Operating Systems & Software Dependencies
* **Tested Operating System**:
  * Windows 10 / 11 (64-bit)
* **Software Dependencies**:
  * Python >= 3.10
  * PyTorch >= 2.0.0
  * Torchvision >= 0.15.0
  * Additional packages listed in [`requirements.txt`](requirements.txt)

### Hardware Tested
* **Workstation Configuration (Used for paper training and experiments)**:
  * CPU: Intel Core i7-12700K
  * GPU: NVIDIA GeForce RTX 4090 (24 GB VRAM)
  * Memory: 32 GB RAM
* **Inference / Demo Requirements**:
  * Tested on both NVIDIA GPU (`--gpu_ids 0`) and standard CPU mode (`--gpu_ids -1`).

---

## 2. Installation Guide

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/lbsiuhr/BCGAN.git
cd BCGAN

# 2. Create and activate Conda environment
conda create -n bcgan python=3.10 -y
conda activate bcgan

# 3. Install PyTorch with CUDA (or CPU version from pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt
```

### Estimated Install Time
* Approximately **5–10 minutes** on a standard desktop computer with normal network connectivity.

---

## 3. Demo

A sample dataset (`jurkat_mini_sample`) and a pre-trained Jurkat deblurring model are included directly in this repository for out-of-the-box demonstration.

### Included Demo Assets
* **Pre-trained Model ($G_A: \text{Blur} \to \text{Sharp}$)**: [`checkpoints/jurkat_bcgan/latest_net_G_A.pth`](checkpoints/jurkat_bcgan/latest_net_G_A.pth)
* **Demo Test Images**: [`datasets/jurkat_mini_sample/testA/`](datasets/jurkat_mini_sample/testA/) (6 microfluidic blurred cell images)

### Running Demo Inference

#### Standard Bash (Linux / macOS):
```bash
python test.py \
  --dataroot "./datasets/jurkat_mini_sample/testA" \
  --checkpoints_dir "./checkpoints" \
  --name jurkat_bcgan \
  --model test \
  --model_suffix _A \
  --input_nc 1 \
  --output_nc 1 \
  --no_dropout \
  --results_dir "./results" \
  --epoch latest \
  --gpu_ids 0
```

#### Windows PowerShell (Single-Line Command):
```powershell
python test.py --dataroot "./datasets/jurkat_mini_sample/testA" --checkpoints_dir "./checkpoints" --name jurkat_bcgan --model test --model_suffix _A --input_nc 1 --output_nc 1 --no_dropout --results_dir "./results" --epoch latest --gpu_ids 0
```
*(For CPU-only inference, set `--gpu_ids -1`)*

### Expected Output
* **Console Output**:
  ```text
  loading the model from ./checkpoints/jurkat_bcgan/latest_net_G_A.pth
  processing (0000)-th image...
  processing (0005)-th image...
  ```
* **Restored Images**: Saved in `./results/jurkat_bcgan/test_latest/images/`
  * `*_real.png`: Original blurred input image.
  * `*_fake.png`: Restored in-focus cell image.
* **HTML Summary**: Generated at `./results/jurkat_bcgan/test_latest/index.html`.

### Expected Run Time
* **End-to-End Demo Execution**: **~2–3 seconds** on an ordinary desktop (including Python/PyTorch cold-start, model weights loading, and disk I/O for the 6 demo cell images).
* **Pure Network Inference Latency (as reported in Section 4.7)**: Measured on an NVIDIA GeForce RTX 4090 GPU (batch size 4), BCGAN achieves a mean network inference latency of **0.48 ± 0.01 ms per image** (equivalent to **2,087 ± 49 frames/s**), fulfilling real-time throughput requirements for microfluidic imaging flow cytometry.

---

## 4. Instructions for Use

### Running on Custom Data
1. Prepare single-cell brightfield images:
   * **Format**: Single-channel 8-bit grayscale (`.png`, `.tif`, or `.jpg`).
   * **Image Dimensions**: Flexible (e.g., 128×128, 256×256, 512×512; arbitrary dimensions supported, preferably multiples of 4).
2. Place images in `./datasets/[your_dataset]/testA/`.
3. Run `test.py` pointing `--dataroot` to your test folder.

### Custom Training
1. Organize your training data:
   ```text
   datasets/[dataset_name]/
   ├── trainA/          # Blurred single-cell images (Domain A)
   ├── trainB/          # In-focus single-cell images (Domain B, unpaired)
   ├── masks_visible/   # Corresponding masks for trainA
   └── testA/           # Blurred cell images for inference (Domain A)
   ```
2. Run training:
   ```bash
   python train.py \
     --dataroot "./datasets/jurkat_mini_sample" \
     --mask_dataroot "./datasets/jurkat_mini_sample/masks_visible" \
     --name jurkat_bcgan \
     --model cycle_gan \
     --display_id 1 \
     --batch_size 1 \
     --input_nc 1 \
     --output_nc 1 \
     --lr 0.0002 \
     --lambda_physics 1.5 \
     --physics_kernels 15,20,23,27 \
     --weight_dice 1.0 \
     --weight_proj 1.5 \
     --phys_warmup_start 0 \
     --phys_warmup_end 3 \
     --phys_decay_start 100 \
     --phys_decay_end 200 \
     --n_epochs 100 \
     --n_epochs_decay 100 \
     --gpu_ids 0
   ```

### Reproduction of Results
Refer to **Section 4.7 (Network training and inference)** in the manuscript for quantitative training procedures, hyperparameter schedules, and evaluation metrics.

---

## 5. Key Options

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--lambda_physics` | `1.5` | Weight of the blurred-morphology constraint |
| `--physics_kernels`| `15,20,23,27` | Kernels simulating velocity blur along the flow axis |
| `--weight_dice` | `1.0` | Weight for Dice mask overlap constraint |
| `--weight_proj` | `1.5` | Weight for vertical projection alignment |
| `--phys_decay_start/end` | `100 / 200` | Start and end epochs for decay of physics constraint |
| `--no_flip` | `True` | Recommended: fluidic flow direction is non-symmetric |

---

## 6. License

This project is licensed under the [MIT License](LICENSE).

---

## 7. Citation

```bibtex
@article{fan2026motion,
  title={Motion-blur-tolerant imaging flow cytometry extends the throughput--sensitivity frontier},
  author={Fan, Hangtian and Chen, Jingjing and Zhang, Hongbo and Mao, Yuxin and Wang, Zi and Xin, Chen and Huang, Ziwei and Zhang, Yue and Wang, Xiaojie and Liu, Guangli and others},
  journal={bioRxiv},
  year={2026}
}
```

---

## 8. Acknowledgments

Based on [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
