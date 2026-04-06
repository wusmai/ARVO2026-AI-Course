# ARVO 2026 AI Course

Hands-on notebooks and utilities for an ophthalmic AI course, covering both **image classification** and **image segmentation** workflows using PyTorch and MONAI.

---

## Overview

This repository provides practical, notebook-based examples for building and evaluating deep learning models for medical imaging.

It includes:

- **Classification** with a modern pretrained vision backbone (DINOv3)
- **Segmentation** using MONAI and U-Net
- **Live training dashboards** for teaching and visualization

---

## Repository Structure

```text
ARVO2026-AI-Course/
├── 1-Classification.ipynb
├── 2-Segmentation.ipynb
└── arvo.py
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/wusmai/ARVO2026-AI-Course.git
cd ARVO2026-AI-Course
```

---

### 2. Create environment

#### Option A: pip (simple)

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
```

#### Option B: conda

```bash
conda create -n arvo-ai python=3.10 -y
conda activate arvo-ai
```

---

### 3. Install dependencies

```bash
pip install torch torchvision torchaudio
pip install monai
pip install transformers
pip install scikit-learn
pip install matplotlib numpy pillow
pip install jupyter ipywidgets tensorboard
```

#### Optional (recommended for performance)

Install CUDA-enabled PyTorch from:
👉 https://pytorch.org/get-started/locally/

---

### 4. Launch notebooks

```bash
jupyter lab
```

Open:

- `1-Classification.ipynb`
- `2-Segmentation.ipynb`

---

## Data Setup (IMPORTANT)

This repo does **not include data**, so you must organize your datasets correctly before running.

---

# 📊 Classification Data Setup

The classification notebook expects a dataset under a `data/` directory.

### Recommended structure:

```text
data/
├── train/
│   ├── class0/
│   │   ├── img1.png
│   │   ├── img2.png
│   ├── class1/
│       ├── img3.png
│       ├── img4.png
│
├── valid/
│   ├── class0/
│   ├── class1/
│
└── test/
    ├── class0/
        ├── class1/
```

### Notes

- Each folder name = class label
- Images can be `.png`, `.jpg`, etc.
- You can modify the dataset loader in the notebook if using CSVs or other formats

---

# 🧠 Segmentation Data Setup

The segmentation notebook expects **paired image/mask files** in this exact structure:

```text
segmentation_data/
├── train-images/
├── train-labels/
├── valid-images/
├── valid-labels/
├── test-images/
└── test-labels/
```

---

### File pairing rule (CRITICAL)

Each image must have a corresponding label with the **same filename**:

```text
train-images/case_001.png
train-labels/case_001.png
```

---

### Label format requirements

- Binary masks:
  - Background = 0
    - Foreground = 255 (or 1 depending on preprocessing)
    - Same spatial size as input image
    - Grayscale (single channel)

    ---

### Example

```text
segmentation_data/
├── train-images/
│   ├── img_001.png
│   ├── img_002.png
│
├── train-labels/
│   ├── img_001.png
│   ├── img_002.png
```

---

## Notebook Details

---

### 1️⃣ Classification

Key features:

- Pretrained model:
  ```
    facebook/dinov3-vits16-pretrain-lvd1689m
      ```
      - Transfer learning vs. scratch comparison
      - AUROC evaluation
      - Occlusion sensitivity for interpretability
      - Weighted sampling for class imbalance
      - Live training visualization

      Config highlights:

      ```python
      image_size = 224
      batch_size = 32
      ```

      ---

### 2️⃣ Segmentation

Pipeline includes:

- MONAI `UNet`
- Dice loss optimization
- Mean Dice validation metric
- TensorBoard logging
- Best model checkpointing

Model saved to:

```text
ckpt/best_model.pt
```

---

## Visualization & Monitoring

---

### Notebook Dashboard (`arvo.py`)

Provides:

- Training + validation curves
- AUC / loss tracking
- Learning rate visualization
- GPU memory usage
- ETA tracking

Designed for **live teaching and demos**

---

### TensorBoard (Segmentation)

Run:

```bash
tensorboard --logdir runs
```

Then open:

```text
http://localhost:6006
```

---

## Tips for Smooth Execution

- Start with small datasets to validate pipeline
- Ensure image sizes are consistent
- Use GPU if available (especially for segmentation)
- If training crashes:
  - reduce batch size
    - check image/mask pairing
      - verify data paths

      ---

## Common Pitfalls

| Issue | Cause |
|------|------|
| File not found | Incorrect directory structure |
| Poor Dice score | Masks not binary or misaligned |
| Training unstable | Learning rate too high |
| CUDA errors | Incorrect PyTorch install |

---

## Suggested Improvements (Future)

- Add `requirements.txt`
- Add sample dataset
- Add pretrained weights
- Add inference-only notebook
- Add CLI training script version

---

## Educational Use

This repo is ideal for:

- Ophthalmology AI courses
- MONAI workshops
- Medical imaging bootcamps
- Intro-to-deep-learning labs

---

## Disclaimer

This repository is intended for **educational and research use only**.

Models and workflows should **not be used for clinical decision-making without proper validation**.

---

## Acknowledgments

- PyTorch
- MONAI
- Hugging Face Transformers
- Meta FAIR (DINOv3)

