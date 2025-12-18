# DeepDance: Posture-Guided Person Image Synthesis

![Project Status](https://img.shields.io/badge/Status-Educational-yellow)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

> **Note:** This project is an educational implementation of pose transfer concepts derived from the "Everybody Dance Now" (Chan et al., 2019) paper, adapted for limited hardware constraints (64x64 output).

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Methodology](#-methodology)
    - [1. Data Pipeline](#1-data-pipeline)
    - [2. Baseline: Nearest Neighbor](#2-baseline-nearest-neighbor)
    - [3. Deep Regression (VanillaNN)](#3-deep-regression-vanillann)
    - [4. Adversarial Generation (GAN)](#4-adversarial-generation-gan)
- [Training & Usage](#-training--usage)
- [Technical Analysis & Limitations](#-technical-analysis--limitations)
- [Future Improvements](#-future-improvements)

## 🔭 Project Overview

This system solves the **Pose-to-Image Translation** problem. Given a source video of a person dancing, we extract the skeletal movements and synthesize a target person performing those exact movements. The project explores the evolution of synthesis techniques from simple retrieval methods to Generative Adversarial Networks (GANs).

## 📂 Project Structure

```text
├── data/                   # Dataset storage
│   ├── taichi1.mp4         # Source video
│   ├── dance_skeletons/    # Cached .npy skeleton files
│   └── checkpoints/        # Saved .pth models
├── src/                    # Source code (implied)
├── VideoSkeleton.py        # Preprocessing & MediaPipe extraction
├── GenNearest.py           # Nearest Neighbor implementation
├── GenVanillaNN.py         # Direct regression models
├── GenGAN.py               # GAN training pipeline
├── DanceDemo.py            # Inference interface
└── requirements.txt        # Dependencies
```

## ⚙️ Installation & Setup

### Prerequisites
- NVIDIA GPU (Recommended: GTX 1050 or better)
- CUDA Toolkit 11.8+

### Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python mediapipe numpy matplotlib scikit-learn tqdm
```

## 🧠 Methodology

### 1. Data Pipeline
The backbone of the project is the extraction of pose vectors using **MediaPipe**.

*   **Extraction:** We extract 13 key joints (shoulders, elbows, wrists, hips, knees, ankles, nose).
*   **Normalization:**
    *   Coordinates are centered relative to the hip center to handle translation.
    *   Scale is normalized by the bounding box height to handle camera zoom variances.
*   **Output:** A vector $v \in \mathbb{R}^{26}$ (13 joints $\times$ (x, y)) or a localized heatmap image.

### 2. Baseline: Nearest Neighbor
*   **Concept:** Serves as a "retrieval-based" lower bound for performance.
*   **Algorithm:**
    1.  Store a dictionary $D = \{(S_i, I_i)\}$ of all Skeleton-Image pairs from the target video.
    2.  For a new query skeleton $S_{query}$, find index $k$ such that:
        $$k = \text{argmin}_i || S_{query} - S_i ||_2$$
    3.  Return image $I_k$.
*   **Pros:** Perfect texture preservation.
*   **Cons:** Cannot generate unseen poses; high temporal jitter.

### 3. Deep Regression (VanillaNN)
We frame the synthesis as a direct regression problem learning the mapping $f: S \rightarrow I$.

**Variant 1: Vector-to-Image**
- **Input:** Flat vector (26,)
- **Architecture:** A sequence of `ConvTranspose2d` layers. This effectively "decodes" a latent vector into an image.
- **Issue:** Lacks spatial context; relies purely on memorizing coordinate-to-pixel mappings.

**Variant 2: SkeletonMap-to-Image**
- **Input:** A sparse tensor $(3, 64, 64)$ where skeletons are drawn as lines.
- **Architecture:** U-Net-like Encoder-Decoder structure.
- **Benefit:** Convolutional filters can exploit the spatial locality of the drawn skeleton structure.

### 4. Adversarial Generation (GAN)
To tackle the "blurriness" caused by L1 loss in VanillaNN, we introduce an adversarial game.

#### Generator ($G$)
Modified DCGAN architecture with skip connections.
*   **Input:** Skeleton Image
*   **Output:** Synthesized RGB Image
*   **Loss:** Weighted combination of Pixel Loss and Adversarial Loss.

#### Discriminator ($D$)
A PatchGAN-style classifier that determines if an image is Real or Fake.
*   **Input:** RGB Image
*   **Output:** Probability map (Single scalar in this simplified implementation).

#### Objective Function
$$ \mathcal{L}_{Total} = \lambda_{L1} \cdot ||G(x) - y||_1 + \mathcal{L}_{GAN}(G, D) $$

**Dynamic Weighting Strategy:**
We utilize a dynamic $\lambda_{L1}$ schedule ($100 \rightarrow 30$) over epochs.
1.  **Early Training:** High L1 weight forces the model to learn the correct *structure* and *colors*.
2.  **Late Training:** Lower L1 weight allows the Discriminator to force high-frequency details (texture/sharpness).

## 🚀 Training & Usage

### 1. Preprocessing
Extract skeletons and normalize data.
```bash
# Syntax: python VideoSkeleton.py <video_path> <force_recompute> <frame_skip>
python VideoSkeleton.py data/taichi1.mp4 True 5
```

### 2. Training
Train the models. Checkpoints are saved automatically.

```bash
# Train Vanilla NN (Variant 2: Skeleton Image Input)
python GenVanillaNN.py data/taichi1.mp4 false 2

# Train GAN (Requires ~2-4 hours on GTX 1050)
python GenGAN.py data/taichi1.mp4
```

### 3. Inference / Demo
Generate a video using a specific method.

```bash
# Arguments:
# --gen_type: 1 (Nearest), 2 (Vanilla Vec), 3 (Vanilla Img), 4 (GAN)
# --video_path: Path to target video

python DanceDemo.py --gen_type 4 --video_path "data/taichi1.mp4"
```

## 📊 Technical Analysis & Limitations

### Results Comparison

| Method | Sharpness | Structural Consistency | Temporal Stability | Training Time |
|:---:|:---:|:---:|:---:|:---:|
| **Nearest Neighbor** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | N/A |
| **Vanilla NN** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Fast |
| **GAN** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Slow |

### Current Challenges
1.  **Mode Collapse / Blurriness:** The face region often lacks detail. This is due to the 64x64 resolution limit and the relative sparsity of facial keypoints in the input data.
2.  **Temporal Jitter:** The model processes every frame independently. There is no memory of the previous frame ($t-1$), leading to flickering background or clothing artifacts.
3.  **Limb Scaling:** When the source skeleton has different limb proportions than the target, the generator struggles to "stretch" or "shrink" the target's appearance naturally.

## 🔮 Future Improvements

To move closer to State-of-the-Art performance, the following upgrades are planned:

1.  **Temporal Coherence (Video-to-Video Synthesis):**
    *   Implement **Temporal Smoothing** by modifying the Generator input to take $S_t, S_{t-1}, S_{t-2}$ to predict $I_t$.
    *   Add a **Temporal Discriminator** (3D Conv) to penalize flickering.

2.  **Architecture Upgrades:**
    *   Replace DCGAN with **Pix2PixHD** or **SPADE (GauGAN)** blocks for better semantic layout preservation.
    *   Implement **FaceGAN** residual blocks specifically to refine facial features after the main generation.

3.  **Perceptual Loss:**
    *   Instead of simple L1 Loss, use **VGG Perceptual Loss** to optimize for high-level feature similarity rather than pixel-by-pixel exactness.

## 📚 References

1.  **Chan, C., et al.** (2019). *Everybody Dance Now*. ICCV.
2.  **Isola, P., et al.** (2017). *Image-to-Image Translation with Conditional Adversarial Networks* (Pix2Pix).
3.  **Wang, T.C., et al.** (2018). *Video-to-Video Synthesis*.

***

### Acknowledgments
Original course material by [Alexandre Meyer](http://alexandre.meyer.pages.univ-lyon1.fr/m2-apprentissage-profond-image/am/tp_dance/). Adapted and enhanced by Asskali Bakr.
