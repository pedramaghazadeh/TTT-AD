# Test-Time Training for Robust Object Classification in Autonomous Driving

This project implements a **Test-Time Training (TTT)** framework to improve the robustness of object classification models in autonomous driving, particularly under distribution shifts caused by weather, lighting, and sensor artifacts.

## 🧠 Core Idea

We augment a pretrained classifier with a self-supervised head that solves an auxiliary task—such as predicting image rotation—at test time.

Given a test-time input $x \sim \mathcal{D}_{\text{test}}$, the model performs:

- $y_{\text{cls}} = f_{\text{cls}}(f_{\theta}(x))$ — supervised classification
- $y_{\text{rot}} = f_{\text{rot}}(f_{\theta}(x'))$ — self-supervised rotation prediction on augmented input $x'$

During inference, we update the shared model parameters $\theta$ using:

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}_{\text{rot}}(f_{\text{rot}}(f_{\theta}(x')))
$$

This enables dynamic adaptation to out-of-distribution (OOD) inputs **without requiring any test-time labels**.

---

## 📦 Datasets

- **BDD100K** and **KITTI**: Object crops are extracted using annotated 2D bounding boxes.
- **Synthetic corruptions**: Simulated test-time effects include fog, rain, snow, blur, and sun flare (using `albumentations`).

Each dataset is split into:
- `train/`
- `val/`
- `test/`

with accompanying label files: `train_labels.json`, `val_labels.json`, `test_labels.json`.

---

## 🏗 Architecture

- **Backbone**: `EfficientNet-B0` (from `timm`)
- **Heads**:
  - Supervised classification head
  - Self-supervised rotation head:
    - 4-class classification (0°, 90°, 180°, 270°)
    - or continuous rotation regression (in radians)

---

## 🔍 Features

- Rotation-based self-supervised learning for TTT
- Robust object-level dataset generation with synthetic OOD shifts
- Dual-head architecture for efficient online test-time adaptation
- Fully PyTorch-compatible with modular design

---

## 🚀 Future Work

- Replace EfficientNet with **Vision Transformers (ViTs)** for better OOD handling
- Explore **JEPA**-style contrastive/predictive adaptation instead of rotation-only tasks
- Extend to **online TTT** with streaming data in real-time driving settings

---

## 📁 Project Structure
