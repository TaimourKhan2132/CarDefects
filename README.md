# Multi-Label Car Defect Detection

This project implements a deep learning pipeline to detect multiple types of damage on vehicle images simultaneously. Utilizing a **ConvNeXt-Tiny** architecture, the model identifies four specific defect classes: `broken_glass`, `dent`, `scratch`, and `wreck`.

The system is designed for high-performance inference on consumer hardware (e.g., RTX 4050) using Mixed Precision (AMP) and achieves an Exact Match Accuracy of **91.7%**.

## 📊 Performance Metrics

| Metric | Score | Description |
| :--- | :--- | :--- |
| **Exact Match Accuracy** | **91.72%** | The strict metric where *all* labels for an image must be correct. |
| **Weighted F1-Score** | **94.65%** | Balanced metric accounting for class imbalance. |
| **Weighted Precision** | **94.64%** | Minimizes false positives. |
| **Weighted Recall** | **94.76%** | Minimizes false negatives (missed defects). |

## 🛠️ Technical Architecture

### Model
* **Backbone:** `ConvNeXt-Tiny` (Pretrained on ImageNet-1K).
* **Head:** Custom Linear Layer for multi-label output (`num_classes=4`).
* **Loss Function:** `BCEWithLogitsLoss` (Binary Cross Entropy) for independent probability estimation per class.

### Training Pipeline
* **Two-Stage Fine-Tuning:**
    1.  **Warmup:** Frozen backbone, trained classifier head only (5 epochs).
    2.  **Fine-Tuning:** Unfrozen full model with a lower learning rate (`1e-4`) (5 epochs).
* **Optimization:** `Adam` optimizer with `GradScaler` for Mixed Precision (FP16) training.
* **Augmentation:** Random horizontal flips, rotation, and color jittering to improve generalization.

## 📂 Dataset Structure

The dataset is a merged compilation of 6 raw sources, processed into a unified format with approx 30k images.

**Class Mapping:**
* 0: `broken_glass`
* 1: `dent`
* 2: `scratch`
* 3: `wreck`

**Input Processing:**
* Images are resized to `224x224`.
* Normalized using ImageNet mean/std standards.

## 🚀 Setup & Usage

### 1. Installation
Clone the repository and install dependencies:
```bash
pip install -r requirements.txt