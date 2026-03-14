# Adversarial ML — FGSM Attack and Adversarial Training

This project explores **adversarial attacks and defenses** on neural networks using PyTorch.

## What You'll Learn

We start by training a normal CNN on MNIST and demonstrating how the model can be fooled by **Fast Gradient Sign Method (FGSM)**, a simple yet powerful adversarial attack technique. Then we implement **adversarial training**, where the model is trained on adversarial examples to improve robustness and resistance to attacks.

The goal is to understand:
- **How attacks work**: The mathematics and mechanics behind gradient-based attacks
- **Why models are vulnerable**: How small perturbations exploit neural network decision boundaries
- **How to defend**: Building robust models through adversarial training

---

# Dataset

The experiments use the MNIST.

Dataset details:

* 60,000 training images
* 10,000 test images
* grayscale images
* size **28 × 28**
* labels **0–9**

The dataset is automatically downloaded using PyTorch.

---

---

# FGSM Attack

The project implements **Fast Gradient Sign Method (FGSM)** introduced by Ian Goodfellow.

FGSM creates adversarial inputs by modifying pixels in the direction that maximizes model error.

## Attack Formula

```
x_adv = x + ε * sign(∇x J(θ, x, y))
```

Where:

* `x` = original input image
* `ε` (epsilon) = attack strength (controls perturbation magnitude)
* `∇x J(θ, x, y)` = gradient of the loss with respect to the input
* `sign()` = sign function (returns -1, 0, or 1)

## Why FGSM Works

Even small perturbations can cause the model to misclassify images because:
1. The attack moves inputs along the gradient direction (steepest ascent of loss)
2. This direction is precisely where the model is most sensitive
3. Neural networks have large linear regions in their decision boundaries
4. Small moves in adversarial directions can cross these boundaries easily

## FGSM Properties

| Property | Value |
|----------|-------|
| Attack Type | Gradient-based, white-box |
| Computational Cost | Very low (single backprop) |
| Effectiveness | High on undefended models |
| Visibility | Perturbations often imperceptible to humans |

---

# Normal Training

In standard neural network training the model only sees **clean images**.

## Training Pipeline

```
image
  ↓
model prediction
  ↓
compute loss
  ↓
backpropagate gradients
  ↓
update weights
```

The model learns patterns from normal data but **never learns how adversarial noise behaves**.

Because of this, small gradient-aligned perturbations can push the input across the model's decision boundary.

## Vulnerability

- **No exposure to attacks**: Model has never seen adversarial examples during training
- **Decision boundaries**: The model's learned boundaries are not robust to gradient-based perturbations
- **Result**: High clean accuracy but extremely vulnerable to FGSM and similar attacks

---

# Adversarial Training

Adversarial training modifies the training process so the model **learns from adversarial examples during training**.

Instead of only training on clean images:

```
(x, y)
```

we generate adversarial images:

```
(x_adv, y)
```

and train on them as well.

## Training Pipeline

```
image
  ↓
compute gradient wrt image
  ↓
generate adversarial image
  ↓
model prediction
  ↓
compute loss
  ↓
update weights
```

This forces the model to learn **more robust decision boundaries**.

## How It Improves Robustness

During adversarial training:
1. The model sees adversarial examples with their true labels
2. It learns to classify adversarial images correctly
3. This teaches the model to discover robust features that aren't fooled by gradient-based perturbations
4. The model develops decision boundaries that are harder for attacks to exploit

---

# Key Difference

| Normal Training                          | Adversarial Training                    |
| ---------------------------------------- | --------------------------------------- |
| trains only on clean images              | trains on adversarial images            |
| model never sees attacks during training | model learns to handle attacks          |
| high accuracy but fragile                | slightly lower accuracy but more robust |

Adversarial training is currently one of the **most effective practical defenses** against adversarial attacks.

---

## Results

We compare the robustness of two models:

* **Standard CNN** trained on clean images
* **Adversarially trained CNN** trained with FGSM examples

### Normal Model (Clean Training)

| Metric               | Accuracy           |
| -------------------- | ------------------ |
| Clean Test Accuracy  | **98.98%**         |
| FGSM Attack Accuracy | **28.91%**         |

**Observation**: The normal model performs very well on clean data (**98.98%**) but is extremely vulnerable to adversarial attacks (**28.91%**). This demonstrates the stark **robustness gap** — high accuracy on clean data does not guarantee robustness to adversarial examples. The model's accuracy drops by **70.07 percentage points** under attack, showing severe vulnerability.

---

### Adversarially Trained Model

| Metric               | Accuracy   |
| -------------------- | ---------- |
| Clean Test Accuracy  | **96.31%** |
| FGSM Attack Accuracy | **92.89%** |

**Observation**: Adversarial training slightly reduces clean accuracy but **significantly improves robustness** against FGSM attacks. The model maintains ~93% accuracy even under attack.

---

### Comparison & Key Insights

| Property             | Normal Model | Adversarial Training |
| -------------------- | ------------ | -------------------- |
| Clean Accuracy       | 98.98%       | 96.31%               |
| Attack Accuracy      | Low (~10%)   | 92.89%               |
| Robustness Gap       | Huge         | Small                |
| Trade-off            | High accuracy, fragile | Balanced robustness |

**Key Observation**: Adversarial training forces the model to learn **decision boundaries that are harder for gradient-based attacks to exploit**. The small accuracy drop on clean data (2.67%) is a worthwhile trade-off for gaining significant robustness (83% improvement in adversarial accuracy).

---

# Project Structure

```
adversarial-ml/
│
├── notebooks/
│   ├── fgsm_attack.ipynb                    # FGSM attack demonstration
│   └── fgsm_adversarial_training.ipynb      # Adversarial training implementation
│
├── data/                                     # Dataset (ignored by git)
│   └── MNIST/
│
├── .gitignore
├── pyproject.toml                            # Project dependencies
└── README.md
```

---

# Setup & Installation

This project uses `uv` for fast, reliable Python environment management.

## Prerequisites

- Python 3.8 or higher
- pip or conda (for alternative installation)

## Installation Steps

**1. Create virtual environment:**

```bash
uv venv
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

**2. Install dependencies:**

```bash
uv pip install torch torchvision matplotlib jupyter
```

**3. Verify installation:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Running the Notebooks

**Start Jupyter:**

```bash
jupyter notebook
```

Then open either:
- `notebooks/fgsm_attack.ipynb` - Learn how FGSM attacks work
- `notebooks/fgsm_adversarial_training.ipynb` - Learn adversarial training defense

---

# What This Project Demonstrates

The notebooks illustrate several core ideas in adversarial machine learning:

* **Neural network gradients**: Understanding how to compute and use gradients for attacks
* **Adversarial example generation**: Creating perturbations that fool neural networks
* **FGSM attack implementation**: Implementing a simple yet effective white-box attack
* **Adversarial training as a defense**: Building robust models through exposure to attacks
* **Robustness vs accuracy trade-offs**: Understanding the cost of improving robustness

---

# Possible Extensions

Future experiments could include:

* **Stronger attacks**: PGD (Projected Gradient Descent), C&W (Carlini & Wagner)
* **Robustness evaluation**: Testing across different epsilon values
* **Advanced training**: Adversarial training with PGD or other stronger attacks
* **Multi-dataset testing**: Evaluating on CIFAR-10, ImageNet
* **Certified robustness**: Computing verifiable robustness bounds
* **Transfer attacks**: Testing if adversarial examples transfer between models

---

# Key Takeaways

1. **Vulnerability is subtle**: Small, imperceptible perturbations can fool state-of-the-art models
2. **Gradients matter**: Attackers can exploit the very mechanism that makes neural networks learnable
3. **Defense requires trade-offs**: Robustness comes at the cost of some clean accuracy
4. **Adversarial training works**: It's simple, practical, and significantly improves robustness
5. **This is an arms race**: As defenses improve, attacks become stronger; ongoing research is essential

---

# References

* Goodfellow, I., Shlens, J., & Szegedy, C. (2015). **Explaining and Harnessing Adversarial Examples**. *arXiv preprint arXiv:1412.6572*
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644) - Carlini & Wagner Attack
* [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1809.02889) - PGD Attack

---

# License

This project is provided as educational material.

---
