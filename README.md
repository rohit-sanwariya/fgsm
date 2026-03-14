# Adversarial ML — FGSM Attack on MNIST

This project reproduces a **basic adversarial attack (FGSM)** against a convolutional neural network trained on the MNIST handwritten digit dataset.

The goal is to understand how small, carefully crafted perturbations to input images can cause a neural network to misclassify them.

This is a minimal educational project focused on the **fundamentals of adversarial machine learning**.

---

## What This Project Demonstrates

1. Training a simple CNN classifier on MNIST
2. Evaluating model accuracy
3. Computing gradients with respect to the input image
4. Generating adversarial examples using the FGSM attack
5. Visualizing original vs adversarial images

The experiment shows how a model that performs well on normal inputs can fail when inputs are slightly modified in a specific direction.

---

## Dataset

This project uses the MNIST.

MNIST contains:

* 60,000 training images
* 10,000 test images
* grayscale images of size **28×28**
* labels **0–9**

The dataset is automatically downloaded through PyTorch.

---

## Attack Implemented

This repository implements **Fast Gradient Sign Method (FGSM)** introduced by Ian Goodfellow.

FGSM creates adversarial images using the gradient of the loss with respect to the input image.

Adversarial example formula:

```
x_adv = x + ε * sign(∇x J(θ, x, y))
```

Where:

* `x` is the original image
* `ε` controls attack strength
* `∇x` is the gradient with respect to the input
* `J` is the loss function

Even very small values of `ε` can cause the model to produce incorrect predictions.

---

## Project Structure

```
adversarial-ml/
│
├── notebooks/
│   └── fgsm_attack.ipynb
│
├── data/          # downloaded datasets (ignored by git)
├── models/        # trained models (ignored by git)
│
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## Setup

This project uses `uv` for environment management.

Create the environment:

```
uv venv
source .venv/bin/activate
```

Install dependencies:

```
uv pip install torch torchvision matplotlib jupyter
```

Start Jupyter:

```
jupyter notebook
```

Open:

```
notebooks/fgsm_attack.ipynb
```

---

## Training the Model

A simple convolutional neural network is trained using PyTorch.

Architecture:

```
Conv2D → ReLU
Conv2D → ReLU
Flatten
Linear → ReLU
Linear → 10 classes
```

After a few epochs the model typically reaches **~97–99% accuracy** on the test set.

---

## Generating Adversarial Examples

Steps:

1. Enable gradients on the input image
2. Compute model loss
3. Backpropagate gradients to the input
4. Modify the image using the gradient sign
5. Re-run the model on the modified image

The adversarial image often appears identical to humans but causes the model to misclassify the digit.

---

## Visualization

The notebook displays:

* Original image
* Adversarial image
* Perturbation (difference)

This helps illustrate how minimal changes to pixels can significantly affect neural network predictions.

---

## Learning Goals

This project helps build intuition for:

* neural network gradients
* adversarial examples
* model robustness
* limitations of deep learning models

It serves as a starting point for exploring more advanced adversarial attacks and defenses.

---

## Possible Extensions

Some natural next steps include:

* Measuring accuracy vs attack strength (epsilon)
* Implementing stronger attacks (PGD)
* Training adversarially robust models
* Testing attacks on other datasets

---

## References

* Explaining and Harnessing Adversarial Examples
* PyTorch
* MNIST

---
