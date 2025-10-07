Purely testing purposes using C-KAN's 
# Convolutional KANs (C-KAN)

## Introduction
This repository contains an implementation of Convolutional Kolmogorov-Arnold Networks (C-KANs), which extend the concept of Kolmogorov-Arnold Networks to convolutional layers. This approach replaces traditional linear convolutions with learnable non-linear activations for each pixel, aiming for improved parameter efficiency and expressive power compared to standard Convolutional Neural Networks (CNNs).

## Original Authors & Source
The core implementation of the Convolutional KANs in this repository is almost entirely derived from the work of Antonio Tepsich and his collaborators.

**Authors:** Alexander Bodner, Antonio Tepsich, Jack Spolski, and Santiago Pourteau.
**GitHub Repository:** [AntonioTepsich/Convolutional-KANs](https://github.com/AntonioTepsich/Convolutional-KANs)
**Academic Paper:** "Convolutional Kolmogorov-Arnold Networks" (submitted June 19, 2024)

Please refer to their original work for a comprehensive understanding of the mathematical explanations and empirical evaluations.

## Files in this Repository
*   `convolution.py`: Low-level convolutional operations for KANs.
*   `KANConv.py`: KAN-based convolutional layer implementations.
*   `KANLinear.py`: Core Kolmogorov-Arnold Network (KAN) linear layer.
*   `test.py`: Defines the `KANC_MLP` model architecture (MNIST-sized).
*   `MNISTexecution.py`: simple trainer for MNIST using `KANC_MLP`.
*   `CIFARexecution.py`: simple trainer for CIFAR-10/100 using C-KAN conv layers.

## Usage

- MNIST
  - Ensure PyTorch and `torchvision` are installed.
  - Run:
    ```bash
    python MNISTexecution.py
    ```

- CIFAR-10 / CIFAR-100
  - The script is self‑contained and uses only files in this repo (it does not depend on `torch-conv-kan-main`).
  - Open `CIFARexecution.py` and set `dataset_name = "cifar10"` or `"cifar100"`.
  - Run:
    ```bash
    python CIFARexecution.py
    ```
  - The script auto-selects device: `cuda` (NVIDIA), `mps` (Apple Silicon), or `cpu`.

## License
This is distributed under the MIT License. See the `LICENSE` file for more details.
