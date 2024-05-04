# Exploring-KANs

In this repository, I explore the performance of KANs on standard tasks in Machine Learning since the paper ["KAN: Kolmogorov-Arnold Networks"](https://arxiv.org/abs/2404.19756) focuses on applications in Physics and Mathematics.

Kolmogorov-Arnold Networks (KANs) are promising alternatives of Multi-Layer Perceptrons (MLPs). KANs have strong mathematical foundations just like MLPs: MLPs are based on the universal approximation theorem, while KANs are based on Kolmogorov-Arnold representation theorem. While MLPs have fixed activation functions on nodes ("neurons"), KANs have learnable activation functions on edges ("weights"). KANs have no linear weights at all -- every weight parameter is replaced by a univariate function parametrized as a spline. They show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability.

The original implementation of KAN is available [here](https://github.com/KindXiaoming/pykan).

I use [Blealtan's](https://github.com/Blealtan) efficient and PyTorch compatible implementation for easy use.

This implementation is available [here](https://github.com/Blealtan/efficient-kan)

Throughtout these experiments, a simple Feedforward Neural Network is also implemented alongside each KAN for comparison.

## Binary Classification On Tabular Data

For this task I chose the [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html). It has **569** instances, and **30** features.

### Architecture

