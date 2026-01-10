# Interpretable and Adaptive Node Classification on Heterophilic Graphs

[![arXiv](https://img.shields.io/badge/arXiv-2512.22221-b31b1b.svg)](https://arxiv.org/abs/2512.22221)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

> **Paper:** [Interpretable and Adaptive Node Classification on Heterophilic Graphs via Combinatorial Scoring and Hybrid Learning](https://arxiv.org/abs/2512.22221)  
> **Author:** Soroush Vahidi  
> **Status:** arXiv Preprint (December 2025)

---

## 🎯 What is this repository for?

This repository contains the **official PyTorch implementation** of the framework proposed in the paper above. It is designed to:

1.  **Reproduce the results** reported in the paper on heterophilic benchmarks (e.g., Chameleon, Squirrel, Actor).
2.  **Provide a lightweight, interpretable alternative** to deep Graph Neural Networks (GNNs). Unlike black-box models, this code allows you to classify nodes using transparent logic (e.g., "predicted Class A because 60% of neighbors are Class A").
3.  **Demonstrate the Hybrid Strategy**, which bridges the gap between simple combinatorial heuristics and neural networks by only using the latter when validation data proves it beneficial.

If you are researching **graph heterophily**, **interpretable machine learning**, or need a **fast, CPU-friendly baseline** for node classification, this codebase is for you.

---

## 📖 Overview

Graph Neural Networks (GNNs) often struggle when connected nodes belong to different classes (**heterophily**). Deep models try to fix this with complex architectures, but often at the cost of interpretability and speed.

We propose a different approach: **Combinatorial Scoring**. Instead of deep message passing, we assign labels using a confidence-ordered greedy procedure driven by an additive scoring function:

$$S(v, c) = \alpha \cdot \Psi_{\text{neigh}}(v, c) + \beta \cdot \Psi_{\text{feat}}(v, c) + \gamma \cdot \Psi_{\text{prior}}(c) + \delta \cdot \Psi_{\text{compat}}(v, c)$$

We also provide a **Validation-Gated Hybrid Strategy** that injects these scores into a lightweight neural model, but *only* if it improves validation accuracy—preserving interpretability whenever possible.

## ✨ Key Features

* **🧩 Fully Interpretable:** Inspect exactly why a node got a label (Neighbors vs. Features vs. Priors).
* **⚖️ Adaptive:** Smoothly interpolates between homophilic (neighbor-reliant) and heterophilic (compatibility-reliant) behaviors via simple hyperparameters.
* **⚡ High Efficiency:** The core combinatorial algorithm runs efficiently on CPU.
* **🛡️ Leakage-Free:** Strict separation of training/validation/test signals.

---

## ⚙️ Installation

```bash
# Clone the repository
git clone [https://github.com/vahidi-research/adaptive-heterophilic-graphs.git](https://github.com/vahidi-research/adaptive-heterophilic-graphs.git)
cd adaptive-heterophilic-graphs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
