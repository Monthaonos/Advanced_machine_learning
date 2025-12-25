# Adversarial Robustness on GTSRB: Extensions of the Madry Framework

## 1. Project Overview

This repository provides a rigorous implementation and extension of the robust optimization framework established by **Madry et al.** (*Towards Deep Learning Models Resistant to Adversarial Attacks*). 

### Motivation and Objectives
While Deep Neural Networks (DNNs) have achieved state-of-the-art performance in computer vision, their vulnerability to **adversarial perturbations**—carefully crafted, often imperceptible changes to inputs—remains a significant barrier to their deployment in safety-critical environments. The primary objective of this research is to evaluate and enhance the resilience of DNNs against such perturbations using a formalized "Min-Max" optimization approach.



### Extension to Real-World Scenarios
While the original framework was primarily validated on standard datasets like MNIST and CIFAR-10, this repository extends the methodology to the **German Traffic Sign Recognition Benchmark (GTSRB)**. By shifting the focus to traffic sign recognition, we simulate the environmental challenges of **autonomous driving**, where robust classification is a prerequisite for safety under variable lighting, perspective shifts, and physical occlusions.

### Scientific Foundation
The framework is built upon the saddle-point problem of robust optimization:

$$\min_{\theta} \rho(\theta), \quad \text{where} \quad \rho(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(f_\theta(x+\delta), y) \right]$$

1.  **Inner Maximization (Attack)**: Identifies the "worst-case" perturbation $\delta$ within a constraint set $\mathcal{S}$ (e.g., $L_\infty$ or $L_0$ norms) that maximizes the classification loss.
2.  **Outer Minimization (Defense)**: Updates the model parameters $\theta$ to minimize this maximum loss, thereby regularizing the decision boundaries for increased stability.

## 2. Technical Features & Extensions

This framework transcends basic robust training by integrating advanced adversarial modules designed for comprehensive security auditing and model hardening.

### Advanced Adversarial Modules
* **Multi-Dataset Support**: Specialized data pipelines for **CIFAR-10** and **GTSRB**. These include dataset-specific normalization layers and geometric augmentations (RandomAffine, ColorJitter) necessary for traffic sign invariance.
* **Momentum Iterative Method (MiM)**: An implementation of momentum-based $L_\infty$ attacks. By integrating a momentum term into the PGD iterations, this module generates more stable adversarial examples that exhibit higher transferability across different architectures.
* **Stochastic Robust Training ($P_{train}$)**: A probabilistic training logic controlled by the `train_prob` parameter. This allows for a controlled mixture of clean and adversarial samples within the same batch, providing a tunable mechanism to navigate the **Accuracy-Robustness Trade-off**.
* **Universal Adversarial Patch ($L_0$ Attack)**: A specialized optimization-based attack (`patch_attack`) that generates localized, high-intensity perturbations. These "stickers" simulate real-world physical occlusions or vandalism on traffic signage, testing the model's resilience to non-global noise.



### Robust Architectures
The repository supports three distinct levels of model complexity to facilitate benchmark comparisons:
1.  **SimpleCNN**: A modular VGG-style backbone for rapid pipeline verification.
2.  **ResNet-18**: A standard residual network adapted for $32 \times 32$ inputs by removing the initial max-pooling and reducing the kernel size.
3.  **WideResNet-28-10**: A high-capacity architecture utilizing pre-activation blocks and an increased width factor. This model is specifically recommended for adversarial training as its increased parameter density allows for the absorption of adversarial noise without collapsing standard accuracy.


## 3. Detailed Project Structure

The repository follows a strictly modular architecture to ensure a clear separation between experimental configuration, data orchestration, and adversarial logic.

### Directory Hierarchy

```text
├── main.py                     # Unified entry point and research orchestrator
├── config.toml                 # Comprehensive research configuration (Phase 1, 2, 3)
├── requirements.txt            # Environment dependencies
├── services/                   # Core modular services
│   ├── dataloaders/            # Data pipeline orchestration
│   │   ├── factory.py          # Unified loader factory (CIFAR-10 / GTSRB)
│   │   ├── cifar10_loader.py   # Specialized CIFAR-10 augmentations
│   │   └── gtsrb_loader.py     # Specialized GTSRB normalization and resizing
│   ├── models/                 # Architecture definitions
│   │   ├── factory.py          # Unified model factory with integrated NormalizeLayer
│   │   ├── simple_cnn.py       # VGG-style baseline model
│   │   ├── resnet.py           # ResNet-18 optimized for 32x32 inputs
│   │   └── wideresnet.py       # High-capacity WideResNet-28-10
│   ├── attacks.py              # Adversarial generators (FGSM, PGD, MIM)
│   ├── training.py             # Phase 1: Robust training engine (Madry logic)
│   ├── evaluation.py           # Phase 2: L-inf benchmarking orchestrator
│   ├── evaluator.py            # Phase 2: Quantitative metric computation
│   ├── patch_service.py        # Phase 3: L-0 universal patch analysis & viz
│   ├── config_manager.py       # TOML parsing and CLI merging logic
│   └── storage_manager.py      # Checkpoint and results persistence utility
├── checkpoints_<prefix>/       # Serialized model weights (.pth)
└── results_<prefix>/           # Quantitative reports (.csv) and qualitative plots (.png)

## 4. Configuration and Usage

The framework utilizes a centralized `TOML` configuration system to ensure experimental reproducibility and simplified version control. All research parameters are managed through a single entry point, allowing for seamless transitions between different datasets and architectures.

### Unified Configuration (config.toml)
Parameters are organized into functional blocks to maintain a clean separation of concerns:

| Section | Parameter | Description |
| :--- | :--- | :--- |
| **`[project]`** | `device`, `storage_path` | Computation backend (`cuda`, `mps`, `cpu`) and root checkpoint directory. |
| **`[data]`** | `dataset`, `batch_size` | Selection between `cifar10` or `gtsrb` and sampling density per iteration. |
| **`[model]`** | `architecture`, `prefix` | Choice of backbone (`resnet18`, etc.) and naming convention for experiment isolation. |
| **`[training]`** | `epochs`, `learning_rate` | Optimization hyperparameters and control flow for model retraining. |
| **`[adversarial]`** | `epsilon`, `train_prob` | Global $L_\infty$ perturbation budget and the probability $P_{train}$ for robust training. |
| **`[patch_attack]`** | `scale`, `number_of_steps` | Spatial coverage ratio and optimization iterations for localized $L_0$ sticker attacks. |
| **`[evaluation]`** | `attacks_to_run` | Enumeration of adversarial generators (e.g., `["Clean", "PGD", "MIM"]`) for benchmarking. |

### Pipeline Execution
The `main.py` orchestrator supports phased execution. You can run all phases sequentially or target specific research stages using flags:

**Phase 1: Robust Training**
```bash
python main.py --config config.toml --train --prefix final_run
```
**Phase 2: Formal Evaluation
```bash
python main.py --config config.toml --eval --prefix final_run
```
**Phase 3: Patch Vulnerability Analysis
```bash
python main.py --config config.toml --patch --prefix final_run
```

### Dynamic Parameter Overrides

The framework implements a strict configuration hierarchy to ensure maximum flexibility during experimentation: **CLI Arguments > TOML File > Internal Defaults**. 

This allows for rapid hyperparameter sweeps or "smoke tests" without the need to manually modify the source `config.toml` file. Any parameter defined in the TOML can be overridden by passing the corresponding flag in the terminal.

```bash
# Example: Overriding epsilon, dataset, and model for a quick verification run
python main.py --dataset cifar10 --epsilon 0.015 --model simple_cnn --train --eval
```

## 5. Mathematical Foundation

The core of this framework is based on the **saddle-point problem** formulation, which provides a unified view of adversarial training as a robust optimization challenge.

### The Min-Max Formulation
The objective is to find model parameters $\theta$ that minimize the risk under the "worst-case" perturbations:

$$\min_{\theta} \rho(\theta), \quad \text{where} \quad \rho(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(f_\theta(x+\delta), y) \right]$$

This optimization is handled through two distinct nested loops:

1.  **Inner Maximization (The Adversary)**: 
    For a given set of parameters $\theta$, we seek the perturbation $\delta$ within a constraint set $\mathcal{S}$ that maximizes the classification loss $L$.
    * **Global Attacks ($L_\infty$)**: Perturbations are constrained by a budget $\epsilon$, where $||\delta||_\infty \le \epsilon$. This is solved iteratively via **PGD** or **MIM**.
    * **Local Attacks ($L_0$)**: Perturbations are restricted to a specific number of pixels (spatial coverage). This is handled by the **Universal Patch** optimization.

2.  **Outer Minimization (The Defender)**: 
    We update the parameters $\theta$ to minimize the loss generated by the adversary. By training on these "hard" examples, the model learns to rectify its decision boundaries, effectively increasing the margin of stability around each data point.


---

## 6. References

This research extension builds upon several foundational works in the field of Adversarial Machine Learning and Computer Vision:

* **Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018).** *Towards Deep Learning Models Resistant to Adversarial Attacks*. International Conference on Learning Representations (ICLR).
* **Dong, Y., Liao, F., Pang, T., Su, H., Zhu, J., Hu, X., & Li, J. (2018).** *Boosting Adversarial Attacks with Momentum*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
* **Zagoruyko, S., & Komodakis, N. (2016).** *Wide Residual Networks*. British Machine Vision Conference (BMVC).
* **Brown, T. B., Mane, D., Roy, A., Abadi, M., & Gilmer, J. (2017).** *Adversarial Patch*. arXiv preprint arXiv:1712.09665.
* **Stallkamp, J., Schlipsing, M., Salmen, J., & Igel, C. (2012).** *Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition*. Neural Networks.