# Adversarial Robustness on GTSRB: Extensions of the Madry Framework

## 1. Project Overview
This repository implements and extends the adversarial training framework established by **Madry et al.** (*Towards Deep Learning Models Resistant to Adversarial Attacks*). 

The primary objective is to evaluate and enhance the robustness of Deep Neural Networks (DNNs) against adversarial perturbations. While the original framework was tested on MNIST and CIFAR-10, this repository extends the methodology to the **German Traffic Sign Recognition Benchmark (GTSRB)**, simulating real-world safety-critical scenarios for autonomous driving.

## 2. Technical Features & Extensions
Beyond the standard PGD-based robust training, this implementation introduces several advanced modules:

* **Multi-Dataset Support:** Optimized pipelines for both CIFAR-10 and GTSRB classification.
* **Momentum Iterative Method (MiM):** Implementation of momentum-based attacks ($L_\infty$ norm) to improve the transferability and stability of adversarial examples.
* **Probabilistic Training (train_prob):** A stochastic training logic controlled by the $P_{train}$ parameter. It dynamically manages the ratio between clean and adversarial samples in each batch.
* **Adversarial Patch (Stickers):** A specialized optimization-based attack (`patch_attack`) generating localized perturbations to simulate physical-world occlusions.

## 3. Detailed Project Structure
The repository is organized into modular components to ensure clear separation between configuration, data handling, and logic:

* **src/attacks/**: Contains adversarial generators (PGD, MiM, and the optimized Patch Attack).
* **src/models/**: Defines neural architectures including `simple_cnn`, `resnet18`, and `wideresnet`.
* **src/data_loader.py**: Manages data pipelines for CIFAR-10 and GTSRB, handling normalization and worker synchronization.
* **src/trainer.py**: Core engine managing robust training loops and the `train_prob` stochastic logic.
* **checkpoints/**: Default storage path for serialized model weights (`.pth`).
* **main.py**: Unified entry point that parses the TOML configuration and executes the research pipeline.

## 4. Configuration and Usage
The project utilizes a `TOML` configuration file for centralized management of research parameters. This ensures experiments are reproducible and easy to version.

### Unified Configuration (config.toml)
All parameters are grouped into functional blocks:

| Section | Parameter | Description |
| :--- | :--- | :--- |
| **[project]** | `device` / `storage_path` | Computation backend (cuda/cpu) and checkpoint directory. |
| **[data]** | `dataset` / `batch_size` | Choice of data (cifar10/gtsrb) and sampling density. |
| **[model]** | `architecture` / `prefix` | Model selection and naming convention for saved weights. |
| **[training]** | `learning_rate` / `force_retrain` | Optimization hyperparameters and execution flow control. |
| **[adversarial]** | `epsilon` / `train_prob` | Attack budget and the probability $P_{train}$ of adversarial training. |
| **[patch_attack]** | `scale` / `number_of_steps` | Spatial coverage and optimization iterations for sticker attacks. |
| **[evaluation]** | `attacks_to_run` | List of attacks to execute during the validation phase. |

### Execution
To run the project using the unified configuration:

```bash
python main.py --config config.toml
```

### Overriding Parameters (Optional)
The CLI also supports direct overrides for rapid testing:
```bash
python main.py --dataset gtsrb --train_prob 0.5 --architecture resnet18
```

## 5. Mathematical Context
The core optimization problem follows the min-max formulation:
$$\min_{\theta} \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\delta \in \mathcal{S}} L(\theta, x + \delta, y) \right]$$
Where $\mathcal{S}$ represents the perturbation set defined in the `[adversarial]` or `[patch_attack]` sections.

## 6. References
* Madry, A., et al. (2018). *Towards Deep Learning Models Resistant to Adversarial Attacks*. ICLR.
* Dong, Y., et al. (2018). *Boosting Adversarial Attacks with Momentum*. CVPR.
* Stallkamp, J., et al. (2012). *Man vs. Computer: Benchmarking Machine Learning Algorithms for Traffic Sign Recognition*.