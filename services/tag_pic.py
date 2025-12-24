import torch
import matplotlib.pyplot as plt
import numpy as np
import os


class TagPicService:
    """
    Visual validation service for evaluating adversarial patches.
    Compares Standard vs. Robust models on patched test samples.
    """

    def __init__(
        self, classifier_clean, classifier_robust, config, classes_map
    ):
        self.c_model = classifier_clean
        self.r_model = classifier_robust
        self.config = config
        self.classes_map = classes_map
        self.device = torch.device(config["project"]["device"])

    def load_patch(self, attack_obj, filename: str):
        """Loads a pre-trained patch from the storage path."""
        path = os.path.join(self.config["project"]["storage_path"], filename)
        if os.path.exists(path):
            attack_obj.patch.data = torch.load(path).to(self.device)
            print(f"Loaded optimized patch: {filename}")
        else:
            raise FileNotFoundError(f"No patch file found at {path}")

    def plot_comparison(self, attack, test_loader, n_images=6):
        """
        Generates a qualitative evaluation grid.
        Displays: Original | Patched | Model Predictions.
        """
        self.c_model.eval()
        self.r_model.eval()

        images, labels = next(iter(test_loader))
        images, labels = (
            images[:n_images].to(self.device),
            labels[:n_images].to(self.device),
        )

        # Apply the static universal patch
        patched_images = attack.apply_patch(images)

        with torch.no_grad():
            outputs_c = self.c_model(patched_images)
            outputs_r = self.r_model(patched_images)

            preds_c = torch.argmax(outputs_c, dim=1)
            preds_r = torch.argmax(outputs_r, dim=1)

        plt.figure(figsize=(18, n_images * 4))
        for i in range(n_images):
            img_clean = np.transpose(images[i].cpu().numpy(), (1, 2, 0))
            img_adv = np.transpose(patched_images[i].cpu().numpy(), (1, 2, 0))

            true_name = self.classes_map.get(
                labels[i].item(), f"ID {labels[i].item()}"
            )
            name_c = self.classes_map.get(preds_c[i].item(), "Unknown")
            name_r = self.classes_map.get(preds_r[i].item(), "Unknown")

            # Plotting
            plt.subplot(n_images, 3, i * 3 + 1)
            plt.imshow(img_clean)
            plt.title(f"Original: {true_name}")
            plt.axis("off")
            plt.subplot(n_images, 3, i * 3 + 2)
            plt.imshow(img_adv)
            plt.title("Patched Input")
            plt.axis("off")

            plt.subplot(n_images, 3, i * 3 + 3)
            plt.axis("off")
            color_c = "red" if preds_c[i] != labels[i] else "green"
            plt.text(
                0.1,
                0.6,
                f"CLEAN MODEL: {name_c}",
                color=color_c,
                fontweight="bold",
                fontsize=12,
            )
            plt.text(
                0.1, 0.2, f"ROBUST MODEL: {name_r}", color="blue", fontsize=12
            )

        plt.tight_layout()
        plt.show()
