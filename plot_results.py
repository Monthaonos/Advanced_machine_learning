import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


def plot_benchmark(csv_file, output_dir="results_large_random"):
    """
    Lit le CSV de résultats et génère deux graphiques :
    1. Accuracy par Attaque et par Modèle
    2. Loss par Attaque et par Modèle
    """
    if not os.path.exists(csv_file):
        print(f"❌ Fichier introuvable : {csv_file}")
        return

    # Création du dossier de sortie pour les images
    os.makedirs(output_dir, exist_ok=True)

    # Chargement des données
    df = pd.read_csv(csv_file)

    # Nettoyage pour l'affichage (optionnel)
    # On arrondit l'epsilon pour que ce soit lisible dans les graphes
    df["Epsilon"] = df["Epsilon"].apply(lambda x: f"{x:.2f}")

    # Configuration du style
    sns.set_theme(style="whitegrid")

    # ==========================================
    # GRAPHIQUE 1 : ACCURACY
    # ==========================================
    plt.figure(figsize=(10, 6))

    # Le barplot magique : x=Attaque, y=Accuracy, hue=Modèle (couleur)
    chart = sns.barplot(
        data=df, x="Attack", y="Accuracy", hue="Model", palette="viridis"
    )

    plt.title("Comparaison de la Robustesse (Accuracy)", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Type d'Attaque", fontsize=12)
    plt.ylim(0, 100)  # On fixe l'échelle de 0 à 100%
    plt.legend(title="Modèle", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Ajout des valeurs sur les barres
    for container in chart.containers:
        chart.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()
    save_path_acc = os.path.join(output_dir, "benchmark_accuracy.png")
    plt.savefig(save_path_acc, dpi=300)
    print(f"✅ Graphique Accuracy sauvegardé : {save_path_acc}")
    plt.close()

    # ==========================================
    # GRAPHIQUE 2 : LOSS (Optionnel mais intéressant)
    # ==========================================
    plt.figure(figsize=(10, 6))

    chart_loss = sns.barplot(
        data=df, x="Attack", y="Loss", hue="Model", palette="magma"
    )

    plt.title("Comparaison de la Stabilité (Loss)", fontsize=16)
    plt.ylabel("Cross Entropy Loss", fontsize=12)
    plt.xlabel("Type d'Attaque", fontsize=12)
    plt.legend(title="Modèle", bbox_to_anchor=(1.05, 1), loc="upper left")

    for container in chart_loss.containers:
        chart_loss.bar_label(container, fmt="%.2f", padding=3)

    plt.tight_layout()
    save_path_loss = os.path.join(output_dir, "benchmark_loss.png")
    plt.savefig(save_path_loss, dpi=300)
    print(f"✅ Graphique Loss sauvegardé : {save_path_loss}")
    plt.close()


if __name__ == "__main__":
    # Remplace par le chemin vers ton fichier CSV généré par evaluate.py
    # Si tu es sur Mac et que tu as utilisé le chemin local :
    plot_benchmark("checkpoints/cifar10_large_rd.csv")
