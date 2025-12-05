import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import argparse
import s3fs

# Configuration S3 (Onyxia / MinIO)
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})


def load_data(input_path):
    """Charge le CSV depuis S3 ou Local."""
    print(f"📂 Lecture des données : {input_path}")

    if input_path.startswith("s3://"):
        try:
            s3_path = input_path.replace("s3://", "")
            with fs.open(s3_path, "rb") as f:
                return pd.read_csv(f)
        except Exception as e:
            print(f"❌ Erreur lecture S3 : {e}")
            return None
    else:
        if not os.path.exists(input_path):
            print(f"❌ Fichier introuvable : {input_path}")
            return None
        return pd.read_csv(input_path)


def save_plot(fig, output_dir, filename):
    """Sauvegarde le graphique en local puis upload sur S3 si nécessaire."""

    # 1. Sauvegarde temporaire locale
    temp_path = os.path.join("/tmp", filename)
    fig.savefig(temp_path, dpi=300, bbox_inches="tight")
    print(f"🖼  Image générée : {temp_path}")

    # 2. Gestion de la destination finale
    if output_dir.startswith("s3://"):
        s3_path = os.path.join(output_dir.replace("s3://", ""), filename)
        print(f"⬆️  Upload S3 vers : {s3_path}")
        try:
            fs.put(temp_path, s3_path)
            print("✅ Upload réussi.")
        except Exception as e:
            print(f"⚠️ Erreur upload : {e}")
    else:
        # Cas Local
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, filename)
        os.rename(temp_path, final_path)
        print(f"✅ Sauvegardé dans : {final_path}")


def plot_benchmark(df, output_dir, prefix_name):
    """Génère les graphiques avec un préfixe personnalisé."""

    # Configuration du style
    sns.set_theme(style="whitegrid")

    if "Epsilon" in df.columns:
        df["Epsilon"] = df["Epsilon"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, float) else x
        )

    # ==========================================
    # GRAPHIQUE 1 : ACCURACY
    # ==========================================
    plt.figure(figsize=(10, 6))
    chart = sns.barplot(
        data=df, x="Attack", y="Accuracy", hue="Model", palette="viridis"
    )

    plt.title("Comparaison de la Robustesse (Accuracy)", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Type d'Attaque", fontsize=12)
    plt.ylim(0, 100)
    plt.legend(title="Modèle", bbox_to_anchor=(1.05, 1), loc="upper left")

    for container in chart.containers:
        chart.bar_label(container, fmt="%.1f%%", padding=3)

    plt.tight_layout()

    # --- MODIFICATION ICI : Utilisation du préfixe ---
    filename_acc = f"{prefix_name}_accuracy.png"
    save_plot(plt, output_dir, filename_acc)
    plt.close()

    # ==========================================
    # GRAPHIQUE 2 : LOSS
    # ==========================================
    if "Loss" in df.columns:
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

        # --- MODIFICATION ICI : Utilisation du préfixe ---
        filename_loss = f"{prefix_name}_loss.png"
        save_plot(plt, output_dir, filename_loss)
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Générateur de graphiques de Benchmark"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Chemin vers le CSV (Local ou s3://...)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Dossier de sortie pour les images (Local ou s3://...)",
    )

    # --- NOUVEL ARGUMENT ---
    parser.add_argument(
        "--plot-name",
        type=str,
        default="benchmark",
        help="Préfixe pour le nom des fichiers (ex: 'mon_test' -> 'mon_test_accuracy.png')",
    )

    args = parser.parse_args()

    # Exécution
    df = load_data(args.input_file)

    if df is not None:
        # On passe le plot_name à la fonction
        plot_benchmark(df, args.output_dir, args.plot_name)
    else:
        print("❌ Impossible de charger les données. Vérifiez le chemin.")


if __name__ == "__main__":
    main()
