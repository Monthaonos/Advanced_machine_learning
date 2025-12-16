import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import s3fs

# Configuration S3 (Onyxia / MinIO)
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})


def load_single_csv(input_path):
    """Charge un fichier CSV unique depuis S3 ou le système de fichiers local."""
    print(f"[INFO] Loading data: {input_path}")

    if input_path.startswith("s3://"):
        try:
            s3_path = input_path.replace("s3://", "")
            with fs.open(s3_path, "rb") as f:
                return pd.read_csv(f)
        except Exception as e:
            print(f"[ERROR] S3 Load Failed for {input_path}: {e}")
            return None
    else:
        if not os.path.exists(input_path):
            print(f"[ERROR] File not found: {input_path}")
            return None
        return pd.read_csv(input_path)


def load_and_merge_data(input_files):
    """Charge et fusionne plusieurs fichiers CSV en un seul DataFrame."""
    dfs = []
    for path in input_files:
        df = load_single_csv(path)
        if df is not None:
            dfs.append(df)

    if not dfs:
        return None

    # Fusion verticale
    return pd.concat(dfs, ignore_index=True)


def save_plot(fig, output_dir, filename):
    """Sauvegarde la figure matplotlib localement ou sur S3."""
    temp_path = os.path.join("/tmp", filename)
    fig.savefig(temp_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Plot generated: {temp_path}")

    if output_dir.startswith("s3://"):
        s3_path = os.path.join(output_dir.replace("s3://", ""), filename)
        print(f"[INFO] Uploading to S3: {s3_path}")
        try:
            fs.put(temp_path, s3_path)
            print("[SUCCESS] Upload complete.")
        except Exception as e:
            print(f"[ERROR] S3 Upload Failed: {e}")
    else:
        os.makedirs(output_dir, exist_ok=True)
        final_path = os.path.join(output_dir, filename)
        os.rename(temp_path, final_path)
        print(f"[SUCCESS] Saved locally: {final_path}")


def parse_args():
    """Analyse les arguments de la ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Benchmark Visualization Tool"
    )

    parser.add_argument(
        "--input-files",
        nargs="+",
        required=True,
        help="List of CSV file paths (Local or S3).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory to save the plots.",
    )

    parser.add_argument(
        "--plot-name",
        type=str,
        default="benchmark_comparison",
        help="Prefix for the output filenames.",
    )

    return parser.parse_args()


def run_plotting(args):
    """Logique principale pour la génération des graphiques."""

    # 1. Chargement des données
    df = load_and_merge_data(args.input_files)

    if df is None or df.empty:
        print("[WARNING] No data available to plot.")
        return

    # Configuration du style
    sns.set_theme(style="whitegrid")

    # Formatage de la colonne Epsilon pour l'affichage (évite les floats trop longs)
    if "Epsilon" in df.columns:
        df["Epsilon"] = df["Epsilon"].apply(
            lambda x: f"{x:.2f}" if isinstance(x, float) else x
        )

    # ==========================================
    # Plot 1 : Accuracy Comparison
    # ==========================================
    plt.figure(figsize=(12, 6))
    chart = sns.barplot(
        data=df, x="Attack", y="Accuracy", hue="Model", palette="viridis"
    )

    plt.title("Model Robustness Comparison (Accuracy)", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Attack Type", fontsize=12)
    plt.ylim(0, 105)  # Marge pour les labels
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

    # Ajout des valeurs sur les barres
    for container in chart.containers:
        chart.bar_label(container, fmt="%.1f%%", padding=3, fontsize=9)

    plt.tight_layout()
    save_plot(plt, args.output_dir, f"{args.plot_name}_accuracy.png")
    plt.close()

    # ==========================================
    # Plot 2 : Loss Comparison (Optionnel)
    # ==========================================
    if "Loss" in df.columns:
        plt.figure(figsize=(12, 6))
        chart_loss = sns.barplot(
            data=df, x="Attack", y="Loss", hue="Model", palette="magma"
        )

        plt.title(
            "Model Stability Comparison (Cross-Entropy Loss)", fontsize=16
        )
        plt.ylabel("Loss", fontsize=12)
        plt.xlabel("Attack Type", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0)

        for container in chart_loss.containers:
            chart_loss.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

        plt.tight_layout()
        save_plot(plt, args.output_dir, f"{args.plot_name}_loss.png")
        plt.close()


def main():
    args = parse_args()
    run_plotting(args)


if __name__ == "__main__":
    main()
