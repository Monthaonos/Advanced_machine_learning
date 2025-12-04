import os
import s3fs
import torch
import pandas as pd
from pathlib import Path
from typing import Optional
from torch import nn

# Configuration for SSP Cloud
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"
fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})  # S3 connection

# A. basic fonctions


def save_model(model: nn.Module, path: str) -> None:
    """Sauvegarde le state_dict d'un modèle PyTorch vers un fichier."""
    path_obj = Path(path)
    if path_obj.suffix == "":
        path_obj = path_obj.with_suffix(".pth")
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path_obj)


def load_model(model: nn.Module, path: str, device: Optional[str] = None) -> nn.Module:
    """Charge les paramètres d'un modèle à partir du disque."""
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")

    map_location = None
    if device is not None:
        map_location = device

    state_dict = torch.load(path_obj, map_location=map_location)
    model.load_state_dict(state_dict)
    return model


# B. Function to create robust paths


def get_local_path(storage_path: str, filename: str) -> str:
    """
    Résout le chemin local absolu pour le fichier (modèle ou csv).
    Garantit que les chemins relatifs sont ancrés à la racine d'exécution (CWD).
    """
    # Extrait le chemin local, même si l'input est S3
    if storage_path.startswith("s3://"):
        # Ex: "s3://bucket/prefix/path" -> "prefix/path"
        parts = storage_path.replace("s3://", "").split("/", 1)
        # Si le path S3 est juste le bucket, on utilise un dossier temporaire
        local_dir = parts[1] if len(parts) > 1 else ".tmp_s3_checkpoints"
    else:
        local_dir = storage_path

    # Rend le chemin absolu et crée les dossiers
    abs_dir = os.path.abspath(local_dir)
    os.makedirs(abs_dir, exist_ok=True)
    return os.path.join(abs_dir, filename)


# C. High-level functions


def manage_checkpoint(model, storage_path, model_filename, device="cpu"):
    """Tries to load the model from storage_path (S3 or Local)."""
    is_s3 = storage_path.startswith("s3://")

    # S3
    if is_s3:
        full_path = f"{storage_path.rstrip('/')}/{model_filename}"
        s3_obj_path = full_path.replace("s3://", "")

        print(f"\n🔎 Recherche S3 : {full_path}")

        try:
            if fs.exists(s3_obj_path):
                temp_filename = os.path.basename(model_filename)
                temp_path = f"/tmp/{temp_filename}"

                print(f"⬇️  Modèle trouvé sur S3. Téléchargement vers {temp_path}...")
                fs.get(s3_obj_path, temp_path)

                load_model(model, temp_path, device=device)
                os.remove(temp_path)
                print("✅ Modèle téléchargé et chargé avec succès.")
                return True
            else:
                print(f"❌ Modèle absent du S3 : {s3_obj_path}")
        except Exception as e:
            print(f"⚠️ Erreur S3 (Connexion/Téléchargement) : {e}")
            return False

    # Local
    else:
        full_path = get_local_path(storage_path, model_filename)
        print(f"\n🔎 Recherche Local : {full_path}")
        try:
            if os.path.exists(full_path):
                load_model(model, full_path, device=device)
                print("✅ Modèle trouvé et chargé localement.")
                return True
            else:
                print("❌ Modèle absent localement.")
        except Exception as e:
            print(f"⚠️ Erreur chargement local : {e}")
            return False

    return False


def save_checkpoint(model, storage_path, model_filename):
    """Saves locally the model and eventually uploads it on S3."""
    is_s3 = storage_path.startswith("s3://")

    # Local save
    local_path = get_local_path(storage_path, model_filename)
    save_model(model, local_path)
    print(f"💾 Modèle sauvegardé localement : {local_path}")

    # S3 upload
    if is_s3:
        full_path = f"{storage_path.rstrip('/')}/{model_filename}"
        s3_obj_path = full_path.replace("s3://", "")

        print(f"⬆️  Upload vers S3 : {full_path}")
        try:
            fs.put(local_path, s3_obj_path)
        except Exception as e:
            print(f"⚠️ Erreur d'Upload S3 : {e}")


def save_results(df: pd.DataFrame, storage_path: str, filename: str):
    """
    Saves a Pandas DataFrame in CSV format.
    Deals with local save/S3 upload.
    """
    is_s3 = storage_path.startswith("s3://")

    # Local save
    local_path = get_local_path(storage_path, filename)

    try:
        df.to_csv(local_path, index=True)
        print(f"📊 Résultats sauvegardés localement : {local_path}")
    except Exception as e:
        print(f"❌ Erreur sauvegarde CSV locale : {e}")
        return

    # S3 upload
    if is_s3:
        s3_path = f"{storage_path.rstrip('/')}/{filename}"
        s3_obj_path = s3_path.replace("s3://", "")

        print(f"⬆️  Upload des résultats vers S3 : {s3_path}")
        try:
            fs.put(local_path, s3_obj_path)
        except Exception as e:
            print(f"⚠️ Erreur Upload CSV S3 : {e}")
