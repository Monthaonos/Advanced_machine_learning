import os
import s3fs
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
from torch import nn

# --- Configuration for SSP Cloud / MinIO ---
# Ideally, these should be loaded from environment variables for security/portability.
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"

# Initialize S3 FileSystem
# Note: This requires appropriate environment variables (AWS_ACCESS_KEY_ID, etc.) to be set.
try:
    fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": S3_ENDPOINT})
except Exception as e:
    print(f"⚠️ Warning: Could not initialize S3 connection: {e}")
    fs = None

# ==========================================
# A. Basic File Operations
# ==========================================


def save_model(model: nn.Module, path: str) -> None:
    """
    Saves the model's state dictionary to a file.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (str): Local destination path.
    """
    path_obj = Path(path)
    # Ensure extension is .pth
    if path_obj.suffix == "":
        path_obj = path_obj.with_suffix(".pth")

    # Create parent directories if they don't exist
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), path_obj)


def load_model(
    model: nn.Module,
    path: str,
    device: Optional[Union[str, torch.device]] = None,
) -> nn.Module:
    """
    Loads model weights from a file.

    Args:
        model (nn.Module): The model architecture instance.
        path (str): Local path to the weights file.
        device (str | torch.device, optional): Device to map the location to (cpu/cuda).

    Returns:
        nn.Module: The model with loaded weights.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")

    # Handle device mapping (crucial when loading a GPU model on CPU)
    state_dict = torch.load(path_obj, map_location=device)
    model.load_state_dict(state_dict)
    return model


# ==========================================
# B. Path Management
# ==========================================


def get_local_path(storage_path: str, filename: str) -> str:
    """
    Resolves the absolute local path for a file.

    If an S3 path is provided, it mirrors the S3 structure locally to avoid
    overwriting files from different buckets/prefixes.

    Args:
        storage_path (str): The base path (local or 's3://...').
        filename (str): The name of the file.

    Returns:
        str: The absolute local path to the file.
    """
    # Extract local directory structure
    if storage_path.startswith("s3://"):
        # Example: "s3://bucket/prefix/path" -> "prefix/path"
        parts = storage_path.replace("s3://", "").split("/", 1)
        # If path is just the bucket, use a temp folder, else use the subfolder structure
        local_dir = parts[1] if len(parts) > 1 else ".tmp_s3_storage"
    else:
        local_dir = storage_path

    # Ensure absolute path and create directories
    abs_dir = os.path.abspath(local_dir)
    os.makedirs(abs_dir, exist_ok=True)

    return os.path.join(abs_dir, filename)


# ==========================================
# C. High-Level Management (S3 + Local)
# ==========================================


def manage_checkpoint(
    model: nn.Module,
    storage_path: str,
    model_filename: str,
    device: Optional[Union[str, torch.device]] = "cpu",
) -> bool:
    """
    Attempts to load a model checkpoint from storage (S3 or Local).

    If S3 is used, it downloads the file to a temporary location first.

    Args:
        model (nn.Module): The model to populate.
        storage_path (str): The directory path (local or s3://).
        model_filename (str): The filename of the checkpoint.
        device (str): Device to load the model onto.

    Returns:
        bool: True if loading was successful, False otherwise.
    """
    is_s3 = storage_path.startswith("s3://")

    # --- S3 Handling ---
    if is_s3:
        if fs is None:
            print("❌ S3 connection not available.")
            return False

        full_path = f"{storage_path.rstrip('/')}/{model_filename}"
        s3_obj_path = full_path.replace("s3://", "")

        print(f"\n🔎 Searching S3: {full_path}")

        try:
            if fs.exists(s3_obj_path):
                temp_filename = os.path.basename(model_filename)
                temp_path = f"/tmp/{temp_filename}"

                print(
                    f"⬇️  Checkpoint found on S3. Downloading to {temp_path}..."
                )
                fs.get(s3_obj_path, temp_path)

                load_model(model, temp_path, device=device)

                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

                print("✅ Model successfully downloaded and loaded.")
                return True
            else:
                print(f"❌ Checkpoint not found on S3: {s3_obj_path}")
        except Exception as e:
            print(f"⚠️ S3 Error (Connection/Download): {e}")
            return False

    # --- Local Handling ---
    else:
        full_path = get_local_path(storage_path, model_filename)
        print(f"\n🔎 Searching Local: {full_path}")
        try:
            if os.path.exists(full_path):
                load_model(model, full_path, device=device)
                print("✅ Model found and loaded locally.")
                return True
            else:
                print("❌ Checkpoint not found locally.")
        except Exception as e:
            print(f"⚠️ Local Loading Error: {e}")
            return False

    return False


def save_checkpoint(
    model: nn.Module, storage_path: str, model_filename: str
) -> None:
    """
    Saves the model locally and optionally uploads it to S3.

    Args:
        model (nn.Module): The model to save.
        storage_path (str): Destination directory (local or s3://).
        model_filename (str): Filename for the checkpoint.
    """
    is_s3 = storage_path.startswith("s3://")

    # 1. Always save locally first (as cache or source for upload)
    local_path = get_local_path(storage_path, model_filename)
    save_model(model, local_path)
    print(f"💾 Model saved locally: {local_path}")

    # 2. Upload to S3 if requested
    if is_s3:
        if fs is None:
            print("⚠️ Skipping S3 upload (No connection).")
            return

        full_path = f"{storage_path.rstrip('/')}/{model_filename}"
        s3_obj_path = full_path.replace("s3://", "")

        print(f"⬆️  Uploading to S3: {full_path}")
        try:
            fs.put(local_path, s3_obj_path)
        except Exception as e:
            print(f"⚠️ S3 Upload Error: {e}")


def save_results(
    df: pd.DataFrame, storage_path: str, filename: str, index: bool = False
) -> None:
    """
    Saves a Pandas DataFrame to CSV (Local + Optional S3).

    Args:
        df (pd.DataFrame): The data to save.
        storage_path (str): Destination directory.
        filename (str): Name of the CSV file.
        index (bool): Whether to include the DataFrame index in the CSV.
    """
    is_s3 = storage_path.startswith("s3://")
    local_path = get_local_path(storage_path, filename)

    try:
        df.to_csv(local_path, index=index)
        print(f"📊 Results saved locally: {local_path}")
    except Exception as e:
        print(f"❌ Error saving local CSV: {e}")
        return

    if is_s3:
        if fs is None:
            return

        s3_path = f"{storage_path.rstrip('/')}/{filename}"
        s3_obj_path = s3_path.replace("s3://", "")

        print(f"⬆️  Uploading results to S3: {s3_path}")
        try:
            fs.put(local_path, s3_obj_path)
        except Exception as e:
            print(f"⚠️ S3 CSV Upload Error: {e}")


def save_training_metrics(
    loss_history: List[float], storage_path: str, filename: str
) -> None:
    """
    Converts loss history list to CSV and saves it.

    Args:
        loss_history (List[float]): List of loss values per epoch.
        storage_path (str): Destination directory.
        filename (str): Name of the CSV file.
    """
    # Create a DataFrame for structured storage
    df = pd.DataFrame(
        {"epoch": range(1, len(loss_history) + 1), "loss": loss_history}
    )

    # Delegate to the generic save_results function
    save_results(df, storage_path, filename, index=False)
