import os
import torch
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
from torch import nn


# --- Optional S3 Support ---
# S3 integration is only initialized when an S3 path is actually used.
_fs = None
_s3_initialized = False
S3_ENDPOINT = "https://minio.lab.sspcloud.fr"


def _get_s3_fs():
    """Lazily initialize the S3 filesystem connection."""
    global _fs, _s3_initialized
    if not _s3_initialized:
        _s3_initialized = True
        try:
            import s3fs

            _fs = s3fs.S3FileSystem(
                client_kwargs={"endpoint_url": S3_ENDPOINT}
            )
        except Exception:
            _fs = None
    return _fs


# ==========================================
# A. Basic File Operations
# ==========================================


def save_model(model: nn.Module, path: str) -> None:
    """
    Saves the model's state dictionary to a file.

    Args:
        model: The PyTorch model to save.
        path: Local destination path.
    """
    path_obj = Path(path)
    if path_obj.suffix == "":
        path_obj = path_obj.with_suffix(".pth")

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
        model: The model architecture instance.
        path: Local path to the weights file.
        device: Device to map the location to (cpu/cuda).

    Returns:
        The model with loaded weights.
    """
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Checkpoint '{path}' does not exist.")

    state_dict = torch.load(path_obj, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model


# ==========================================
# B. Path Management
# ==========================================


def get_local_path(storage_path: str, filename: str) -> str:
    """
    Resolves the absolute local path for a file.

    If an S3 path is provided, it mirrors the S3 structure locally.

    Args:
        storage_path: The base path (local or 's3://...').
        filename: The name of the file.

    Returns:
        The absolute local path to the file.
    """
    if storage_path.startswith("s3://"):
        parts = storage_path.replace("s3://", "").split("/", 1)
        local_dir = parts[1] if len(parts) > 1 else ".tmp_s3_storage"
    else:
        local_dir = storage_path

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
    Attempts to load a model checkpoint from storage (S3 or local).

    Args:
        model: The model to populate.
        storage_path: The directory path (local or s3://).
        model_filename: The filename of the checkpoint.
        device: Device to load the model onto.

    Returns:
        True if loading was successful, False otherwise.
    """
    is_s3 = storage_path.startswith("s3://")

    if is_s3:
        fs = _get_s3_fs()
        if fs is None:
            print("[!] S3 connection not available.")
            return False

        full_path = f"{storage_path.rstrip('/')}/{model_filename}"
        s3_obj_path = full_path.replace("s3://", "")

        print(f"\n[*] Searching S3: {full_path}")

        try:
            if fs.exists(s3_obj_path):
                temp_path = f"/tmp/{os.path.basename(model_filename)}"

                print(f"[+] Checkpoint found on S3. Downloading to {temp_path}...")
                fs.get(s3_obj_path, temp_path)
                load_model(model, temp_path, device=device)

                if os.path.exists(temp_path):
                    os.remove(temp_path)

                print("[+] Model successfully downloaded and loaded.")
                return True
            else:
                print(f"[-] Checkpoint not found on S3: {s3_obj_path}")
        except Exception as e:
            print(f"[!] S3 Error: {e}")
            return False

    else:
        full_path = get_local_path(storage_path, model_filename)
        print(f"\n[*] Searching local: {full_path}")
        try:
            if os.path.exists(full_path):
                load_model(model, full_path, device=device)
                print("[+] Model found and loaded locally.")
                return True
            else:
                print("[-] Checkpoint not found locally.")
        except Exception as e:
            print(f"[!] Local loading error: {e}")
            return False

    return False


def save_checkpoint(
    model: nn.Module, storage_path: str, model_filename: str
) -> None:
    """
    Saves the model locally and optionally uploads it to S3.

    Args:
        model: The model to save.
        storage_path: Destination directory (local or s3://).
        model_filename: Filename for the checkpoint.
    """
    is_s3 = storage_path.startswith("s3://")

    local_path = get_local_path(storage_path, model_filename)
    save_model(model, local_path)
    print(f"[+] Model saved locally: {local_path}")

    if is_s3:
        fs = _get_s3_fs()
        if fs is None:
            print("[!] Skipping S3 upload (no connection).")
            return

        full_path = f"{storage_path.rstrip('/')}/{model_filename}"
        s3_obj_path = full_path.replace("s3://", "")

        print(f"[+] Uploading to S3: {full_path}")
        try:
            fs.put(local_path, s3_obj_path)
        except Exception as e:
            print(f"[!] S3 upload error: {e}")


def save_results(
    df: pd.DataFrame, storage_path: str, filename: str, index: bool = False
) -> None:
    """
    Saves a Pandas DataFrame to CSV (local + optional S3).

    Args:
        df: The data to save.
        storage_path: Destination directory.
        filename: Name of the CSV file.
        index: Whether to include the DataFrame index in the CSV.
    """
    is_s3 = storage_path.startswith("s3://")
    local_path = get_local_path(storage_path, filename)

    try:
        df.to_csv(local_path, index=index)
        print(f"[+] Results saved locally: {local_path}")
    except Exception as e:
        print(f"[-] Error saving local CSV: {e}")
        return

    if is_s3:
        fs = _get_s3_fs()
        if fs is None:
            return

        s3_path = f"{storage_path.rstrip('/')}/{filename}"
        s3_obj_path = s3_path.replace("s3://", "")

        print(f"[+] Uploading results to S3: {s3_path}")
        try:
            fs.put(local_path, s3_obj_path)
        except Exception as e:
            print(f"[!] S3 CSV upload error: {e}")


def save_training_metrics(
    loss_history: List[float], storage_path: str, filename: str
) -> None:
    """
    Converts loss history list to CSV and saves it.

    Args:
        loss_history: List of loss values per epoch.
        storage_path: Destination directory.
        filename: Name of the CSV file.
    """
    df = pd.DataFrame(
        {"epoch": range(1, len(loss_history) + 1), "loss": loss_history}
    )
    save_results(df, storage_path, filename, index=False)
