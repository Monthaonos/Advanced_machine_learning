import os
import sys

# Gestion de compatibilité Python < 3.11
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Pour lire le config.toml, installez tomli : 'pip install tomli'"
        )


def load_config(config_path: str = "config.toml") -> dict:
    """
    Charge le fichier de configuration TOML et renvoie un dictionnaire.
    Vérifie que le fichier existe.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Fichier de configuration introuvable : {config_path}"
        )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config
