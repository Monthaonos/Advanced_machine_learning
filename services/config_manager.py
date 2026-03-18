import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        raise ImportError(
            "Python < 3.11 requires the 'tomli' package: pip install tomli"
        )


def load_config(config_path: str = "config.toml") -> dict:
    """Load a TOML configuration file and return it as a dictionary."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "rb") as f:
        config = tomllib.load(f)

    return config
