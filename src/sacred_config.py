"""Shared Sacred/YAML configuration helpers for evaluation entrypoints."""

from __future__ import annotations

import argparse
import copy
import os
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    from sacred import Experiment
except ImportError:
    Experiment = None


def ensure_sacred_available() -> None:
    """Raise a friendly error when Sacred is missing."""
    if Experiment is None:
        raise RuntimeError(
            "Sacred is required for configuration management. "
            "Please install it with: pip install sacred"
        )


def parse_bootstrap_args(
    argv: Sequence[str],
    *,
    description: str,
    options: Iterable[Tuple[str, Dict[str, Any]]],
) -> Tuple[argparse.Namespace, list[str]]:
    """Parse a small set of startup args and leave Sacred overrides untouched."""
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("-h", "--help", action="store_true", dest="_show_help")
    for name, kwargs in options:
        parser.add_argument(name, **kwargs)
    return parser.parse_known_args(list(argv))


def load_yaml(path: Optional[str]) -> dict:
    """Load a YAML file; return an empty dict when unavailable or invalid."""
    if not yaml or not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def resolve_named_yaml(raw: Optional[str], config_subdir: str, *, root_dir: str) -> Optional[str]:
    """Resolve a config name to ``<root_dir>/src/config/<subdir>/<name>.yaml``."""
    if not raw:
        return None
    value = raw.strip()
    if not value:
        return None
    if "/" in value or "\\" in value:
        print(
            f"[config] Warning: --{config_subdir} expects a file name only; "
            f"ignored: {raw!r}"
        )
        return None
    stem = value
    if stem.lower().endswith(".yaml"):
        stem = stem[:-5]
    elif stem.lower().endswith(".yml"):
        stem = stem[:-4]
    if not stem:
        return None
    base_dir = os.path.join(root_dir, "src", "config", config_subdir)
    for extension in (".yaml", ".yml"):
        candidate = os.path.join(base_dir, stem + extension)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(base_dir, stem + ".yaml")


def build_experiment(name: str, base_config: Dict[str, Any]) -> "Experiment":
    """Create a Sacred experiment seeded with a plain dictionary config."""
    ensure_sacred_available()
    experiment = Experiment(name)
    experiment.add_config(copy.deepcopy(base_config))
    return experiment


def dict_to_namespace(config: Dict[str, Any]) -> SimpleNamespace:
    """Convert a config dict to an attribute-style namespace."""
    return SimpleNamespace(**config)
