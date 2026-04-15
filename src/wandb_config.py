"""Shared Weights & Biases CLI/config helpers.

This module keeps argparse-based scripts backward compatible while enabling:
1) Structured run config tracking in W&B.
2) Hyperparameter override via W&B Sweeps (wandb.config).
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple


def add_wandb_args(parser: argparse.ArgumentParser) -> None:
    """Attach a standard W&B argument group to a parser."""
    group = parser.add_argument_group("Weights & Biases")
    group.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Enable Weights & Biases tracking and config management.",
    )
    group.add_argument(
        "--wandb_project",
        type=str,
        default="benchfortir",
        help="W&B project name.",
    )
    group.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity/team (optional).",
    )
    group.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (optional).",
    )
    group.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="W&B run tags.",
    )
    group.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode: online/offline/disabled.",
    )


def maybe_init_wandb(
    args: argparse.Namespace,
    *,
    job_type: str,
) -> Tuple[argparse.Namespace, Optional[Any]]:
    """Initialize W&B when enabled and merge wandb.config back into args.

    Returns:
        (possibly-updated args, wandb_run_or_none)
    """
    if not getattr(args, "use_wandb", False):
        return args, None

    try:
        import wandb  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "W&B is enabled but not installed. Please run: pip install wandb"
        ) from e

    run = wandb.init(
        project=getattr(args, "wandb_project", "benchfortir"),
        entity=getattr(args, "wandb_entity", None),
        name=getattr(args, "wandb_run_name", None),
        tags=getattr(args, "wandb_tags", None),
        mode=getattr(args, "wandb_mode", "online"),
        config=vars(args),
        job_type=job_type,
    )

    # Apply sweep/runtime overrides back to argparse namespace.
    cfg = dict(wandb.config)
    for key, value in cfg.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args, run


def wandb_log(run: Optional[Any], payload: Dict[str, Any]) -> None:
    """Log metrics safely when W&B is active."""
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:
        pass


def wandb_finish(
    run: Optional[Any],
    *,
    status: str,
    summary: Optional[Dict[str, Any]] = None,
) -> None:
    """Finalize a W&B run safely."""
    if run is None:
        return
    if summary:
        for key, value in summary.items():
            try:
                run.summary[key] = value
            except Exception:
                continue
    try:
        run.finish(exit_code=0 if status == "success" else 1)
    except Exception:
        # Backward compatibility for older wandb versions
        try:
            run.finish()
        except Exception:
            pass
