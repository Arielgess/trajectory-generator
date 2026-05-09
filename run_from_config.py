"""Run trajectory generation from a JSON config file.

The config file must contain a ``"type"`` key set to either ``"equations"``
or ``"px4"``.  All other keys are forwarded as keyword arguments to the
corresponding generation function in :mod:`generate`.

Usage::

    python run_from_config.py path/to/config.json

Example configs are provided in the ``configs/`` directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from generate import generate_equations, generate_px4

_GENERATORS = {
    "equations": generate_equations,
    "px4": generate_px4,
}


def run_config(config_path: str | Path) -> str:
    """Load *config_path* and run the appropriate generator.

    Parameters
    ----------
    config_path:
        Path to a JSON file containing a ``"type"`` key and generation
        parameters.

    Returns
    -------
    str
        Path to the written ``.jsonl`` output file.

    Raises
    ------
    KeyError
        If the config does not contain a ``"type"`` key.
    ValueError
        If ``"type"`` is not ``"equations"`` or ``"px4"``.
    """
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))

    generator_type = config.pop("type", None)
    if generator_type is None:
        raise KeyError("Config must contain a 'type' key ('equations' or 'px4').")
    if generator_type not in _GENERATORS:
        raise ValueError(
            f"Unknown type {generator_type!r}. Must be one of: {list(_GENERATORS)}."
        )

    output_path = _GENERATORS[generator_type](**config)
    return output_path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python run_from_config.py <config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    try:
        output_path = run_config(config_path)
    except (KeyError, ValueError, TypeError) as exc:
        print(f"Config error: {exc}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        sys.exit(1)

    print(f"Done. Output written to: {output_path}")


if __name__ == "__main__":
    main()
