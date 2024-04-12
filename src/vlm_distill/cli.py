"""Small shared CLI helpers used by every entrypoint.

Keeps ``scripts/*.py`` and the ``src`` ``main()`` functions thin and consistent:
a common argument parser (``--config``, ``--dry-run``) and a marker exception
for steps that are scaffolded but not yet implemented in the current phase.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


class PhaseNotImplementedError(NotImplementedError):
    """Raised by a scaffolded entrypoint whose phase has not landed yet."""

    def __init__(self, step: str, phase: int) -> None:
        super().__init__(
            f"'{step}' is scaffolded but not implemented yet (lands in SPEC Phase {phase})."
        )
        self.step = step
        self.phase = phase


@dataclass
class CommonArgs:
    """Parsed values common to all entrypoints."""

    config: str | None
    dry_run: bool


def build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    """Build an argument parser preloaded with the common flags."""
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config; defaults to the matching file under configs/.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a minimal CPU-only pass to validate wiring without heavy compute.",
    )
    return parser


def parse_common(parser: argparse.ArgumentParser, argv: list[str] | None) -> CommonArgs:
    """Parse argv and return the common arguments in a typed container."""
    ns = parser.parse_args(argv)
    return CommonArgs(config=ns.config, dry_run=bool(ns.dry_run))
