"""Permite que `python -m lungseg` despache a la CLI de typer."""

from __future__ import annotations

from lungseg.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
