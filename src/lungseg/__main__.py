"""Allow `python -m lungseg` to dispatch to the typer CLI."""

from __future__ import annotations

from lungseg.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
