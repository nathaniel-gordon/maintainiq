"""No-install launcher for the src/ layout: ``python . <subcommand>``.

The package lives under ``src/`` so that tests and tooling always import the
installed-shaped package rather than whatever happens to sit in the working
directory. Running the project directory directly puts ``src`` on the path and
hands off to :mod:`pme.__main__`, so the daemon is startable from a clean
checkout with no build step. ``pip install -e .`` additionally enables
``python -m pme`` and the ``pme`` console script.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pme.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
