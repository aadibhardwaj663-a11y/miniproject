"""Backward-compatible entry point for the authoritative 286-drug identity audit.

Do not overwrite the authoritative audit with the older 20-structure pilot analysis.
"""
from pathlib import Path
from identity_resolution import build


def run(root):
    return build(Path(root))


if __name__ == '__main__':
    run(Path(__file__).resolve().parents[1])
