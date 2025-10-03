"""
Backward-compatibility shim for tests and legacy imports.

Allows:
    from gam.core.cli import cli

by re-exporting the Click CLI object defined in gam.cli.cli.
"""

from gam.cli.cli import cli

__all__ = ["cli"]

if __name__ == "__main__":
    # Optional: enable running this module directly
    cli()