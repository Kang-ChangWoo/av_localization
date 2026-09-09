"""Import bootstrap for the vendored F3Loc baseline.

The vendored tree under ``third_party/f3loc`` keeps upstream's absolute imports
(``from modules...``, ``from utils...``) unchanged so it stays diffable against
the pinned revision. Those imports only resolve when the vendored root itself is
on ``sys.path``, which is what this module arranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "third_party" / "f3loc"
PINNED_REVISION_FILE = VENDOR_ROOT / "PINNED_REVISION"


def pinned_revision() -> str:
    """Upstream f3loc commit the vendored copy was taken from."""
    return PINNED_REVISION_FILE.read_text().strip()


def ensure_on_path() -> Path:
    """Put the vendored f3loc root on ``sys.path`` exactly once."""
    if not VENDOR_ROOT.is_dir():
        raise FileNotFoundError(
            f"vendored f3loc baseline missing at {VENDOR_ROOT}; "
            "re-copy modules/ and utils/ from the pinned upstream checkout"
        )
    path = str(VENDOR_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
    return VENDOR_ROOT


ensure_on_path()
