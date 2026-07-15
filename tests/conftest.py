"""Shared test setup.

Put the package root (the ``cli-py`` dir that contains ``wuji/``) on sys.path so
``import wuji`` works under pytest's default prepend import mode without needing
an editable install. Only this tests/ tree is touched — no source/config edits.
"""

import pathlib
import sys

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
