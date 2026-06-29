"""Pytest configuration for the test suite (located under ``aml/tests``).

The tests import the project's modules as ``aml.<module>`` and the canonical
runner scripts as ``scripts.<module>`` (the top-level ``scripts`` package). Both
require the *project root* (the parent of ``aml/``) to be on ``sys.path``, so
insert it here regardless of where pytest is invoked from.

The package was renamed from ``code`` to ``aml`` precisely so it no longer
shadows Python's standard-library ``code`` module: on Python 3.13 pytest imports
``pdb`` (which references ``code.InteractiveConsole``) at startup, and a ``code``
package on the path made that resolve to this project and crash.
"""

import sys
from pathlib import Path

# aml/tests/conftest.py -> parents[0]=tests, [1]=aml, [2]=project root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
