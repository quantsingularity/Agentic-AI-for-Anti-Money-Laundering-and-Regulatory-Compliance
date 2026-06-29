"""Wrapper to maintain the historical `code.scripts` import path.

The canonical implementation now resides in `scripts.generate_deterministic_results`.
This thin shim imports from `scripts` to preserve compatibility with code
that imports `code.scripts.generate_deterministic_results`.
"""

import warnings

warnings.warn(
    "`code.scripts.generate_deterministic_results` is deprecated; the canonical implementation is `scripts.generate_deterministic_results`.",
    DeprecationWarning,
)

from scripts.generate_deterministic_results import main

if __name__ == "__main__":
    main()
