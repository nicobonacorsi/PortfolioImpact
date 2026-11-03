"""Portable pytest IDs for large input-boundary regression cases.

Pytest exports the *full* parameterized test node ID through the
PYTEST_CURRENT_TEST environment variable. Windows rejects environment values
longer than 32,767 characters, so an intentional multi-megabyte test input can
fail at setup before the application runs. Only the *display identifier* is
shortened here; pytest passes the unchanged parameter bytes to the test.

This uses pytest's public parametrization-ID hook rather than patching pytest,
changing test data, excluding a test, or changing its assertions.
"""

from __future__ import annotations

from hashlib import sha256


MAX_INLINE_PARAMETER_ID = 1024


def pytest_make_parametrize_id(config, val, argname):
    """Replace enormous byte/text test IDs with stable length/hash labels."""
    if not isinstance(val, (bytes, str)) or len(val) <= MAX_INLINE_PARAMETER_ID:
        return None
    value = val if isinstance(val, bytes) else val.encode("utf-8", errors="surrogatepass")
    kind = "bytes" if isinstance(val, bytes) else "text"
    digest = sha256(value).hexdigest()[:20]
    return f"{argname}-{kind}-length-{len(val)}-sha256-{digest}"
