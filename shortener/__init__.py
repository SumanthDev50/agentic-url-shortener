"""URL shortener service (FastAPI + SQLite).

Layering: ``api`` -> ``services`` -> ``domain`` / ``storage``.  ``api`` never touches SQL and
``domain`` performs no I/O.  See ``docs/product-spec.md`` for the behavioural contract.
"""

from __future__ import annotations

__version__ = "0.1.0"
