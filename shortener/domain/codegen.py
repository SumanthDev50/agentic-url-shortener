"""Short code generation (NFR-03).

Codes are random base62 strings drawn from ``secrets`` (unguessable, so private links cannot be
enumerated).  Uniqueness is *not* decided here: the database unique constraint is the source of
truth and the caller retries with the next candidate.  The retry budget is bounded and the
length is bumped after a few collisions so a dense keyspace degrades instead of looping.
"""

from __future__ import annotations

import secrets
import string
from collections.abc import Callable, Iterator

from shortener.domain.validation import RESERVED_ALIASES

BASE62 = string.digits + string.ascii_letters
DEFAULT_CODE_LENGTH = 7
MAX_ATTEMPTS = 5
COLLISIONS_BEFORE_LENGTH_BUMP = 3


class CodeGenerator:
    """Produces a bounded sequence of candidate codes for one creation attempt."""

    def __init__(
        self,
        choice: Callable[[str], str] = secrets.choice,
        *,
        length: int = DEFAULT_CODE_LENGTH,
        max_attempts: int = MAX_ATTEMPTS,
        collisions_before_bump: int = COLLISIONS_BEFORE_LENGTH_BUMP,
    ) -> None:
        if length < 3 or max_attempts < 1 or collisions_before_bump < 1:
            raise ValueError("invalid code generator parameters")
        self._choice = choice
        self._length = length
        self._max_attempts = max_attempts
        self._collisions_before_bump = collisions_before_bump

    def generate(self, length: int | None = None) -> str:
        """One random base62 code of ``length`` characters."""
        size = length if length is not None else self._length
        return "".join(self._choice(BASE62) for _ in range(size))

    def candidates(self) -> Iterator[str]:
        """Yield at most ``max_attempts`` candidates, longer after repeated collisions.

        A candidate that happens to spell a reserved word is skipped, but it still consumes an
        attempt so a misbehaving random source cannot make the loop unbounded.
        """
        for attempt in range(self._max_attempts):
            bumped = attempt >= self._collisions_before_bump
            code = self.generate(self._length + 1 if bumped else self._length)
            if code.lower() not in RESERVED_ALIASES:
                yield code
