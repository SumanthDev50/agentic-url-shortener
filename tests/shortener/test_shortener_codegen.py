"""Code generation: alphabet, bounded retry plan, length bump, reserved words."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from shortener.domain.codegen import BASE62, CodeGenerator
from shortener.domain.validation import is_valid_code_format


def constant(char: str) -> Callable[[str], str]:
    return lambda _alphabet: char


def scripted(text: str) -> Callable[[str], str]:
    """A fake RNG that returns the given characters one after another (then repeats 'a')."""
    stream: Iterator[str] = iter(text)
    return lambda _alphabet: next(stream, "a")


def test_default_generator_uses_base62_and_default_length() -> None:
    generator = CodeGenerator()
    codes = {generator.generate() for _ in range(200)}
    assert len(codes) == 200  # 62**7 possibilities: a collision here means a broken RNG
    for code in codes:
        assert len(code) == 7
        assert set(code) <= set(BASE62)
        assert is_valid_code_format(code)


def test_alphabet_is_exactly_base62() -> None:
    assert len(BASE62) == 62
    assert len(set(BASE62)) == 62
    assert BASE62.isalnum()


def test_candidates_are_bounded_and_length_is_bumped_after_three_collisions() -> None:
    codes = list(CodeGenerator(constant("a")).candidates())
    assert codes == ["a" * 7] * 3 + ["a" * 8] * 2


def test_candidates_are_lazy() -> None:
    drawn: list[str] = []

    def spy(_alphabet: str) -> str:
        drawn.append("x")
        return "a"

    candidates = CodeGenerator(spy).candidates()
    assert drawn == []
    next(candidates)
    assert len(drawn) == 7  # only the first candidate was generated


def test_reserved_words_are_skipped_but_consume_an_attempt() -> None:
    # The first candidate spells the reserved word "healthz" (case-insensitively "HealthZ").
    generator = CodeGenerator(scripted("HealthZ"))
    codes = list(generator.candidates())
    assert "HealthZ" not in codes
    assert len(codes) == 4  # 5 attempts, one of them wasted on the reserved word


def test_a_source_that_only_yields_reserved_words_still_terminates() -> None:
    generator = CodeGenerator(constant("a"), length=3)
    assert len(list(generator.candidates())) == 5
    only_reserved = CodeGenerator(scripted("api" * 10), length=3, max_attempts=3)
    assert list(only_reserved.candidates()) == []


def test_custom_budget_and_length() -> None:
    generator = CodeGenerator(constant("z"), length=4, max_attempts=2, collisions_before_bump=1)
    assert list(generator.candidates()) == ["zzzz", "zzzzz"]


@pytest.mark.parametrize(
    "kwargs",
    [{"length": 2}, {"max_attempts": 0}, {"collisions_before_bump": 0}],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="invalid code generator parameters"):
        CodeGenerator(**kwargs)
