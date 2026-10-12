"""serde: deterministic ASCII JSON, hashing, strict parsing of untrusted JSON."""

from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any

import pytest

from orchestrator.core.serde import (
    MAX_JSON_DEPTH_DEFAULT,
    canonical_json,
    hash_jsonable,
    loads_strict,
    parse_datetime,
    sha256_hex,
    to_jsonable,
)
from orchestrator.core.types import NodeStatus, RiskLevel, Severity


class Color(enum.Enum):
    RED = "red"


class Level(enum.IntEnum):
    ONE = 1


@dataclass
class Plain:
    a: int
    b: tuple[int, ...] = ()
    c: dict[str, Any] = field(default_factory=dict)


@dataclass
class WithToDict:
    x: int

    def to_dict(self) -> dict[str, Any]:
        return {"custom": self.x}


# =============================================================================
# to_jsonable
# =============================================================================


class TestToJsonable:
    def test_primitives_pass_through(self) -> None:
        assert to_jsonable([None, True, 3, 1.5, "x"]) == [None, True, 3, 1.5, "x"]

    def test_int_enums_serialise_by_name_everywhere(self) -> None:
        """Enums are checked before primitives, so an IntEnum never leaks as a bare int."""
        assert to_jsonable(Severity.HIGH) == "HIGH"
        assert to_jsonable(RiskLevel.MEDIUM) == "MEDIUM"
        assert to_jsonable(Level.ONE) == "ONE"
        assert to_jsonable({"s": Severity.LOW, "r": [RiskLevel.HIGH]}) == {
            "s": "LOW",
            "r": ["HIGH"],
        }
        assert canonical_json({"s": Severity.LOW}) == b'{"s":"LOW"}'

    def test_str_and_plain_enums_serialise_by_value(self) -> None:
        assert to_jsonable(NodeStatus.AWAITING_APPROVAL) == "awaiting_approval"
        assert to_jsonable(Color.RED) == "red"

    def test_enum_and_non_string_mapping_keys(self) -> None:
        assert to_jsonable({Severity.HIGH: 1, NodeStatus.READY: 2, 5: 3, None: 4}) == {
            "HIGH": 1,
            "ready": 2,
            "5": 3,
            "None": 4,
        }

    def test_hash_of_an_int_enum_equals_hash_of_its_name(self) -> None:
        assert hash_jsonable(Severity.HIGH) == hash_jsonable("HIGH")
        assert hash_jsonable(RiskLevel.MEDIUM) != hash_jsonable(2)

    def test_datetimes_paths_and_bytes(self) -> None:
        moment = datetime(2026, 1, 2, 3, 4, 5, 678, tzinfo=UTC)
        assert to_jsonable(moment) == "2026-01-02T03:04:05.000678+00:00"
        assert to_jsonable(PureWindowsPath("a\\b\\c.txt")) == "a/b/c.txt"
        assert to_jsonable(PurePosixPath("a/b")) == "a/b"
        assert to_jsonable(b"\x00\x01") == {
            "__type__": "bytes",
            "sha256": hashlib.sha256(b"\x00\x01").hexdigest(),
            "len": 2,
        }

    def test_dataclasses_use_to_dict_when_present_else_their_fields(self) -> None:
        assert to_jsonable(WithToDict(3)) == {"custom": 3}
        assert to_jsonable(Plain(1, (2, 3), {"k": Severity.INFO})) == {
            "a": 1,
            "b": [2, 3],
            "c": {"k": "INFO"},
        }

    def test_a_dataclass_type_is_not_a_dataclass_instance(self) -> None:
        with pytest.raises(TypeError, match="Cannot serialise"):
            to_jsonable(Plain)

    def test_sets_are_sorted_deterministically(self) -> None:
        assert to_jsonable({3, 1, 2}) == [1, 2, 3]
        assert to_jsonable(frozenset({"b", "a"})) == ["a", "b"]
        assert to_jsonable({(1, 2), (0, 5)}) == [[0, 5], [1, 2]]
        assert canonical_json({"x": {"b", "a", "c"}}) == canonical_json({"x": {"c", "a", "b"}})

    def test_tuples_become_lists(self) -> None:
        assert to_jsonable(((1, 2), [3])) == [[1, 2], [3]]

    @pytest.mark.parametrize("bad", [object(), 1j, {"a": object()}, [lambda: 1], range(3)])
    def test_unsupported_types_are_rejected(self, bad: Any) -> None:
        with pytest.raises(TypeError, match="Cannot serialise"):
            to_jsonable(bad)


# =============================================================================
# canonical_json / hashing
# =============================================================================


class TestCanonicalJson:
    def test_keys_are_sorted_and_whitespace_free(self) -> None:
        assert canonical_json({"b": 1, "a": [1, 2], "c": {"z": None, "y": True}}) == (
            b'{"a":[1,2],"b":1,"c":{"y":true,"z":null}}'
        )

    def test_insertion_order_never_matters(self) -> None:
        one = {"a": 1, "b": {"x": 1, "y": 2}}
        other = {"b": {"y": 2, "x": 1}, "a": 1}
        assert canonical_json(one) == canonical_json(other)
        assert hash_jsonable(one) == hash_jsonable(other)

    def test_output_is_pure_ascii_so_line_terminators_cannot_split_a_log(self) -> None:
        nasty = "caf\u00e9 \u2028 \u2029 \u0085 \x00 \n \r \t \x1b \U0001f600 \ud800"
        out = canonical_json({"k": nasty})
        assert out.isascii()
        assert b"\n" not in out and b"\r" not in out and b"\x85" not in out
        assert b"\\u2028" in out and b"\\u2029" in out and b"\\u0085" in out
        assert b"\\ud83d\\ude00" in out  # astral characters as a surrogate pair
        assert b"\\ud800" in out  # a lone surrogate is escaped, never raised or emitted raw
        assert json.loads(out)["k"] == nasty  # and it round-trips

    def test_lone_surrogates_can_be_serialised_but_not_hashed_as_text(self) -> None:
        assert canonical_json("\udc80") == b'"\\udc80"'
        with pytest.raises(ValueError, match="not valid UTF-8"):
            sha256_hex("\udc80")
        assert len(hash_jsonable("\udc80")) == 64  # hashing the ASCII encoding is fine

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf"), {"a": [float("nan")]}]
    )
    def test_non_finite_numbers_are_rejected(self, bad: Any) -> None:
        with pytest.raises(ValueError, match=r"Out of range|not JSON compliant"):
            canonical_json(bad)

    def test_returns_bytes(self) -> None:
        out = canonical_json([1])
        assert isinstance(out, bytes) and out == b"[1]"

    def test_known_digests(self) -> None:
        assert sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert sha256_hex(b"abc") == hashlib.sha256(b"abc").hexdigest() == sha256_hex("abc")
        assert hash_jsonable({"a": 1}) == hashlib.sha256(b'{"a":1}').hexdigest()

    def test_hash_changes_with_any_value_change(self) -> None:
        base = {"a": [1, 2, {"b": "c"}]}
        assert hash_jsonable(base) != hash_jsonable({"a": [1, 2, {"b": "d"}]})
        assert hash_jsonable(base) != hash_jsonable({"a": [2, 1, {"b": "c"}]})


# =============================================================================
# parse_datetime
# =============================================================================


class TestParseDatetime:
    def test_aware_timestamps_are_accepted(self) -> None:
        assert parse_datetime("2026-01-01T00:00:00+00:00") == datetime(2026, 1, 1, tzinfo=UTC)
        assert parse_datetime("2026-01-01T00:00:00Z") == datetime(2026, 1, 1, tzinfo=UTC)
        offset = parse_datetime("2026-01-01T05:30:00+05:30")
        assert offset.utcoffset() == timedelta(hours=5, minutes=30)
        assert offset == datetime(2026, 1, 1, tzinfo=UTC)

    def test_aware_datetime_objects_pass_through(self) -> None:
        moment = datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2)))
        assert parse_datetime(moment) is moment

    @pytest.mark.parametrize("naive", ["2026-01-01T00:00:00", "2026-01-01", "2026-01-01 12:00"])
    def test_naive_timestamps_are_rejected(self, naive: str) -> None:
        with pytest.raises(ValueError, match="naive"):
            parse_datetime(naive)

    def test_naive_datetime_objects_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive"):
            parse_datetime(datetime(2026, 1, 1))

    @pytest.mark.parametrize(
        "bad", ["", "yesterday", "2026-13-01T00:00:00+00:00", "2026-01-01T25:00:00+00:00"]
    )
    def test_garbage_is_rejected(self, bad: str) -> None:
        with pytest.raises(ValueError):
            parse_datetime(bad)


# =============================================================================
# loads_strict
# =============================================================================


class TestLoadsStrict:
    def test_parses_documents_given_as_str_or_bytes(self) -> None:
        assert loads_strict('{"a": [1, 2.5, "x", null, true]}') == {"a": [1, 2.5, "x", None, True]}
        assert loads_strict(b'{"a": 1}') == {"a": 1}
        assert loads_strict("  [1]  ") == [1]
        assert loads_strict("12345678901234567890") == 12345678901234567890

    def test_duplicate_keys_are_rejected_at_any_depth(self) -> None:
        for doc in ('{"a": 1, "a": 2}', '{"x": {"a": 1, "a": 1}}', '[{"k": 1, "k": 1}]'):
            with pytest.raises(ValueError, match="duplicate JSON key"):
                loads_strict(doc)

    def test_keys_that_differ_only_after_unescaping_are_still_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            loads_strict('{"a": 1, "\\u0061": 2}')

    @pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_constants_are_rejected(self, constant: str) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            loads_strict(f'{{"x": {constant}}}')

    @pytest.mark.parametrize("number", ["1e999", "-1e999", "1E400", "123e1000000"])
    def test_numbers_that_overflow_to_infinity_are_rejected(self, number: str) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            loads_strict(f"[{number}]")

    def test_large_but_finite_floats_are_fine(self) -> None:
        assert loads_strict("[1e308, -1e308, 1e-400]") == [1e308, -1e308, 0.0]

    def test_depth_limit(self) -> None:
        """Levels are zero-based and a scalar counts as one level below its container."""
        limit = MAX_JSON_DEPTH_DEFAULT
        accepted = "[" * (limit + 1) + "]" * (
            limit + 1
        )  # innermost container sits at level == limit
        assert loads_strict(accepted) is not None
        rejected = "[" * (limit + 2) + "]" * (limit + 2)
        with pytest.raises(ValueError, match="too deep"):
            loads_strict(rejected)
        assert loads_strict("[[1]]", max_depth=2) == [[1]]  # levels 0, 1 and the scalar at 2
        with pytest.raises(ValueError, match="too deep"):
            loads_strict("[[1]]", max_depth=1)
        assert loads_strict('{"a": {"b": {}}}', max_depth=2) == {"a": {"b": {}}}
        with pytest.raises(ValueError, match="too deep"):
            loads_strict('{"a": {"b": {"c": 1}}}', max_depth=2)

    def test_pathological_nesting_is_a_value_error_not_a_crash(self) -> None:
        with pytest.raises(ValueError, match="too deep"):
            loads_strict("[" * 200_000 + "]" * 200_000, max_bytes=10_000_000)
        with pytest.raises(ValueError, match="too deep"):
            loads_strict('{"a":' * 50_000 + "1" + "}" * 50_000, max_bytes=10_000_000)

    def test_size_limit_counts_bytes(self) -> None:
        doc = '"' + "x" * 10 + '"'
        assert loads_strict(doc, max_bytes=len(doc)) == "x" * 10
        with pytest.raises(ValueError, match="too large"):
            loads_strict(doc, max_bytes=len(doc) - 1)
        with pytest.raises(ValueError, match="too large"):
            loads_strict('"\u00e9\u00e9"', max_bytes=5)  # 4 chars but 6 bytes

    def test_the_default_size_limit_applies(self) -> None:
        with pytest.raises(ValueError, match="too large"):
            loads_strict(b" " * (16 * 1024 * 1024 + 1))

    @pytest.mark.parametrize(
        "bad",
        [
            b"",
            b"   ",
            b"{",
            b'{"a": 1,}',
            b"[1] x",
            b"{'a': 1}",
            b"\xef\xbb\xbf[1]",  # a BOM is not JSON
            b"\xff\xfe[1]",
            b'"\xff"',
            b"[1, 2",
            b"nul",
        ],
    )
    def test_malformed_documents_are_value_errors(self, bad: bytes) -> None:
        with pytest.raises(ValueError):
            loads_strict(bad)

    def test_a_lone_surrogate_in_a_str_input_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            loads_strict('"\ud800"')

    def test_integer_digit_bomb_is_a_value_error(self) -> None:
        with pytest.raises(ValueError):
            loads_strict("1" * 5000)

    def test_unicode_escapes_round_trip(self) -> None:
        assert loads_strict('"\\u00e9\\ud83d\\ude00"') == "\u00e9\U0001f600"
