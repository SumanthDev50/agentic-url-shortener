"""AuditLog: hash chain, tamper detection, recovery, segments, listeners, concurrency, redaction."""

from __future__ import annotations

import gc
import itertools
import json
import logging
import os
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core import audit as audit_module
from orchestrator.core.audit import (
    AUDIT_FILENAME,
    GENESIS_HASH,
    GOVERNANCE_EVENTS,
    AuditIntegrityError,
    AuditLog,
    VerifyResult,
    read_run_events,
    segment_paths,
    verify_run,
)
from orchestrator.core.clock import FakeClock
from orchestrator.core.errors import ConfigError, IntegrityError, StateError
from orchestrator.core.serde import canonical_json, hash_jsonable
from orchestrator.core.types import Actor, AuditEvent, AuditSink, EventType

RUN = "run-1"
ET = EventType

SCRIPT: list[tuple[EventType, str | None, int | None, dict[str, Any]]] = [
    (ET.RUN_CREATED, None, None, {"scenario": "demo"}),
    (ET.NODE_STARTED, "design", 1, {"stage": "design"}),
    (ET.TOOL_CALL, "design", 1, {"tool": "read"}),
    (ET.NODE_SUCCEEDED, "design", 1, {"outputs": []}),
    (ET.NODE_STARTED, "build", 1, {"stage": "implementation"}),
    (ET.RUN_SUCCEEDED, None, None, {}),
]


def new_log(path: Path, *, clock: FakeClock | None = None, **kwargs: Any) -> AuditLog:
    return AuditLog(path, run_id=kwargs.pop("run_id", RUN), clock=clock or FakeClock(), **kwargs)


def fill(log: AuditLog, script: list[tuple[Any, ...]] | None = None) -> list[AuditEvent]:
    out = []
    for event_type, node_id, attempt, data in script or SCRIPT:
        out.append(log.emit(event_type, node_id=node_id, attempt=attempt, data=data))
    return out


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / AUDIT_FILENAME


@pytest.fixture
def filled(path: Path) -> list[AuditEvent]:
    return fill(new_log(path))


def read_lines(path: Path) -> list[bytes]:
    data = path.read_bytes()
    assert data.endswith(b"\n")
    return data.split(b"\n")[:-1]


def write_lines(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(b"\n".join(lines) + b"\n")


def reseal(records: list[dict[str, Any]], start: int = 0) -> list[bytes]:
    """What a privileged attacker can do: recompute the chain from ``start`` on."""
    prev = records[start - 1]["hash"] if start else GENESIS_HASH
    for record in records[start:]:
        record["prev_hash"] = prev
        record.pop("hash", None)
        record["hash"] = hash_jsonable(record)
        prev = record["hash"]
    return [canonical_json(r) for r in records]


def records_of(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in read_lines(path)]


# --------------------------------------------------------------------------- chain basics


class TestChain:
    def test_events_form_a_valid_hash_chain(self, path: Path, filled: list[AuditEvent]) -> None:
        assert [e.seq for e in filled] == [1, 2, 3, 4, 5, 6]
        assert filled[0].prev_hash == GENESIS_HASH
        for before, after in itertools.pairwise(filled):
            assert after.prev_hash == before.hash
        for event in filled:
            body = {k: v for k, v in event.to_dict().items() if k != "hash"}
            assert event.hash == hash_jsonable(body)
        result = AuditLog.verify_file(path)
        assert result == VerifyResult(True, 6, None, "", 6, filled[-1].hash, False)

    def test_file_is_canonical_ascii_one_event_per_line(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        raw = path.read_bytes()
        assert raw.isascii() and b"\r" not in raw
        assert [canonical_json(json.loads(line)) for line in read_lines(path)] == read_lines(path)
        assert [AuditEvent.from_dict(json.loads(line)) for line in read_lines(path)] == filled

    def test_emit_returns_what_was_persisted(self, path: Path) -> None:
        log = new_log(path)
        event = log.emit(
            ET.NODE_STARTED, node_id="n", attempt=2, actor=Actor.agent("a"), data={"x": (1, 2)}
        )
        assert event == log.events()[0] == AuditEvent.from_dict(records_of(path)[0])
        assert event.data == {"x": [1, 2]} and event.actor == Actor.agent("a")

    def test_spans_are_derived_from_node_and_attempt(self, path: Path) -> None:
        log = new_log(path)
        run = log.emit(ET.RUN_CREATED)
        node = log.emit(ET.NODE_READY, node_id="design")
        attempt = log.emit(ET.NODE_STARTED, node_id="design", attempt=3)
        assert (run.span_id, run.parent_span_id) == ("run", None)
        assert (node.span_id, node.parent_span_id) == ("n:design", "run")
        assert (attempt.span_id, attempt.parent_span_id) == ("n:design#3", "n:design")

    def test_ts_comes_from_the_injected_clock_inside_the_lock(self, path: Path) -> None:
        clock = FakeClock()
        log = new_log(path, clock=clock)
        first = log.emit(ET.RUN_CREATED)
        clock.advance(2.5)
        second = log.emit(ET.RUN_STARTED)
        assert first.ts == FakeClock().now() and (second.ts - first.ts).total_seconds() == 2.5

    def test_identical_inputs_give_byte_identical_logs(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a" / "audit.jsonl", tmp_path / "b" / "audit.jsonl"
        fill(new_log(a))
        fill(new_log(b))
        assert a.read_bytes() == b.read_bytes()

    def test_head(self, path: Path) -> None:
        log = new_log(path)
        assert log.head() == (0, GENESIS_HASH)
        events = fill(log)
        assert log.head() == (6, events[-1].hash)

    def test_events_returns_a_copy(self, path: Path, filled: list[AuditEvent]) -> None:
        log = new_log(path)
        log.events().clear()
        assert len(log.events()) == 6

    def test_trace_id_is_derived_from_the_run_and_adopted_on_reopen(self, path: Path) -> None:
        log = new_log(path)
        event = log.emit(ET.RUN_CREATED)
        assert event.trace_id == new_log(path.parent / "other.jsonl").emit(ET.RUN_CREATED).trace_id
        explicit = new_log(path.parent / "x.jsonl", trace_id="trace-abc").emit(ET.RUN_CREATED)
        assert explicit.trace_id == "trace-abc"
        reopened = new_log(path)
        assert reopened.emit(ET.RUN_STARTED).trace_id == event.trace_id
        assert new_log(path, trace_id="trace-zzz").emit(ET.RUN_RESUMED).trace_id == "trace-zzz"

    def test_default_actor_is_the_engine_and_actor_ids_are_capped(self, path: Path) -> None:
        log = new_log(path)
        assert log.emit(ET.RUN_CREATED).actor == Actor.system()
        assert len(log.emit(ET.RUN_STARTED, actor=Actor.human("h" * 1000)).actor.id) == 128

    def test_actor_ids_are_redacted_too(self, path: Path) -> None:
        secret = "gh" + "p_" + "Q9w8" * 9
        event = new_log(path).emit(ET.RUN_CREATED, actor=Actor.human(secret))
        assert secret not in event.actor.id and secret.encode() not in path.read_bytes()

    def test_event_types_may_be_given_as_values(self, path: Path) -> None:
        assert new_log(path).emit("run.created").type is ET.RUN_CREATED  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            new_log(path.parent / "z.jsonl").emit("not.an.event")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("node_id", "attempt"),
        [
            ("Bad Id", None),
            ("a\n", None),
            ("n:x", None),
            (7, None),
            (None, 1),
            ("a", -1),
            ("a", True),
            ("a", 1.5),
            ("a", "1"),
        ],
    )
    def test_invalid_targets_are_rejected(self, path: Path, node_id: Any, attempt: Any) -> None:
        log = new_log(path)
        with pytest.raises(ValueError):
            log.emit(ET.NODE_STARTED, node_id=node_id, attempt=attempt)
        assert not path.exists() and log.head() == (0, GENESIS_HASH)

    def test_actor_must_be_an_actor(self, path: Path) -> None:
        with pytest.raises(TypeError):
            new_log(path).emit(ET.RUN_CREATED, actor="alice")  # type: ignore[arg-type]

    @pytest.mark.parametrize("run_id", ["", "../x", "run\n", "9run", "a b", "x" * 80, 5])
    def test_invalid_run_ids(self, path: Path, run_id: Any) -> None:
        with pytest.raises(ConfigError):
            AuditLog(path, run_id=run_id, clock=FakeClock())

    def test_invalid_mode_and_trace_id(self, path: Path) -> None:
        with pytest.raises(ConfigError):
            AuditLog(path, run_id=RUN, clock=FakeClock(), mode="lenient")  # type: ignore[arg-type]
        with pytest.raises(ConfigError):
            AuditLog(path, run_id=RUN, clock=FakeClock(), trace_id="bad id\n")

    def test_a_linked_log_path_is_refused_even_where_symlinks_cannot_be_created(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(audit_module, "is_link_or_reparse", lambda candidate: True)
        with pytest.raises(ConfigError, match="symlink"):
            new_log(path)

    def test_symlinked_log_path_is_refused(self, tmp_path: Path) -> None:
        target = tmp_path / "elsewhere.jsonl"
        target.write_bytes(b"")
        link = tmp_path / "audit.jsonl"
        try:
            os.symlink(target, link)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks unavailable")
        with pytest.raises(ConfigError, match="symlink"):
            new_log(link)


class TestReopen:
    def test_reopening_verifies_and_continues_the_chain(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        reopened = new_log(path)
        assert reopened.events() == filled and reopened.head() == (6, filled[-1].hash)
        extra = reopened.emit(ET.RUN_RESUMED)
        assert extra.seq == 7 and extra.prev_hash == filled[-1].hash
        assert AuditLog.verify_file(path).ok and AuditLog.verify_file(path).count == 7

    def test_a_log_of_another_run_is_refused(self, path: Path, filled: list[AuditEvent]) -> None:
        with pytest.raises(AuditIntegrityError, match="different run"):
            new_log(path, run_id="run-2")

    def test_empty_file_is_an_empty_log(self, path: Path) -> None:
        path.write_bytes(b"")
        log = new_log(path)
        assert log.head() == (0, GENESIS_HASH)
        assert log.emit(ET.RUN_CREATED).seq == 1


# --------------------------------------------------------------------------- hostile content


class TestHostileData:
    HOSTILE = "a\u2028b\u2029c\x85d\ud800e\x00f\r\ng\u202eh\x0bi\x0cj\x1ck"

    def test_unicode_line_separators_never_split_a_record(self, path: Path) -> None:
        log = new_log(path)
        log.emit(
            ET.AGENT_NOTE, data={"text": self.HOSTILE, "list": [self.HOSTILE], self.HOSTILE: 1}
        )
        log.emit(ET.RUN_STARTED)
        raw = path.read_bytes()
        assert raw.isascii()
        assert len(raw.split(b"\n")) == 3  # two lines plus the empty tail
        assert len(raw.decode("ascii").splitlines()) == 2  # not even str.splitlines can split it
        assert len(raw.splitlines()) == 2
        reopened = new_log(path)
        assert reopened.events()[0].data["text"] == self.HOSTILE
        assert AuditLog.verify_file(path).ok

    def test_nul_and_lone_surrogates_in_data_round_trip(self, path: Path) -> None:
        log = new_log(path)
        event = log.emit(ET.AGENT_NOTE, data={"s": "\x00𐏿"})
        assert new_log(path).events()[0] == event

    def test_unserialisable_data_is_replaced_not_raised(self, path: Path) -> None:
        log = new_log(path)
        for data in ({"x": object()}, {"x": {1, object()}}):
            event = log.emit(ET.AGENT_NOTE, data=data)
            assert event.data == {"_error": "audit data was not serialisable"}
        not_a_mapping: Any = ["not", "a", "mapping"]
        assert log.emit(ET.AGENT_NOTE, data=not_a_mapping).data == {
            "_error": "audit data was not serialisable"
        }
        assert AuditLog.verify_file(path).ok

    def test_deeply_nested_data_cannot_crash_emit(self, path: Path) -> None:
        nested: Any = "leaf"
        for _ in range(5000):
            nested = [nested]
        log = new_log(path)
        event = log.emit(ET.AGENT_NOTE, data={"deep": nested})
        assert event.data == {"_error": "audit data was not serialisable"}

    def test_non_finite_numbers_are_recorded_as_text(self, path: Path) -> None:
        event = new_log(path).emit(ET.AGENT_NOTE, data={"a": float("nan"), "b": float("inf")})
        assert event.data == {"a": "nan", "b": "inf"} and AuditLog.verify_file(path).ok

    def test_oversized_data_is_replaced_by_a_digest(self, path: Path) -> None:
        big = {f"k{i}": "v" * 4000 for i in range(200)}  # ~800 kB after redaction caps
        event = new_log(path).emit(ET.AGENT_NOTE, data=big)
        assert (
            event.data["_truncated"] is True
            and event.data["original_bytes"] > audit_module.MAX_EVENT_BYTES
        )
        assert len(path.read_bytes()) < 2000

    def test_data_is_deep_copied_into_plain_json(self, path: Path) -> None:
        payload = {"items": [1, 2]}
        event = new_log(path).emit(ET.AGENT_NOTE, data=payload)
        payload["items"].append(3)
        assert event.data == {"items": [1, 2]}


class TestRedactionInTheLog:
    SECRET = "gh" + "p_" + "Zq7r" * 9
    AWS = "AK" + "IA" + "M" * 16

    def test_secrets_never_reach_the_file_or_the_hash_input(self, path: Path) -> None:
        log = new_log(path)
        event = log.emit(
            ET.AGENT_NOTE,
            data={
                "message": f"cloning with token {self.SECRET} done",
                "nested": [{"password": "hunter2", "cfg": {"api_key": self.AWS, "ok": 1}}],
                "url": "https://u:pw@example.test/x",
            },
        )
        raw = path.read_bytes()
        for leaked in (self.SECRET, self.AWS, "hunter2", "u:pw"):
            assert leaked.encode() not in raw
        assert event.data["nested"] == [
            {"password": "[REDACTED]", "cfg": {"api_key": "[REDACTED]", "ok": 1}}
        ]
        assert event.data["url"] == "https://[REDACTED]@example.test/x"
        assert AuditLog.verify_file(path).ok  # the stored (redacted) form is what the chain covers

    def test_listeners_only_ever_see_the_redacted_event(self, path: Path) -> None:
        log = new_log(path)
        seen: list[AuditEvent] = []
        log.add_listener(seen.append)
        log.emit(ET.AGENT_NOTE, data={"leak": self.SECRET})
        assert log.flush()
        assert self.SECRET not in json.dumps(seen[0].to_dict())

    def test_custom_redactor_is_used(self, path: Path) -> None:
        from orchestrator.core.audit import Redactor

        log = new_log(path, redactor=Redactor([r"ACME-[0-9]{4}"]))
        assert log.emit(ET.AGENT_NOTE, data={"m": "ticket ACME-1234"}).data == {
            "m": "ticket [REDACTED]"
        }

    def test_long_strings_are_capped_in_the_log(self, path: Path) -> None:
        event = new_log(path).emit(ET.AGENT_NOTE, data={"blob": "x" * 50_000})
        assert len(event.data["blob"]) < 8400 and "truncated" in event.data["blob"]


# --------------------------------------------------------------------------- tampering


class TestTamperDetection:
    def check(self, path: Path, broken_at: int) -> VerifyResult:
        result = AuditLog.verify_file(path)
        assert not result.ok and result.broken_at == broken_at, result
        for mode in ("strict", "recover"):
            with pytest.raises(AuditIntegrityError) as info:
                new_log(path, mode=mode)
            assert info.value.broken_at == broken_at
            assert info.value.last_good_seq == broken_at - 1
            assert isinstance(info.value, IntegrityError)
        return result

    def test_edited_event(self, path: Path, filled: list[AuditEvent]) -> None:
        lines = read_lines(path)
        lines[2] = lines[2].replace(b'"tool":"read"', b'"tool":"nope"')
        write_lines(path, lines)
        assert "hash mismatch" in self.check(path, 3).reason

    def test_deleted_middle_event(self, path: Path, filled: list[AuditEvent]) -> None:
        lines = read_lines(path)
        del lines[2]
        write_lines(path, lines)
        assert "sequence break" in self.check(path, 3).reason

    def test_duplicated_event(self, path: Path, filled: list[AuditEvent]) -> None:
        lines = read_lines(path)
        lines.insert(2, lines[1])
        write_lines(path, lines)
        self.check(path, 3)

    def test_forged_inserted_event_with_a_valid_own_hash(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        records = records_of(path)
        forged = dict(records[1], seq=3, prev_hash=records[1]["hash"], data={"forged": True})
        forged.pop("hash")
        forged["hash"] = hash_jsonable(forged)
        lines = read_lines(path)
        lines.insert(2, canonical_json(forged))
        write_lines(path, lines)
        self.check(path, 4)  # the genuine event 3 now follows a forged one

    def test_reordered_events(self, path: Path, filled: list[AuditEvent]) -> None:
        lines = read_lines(path)
        lines[2], lines[3] = lines[3], lines[2]
        write_lines(path, lines)
        self.check(path, 3)

    def test_edited_last_event(self, path: Path, filled: list[AuditEvent]) -> None:
        lines = read_lines(path)
        lines[-1] = lines[-1].replace(b"run.succeeded", b"run.failed")
        write_lines(path, lines)
        self.check(path, 6)

    @pytest.mark.parametrize("position", [0, 3, 5])
    def test_unknown_extra_key_breaks_the_hash(
        self, path: Path, filled: list[AuditEvent], position: int
    ) -> None:
        records = records_of(path)
        records[position]["smuggled"] = "x"
        write_lines(path, [canonical_json(r) for r in records])
        assert not AuditLog.verify_file(path).ok

    def test_extra_key_is_rejected_even_when_the_hash_is_recomputed_on_the_last_event(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        records = records_of(path)
        records[-1]["smuggled"] = "x"
        write_lines(path, reseal(records, start=5))
        result = self.check(path, 6)
        assert "unexpected keys" in result.reason

    def test_missing_key_is_rejected(self, path: Path, filled: list[AuditEvent]) -> None:
        records = records_of(path)
        del records[-1]["actor"]
        write_lines(path, reseal(records, start=5))
        self.check(path, 6)

    def test_non_canonical_serialisation_is_rejected(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        records = records_of(path)
        lines = read_lines(path)
        lines[1] = json.dumps(records[1], sort_keys=True).encode()  # same data, extra whitespace
        write_lines(path, lines)
        assert "canonical" in self.check(path, 2).reason

    def test_crlf_line_endings_are_rejected(self, path: Path, filled: list[AuditEvent]) -> None:
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.check(path, 1)

    def test_forged_hash_field(self, path: Path, filled: list[AuditEvent]) -> None:
        records = records_of(path)
        records[2]["hash"] = "zz" + records[2]["hash"][2:]
        write_lines(path, [canonical_json(r) for r in records])
        assert "malformed hash" in self.check(path, 3).reason

    def test_event_of_another_run_inside_the_chain(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        records = records_of(path)
        records[3]["run_id"] = "run-2"
        write_lines(path, reseal(records, start=3))
        assert "different run" in self.check(path, 4).reason

    def test_unknown_event_type(self, path: Path, filled: list[AuditEvent]) -> None:
        records = records_of(path)
        records[-1]["type"] = "totally.unknown"
        write_lines(path, reseal(records, start=5))
        assert "malformed event" in self.check(path, 6).reason

    def test_bad_sequence_numbers(self, path: Path, filled: list[AuditEvent]) -> None:
        records = records_of(path)
        records[0]["seq"] = True
        write_lines(path, reseal(records, start=0))
        assert not AuditLog.verify_file(path).ok

    def test_garbage_and_bom_at_the_start(self, path: Path, filled: list[AuditEvent]) -> None:
        path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
        self.check(path, 1)

    def test_duplicate_json_keys_in_a_middle_line(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        lines = read_lines(path)
        lines[2] = b'{"seq":3,"seq":3}'
        write_lines(path, lines)
        self.check(path, 3)

    def test_first_event_must_be_genesis_or_a_segment_start(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        lines = read_lines(path)
        write_lines(path, lines[1:])  # drop the genesis event
        assert not AuditLog.verify_file(path).ok

    @pytest.mark.parametrize("line", [b"[1,2]", b'"text"', b"123", b"null"])
    def test_lines_that_are_not_json_objects(
        self, path: Path, filled: list[AuditEvent], line: bytes
    ) -> None:
        lines = read_lines(path)
        lines[2] = line
        write_lines(path, lines)
        result = self.check(path, 3)
        assert "not a JSON object" in result.reason

    def test_wrong_prev_hash_with_a_valid_sequence_and_hash(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        records = records_of(path)
        records[3]["prev_hash"] = "f" * 64
        records[3].pop("hash")
        records[3]["hash"] = hash_jsonable(records[3])  # self-consistent, but it links to nothing
        lines = read_lines(path)
        lines[3] = canonical_json(records[3])
        write_lines(path, lines)
        assert "prev_hash" in self.check(path, 4).reason

    def test_an_event_that_cannot_be_canonicalised_is_a_verification_failure(
        self, path: Path, filled: list[AuditEvent], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(obj: Any) -> str:
            raise TypeError("boom")

        monkeypatch.setattr(audit_module, "hash_jsonable", broken)
        result = AuditLog.verify_file(path)
        assert not result.ok and result.broken_at == 1 and "canonicalisable" in result.reason

    def test_clean_tail_truncation_needs_an_anchor(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        head = (6, filled[-1].hash)
        write_lines(path, read_lines(path)[:4])
        assert AuditLog.verify_file(path).ok  # a clean cut is invisible to the chain alone...
        result = AuditLog.verify_file(path, anchor=head)
        assert (
            not result.ok and result.broken_at is None and "anchor" in result.reason
        )  # ...but not to an anchor
        reopened = new_log(path)
        assert (
            not reopened.verify(anchor=head).ok and reopened.verify(anchor=(4, filled[3].hash)).ok
        )

    def test_resealed_history_is_valid_but_the_anchor_exposes_it(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        anchor = {"seq": 6, "hash": filled[-1].hash}
        records = records_of(path)
        records[2]["data"] = {"tool": "rewritten"}
        write_lines(path, reseal(records, start=2))
        assert AuditLog.verify_file(
            path
        ).ok  # documented limitation: the chain alone cannot catch a full rewrite
        assert not AuditLog.verify_file(path, anchor=anchor).ok

    def test_live_log_notices_tampering_after_it_was_opened(self, path: Path) -> None:
        log = new_log(path)
        fill(log)
        assert log.verify().ok
        lines = read_lines(path)
        lines[1] = lines[1].replace(b'"stage":"design"', b'"stage":"hacked"')
        write_lines(path, lines)
        assert not log.verify().ok


class TestAnchors:
    def test_all_anchor_forms(self, path: Path, filled: list[AuditEvent]) -> None:
        digest = filled[-1].hash
        for anchor in (
            digest,
            digest.upper(),
            f" {digest} ",
            (6, digest),
            {"seq": 6, "hash": digest},
        ):
            assert AuditLog.verify_file(path, anchor=anchor).ok, anchor
        assert AuditLog.verify_file(
            path, anchor=filled[2].hash
        ).ok  # any event of the chain is a valid anchor
        assert AuditLog.verify_file(path, anchor=(3, filled[2].hash)).ok

    @pytest.mark.parametrize(
        "anchor",
        [
            "0" * 63 + "1",
            (5, "ab" * 32),
            {"seq": 6, "hash": "cd" * 32},
            "not a hash",
            (6, 7),
            {"seq": "x", "hash": "ab" * 32},
            12,
            None.__class__,
        ],
    )
    def test_wrong_or_unusable_anchors_fail(
        self, path: Path, filled: list[AuditEvent], anchor: Any
    ) -> None:
        result = AuditLog.verify_file(path, anchor=anchor)
        assert not result.ok and result.count == 6 and result.broken_at is None

    def test_right_hash_with_wrong_sequence_fails(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        assert not AuditLog.verify_file(path, anchor=(2, filled[-1].hash)).ok

    def test_anchor_of_an_empty_log_is_trivial(self, path: Path, filled: list[AuditEvent]) -> None:
        assert AuditLog.verify_file(path, anchor={"seq": 0, "hash": ""}).ok
        assert AuditLog.verify_file(path, anchor=(0, GENESIS_HASH)).ok

    def test_missing_and_empty_files(self, tmp_path: Path) -> None:
        missing = AuditLog.verify_file(tmp_path / "nope.jsonl")
        assert not missing.ok and "not found" in missing.reason
        empty = tmp_path / "empty.jsonl"
        empty.write_bytes(b"")
        assert AuditLog.verify_file(empty) == VerifyResult(
            True, 0, None, "", 0, GENESIS_HASH, False
        )

    def test_unusable_paths_are_reported_not_raised(self, tmp_path: Path) -> None:
        result = AuditLog.verify_file(str(tmp_path / ("bad" + chr(0) + "name")))
        assert not result.ok and "unreadable" in result.reason


# --------------------------------------------------------------------------- torn tails


class TestTornTail:
    def torn(self, path: Path, keep: int = 5, extra: bytes = b'{"seq":6,"ts":"2026-01-01T0') -> int:
        lines = read_lines(path)
        path.write_bytes(b"\n".join(lines[:keep]) + b"\n" + extra)
        return keep

    def test_strict_mode_refuses_a_torn_tail(self, path: Path, filled: list[AuditEvent]) -> None:
        self.torn(path)
        with pytest.raises(AuditIntegrityError) as info:
            new_log(path)
        assert (
            info.value.result.torn_tail
            and info.value.broken_at == 6
            and info.value.last_good_seq == 5
        )
        assert AuditLog.verify_file(path).torn_tail

    def test_recover_mode_truncates_continues_and_records_the_repair(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        garbage = b'{"seq":6,"ts":"2026-01-01T0'
        self.torn(path, extra=garbage)
        log = new_log(path, mode="recover")
        events = log.events()
        assert [e.type for e in events[:5]] == [e.type for e in filled[:5]]
        repair = events[5]
        assert (
            repair.type is ET.AUDIT_TAIL_REPAIRED
            and repair.seq == 6
            and repair.prev_hash == filled[4].hash
        )
        assert repair.data["repair"] == "truncated_torn_tail"
        assert repair.data["dropped_bytes"] == len(garbage)
        assert repair.data["last_good_seq"] == 5 and len(repair.data["dropped_sha256"]) == 64
        assert garbage not in path.read_bytes()
        assert AuditLog.verify_file(path).ok
        assert log.emit(ET.RUN_RESUMED).seq == 7 and AuditLog.verify_file(path).ok

    def test_torn_tail_that_is_only_missing_its_newline_when_complete(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        path.write_bytes(path.read_bytes()[:-1])  # the last event is intact but unterminated
        with pytest.raises(AuditIntegrityError):
            new_log(path)
        log = new_log(path, mode="recover")
        assert [e.seq for e in log.events()] == [1, 2, 3, 4, 5, 6, 7]
        assert log.events()[-1].data["repair"] == "added_missing_newline"
        assert log.events()[:6] == filled
        assert AuditLog.verify_file(path).ok

    def test_whole_file_of_garbage_is_recovered_as_an_empty_log(self, path: Path) -> None:
        path.write_bytes(b'{"seq":1,"ts"')
        log = new_log(path, mode="recover")
        assert [e.type for e in log.events()] == [ET.AUDIT_TAIL_REPAIRED] and log.events()[
            0
        ].seq == 1
        assert AuditLog.verify_file(path).ok

    def test_nul_padding_after_a_crash_is_a_torn_tail(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        path.write_bytes(path.read_bytes() + b"\x00" * 64)
        new_log(path, mode="recover")
        assert AuditLog.verify_file(path).ok

    def test_recover_does_not_hide_mid_chain_damage(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        lines = read_lines(path)
        lines[2] = b"{garbage"
        write_lines(path, lines)
        with pytest.raises(AuditIntegrityError):
            new_log(path, mode="recover")

    def test_recover_does_not_hide_a_tampered_last_event(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        lines = read_lines(path)
        lines[-1] = lines[-1].replace(
            b"run.succeeded", b"run.failed"
        )  # parseable, wrong hash: tamper, not tear
        write_lines(path, lines)
        with pytest.raises(AuditIntegrityError):
            new_log(path, mode="recover")

    def test_two_garbage_lines_at_the_tail_are_not_a_torn_write(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        path.write_bytes(path.read_bytes() + b"{bad\n{worse")
        with pytest.raises(AuditIntegrityError):
            new_log(path, mode="recover")

    def test_recovery_keeps_the_original_bytes_before_the_tear(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        before = b"".join(line + b"\n" for line in read_lines(path)[:5])
        self.torn(path)
        new_log(path, mode="recover")
        assert path.read_bytes().startswith(before)


# --------------------------------------------------------------------------- segments


class TestSegments:
    def break_chain(self, path: Path) -> AuditIntegrityError:
        lines = read_lines(path)
        lines[3] = lines[3].replace(b'"outputs":[]', b'"outputs":[1]')
        write_lines(path, lines)
        with pytest.raises(AuditIntegrityError) as info:
            new_log(path)
        return info.value

    def test_new_segment_continues_from_the_last_good_event(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        error = self.break_chain(path)
        damaged = path.read_bytes()
        segment = AuditLog.open_new_segment(
            path,
            broken_at=error.broken_at or 0,
            last_good_hash=error.last_good_hash,
            run_id=RUN,
            clock=FakeClock(),
        )
        assert segment.path.name == "audit.1.jsonl" and path.read_bytes() == damaged
        first = segment.events()[0]
        assert (
            first.type is ET.AUDIT_TAIL_REPAIRED
            and first.seq == 4
            and first.prev_hash == filled[2].hash
        )
        assert first.data["broken_at"] == 4 and first.data["last_good_hash"] == filled[2].hash
        assert first.data["broken_file"] == "audit.jsonl"
        segment.emit(
            ET.SAFE_STOP_TRIGGERED, data={"trigger": "integrity", "reason": "audit chain broken"}
        )
        assert AuditLog.verify_file(segment.path).ok
        events = read_run_events(path.parent)
        assert [e.seq for e in events] == [1, 2, 3, 4, 5]
        assert events[:3] == filled[:3] and events[3] == first
        assert events[-1].type is ET.SAFE_STOP_TRIGGERED

    def test_segment_files_can_be_reopened_and_extended(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        error = self.break_chain(path)
        segment = AuditLog.open_new_segment(
            path,
            broken_at=error.broken_at or 0,
            last_good_hash=error.last_good_hash,
            run_id=RUN,
            clock=FakeClock(),
        )
        reopened = new_log(segment.path)
        assert reopened.head() == segment.head()
        assert reopened.emit(ET.RUN_RESUMED).seq == 5

    def test_a_second_break_creates_the_next_segment(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        error = self.break_chain(path)
        first = AuditLog.open_new_segment(
            path,
            broken_at=error.broken_at or 0,
            last_good_hash=error.last_good_hash,
            run_id=RUN,
            clock=FakeClock(),
        )
        first.emit(ET.RUN_RESUMED)
        second = AuditLog.open_new_segment(
            first.path,
            broken_at=5,
            last_good_hash=first.events()[0].hash,
            run_id=RUN,
            clock=FakeClock(),
        )
        assert second.path.name == "audit.2.jsonl"
        assert [p.name for p in segment_paths(path.parent)] == [
            "audit.jsonl",
            "audit.1.jsonl",
            "audit.2.jsonl",
        ]
        assert [e.seq for e in read_run_events(path.parent)] == [1, 2, 3, 4, 5]

    @pytest.mark.parametrize(
        ("broken_at", "last_good_hash"),
        [
            (0, GENESIS_HASH),
            (-1, "a" * 64),
            (True, "a" * 64),
            ("4", "a" * 64),
            (4, "short"),
            (4, GENESIS_HASH),
            (1, "a" * 64),
            (4, "A" * 64),
        ],
    )
    def test_segment_arguments_are_validated(
        self, path: Path, broken_at: Any, last_good_hash: Any
    ) -> None:
        with pytest.raises(ConfigError):
            AuditLog.open_new_segment(
                path,
                broken_at=broken_at,
                last_good_hash=last_good_hash,
                run_id=RUN,
                clock=FakeClock(),
            )
        assert segment_paths(path.parent) == []

    def test_a_segment_from_the_very_first_event(self, path: Path) -> None:
        path.write_bytes(b"garbage\n")
        segment = AuditLog.open_new_segment(
            path, broken_at=1, last_good_hash=GENESIS_HASH, run_id=RUN, clock=FakeClock()
        )
        assert segment.events()[0].seq == 1
        assert [e.seq for e in read_run_events(path.parent)] == [1]

    def test_read_run_events_rejects_a_segment_that_does_not_link(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        stranger = AuditLog.open_new_segment(
            path, broken_at=4, last_good_hash="e" * 64, run_id=RUN, clock=FakeClock()
        )
        assert stranger.path.exists()
        with pytest.raises(IntegrityError, match="does not continue"):
            read_run_events(path.parent)

    def test_read_run_events_rejects_a_genesis_file_posing_as_a_segment(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        other = path.parent / "audit.1.jsonl"
        fill(new_log(other))
        with pytest.raises(IntegrityError):
            read_run_events(path.parent)

    def test_read_run_events_tolerates_a_torn_tail_but_not_other_damage(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        path.write_bytes(path.read_bytes() + b'{"seq":7,"ts')
        assert [e.seq for e in read_run_events(path.parent)] == [1, 2, 3, 4, 5, 6]
        damaged = path.read_bytes().replace(b'"outputs":[]', b'"outputs":[9]')
        path.write_bytes(damaged)
        with pytest.raises(AuditIntegrityError):
            read_run_events(path.parent)

    def test_read_run_events_without_logs(self, tmp_path: Path) -> None:
        assert read_run_events(tmp_path) == [] and segment_paths(tmp_path / "missing") == []

    def test_segment_paths_ignore_look_alikes_and_links(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        for name in (
            "audit.x.jsonl",
            "audit..jsonl",
            "audit.1.jsonl.bak",
            "audit.01.json",
            "xaudit.2.jsonl",
        ):
            (path.parent / name).write_bytes(b"")
        (path.parent / "audit.3.jsonl").mkdir()
        assert [p.name for p in segment_paths(path.parent)] == ["audit.jsonl"]


class TestVerifyRun:
    def test_intact_run(self, path: Path, filled: list[AuditEvent]) -> None:
        result = verify_run(path.parent)
        assert result.ok and result.count == 6 and result.last_good_hash == filled[-1].hash
        assert verify_run(path.parent, anchor=(6, filled[-1].hash)).ok
        assert not verify_run(path.parent, anchor=(6, "ab" * 32)).ok

    def test_no_log(self, tmp_path: Path) -> None:
        result = verify_run(tmp_path)
        assert not result.ok and "not found" in result.reason

    def test_damage_without_a_continuing_segment(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        lines = read_lines(path)
        lines[3] = lines[3].replace(b'"outputs":[]', b'"outputs":[1]')
        write_lines(path, lines)
        result = verify_run(path.parent)
        assert not result.ok and result.broken_at == 4 and result.last_good_seq == 3

    def test_torn_tail_is_reported_by_verify_but_tolerated_by_read(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        path.write_bytes(path.read_bytes() + b'{"seq":7,"ts')
        result = verify_run(path.parent)
        assert not result.ok and result.torn_tail and result.count == 6
        assert len(read_run_events(path.parent)) == 6

    def test_a_forked_run_verifies_as_one_chain_and_anchors_may_span_segments(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        lines = read_lines(path)
        lines[3] = lines[3].replace(b'"outputs":[]', b'"outputs":[1]')
        write_lines(path, lines)
        failure = AuditLog.verify_file(path)
        segment = AuditLog.open_new_segment(
            path,
            broken_at=failure.broken_at or 0,
            last_good_hash=failure.last_good_hash,
            run_id=RUN,
            clock=FakeClock(),
        )
        segment.emit(ET.RUN_FAILED)
        result = verify_run(path.parent)
        assert result.ok and result.count == 5
        assert verify_run(
            path.parent, anchor=segment.head()
        ).ok  # the head lives in the newest segment
        assert verify_run(
            path.parent, anchor=(3, filled[2].hash)
        ).ok  # and an old anchor in the old one
        assert not verify_run(
            path.parent, anchor=(4, filled[3].hash)
        ).ok  # the replaced event is gone

    def test_an_unlinked_segment_is_a_failure_not_an_exception(
        self, path: Path, filled: list[AuditEvent]
    ) -> None:
        AuditLog.open_new_segment(
            path, broken_at=4, last_good_hash="e" * 64, run_id=RUN, clock=FakeClock()
        )
        result = verify_run(path.parent)
        assert not result.ok and "does not continue" in result.reason


# ------------------------------------------------------------------- durability and failures


class TestDurabilityAndFailures:
    def test_governance_events_are_fsynced_before_returning(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []
        real = os.fsync

        def counting(fd: int) -> None:
            calls.append(fd)
            real(fd)

        monkeypatch.setattr(os, "fsync", counting)
        log = new_log(path)
        for event_type in sorted(ET, key=lambda t: t.value):
            before = len(calls)
            log.emit(event_type)
            assert (len(calls) > before) == (event_type in GOVERNANCE_EVENTS), event_type

    def test_governance_set_covers_the_documented_events(self) -> None:
        required = {
            *(ET.APPROVAL_REQUESTED, ET.APPROVAL_VOTE, ET.APPROVAL_GRANTED, ET.APPROVAL_REJECTED),
            *(
                ET.APPROVAL_AUTO,
                ET.APPROVAL_INVALIDATED,
                ET.APPROVAL_CANCELLED,
                ET.APPROVAL_REFUSED,
            ),
            *(ET.SAFE_STOP_TRIGGERED, ET.SAFE_STOP_STEP, ET.SAFE_STOP_OVERRIDDEN),
            *(ET.KILL_SWITCH_DETECTED, ET.NODE_PUBLISHING, ET.RUN_SEALED),
            *(ET.RUN_SUCCEEDED, ET.RUN_FAILED, ET.RUN_SAFE_STOPPED),
            *(ET.EFFECT_STARTED, ET.EFFECT_COMPLETED, ET.AUDIT_TAIL_REPAIRED),
        }
        assert required <= GOVERNANCE_EVENTS
        assert ET.TOOL_CALL not in GOVERNANCE_EVENTS and ET.NODE_STARTED not in GOVERNANCE_EVENTS

    def test_a_write_error_poisons_the_log(
        self, path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log = new_log(path)
        log.emit(ET.RUN_CREATED)
        head = log.head()

        def broken(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full: /secret/path")

        monkeypatch.setattr(audit_module, "append_line", broken)
        with pytest.raises(StateError) as info:
            log.emit(ET.RUN_STARTED)
        assert "/secret/path" not in str(info.value)
        assert log.head() == head and len(log.events()) == 1
        monkeypatch.undo()
        with pytest.raises(StateError, match="failed state"):
            log.emit(ET.RUN_STARTED)  # never append behind a possibly torn write

    def test_naive_clock_is_rejected_before_anything_is_written(self, path: Path) -> None:
        class NaiveClock:
            def now(self) -> datetime:
                return datetime(2026, 1, 1)

            def monotonic(self) -> float:
                return 0.0

            def sleep(self, seconds: float) -> None:
                return None

        log = AuditLog(path, run_id=RUN, clock=NaiveClock())
        with pytest.raises(StateError, match="timezone-aware"):
            log.emit(ET.RUN_CREATED)
        assert not path.exists() and log.head() == (0, GENESIS_HASH)

    def test_verify_before_anything_was_written(self, path: Path) -> None:
        assert new_log(path).verify() == VerifyResult(True, 0, None, "", 0, GENESIS_HASH, False)

    def test_closed_log_refuses_appends(self, path: Path) -> None:
        log = new_log(path)
        log.emit(ET.RUN_CREATED)
        log.close()
        with pytest.raises(StateError, match="closed"):
            log.emit(ET.RUN_STARTED)
        assert AuditLog.verify_file(path).ok


# --------------------------------------------------------------------------- AuditSink


class TestAuditSink:
    def test_audit_log_is_usable_as_an_audit_sink(self, path: Path) -> None:
        log = new_log(path)
        sink: AuditSink = log
        sink(
            ET.TOOL_CALL,
            node_id="design",
            attempt=2,
            actor=Actor.agent("architect"),
            data={"tool": "read_file"},
        )
        event = log.events()[0]
        assert (event.type, event.node_id, event.attempt, event.actor) == (
            ET.TOOL_CALL,
            "design",
            2,
            Actor.agent("architect"),
        )
        assert event.data == {"tool": "read_file"}

    def test_sink_keywords_have_defaults(self, path: Path) -> None:
        new_log(path)(ET.RUN_CREATED)
        assert records_of(path)[0]["node_id"] is None


# --------------------------------------------------------------------------- listeners


def within(seconds: float, fn: Callable[[], Any]) -> Any:
    """Run ``fn`` in a thread; fail the test (instead of hanging) if it deadlocks."""
    box: list[Any] = []
    worker = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    worker.start()
    worker.join(seconds)
    assert not worker.is_alive(), "deadlock"
    return box[0] if box else None


class TestListeners:
    def test_listeners_receive_every_later_event_in_order(self, path: Path) -> None:
        log = new_log(path)
        early = log.emit(ET.RUN_CREATED)
        seen: list[AuditEvent] = []
        log.add_listener(seen.append)
        late = fill(log)
        assert log.flush()
        assert seen == late and early not in seen

    def test_several_listeners_all_get_each_event(self, path: Path) -> None:
        log = new_log(path)
        a: list[int] = []
        b: list[int] = []
        log.add_listener(lambda e: a.append(e.seq))
        log.add_listener(lambda e: b.append(e.seq))
        fill(log)
        assert log.flush() and a == b == [1, 2, 3, 4, 5, 6]

    def test_dispatch_happens_after_the_append_and_outside_the_lock(self, path: Path) -> None:
        log = new_log(path)
        observations: list[tuple[tuple[int, str], bool, int]] = []

        def listener(
            event: AuditEvent,
        ) -> None:  # would deadlock if called under the (non-reentrant) lock
            observations.append((log.head(), log.verify().ok, len(log.events())))

        log.add_listener(listener)
        event = within(5, lambda: log.emit(ET.RUN_CREATED))
        assert log.flush()
        head, verified, count = observations[0]
        assert head[0] >= event.seq and verified and count >= 1  # the event was already durable

    def test_a_listener_may_append_without_deadlocking(self, path: Path) -> None:
        log = new_log(path)
        seen: list[EventType] = []

        def listener(event: AuditEvent) -> None:
            seen.append(event.type)
            if event.type is ET.NODE_STARTED:
                log.emit(
                    ET.AGENT_NOTE,
                    node_id=event.node_id,
                    attempt=event.attempt,
                    data={"reply": event.seq},
                )

        log.add_listener(listener)
        within(5, lambda: log.emit(ET.NODE_STARTED, node_id="design", attempt=1))
        assert log.flush() and log.flush()
        assert seen == [ET.NODE_STARTED, ET.AGENT_NOTE]
        assert [e.type for e in log.events()] == [ET.NODE_STARTED, ET.AGENT_NOTE]
        assert AuditLog.verify_file(path).ok

    def test_listener_exceptions_are_swallowed_and_logged_without_their_message(
        self, path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        log = new_log(path)
        received: list[int] = []

        def failing(event: AuditEvent) -> None:
            raise RuntimeError("password=hunter2 should not be logged")

        log.add_listener(failing)
        log.add_listener(lambda e: received.append(e.seq))
        with caplog.at_level(logging.WARNING, logger="orchestrator.core.audit"):
            fill(log)
            assert log.flush()
        assert received == [1, 2, 3, 4, 5, 6] and AuditLog.verify_file(path).ok
        assert any("RuntimeError" in r.getMessage() for r in caplog.records)
        assert "hunter2" not in caplog.text

    def test_flush_from_a_listener_is_refused(self, path: Path) -> None:
        log = new_log(path)
        outcome: list[str] = []

        def listener(event: AuditEvent) -> None:
            try:
                log.flush()
            except StateError as exc:
                outcome.append(str(exc))

        log.add_listener(listener)
        log.emit(ET.RUN_CREATED)
        assert log.flush()
        assert outcome and "deadlock" in outcome[0]

    def test_flush_without_listeners_is_trivially_true(self, path: Path) -> None:
        assert new_log(path).flush()

    def test_flush_times_out_on_a_stuck_listener(self, path: Path) -> None:
        log = new_log(path)
        release = threading.Event()

        def stuck(event: AuditEvent) -> None:
            release.wait(5)

        log.add_listener(stuck)
        log.emit(ET.RUN_CREATED)
        assert log.flush(timeout=0.02) is False
        release.set()
        assert log.flush()

    def test_dispatch_order_equals_chain_order_under_concurrency(self, path: Path) -> None:
        log = new_log(path)
        order: list[int] = []
        log.add_listener(lambda e: order.append(e.seq))

        def worker(index: int) -> None:
            for step in range(30):
                log.emit(ET.AGENT_NOTE, data={"w": index, "i": step})

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert log.flush()
        assert order == list(range(1, 181))

    def test_close_stops_the_dispatcher_after_draining(self, path: Path) -> None:
        before = set(threading.enumerate())
        log = new_log(path)
        seen: list[int] = []
        log.add_listener(lambda e: seen.append(e.seq))
        spawned = set(threading.enumerate()) - before
        assert len(spawned) == 1
        fill(log)
        log.close()
        assert seen == [1, 2, 3, 4, 5, 6]
        assert not any(t.is_alive() for t in spawned)

    def test_an_abandoned_log_releases_its_dispatcher_thread(self, path: Path) -> None:
        before = set(threading.enumerate())
        log = new_log(path)
        log.add_listener(lambda e: None)
        (thread,) = set(threading.enumerate()) - before
        del log
        gc.collect()
        thread.join(2)
        assert not thread.is_alive()

    def test_flush_after_close_is_trivially_true(self, path: Path) -> None:
        log = new_log(path)
        log.add_listener(lambda e: None)
        log.close()
        assert log.flush()

    def test_non_callable_listener_is_rejected(self, path: Path) -> None:
        with pytest.raises(TypeError):
            new_log(path).add_listener("nope")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- concurrency


class TestConcurrency:
    def test_many_threads_produce_one_valid_contiguous_chain(self, path: Path) -> None:
        clock = FakeClock()
        log = new_log(path, clock=clock)
        threads, per_thread = 8, 30

        def worker(index: int) -> None:
            for step in range(per_thread):
                clock.advance(0.001)
                log.emit(
                    ET.AGENT_NOTE, node_id="n1", attempt=index, data={"thread": index, "step": step}
                )

        pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        events = log.events()
        total = threads * per_thread
        assert [e.seq for e in events] == list(range(1, total + 1))
        assert all(b.prev_hash == a.hash for a, b in itertools.pairwise(events))
        assert all(a.ts <= b.ts for a, b in itertools.pairwise(events))  # ts assigned in the lock
        result = AuditLog.verify_file(path)
        assert result.ok and result.count == total
        per_thread_steps = {
            i: [e.data["step"] for e in events if e.data["thread"] == i] for i in range(threads)
        }
        assert all(
            steps == list(range(per_thread)) for steps in per_thread_steps.values()
        )  # per-thread order kept
        assert new_log(path).head() == log.head()

    def test_verify_and_events_are_safe_while_appending(self, path: Path) -> None:
        log = new_log(path)
        stop = threading.Event()
        failures: list[VerifyResult] = []

        def appender() -> None:
            while not stop.is_set():
                log.emit(ET.AGENT_NOTE, data={"x": 1})

        thread = threading.Thread(target=appender)
        thread.start()
        try:
            for _ in range(25):
                result = log.verify()
                if not result.ok:
                    failures.append(result)
                log.events()
        finally:
            stop.set()
            thread.join()
        assert failures == []
