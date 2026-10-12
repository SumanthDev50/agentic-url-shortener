"""DecisionLog: ids, durability, audit binding, sanitisation, validation, corruption, threads."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core import decisions as decisions_module
from orchestrator.core.decisions import MAX_RATIONALE_CHARS, MAX_SUMMARY_CHARS, DecisionLog
from orchestrator.core.errors import ConfigError, IntegrityError, StateError
from orchestrator.core.serde import canonical_json, hash_jsonable
from orchestrator.core.types import (
    Actor,
    ActorType,
    ArtifactRef,
    Decision,
    DecisionKind,
    EventType,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)
HUMAN = Actor.human("alice")
AGENT = Actor.agent("architect")


class StepClock:
    def __init__(self, naive: bool = False) -> None:
        self._t = T0
        self._naive = naive
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            self._t += timedelta(seconds=1)
            return self._t.replace(tzinfo=None) if self._naive else self._t

    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        return None


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[tuple[EventType, str | None, int | None, Actor, dict[str, Any]]] = []
        self._lock = threading.Lock()

    def __call__(
        self, type: EventType, *, node_id: str | None, attempt: int | None, actor: Actor, data: Any
    ) -> None:
        with self._lock:
            self.calls.append((type, node_id, attempt, actor, dict(data)))


def ref(kind: str = "design_doc", version: int = 1, key: str = "") -> ArtifactRef:
    ident = f"{kind}{'.' + key if key else ''}@v{version}"
    return ArtifactRef(ident, kind, key, version, "a" * 64)


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "fsync", lambda fd: None)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "run" / "decisions.jsonl"


@pytest.fixture
def log(log_path: Path, sink: RecordingSink) -> DecisionLog:
    return DecisionLog(log_path, run_id="run1", clock=StepClock(), audit=sink)


def reopen(path: Path, **kw: Any) -> DecisionLog:
    return DecisionLog(path, run_id=kw.pop("run_id", "run1"), clock=StepClock(), **kw)


def lines_of(path: Path) -> list[bytes]:
    return path.read_bytes().split(b"\n")[:-1]


# =============================================================================
# recording
# =============================================================================


class TestRecord:
    def test_ids_are_sequential_and_fields_are_stored(self, log: DecisionLog) -> None:
        first = log.record(DecisionKind.DESIGN, "use SQLite", actor=AGENT, node_id="design")
        second = log.record(
            DecisionKind.ASSUMPTION,
            "TTL is 24h",
            actor=HUMAN,
            rationale="not specified",
            inputs=(ref("requirements"),),
            outputs=(ref("design_doc"),),
            alternatives=("1h", "7d"),
            outcome="accepted",
            parents=(first.id,),
            refs=("R-001", "ADR-002"),
            data={"k": [1, 2]},
        )
        assert (first.id, second.id) == ("D-0001", "D-0002")
        assert (first.run_id, first.kind, first.actor, first.node_id) == (
            "run1",
            DecisionKind.DESIGN,
            AGENT,
            "design",
        )
        assert second.ts > first.ts and second.ts.tzinfo is not None
        assert second.inputs == (ref("requirements"),)
        assert second.alternatives == ("1h", "7d")
        assert second.parents == ("D-0001",)
        assert second.refs == ("R-001", "ADR-002")
        assert dict(second.data) == {"k": (1, 2)}

    def test_kind_may_be_given_as_its_string_value(self, log: DecisionLog) -> None:
        assert log.record("scope", "x", actor=HUMAN).kind == DecisionKind.SCOPE  # type: ignore[arg-type]

    def test_the_log_file_is_canonical_ascii_jsonl_with_lf_only(
        self, log: DecisionLog, log_path: Path
    ) -> None:
        log.record(
            DecisionKind.DESIGN, "na\u00efve\u2028summary", actor=HUMAN, data={"\u00e9": "\u00fc"}
        )
        raw = log_path.read_bytes()
        assert raw.endswith(b"\n") and raw.count(b"\n") == 1 and b"\r" not in raw
        assert raw.isascii()
        assert json.loads(raw)["id"] == "D-0001"

    def test_nothing_is_written_for_a_rejected_decision(
        self, log: DecisionLog, log_path: Path
    ) -> None:
        with pytest.raises(ConfigError):
            log.record(DecisionKind.DESIGN, "", actor=HUMAN)
        assert not log_path.exists()
        assert log.all() == ()
        assert (
            log.record(DecisionKind.DESIGN, "ok", actor=HUMAN).id == "D-0001"
        )  # the id was not consumed

    def test_repeated_refs_parents_inputs_and_outputs_are_deduplicated_in_order(
        self, log: DecisionLog
    ) -> None:
        a = log.record(DecisionKind.DESIGN, "a", actor=HUMAN)
        d = log.record(
            DecisionKind.DESIGN,
            "b",
            actor=HUMAN,
            parents=(a.id, a.id),
            refs=("R-001", "R-002", "R-001"),
            inputs=(ref("x"), ref("x")),
            outputs=(ref("y"), ref("y")),
        )
        assert (d.parents, d.refs, d.inputs, d.outputs) == (
            (a.id,),
            ("R-001", "R-002"),
            (ref("x"),),
            (ref("y"),),
        )

    def test_data_is_canonicalised_and_immutable(self, log: DecisionLog) -> None:
        d = log.record(DecisionKind.DESIGN, "x", actor=HUMAN, data={1: {"a": (1, 2)}})
        assert dict(d.data) == {"1": {"a": (1, 2)}}
        with pytest.raises(TypeError):
            d.data["new"] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            d.data["1"]["a"] = 1  # type: ignore[index]


class TestSanitisation:
    def test_summary_is_single_line_control_free_and_capped(self, log: DecisionLog) -> None:
        d = log.record(
            DecisionKind.DESIGN, "\x1b[31mred\x1b[0m\r\nline\u202e\u200b" + "x" * 2000, actor=HUMAN
        )
        assert "\x1b" not in d.summary and "\n" not in d.summary and "\r" not in d.summary
        assert "\u202e" not in d.summary and "\u200b" not in d.summary
        assert len(d.summary) <= MAX_SUMMARY_CHARS
        assert d.summary.startswith("redline")

    def test_rationale_keeps_newlines_but_not_escape_sequences(self, log: DecisionLog) -> None:
        d = log.record(
            DecisionKind.DESIGN,
            "x",
            actor=HUMAN,
            rationale="line 1\n\x1b]0;title\x07line 2\x00" + "y" * 9000,
        )
        assert d.rationale.startswith("line 1\nline 2")
        assert "\x1b" not in d.rationale and "\x00" not in d.rationale
        assert len(d.rationale) <= MAX_RATIONALE_CHARS

    def test_alternatives_and_outcome_are_sanitised(self, log: DecisionLog) -> None:
        d = log.record(
            DecisionKind.DESIGN, "x", actor=HUMAN, alternatives=("a\x1b[1m\n",), outcome="o\x07k"
        )
        assert d.alternatives == ("a",)
        assert d.outcome == "ok"

    def test_a_summary_that_sanitises_to_nothing_is_rejected(self, log: DecisionLog) -> None:
        with pytest.raises(ConfigError, match="summary"):
            log.record(DecisionKind.DESIGN, "\x1b[0m\u200b", actor=HUMAN)


class TestValidation:
    @pytest.mark.parametrize("kind", ["nonsense", "", None, 3])
    def test_unknown_kinds_are_rejected(self, log: DecisionLog, kind: Any) -> None:
        with pytest.raises(ConfigError, match="invalid decision"):
            log.record(kind, "x", actor=HUMAN)

    @pytest.mark.parametrize("actor_id", ["", " bad", "a b", "x" * 101, "a\nb", "-x"])
    def test_actor_ids_are_validated(self, log: DecisionLog, actor_id: str) -> None:
        with pytest.raises(ConfigError, match="actor"):
            log.record(DecisionKind.DESIGN, "x", actor=Actor(ActorType.HUMAN, actor_id))

    def test_actor_must_be_an_actor(self, log: DecisionLog) -> None:
        with pytest.raises(ConfigError, match="actor"):
            log.record(DecisionKind.DESIGN, "x", actor="alice")  # type: ignore[arg-type]

    @pytest.mark.parametrize("node_id", ["Design", "a:b", "../x", "", "n\n"])
    def test_node_ids_are_validated(self, log: DecisionLog, node_id: str) -> None:
        with pytest.raises(ConfigError, match="node id"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, node_id=node_id)

    @pytest.mark.parametrize("bad", ["", "../x", "R 1", "R-1\n", "x" * 65, "\x1b[0m", "-R"])
    def test_refs_are_validated_not_sanitised(self, log: DecisionLog, bad: str) -> None:
        with pytest.raises(ConfigError, match="refs"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, refs=(bad,))

    def test_counts_are_bounded(self, log: DecisionLog) -> None:
        with pytest.raises(ConfigError, match="alternatives"):
            log.record(
                DecisionKind.DESIGN,
                "x",
                actor=HUMAN,
                alternatives=tuple(f"a{i}" for i in range(21)),
            )
        with pytest.raises(ConfigError, match="refs"):
            log.record(
                DecisionKind.DESIGN, "x", actor=HUMAN, refs=tuple(f"R-{i:03d}" for i in range(51))
            )
        with pytest.raises(ConfigError, match="inputs"):
            log.record(
                DecisionKind.DESIGN,
                "x",
                actor=HUMAN,
                inputs=tuple(ref(version=i) for i in range(1, 202)),
            )

    def test_artifact_refs_must_be_artifact_refs(self, log: DecisionLog) -> None:
        with pytest.raises(ConfigError, match="inputs"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, inputs=("design_doc@v1",))  # type: ignore[arg-type]
        with pytest.raises(ConfigError, match="outputs"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, outputs=(5,))  # type: ignore[arg-type]
        with pytest.raises(ConfigError):  # unhashable entries are rejected too
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, outputs=({"artifact_id": "x"},))  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", ["alternatives", "inputs", "outputs", "parents", "refs"])
    def test_a_bare_string_is_not_a_sequence(self, log: DecisionLog, field: str) -> None:
        with pytest.raises(ConfigError, match="must be a sequence"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, **{field: "R-001"})

    def test_parents_must_exist_and_be_well_formed(self, log: DecisionLog) -> None:
        with pytest.raises(StateError, match="unknown parent"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, parents=("D-0042",))
        with pytest.raises(ConfigError, match="parents"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, parents=("../etc",))
        assert log.all() == ()

    @pytest.mark.parametrize(
        "data", [{"a": object()}, {"a": float("nan")}, {"a": float("inf")}, [1, 2], "str"]
    )
    def test_data_must_be_a_json_mapping(self, log: DecisionLog, data: Any) -> None:
        with pytest.raises(ConfigError, match="invalid decision"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, data=data)

    def test_data_is_size_bounded(self, log: DecisionLog) -> None:
        with pytest.raises(ConfigError, match="data"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, data={"x": "y" * 20_000})

    def test_deeply_nested_data_is_rejected_without_crashing(self, log: DecisionLog) -> None:
        deep: Any = 1
        for _ in range(3000):
            deep = {"k": deep}
        with pytest.raises(ConfigError):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN, data=deep)

    def test_run_id_is_validated(self, log_path: Path) -> None:
        for bad in ("", "../x", "a b", "run\n", "9run"):
            with pytest.raises(ConfigError, match="run id"):
                DecisionLog(log_path, run_id=bad, clock=StepClock())

    def test_a_naive_clock_is_rejected(self, log_path: Path) -> None:
        naive = DecisionLog(log_path, run_id="run1", clock=StepClock(naive=True))
        with pytest.raises(ConfigError, match="timezone-aware"):
            naive.record(DecisionKind.DESIGN, "x", actor=HUMAN)

    def test_the_log_is_bounded(self, log: DecisionLog, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(decisions_module, "MAX_DECISIONS", 2)
        log.record(DecisionKind.DESIGN, "a", actor=HUMAN)
        log.record(DecisionKind.DESIGN, "b", actor=HUMAN)
        with pytest.raises(StateError, match="full"):
            log.record(DecisionKind.DESIGN, "c", actor=HUMAN)


# =============================================================================
# audit binding and durability
# =============================================================================


class TestAuditBinding:
    def test_decision_recorded_event_carries_id_kind_and_hash(
        self, log: DecisionLog, sink: RecordingSink
    ) -> None:
        d = log.record(DecisionKind.REPLAN, "re-plan", actor=AGENT, node_id="decompose")
        ((etype, node_id, attempt, actor, data),) = sink.calls
        assert etype == EventType.DECISION_RECORDED
        assert (node_id, attempt, actor) == ("decompose", None, AGENT)
        assert data == {
            "decision_id": "D-0001",
            "kind": "replan",
            "decision_hash": hash_jsonable(d),
        }
        assert hash_jsonable(d) == hash_jsonable(d.to_dict())

    def test_the_hash_can_be_re_derived_from_the_file(
        self, log: DecisionLog, log_path: Path, sink: RecordingSink
    ) -> None:
        d = log.record(DecisionKind.DESIGN, "x", actor=HUMAN, data={"n": 1}, outputs=(ref(),))
        loaded = Decision.from_dict(json.loads(log_path.read_bytes()))
        assert hash_jsonable(loaded) == sink.calls[0][4]["decision_hash"] == hash_jsonable(d)

    def test_the_event_is_emitted_only_after_the_decision_is_durable(self, log_path: Path) -> None:
        seen: list[int] = []

        def sink(type: EventType, **kw: Any) -> None:
            seen.append(len(lines_of(log_path)))

        durable = DecisionLog(log_path, run_id="run1", clock=StepClock(), audit=sink)  # type: ignore[arg-type]
        durable.record(DecisionKind.DESIGN, "a", actor=HUMAN)
        durable.record(DecisionKind.DESIGN, "b", actor=HUMAN)
        assert seen == [1, 2]

    def test_a_failing_sink_leaves_the_decision_recorded(self, log_path: Path) -> None:
        def broken(type: EventType, **kw: Any) -> None:
            raise OSError("audit disk full")

        failing = DecisionLog(log_path, run_id="run1", clock=StepClock(), audit=broken)  # type: ignore[arg-type]
        with pytest.raises(OSError, match="audit disk full"):
            failing.record(DecisionKind.DESIGN, "x", actor=HUMAN)
        assert [d.id for d in reopen(log_path).all()] == [
            "D-0001"
        ]  # the audit chain will flag the gap

    def test_a_failed_append_leaves_memory_and_audit_untouched(
        self, log: DecisionLog, sink: RecordingSink, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(path: Any, line: bytes, *, fsync: bool = False) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(decisions_module, "append_line", boom)
        with pytest.raises(OSError, match="disk full"):
            log.record(DecisionKind.DESIGN, "x", actor=HUMAN)
        monkeypatch.undo()
        assert log.all() == () and sink.calls == []
        assert log.record(DecisionKind.DESIGN, "x", actor=HUMAN).id == "D-0001"

    def test_audit_is_optional(self, log_path: Path) -> None:
        quiet = DecisionLog(log_path, run_id="run1", clock=StepClock())
        assert quiet.record(DecisionKind.DESIGN, "x", actor=HUMAN).id == "D-0001"


# =============================================================================
# queries
# =============================================================================


class TestQueries:
    def test_lookup_helpers(self, log: DecisionLog) -> None:
        a = log.record(
            DecisionKind.DESIGN, "a", actor=AGENT, node_id="design", outputs=(ref("design_doc"),)
        )
        b = log.record(DecisionKind.ASSUMPTION, "b", actor=AGENT, node_id="requirements")
        c = log.record(
            DecisionKind.DESIGN,
            "c",
            actor=HUMAN,
            node_id="design",
            outputs=(ref("design_doc"), ref("api_contract")),
        )
        assert log.all() == (a, b, c)
        assert (log.get("D-0002"), len(log)) == (b, 3)
        assert log.for_node("design") == (a, c)
        assert log.for_node("nope") == ()
        assert log.of_kind(DecisionKind.DESIGN) == (a, c)
        assert log.of_kind(DecisionKind.SAFE_STOP) == ()
        assert log.decisions_for(ref("design_doc")) == (a, c)
        assert log.decisions_for(ref("api_contract")) == (c,)
        assert log.decisions_for(ref("design_doc", version=2)) == ()
        assert "D-0001" in log and "D-0009" not in log and 5 not in log

    def test_results_are_immutable_snapshots(self, log: DecisionLog) -> None:
        log.record(DecisionKind.DESIGN, "a", actor=HUMAN)
        snapshot = log.all()
        log.record(DecisionKind.DESIGN, "b", actor=HUMAN)
        assert len(snapshot) == 1 and isinstance(snapshot, tuple)

    @pytest.mark.parametrize("bad", ["D-9999", "", "../x", "D-0001\x1b[31m", None, 1])
    def test_unknown_ids_raise_state_error_with_a_sanitised_message(
        self, log: DecisionLog, bad: Any
    ) -> None:
        with pytest.raises(StateError, match="unknown decision") as err:
            log.get(bad)
        assert "\x1b" not in str(err.value)


# =============================================================================
# persistence, corruption, torn tail
# =============================================================================


def _line(index: int, **override: Any) -> bytes:
    d = Decision(
        id=f"D-{index:04d}",
        run_id="run1",
        ts=T0,
        kind=DecisionKind.DESIGN,
        actor=HUMAN,
        node_id=None,
        summary=f"s{index}",
    ).to_dict()
    d.update(override)
    return canonical_json(d)


def _write(path: Path, *lines: bytes, tail: bytes = b"") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(line + b"\n" for line in lines) + tail)


class TestPersistence:
    def test_reopening_restores_everything_and_continues_the_sequence(
        self, log: DecisionLog, log_path: Path
    ) -> None:
        a = log.record(DecisionKind.DESIGN, "a", actor=HUMAN, outputs=(ref(),), data={"k": 1})
        b = log.record(DecisionKind.RETRY, "b", actor=AGENT, parents=(a.id,), node_id="design")
        again = reopen(log_path)
        assert again.all() == (a, b)
        assert again.decisions_for(ref()) == (a,)
        assert again.repaired_tail is False
        assert again.record(DecisionKind.DESIGN, "c", actor=HUMAN, parents=(b.id,)).id == "D-0003"
        assert [d.id for d in reopen(log_path).all()] == ["D-0001", "D-0002", "D-0003"]

    def test_a_missing_or_empty_file_is_an_empty_log(self, log_path: Path) -> None:
        assert reopen(log_path).all() == ()
        log_path.parent.mkdir(parents=True)
        log_path.write_bytes(b"")
        assert reopen(log_path).all() == ()

    def test_the_log_is_append_only(self, log: DecisionLog, log_path: Path) -> None:
        log.record(DecisionKind.DESIGN, "a", actor=HUMAN)
        before = log_path.read_bytes()
        log.record(DecisionKind.DESIGN, "b", actor=HUMAN)
        assert log_path.read_bytes().startswith(before)

    @pytest.mark.parametrize(
        ("lines", "why"),
        [
            ([b"not json"], "corrupt"),
            ([b"[]"], "corrupt"),
            ([b'"x"'], "corrupt"),
            ([b"{}"], "corrupt"),
            ([b""], "corrupt"),
            ([_line(1), b""], "corrupt"),  # an empty line in the middle
            ([_line(2)], "expected sequence"),
            ([_line(1), _line(1)], "expected sequence"),
            ([_line(1), _line(3)], "expected sequence"),
            ([_line(1, run_id="other")], "another run"),
            ([_line(1, parents=["D-0002"])], "unknown parent"),
            (
                [_line(1, parents=["D-0001"])],
                "unknown parent",
            ),  # a decision cannot be its own parent
            ([_line(1, summary="x\x1b[31m")], "control"),
            ([_line(1, summary="")], "summary"),
            ([_line(1, summary="s" * 501)], "summary"),
            ([_line(1, kind="mystery")], "corrupt"),
            ([_line(1, ts="2026-01-01T00:00:00")], "corrupt"),  # naive timestamp
            ([_line(1, actor={"type": "human", "id": "a b"})], "actor"),
            ([_line(1, node_id="../x")], "node id"),
            ([_line(1, refs=["../x"])], "refs"),
            ([_line(1, data={"x": "y" * 20000})], "data"),
            ([b'{"id": "D-0001", "id": "D-0001"}'], "duplicate"),
            ([_line(1).replace(b'"data":{}', b'"data":{"n":NaN}')], "non-finite"),
            ([_line(1).replace(b'"data":{}', b'"data":{"n":1e999}')], "non-finite"),
            ([b" " * 300_000], "too large"),
        ],
    )
    def test_corrupt_or_forged_logs_are_rejected(
        self, log_path: Path, lines: list[bytes], why: str
    ) -> None:
        _write(log_path, *lines)
        with pytest.raises(IntegrityError, match=why):
            reopen(log_path)

    def test_the_loaded_log_is_deeply_immutable(self, log_path: Path) -> None:
        _write(log_path, _line(1, data={"a": {"b": [1]}}))
        loaded = reopen(log_path).get("D-0001")
        with pytest.raises(TypeError):
            loaded.data["a"]["b"] = 2  # type: ignore[index]


class TestTornTail:
    def test_a_partial_final_line_is_dropped_and_the_file_repaired(
        self, log_path: Path, sink: RecordingSink
    ) -> None:
        _write(log_path, _line(1), _line(2), tail=_line(3)[:25])
        repaired = reopen(log_path, audit=sink)
        assert repaired.repaired_tail is True
        assert [d.id for d in repaired.all()] == ["D-0001", "D-0002"]
        assert log_path.read_bytes() == _line(1) + b"\n" + _line(2) + b"\n"
        assert repaired.record(DecisionKind.DESIGN, "next", actor=HUMAN).id == "D-0003"
        assert [d.id for d in reopen(log_path).all()] == ["D-0001", "D-0002", "D-0003"]

    def test_a_complete_but_unterminated_final_line_was_never_announced_either(
        self, log_path: Path
    ) -> None:
        _write(log_path, _line(1), tail=_line(2))  # no trailing newline: the append never finished
        repaired = reopen(log_path)
        assert repaired.repaired_tail is True
        assert [d.id for d in repaired.all()] == ["D-0001"]

    def test_a_file_that_is_only_a_torn_line_becomes_empty(self, log_path: Path) -> None:
        _write(log_path, tail=b'{"id": "D-0001", "ru')
        repaired = reopen(log_path)
        assert repaired.repaired_tail is True and repaired.all() == ()
        assert log_path.read_bytes() == b""

    def test_corruption_before_the_tail_is_still_an_error(self, log_path: Path) -> None:
        _write(log_path, _line(1), b"garbage", tail=b"{")
        with pytest.raises(IntegrityError, match="line 2"):
            reopen(log_path)


# =============================================================================
# concurrency
# =============================================================================


class TestConcurrency:
    def test_concurrent_records_get_unique_contiguous_ids_and_ordered_events(
        self, log: DecisionLog, log_path: Path, sink: RecordingSink
    ) -> None:
        threads, per_thread = 6, 8
        errors: list[BaseException] = []
        barrier = threading.Barrier(threads)

        def work(i: int) -> None:
            try:
                barrier.wait(timeout=10)
                for n in range(per_thread):
                    log.record(
                        DecisionKind.RETRY,
                        f"t{i} n{n}",
                        actor=AGENT,
                        node_id="design",
                        refs=(f"R-{i:03d}",),
                    )
            except BaseException as exc:
                errors.append(exc)

        workers = [threading.Thread(target=work, args=(i,)) for i in range(threads)]
        for w in workers:
            w.start()
        for w in workers:
            w.join(timeout=60)
        assert errors == []
        total = threads * per_thread
        ids = [d.id for d in log.all()]
        assert ids == [f"D-{n:04d}" for n in range(1, total + 1)]
        assert [c[4]["decision_id"] for c in sink.calls] == ids  # audit order == id order
        assert [d.id for d in reopen(log_path).all()] == ids
        assert len(lines_of(log_path)) == total

    def test_readers_can_query_while_writers_append(self, log: DecisionLog) -> None:
        stop = threading.Event()
        errors: list[BaseException] = []

        def reader() -> None:
            try:
                while not stop.is_set():
                    snapshot = log.all()
                    assert [d.id for d in snapshot] == [
                        f"D-{n:04d}" for n in range(1, len(snapshot) + 1)
                    ]
                    log.of_kind(DecisionKind.RETRY)
            except BaseException as exc:
                errors.append(exc)

        watchers = [threading.Thread(target=reader) for _ in range(3)]
        for w in watchers:
            w.start()
        for n in range(25):
            log.record(DecisionKind.RETRY, f"n{n}", actor=AGENT)
        stop.set()
        for w in watchers:
            w.join(timeout=10)
        assert errors == []
