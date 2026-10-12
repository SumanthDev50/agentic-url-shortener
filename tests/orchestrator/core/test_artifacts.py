"""ArtifactStore: layout, versioning, publication, quarantine, integrity, pinned reads, threads."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core import artifacts as artifacts_module
from orchestrator.core.artifacts import TOMBSTONE, ArtifactStore, PinnedReader
from orchestrator.core.errors import (
    ConfigError,
    IntegrityError,
    StateError,
    UndeclaredInputError,
)
from orchestrator.core.types import (
    EXTERNAL_PRODUCER,
    Actor,
    ActorType,
    Artifact,
    ArtifactDraft,
    ArtifactReader,
    ArtifactRef,
    ArtifactStatus,
    AuditEvent,
    EventType,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


class StepClock:
    """Deterministic clock: every ``now()`` is one second later (unique timestamps)."""

    def __init__(self) -> None:
        self._t = T0
        self._lock = threading.Lock()

    def now(self) -> datetime:
        with self._lock:
            self._t += timedelta(seconds=1)
            return self._t

    def monotonic(self) -> float:
        return 0.0

    def sleep(self, seconds: float) -> None:
        return None


class NaiveClock(StepClock):
    def now(self) -> datetime:
        return datetime(2026, 1, 1)  # deliberately naive


class Record:
    def __init__(
        self,
        type: EventType,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: dict[str, Any],
    ) -> None:
        self.type = type
        self.node_id = node_id
        self.attempt = attempt
        self.actor = actor
        self.data = data


class RecordingSink:
    """An ``AuditSink`` that remembers every call (thread-safe)."""

    def __init__(self) -> None:
        self.records: list[Record] = []
        self._lock = threading.Lock()

    def __call__(
        self,
        type: EventType,
        *,
        node_id: str | None,
        attempt: int | None,
        actor: Actor,
        data: Any,
    ) -> None:
        with self._lock:
            self.records.append(Record(type, node_id, attempt, actor, dict(data)))

    def of(self, type: EventType) -> list[Record]:
        return [r for r in self.records if r.type == type]

    def audit_events(self) -> list[AuditEvent]:
        zero = "0" * 64
        return [
            AuditEvent(
                i + 1,
                T0,
                "run1",
                "t",
                "run",
                None,
                r.type,
                r.node_id,
                r.attempt,
                r.actor,
                r.data,
                zero,
                zero,
            )
            for i, r in enumerate(self.records)
        ]


def make_dir_link(link: Path, target: Path) -> bool:
    """Create a directory symlink, or a junction on Windows; False if the OS forbids both."""
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if sys.platform != "win32":
            return False
        done = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True, check=False
        )
        return done.returncode == 0
    return True


def draft(kind: str = "design_doc", content: Any = None, key: str = "", **kw: Any) -> ArtifactDraft:
    return ArtifactDraft(kind, {"schema": 1, "v": 1} if content is None else content, key=key, **kw)


@pytest.fixture(autouse=True)
def _no_fsync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Durability is not under test here and fsync dominates the run time on Windows."""
    monkeypatch.setattr(os, "fsync", lambda fd: None)


@pytest.fixture
def sink() -> RecordingSink:
    return RecordingSink()


@pytest.fixture
def store(tmp_path: Path, sink: RecordingSink) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts", clock=StepClock(), audit=sink)


def reopen(store: ArtifactStore, **kw: Any) -> ArtifactStore:
    return ArtifactStore(store.root, clock=StepClock(), **kw)


def stage(
    store: ArtifactStore, kind: str = "design_doc", content: Any = None, **kw: Any
) -> Artifact:
    key = kw.pop("key", "")
    producer = kw.pop("producer", "design")
    return store.put(draft(kind, content, key), producer=producer, staged=True, **kw)


def publish(
    store: ArtifactStore, kind: str = "design_doc", content: Any = None, **kw: Any
) -> Artifact:
    art = stage(store, kind, content, **kw)
    return store.activate([art.id])[0]


# =============================================================================
# put: layout and content
# =============================================================================


class TestPutLayout:
    def test_text_bytes_and_json_are_stored_verbatim_or_canonically(
        self, store: ArtifactStore
    ) -> None:
        text = store.put(
            ArtifactDraft("request", "hello\r\nworld\n", media_type="text/markdown"),
            producer=EXTERNAL_PRODUCER,
        )
        raw = store.put(ArtifactDraft("release_patch", b"\x00\x01binary"), producer="promote")
        js = store.put(
            ArtifactDraft("design_doc", {"b": 2, "a": [1, {"z": None}]}), producer="design"
        )
        assert store.read_bytes(text) == b"hello\r\nworld\n"  # no newline normalisation
        assert store.read_bytes(raw) == b"\x00\x01binary"
        assert (
            store.read_bytes(js)
            == b'{\n  "a": [\n    1,\n    {\n      "z": null\n    }\n  ],\n  "b": 2\n}\n'
        )
        assert (text.media_type, raw.media_type, js.media_type) == (
            "text/markdown",
            "application/octet-stream",
            "application/json",
        )

    def test_json_bytes_are_ascii_and_sorted_so_hashes_are_stable(
        self, store: ArtifactStore
    ) -> None:
        a = store.put(
            ArtifactDraft("design_doc", {"k": "\u00e9\u2028", "a": 1}),
            producer="design",
            staged=True,
        )
        b = store.put(
            ArtifactDraft("design_doc", {"a": 1, "k": "\u00e9\u2028"}),
            producer="design",
            staged=True,
        )
        assert a.content_hash == b.content_hash
        assert store.read_bytes(a).isascii()
        assert store.read_json(a) == {"a": 1, "k": "\u00e9\u2028"}

    def test_storage_path_uses_slug_and_hash_never_the_raw_key(self, store: ArtifactStore) -> None:
        art = stage(
            store, "changeset", {"ops": []}, key="implement.core", producer="implement.core"
        )
        digest = hashlib.sha256(b"implement.core").hexdigest()[:8]
        assert art.path == f"changeset/implement-core-{digest}/v1/content.json"
        assert art.id == "changeset.implement.core@v1"
        assert (store.root / art.path).is_file()

    def test_empty_key_has_a_valid_path(self, store: ArtifactStore) -> None:
        art = stage(store, "request", "x", producer=EXTERNAL_PRODUCER)
        assert art.path == f"request/_-{hashlib.sha256(b'').hexdigest()[:8]}/v1/content.txt"

    def test_keys_differing_only_in_case_get_distinct_directories(
        self, store: ArtifactStore
    ) -> None:
        a = stage(store, "changeset", {"ops": []}, key="Core", producer="implement.core")
        b = stage(store, "changeset", {"ops": []}, key="core", producer="implement.core")
        assert (
            a.path.split("/")[1] != b.path.split("/")[1]
        )  # case-insensitive file systems stay safe

    def test_agent_filename_is_metadata_only_and_never_a_path_component(
        self, store: ArtifactStore
    ) -> None:
        art = store.put(
            ArtifactDraft("release_patch", "diff", filename="evil-name.py"),
            producer="promote",
        )
        assert "evil-name" not in art.path
        assert art.path.endswith("/content.py")  # only the allow-listed suffix is honoured
        assert art.metadata["filename"] == "evil-name.py"

    @pytest.mark.parametrize(
        ("kwargs", "ext"),
        [
            ({"media_type": "text/markdown"}, "md"),
            ({"media_type": "application/sql"}, "sql"),
            ({"media_type": "text/x-patch"}, "patch"),
            ({"filename": "x.diff"}, "diff"),
            ({"filename": "x.exe"}, "txt"),
            ({"filename": "noext"}, "txt"),
            ({}, "txt"),
        ],
    )
    def test_extension_comes_from_a_fixed_allow_list(
        self, store: ArtifactStore, kwargs: dict[str, str], ext: str
    ) -> None:
        art = store.put(
            ArtifactDraft("release_patch", "x", **kwargs), producer="promote", staged=True
        )
        assert art.path.endswith(f"/content.{ext}")

    def test_versions_are_monotonic_per_kind_and_key(self, store: ArtifactStore) -> None:
        versions = [stage(store, "design_doc", {"n": i}).version for i in range(3)]
        other = stage(store, "task_plan", {"n": 0}, producer="decompose")
        assert versions == [1, 2, 3]
        assert other.version == 1
        assert [a.id for a in store.versions("design_doc")] == [
            "design_doc@v1",
            "design_doc@v2",
            "design_doc@v3",
        ]

    def test_created_at_and_provenance_are_recorded(self, store: ArtifactStore) -> None:
        ref = ArtifactRef("request@v1", "request", "", 1, "a" * 64)
        art = stage(store, attempt=3, agent_id="architect", inputs=(ref,))
        assert art.created_at.tzinfo is not None
        assert (art.producer, art.attempt, art.agent_id, art.inputs) == (
            "design",
            3,
            "architect",
            (ref,),
        )
        assert art.status == ArtifactStatus.STAGED

    def test_size_and_hash_describe_the_stored_bytes(self, store: ArtifactStore) -> None:
        art = stage(store, "release_patch", "h\u00e9llo", producer="promote")
        data = "h\u00e9llo".encode()
        assert (art.size, art.content_hash) == (len(data), hashlib.sha256(data).hexdigest())


class TestPutValidation:
    def test_naive_clock_is_rejected(self, tmp_path: Path) -> None:
        bad = ArtifactStore(tmp_path / "a", clock=NaiveClock())
        with pytest.raises(ConfigError, match="timezone-aware"):
            bad.put(draft(), producer="design")

    @pytest.mark.parametrize("producer", ["", "Design", "a:b", "../x", "design\n", "x" * 100])
    def test_producer_must_be_a_node_id_or_external(
        self, store: ArtifactStore, producer: str
    ) -> None:
        with pytest.raises(ConfigError, match="producer"):
            store.put(draft(), producer=producer)

    @pytest.mark.parametrize("attempt", [-1, True, 1.5])
    def test_attempt_must_be_a_non_negative_int(self, store: ArtifactStore, attempt: Any) -> None:
        with pytest.raises(ConfigError, match="attempt"):
            store.put(draft(), producer="design", attempt=attempt)

    @pytest.mark.parametrize("agent_id", ["bad id", "a/b", "\x1b[31m", "x" * 100, "-lead"])
    def test_agent_id_is_validated(self, store: ArtifactStore, agent_id: str) -> None:
        with pytest.raises(ConfigError, match="agent id"):
            store.put(draft(), producer="design", agent_id=agent_id)

    def test_dotted_agent_ids_are_allowed(self, store: ArtifactStore) -> None:
        assert stage(store, agent_id="implementer.safe").agent_id == "implementer.safe"

    def test_inputs_must_be_artifact_refs_and_bounded(self, store: ArtifactStore) -> None:
        with pytest.raises(ConfigError, match="ArtifactRef"):
            store.put(draft(), producer="design", inputs=("request@v1",))  # type: ignore[arg-type]
        ref = ArtifactRef("request@v1", "request", "", 1, "a" * 64)
        with pytest.raises(ConfigError, match="too many"):
            store.put(draft(), producer="design", inputs=(ref,) * 501)

    def test_mutated_draft_is_revalidated(self, store: ArtifactStore) -> None:
        d = draft()
        d.kind = "../etc"
        with pytest.raises(ConfigError, match="kind"):
            store.put(d, producer="design")
        d2 = draft()
        d2.key = "a/b"
        with pytest.raises(ConfigError, match="key"):
            store.put(d2, producer="design")
        d3 = draft()
        d3.filename = "../x"
        with pytest.raises(ConfigError, match="filename"):
            store.put(d3, producer="design")

    def test_content_limits_and_types(self, tmp_path: Path) -> None:
        small = ArtifactStore(tmp_path / "s", clock=StepClock(), max_artifact_bytes=64)
        small.put(ArtifactDraft("release_patch", "x" * 64), producer="promote")
        with pytest.raises(ConfigError, match="exceeds"):
            small.put(ArtifactDraft("release_patch", "x" * 65), producer="promote")
        with pytest.raises(ConfigError, match="max_artifact_bytes"):
            ArtifactStore(tmp_path / "z", clock=StepClock(), max_artifact_bytes=0)

    @pytest.mark.parametrize(
        "content",
        ["\ud800", {"a": float("nan")}, {"a": float("inf")}, 42, None, {"a": object()}],
    )
    def test_unstorable_content_is_rejected(self, store: ArtifactStore, content: Any) -> None:
        with pytest.raises(ConfigError):
            store.put(ArtifactDraft("design_doc", content), producer="design")
        assert store.all(include_inactive=True) == []

    def test_deeply_nested_json_is_rejected_because_read_json_could_not_read_it(
        self, store: ArtifactStore
    ) -> None:
        deep: Any = "x"
        for _ in range(100):
            deep = {"k": deep}
        with pytest.raises(ConfigError, match="not acceptable"):
            store.put(ArtifactDraft("design_doc", deep), producer="design")

    @pytest.mark.parametrize(
        "media", ["text", "TEXT/ plain", "a/b;charset=x", "../../x/y", "x/y\n", "a/" + "b" * 80]
    )
    def test_media_type_is_validated(self, store: ArtifactStore, media: str) -> None:
        with pytest.raises(ConfigError, match="media type"):
            store.put(ArtifactDraft("release_patch", "x", media_type=media), producer="promote")

    def test_windows_reserved_kind_names_cannot_be_stored(self, store: ArtifactStore) -> None:
        with pytest.raises(ConfigError, match="cannot be stored"):
            store.put(ArtifactDraft("con", "x"), producer="design")
        assert store.all(include_inactive=True) == []


class TestMetadata:
    def test_reserved_keys_are_stripped_and_cannot_forge_store_flags(
        self, store: ArtifactStore
    ) -> None:
        art = store.put(
            ArtifactDraft(
                "design_doc",
                {"a": 1},
                metadata={"_tombstone": True, "_deduped_to": "x@v1", "note": "ok", "n": (1, 2)},
            ),
            producer="design",
            staged=True,
        )
        assert dict(art.metadata) == {"note": "ok", "n": (1, 2)}
        assert store.read_json(art) == {"a": 1}  # a forged _tombstone did not disable reads

    def test_metadata_is_canonical_json_and_excluded_from_the_content_hash(
        self, store: ArtifactStore
    ) -> None:
        a = store.put(
            ArtifactDraft("design_doc", {"a": 1}, metadata={"took_s": 1.5}),
            producer="design",
            staged=True,
        )
        b = store.put(
            ArtifactDraft("design_doc", {"a": 1}, metadata={"took_s": 9.5}),
            producer="design",
            staged=True,
        )
        assert a.content_hash == b.content_hash
        assert reopen(store).get(a.id).metadata["took_s"] == 1.5

    def test_metadata_is_deeply_immutable(self, store: ArtifactStore) -> None:
        art = store.put(
            ArtifactDraft("design_doc", {"a": 1}, metadata={"deep": {"x": [1, 2]}}),
            producer="design",
        )
        with pytest.raises(TypeError):
            art.metadata["new"] = 1  # type: ignore[index]
        with pytest.raises(TypeError):
            art.metadata["deep"]["x"] = 1  # type: ignore[index]
        assert art.metadata["deep"]["x"] == (1, 2)

    def test_hostile_metadata_is_rejected(self, store: ArtifactStore) -> None:
        with pytest.raises(ConfigError, match="metadata"):
            store.put(
                ArtifactDraft("design_doc", {"a": 1}, metadata={"x": "y" * 20000}),
                producer="design",
            )
        with pytest.raises(ConfigError, match="metadata"):
            store.put(
                ArtifactDraft("design_doc", {"a": 1}, metadata={"x": float("nan")}),
                producer="design",
            )
        with pytest.raises(ConfigError, match="metadata"):
            store.put(
                ArtifactDraft("design_doc", {"a": 1}, metadata={"x": object()}), producer="design"
            )


# =============================================================================
# publication
# =============================================================================


class TestStagedAndDirectPut:
    def test_staged_artifacts_are_invisible_downstream(self, store: ArtifactStore) -> None:
        art = stage(store)
        assert art.status == ArtifactStatus.STAGED
        assert store.latest("design_doc") is None
        assert store.active("design_doc") == []
        assert [a.id for a in store.staged()] == [art.id]

    def test_direct_put_is_published_and_supersedes_the_previous_version(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        first = store.put(draft("request", "v1"), producer=EXTERNAL_PRODUCER)
        second = store.put(draft("request", "v2"), producer=EXTERNAL_PRODUCER)
        assert first.status == second.status == ArtifactStatus.ACTIVE
        assert store.latest("request") == second
        assert store.get(first.id).status == ArtifactStatus.SUPERSEDED
        types = [r.type for r in sink.records]
        assert types == [
            EventType.ARTIFACT_CREATED,
            EventType.ARTIFACT_PUBLISHED,
            EventType.ARTIFACT_CREATED,
            EventType.ARTIFACT_PUBLISHED,
            EventType.ARTIFACT_SUPERSEDED,
        ]

    def test_direct_put_refuses_a_foreign_producer_before_writing_anything(
        self, store: ArtifactStore
    ) -> None:
        publish(store, producer="design")
        before = sorted(os.listdir(store.root))
        with pytest.raises(StateError, match="owned by"):
            store.put(draft(), producer="decompose")
        assert sorted(os.listdir(store.root)) == before
        assert [a.id for a in store.versions("design_doc")] == ["design_doc@v1"]

    def test_keyed_kinds_hold_one_active_artifact_per_key_sorted_by_key(
        self, store: ArtifactStore
    ) -> None:
        for key in ("implement.b", "implement.a", "author-tests"):
            publish(store, "changeset", {"ops": [], "k": key}, key=key, producer=key)
        assert [a.key for a in store.active("changeset")] == [
            "author-tests",
            "implement.a",
            "implement.b",
        ]
        assert store.latest("changeset", "implement.a") is not None
        assert store.latest("changeset") is None  # key "" is a different (kind, key)


class TestActivate:
    def test_activation_publishes_and_supersedes_the_previous_active(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        v1 = publish(store, content={"v": 1})
        v2 = stage(store, content={"v": 2})
        (result,) = store.activate([v2.id])
        assert result.status == ArtifactStatus.ACTIVE
        assert store.latest("design_doc").id == v2.id  # type: ignore[union-attr]
        assert store.get(v1.id).status == ArtifactStatus.SUPERSEDED
        assert store.get(v1.id).metadata["_superseded_by"] == v2.id
        assert [r.data["artifact_id"] for r in sink.of(EventType.ARTIFACT_SUPERSEDED)] == [v1.id]

    def test_activation_is_idempotent_and_emits_nothing_the_second_time(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        art = stage(store)
        first = store.activate([art.id])
        count = len(sink.records)
        second = store.activate([art.id, art.id])
        assert first == second
        assert len(sink.records) == count
        assert len(store.versions("design_doc")) == 1

    def test_results_follow_the_order_of_the_ids(self, store: ArtifactStore) -> None:
        a = stage(store, "design_doc")
        b = stage(store, "task_plan", producer="decompose")
        c = stage(store, "requirements", producer="requirements")
        assert [x.id for x in store.activate([c.id, a.id, b.id])] == [c.id, a.id, b.id]

    def test_activation_is_all_or_nothing(self, store: ArtifactStore) -> None:
        publish(store, "task_plan", producer="decompose")
        good = stage(store, "design_doc")
        foreign = stage(
            store, "task_plan", {"x": 1}, producer="design"
        )  # task_plan is owned by decompose
        with pytest.raises(StateError, match="owned by"):
            store.activate([good.id, foreign.id])
        assert store.get(good.id).status == ArtifactStatus.STAGED
        assert store.get(foreign.id).status == ArtifactStatus.STAGED
        assert reopen(store).get(good.id).status == ArtifactStatus.STAGED

    def test_unknown_ids_are_refused_with_a_sanitised_message(self, store: ArtifactStore) -> None:
        with pytest.raises(StateError) as err:
            store.activate(["design_doc@v9\x1b[31m"])
        assert "\x1b" not in str(err.value)

    @pytest.mark.parametrize("status", ["quarantined", "superseded"])
    def test_artifacts_that_are_not_staged_cannot_be_activated(
        self, store: ArtifactStore, status: str
    ) -> None:
        art = stage(store)
        if status == "quarantined":
            store.quarantine([art.id], "rejected")
        else:
            other = stage(store, content={"v": 2})
            store.activate([art.id])
            store.activate([other.id])  # supersedes art
        with pytest.raises(StateError, match="cannot activate"):
            store.activate([art.id])

    def test_a_bare_string_is_not_an_id_list(self, store: ArtifactStore) -> None:
        with pytest.raises(TypeError, match="not a single string"):
            store.activate("design_doc@v1")
        with pytest.raises(TypeError, match="strings"):
            store.activate([1])  # type: ignore[list-item]

    def test_two_versions_of_one_key_in_one_activation_are_refused(
        self, store: ArtifactStore
    ) -> None:
        a = stage(store, content={"v": 1})
        b = stage(store, content={"v": 2})
        with pytest.raises(StateError, match="two versions"):
            store.activate([a.id, b.id])

    def test_an_older_version_cannot_replace_a_newer_active_one(self, store: ArtifactStore) -> None:
        old = stage(store, content={"v": 1})
        new = stage(store, content={"v": 2})
        store.activate([new.id])
        with pytest.raises(StateError, match="older than"):
            store.activate([old.id])
        assert store.latest("design_doc").id == new.id  # type: ignore[union-attr]

    def test_tampered_staged_content_never_becomes_visible(self, store: ArtifactStore) -> None:
        art = stage(store)
        (store.root / art.path).write_bytes(b"{}")
        with pytest.raises(IntegrityError):
            store.activate([art.id])
        assert store.latest("design_doc") is None
        assert store.get(art.id).status == ArtifactStatus.STAGED


class TestDedupe:
    def test_identical_content_keeps_the_existing_ref_and_emits_unchanged(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        v1 = publish(store, content={"same": True})
        v2 = stage(store, content={"same": True})
        (kept,) = store.activate([v2.id])
        assert kept.id == v1.id
        assert store.latest("design_doc").id == v1.id  # type: ignore[union-attr]
        assert store.get(v2.id).status == ArtifactStatus.SUPERSEDED
        (event,) = sink.of(EventType.ARTIFACT_UNCHANGED)
        assert event.data == {"artifact_id": v2.id, "kept": v1.id, "content_hash": v1.content_hash}
        assert len(sink.of(EventType.ARTIFACT_PUBLISHED)) == 1  # v2 was never published

    def test_deduplicated_activation_is_idempotent(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        v1 = publish(store, content={"same": True})
        v2 = stage(store, content={"same": True})
        store.activate([v2.id])
        count = len(sink.records)
        assert store.activate([v2.id]) == [v1]
        assert len(sink.records) == count

    def test_a_deduplicated_artifact_cannot_be_activated_once_its_target_was_replaced(
        self, store: ArtifactStore
    ) -> None:
        publish(store, content={"same": True})
        v2 = stage(store, content={"same": True})
        store.activate([v2.id])
        publish(store, content={"different": True})
        with pytest.raises(StateError, match="cannot activate"):
            store.activate([v2.id])

    def test_different_content_is_a_new_version(self, store: ArtifactStore) -> None:
        publish(store, content={"v": 1})
        v2 = publish(store, content={"v": 2})
        assert v2.version == 2
        assert [a.status for a in store.versions("design_doc")] == [
            ArtifactStatus.SUPERSEDED,
            ArtifactStatus.ACTIVE,
        ]


class TestQuarantineAndSupersede:
    def test_quarantine_hides_the_artifact_and_records_a_sanitised_reason(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        art = stage(store)
        (done,) = store.quarantine([art.id], "gate \x1b[31mrejected\x1b[0m\nSEC-001" + "x" * 1000)
        assert done.status == ArtifactStatus.QUARANTINED
        reason = done.metadata["_quarantine_reason"]
        assert "\x1b" not in reason and "\n" not in reason and len(reason) <= 300
        (event,) = sink.of(EventType.ARTIFACT_QUARANTINED)
        assert event.data == {"artifact_id": art.id, "reason": reason, "tombstone": False}
        assert store.staged() == []
        assert store.read_json(done) == {"schema": 1, "v": 1}  # not tombstoned: still inspectable

    def test_quarantine_is_idempotent_and_all_or_nothing(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        art = stage(store)
        store.quarantine([art.id], "r")
        count = len(sink.records)
        store.quarantine([art.id], "again")
        assert len(sink.records) == count
        other = stage(store, content={"v": 2})
        with pytest.raises(StateError, match="unknown"):
            store.quarantine([other.id, "design_doc@v99"], "r")
        assert store.get(other.id).status == ArtifactStatus.STAGED

    def test_tombstone_replaces_the_bytes_but_keeps_hash_size_and_metadata(
        self, store: ArtifactStore
    ) -> None:
        secret_like = "AK" + "IA" + "X" * 16
        art = store.put(
            ArtifactDraft("release_patch", f"key={secret_like}", metadata={"note": "kept"}),
            producer="promote",
            staged=True,
        )
        (done,) = store.quarantine([art.id], "SEC-001", tombstone=True)
        assert (store.root / art.path).read_bytes() == TOMBSTONE
        assert secret_like.encode() not in (store.root / art.path).read_bytes()
        assert (done.content_hash, done.size, done.metadata["note"]) == (
            art.content_hash,
            art.size,
            "kept",
        )
        assert done.metadata["_tombstone"] is True
        with pytest.raises(StateError, match="tombstone"):
            store.read_text(done)
        assert store.verify_all() == []  # a tombstone is a legitimate state, not corruption

    def test_a_quarantined_artifact_can_be_tombstoned_later(self, store: ArtifactStore) -> None:
        art = stage(store, "release_patch", "secret", producer="promote")
        store.quarantine([art.id], "first")
        (done,) = store.quarantine([art.id], "second", tombstone=True)
        assert (store.root / art.path).read_bytes() == TOMBSTONE
        assert done.metadata["_quarantine_reason"] == "first"  # the original reason is kept

    def test_an_active_artifact_can_be_quarantined(self, store: ArtifactStore) -> None:
        art = publish(store)
        store.quarantine([art.id], "rolled back")
        assert store.latest("design_doc") is None

    def test_supersede_is_for_active_artifacts_and_idempotent(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        active = publish(store)
        done = store.supersede(active.id, "node removed")
        assert done.status == ArtifactStatus.SUPERSEDED
        assert store.supersede(active.id, "again") == done
        assert len(sink.of(EventType.ARTIFACT_SUPERSEDED)) == 1
        assert sink.of(EventType.ARTIFACT_SUPERSEDED)[0].data["superseded_by"] is None
        staged = stage(store, content={"v": 2})
        with pytest.raises(StateError, match="cannot supersede"):
            store.supersede(staged.id, "x")


# =============================================================================
# queries and reads
# =============================================================================


class TestQueries:
    def test_get_unknown_raises_state_error(self, store: ArtifactStore) -> None:
        with pytest.raises(StateError, match="unknown artifact"):
            store.get("nope@v1")
        with pytest.raises(StateError):
            store.get(None)  # type: ignore[arg-type]

    def test_all_defaults_to_active_and_is_sorted(self, store: ArtifactStore) -> None:
        a = publish(store, "task_plan", producer="decompose")
        b = publish(store, "design_doc")
        stage(store, "requirements", producer="requirements")
        old = publish(store, "design_doc", {"v": 2})
        assert [x.id for x in store.all()] == [old.id, a.id]
        assert {x.id for x in store.all(include_inactive=True)} >= {a.id, b.id, old.id}
        assert [x.id for x in store.all(include_inactive=True)] == sorted(
            (x.id for x in store.all(include_inactive=True)),
            key=lambda i: (i.split("@")[0], int(i.split("@v")[1])),
        )

    def test_versions_of_unknown_kind_is_empty(self, store: ArtifactStore) -> None:
        assert store.versions("nothing") == []
        assert store.latest("nothing") is None


class TestReads:
    def test_reads_verify_the_content_hash(self, store: ArtifactStore) -> None:
        art = publish(store)
        (store.root / art.path).write_bytes(b'{"schema": 1, "v": 2}\n')  # same length is not enough
        (store.root / art.path).write_bytes(b"x" * art.size)
        with pytest.raises(IntegrityError, match="hash"):
            store.read_bytes(art)

    def test_size_mismatch_is_detected_before_reading_the_file(self, store: ArtifactStore) -> None:
        art = publish(store)
        (store.root / art.path).write_bytes(b"x" * (art.size + 1_000_000))
        with pytest.raises(IntegrityError, match="size"):
            store.read_bytes(art)

    def test_missing_content_is_an_integrity_error(self, store: ArtifactStore) -> None:
        art = publish(store)
        (store.root / art.path).unlink()
        with pytest.raises(IntegrityError, match="missing"):
            store.read_bytes(art)

    def test_a_forged_artifact_object_is_not_trusted(self, store: ArtifactStore) -> None:
        art = publish(store)
        forged_path = Artifact(**{**art.__dict__, "path": "design_doc/../../../etc/passwd"})
        with pytest.raises(IntegrityError, match="does not match"):
            store.read_bytes(forged_path)
        forged_hash = Artifact(**{**art.__dict__, "content_hash": "0" * 64})
        with pytest.raises(IntegrityError, match="does not match"):
            store.read_bytes(forged_hash)
        with pytest.raises(StateError):
            store.read_bytes(Artifact(**{**art.__dict__, "id": "design_doc@v7", "version": 7}))

    def test_read_json_is_strict(self, store: ArtifactStore) -> None:
        dup = store.put(
            ArtifactDraft("design_doc", b'{"a": 1, "a": 2}', media_type="application/json"),
            producer="design",
            staged=True,
        )
        nan = store.put(
            ArtifactDraft("design_doc", b'{"a": NaN}', media_type="application/json"),
            producer="design",
            staged=True,
        )
        with pytest.raises(ValueError, match="duplicate"):
            store.read_json(dup)
        with pytest.raises(ValueError, match="non-finite"):
            store.read_json(nan)

    def test_read_text_rejects_binary(self, store: ArtifactStore) -> None:
        art = store.put(ArtifactDraft("release_patch", b"\xff\xfe\x00"), producer="promote")
        with pytest.raises(UnicodeDecodeError):
            store.read_text(art)

    def test_links_inside_the_store_are_refused(self, store: ArtifactStore, tmp_path: Path) -> None:
        art = publish(store)
        outside = tmp_path / "outside"
        (store.root / "design_doc").rename(outside)
        if not make_dir_link(store.root / "design_doc", outside):
            pytest.skip("this OS/user may not create symlinks or junctions")
        with pytest.raises(IntegrityError, match="rejected"):
            store.read_bytes(art)


# =============================================================================
# persistence, tampering, crash safety
# =============================================================================


class TestPersistence:
    def test_state_survives_reopening(self, store: ArtifactStore) -> None:
        a = publish(store)
        b = stage(store, content={"v": 2})
        again = reopen(store)
        assert again.get(a.id) == a
        assert again.get(b.id).status == ArtifactStatus.STAGED
        assert again.read_json(a) == store.read_json(a)
        assert stage(again, content={"v": 3}).version == 3  # versions continue, never reused

    def test_the_index_is_canonical_json_written_atomically(self, store: ArtifactStore) -> None:
        publish(store)
        raw = (store.root / "index.json").read_bytes()
        assert raw.endswith(b"\n") and b"\r" not in raw and raw.isascii()
        assert json.loads(raw)["schema"] == 1

    def test_all_the_stores_files_stay_inside_the_root(self, store: ArtifactStore) -> None:
        publish(store, "changeset", {"ops": []}, key="implement.core", producer="implement.core")
        for dirpath, _dirs, files in os.walk(store.root):
            assert Path(dirpath).resolve().is_relative_to(store.root)
            assert all(not f.endswith(".tmp") for f in files)


@pytest.fixture(scope="module")
def template_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A valid store with two published artifacts, built once and copied per tampering test."""
    root = tmp_path_factory.mktemp("template") / "artifacts"
    built = ArtifactStore(root, clock=StepClock())
    publish(built, "design_doc")
    publish(built, "task_plan", producer="decompose")
    return root


@pytest.fixture
def victim(template_root: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "victim"
    shutil.copytree(template_root, copy)
    return copy


def _tamper_root(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    index = root / "index.json"
    doc = json.loads(index.read_text(encoding="utf-8"))
    mutate(doc)
    index.write_text(json.dumps(doc), encoding="utf-8")


def _open(root: Path) -> ArtifactStore:
    return ArtifactStore(root, clock=StepClock())


class TestIndexTampering:
    def test_the_untouched_template_loads(self, victim: Path) -> None:
        assert [a.id for a in _open(victim).all()] == ["design_doc@v1", "task_plan@v1"]

    @pytest.mark.parametrize(
        "path",
        [
            "../outside/v1/content.json",
            "/etc/passwd",
            "C:/Windows/x.json",
            "design_doc/../../x.json",
            "design_doc\\x\\v1\\content.json",
            "index.json",
            "design_doc/_-e3b0c442/v1/content.exe",
            "design_doc/_-e3b0c442/v2/content.json",  # a valid-looking path of ANOTHER version
            "task_plan/_-e3b0c442/v1/content.json",  # another artifact's file
            "design_doc/_-e3b0c442/v1/content.json\n",
            "",
        ],
    )
    def test_hostile_storage_paths_are_rejected_on_load(self, victim: Path, path: str) -> None:
        _tamper_root(victim, lambda d: d["artifacts"][0].__setitem__("path", path))
        with pytest.raises(IntegrityError):
            _open(victim)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("id", "design_doc@v2"),
            ("kind", "Design"),
            ("kind", "../x"),
            ("key", "a/b"),
            ("version", 0),
            ("version", "one"),
            ("content_hash", "xyz"),
            ("content_hash", "A" * 64),
            ("size", -1),
            ("attempt", -1),
            ("producer", "../x"),
            ("agent_id", "a b"),
            ("media_type", "nope"),
            ("created_at", "2026-01-01T00:00:00"),  # naive
            ("created_at", "not a date"),
            ("status", "deleted"),
            ("inputs", "oops"),
            ("inputs", [{"artifact_id": 5, "kind": "k", "version": 1, "content_hash": "h"}]),
            ("metadata", 5),
        ],
    )
    def test_invalid_entries_are_rejected_on_load(
        self, victim: Path, field: str, value: Any
    ) -> None:
        _tamper_root(victim, lambda d: d["artifacts"][0].__setitem__(field, value))
        with pytest.raises(IntegrityError):
            _open(victim)

    def test_duplicate_ids_are_rejected(self, victim: Path) -> None:
        _tamper_root(victim, lambda d: d["artifacts"].append(dict(d["artifacts"][0])))
        with pytest.raises(IntegrityError, match="duplicate id"):
            _open(victim)

    def test_two_active_versions_of_one_key_are_rejected(self, victim: Path) -> None:
        def second_active_version(doc: dict[str, Any]) -> None:
            first = doc["artifacts"][0]
            ver2 = {**first, "id": "design_doc@v2", "version": 2}
            ver2["path"] = first["path"].replace("/v1/", "/v2/")
            doc["artifacts"].append(ver2)

        _tamper_root(victim, second_active_version)
        with pytest.raises(IntegrityError, match="several ACTIVE"):
            _open(victim)

    def test_two_entries_sharing_one_file_are_rejected(self, victim: Path) -> None:
        _tamper_root(
            victim,
            lambda d: d["artifacts"].append(
                {**d["artifacts"][0], "id": "design_doc@v2", "version": 2}
            ),
        )
        with pytest.raises(
            IntegrityError
        ):  # path is not the canonical path of v2 (and would collide)
            _open(victim)

    @pytest.mark.parametrize(
        "raw",
        [
            b"not json",
            b"[]",
            b'{"schema": 2, "artifacts": []}',
            b'{"schema": 1}',
            b'{"schema": 1, "artifacts": {}}',
            b'{"schema": 1, "schema": 1, "artifacts": []}',
            b'{"schema": 1, "artifacts": [5]}',
            b'{"schema": 1, "artifacts": [{}]}',
            b'{"schema": 1, "artifacts": [], "x": NaN}',
            b'{"schema": 1, "artifacts": [], "x": 1e999}',
            b"\xff\xfe",
        ],
    )
    def test_corrupt_index_documents_are_rejected(self, victim: Path, raw: bytes) -> None:
        (victim / "index.json").write_bytes(raw)
        with pytest.raises(IntegrityError):
            _open(victim)

    def test_oversized_index_is_rejected(self, victim: Path) -> None:
        (victim / "index.json").write_bytes(b" " * (17 * 1024 * 1024))
        with pytest.raises(IntegrityError, match="too large"):
            _open(victim)

    def test_a_storage_path_collision_is_refused_at_put_time(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # force two different keys onto one directory (same slug, same 8-hex hash)
        monkeypatch.setattr(artifacts_module, "sha256_hex", lambda _s: "0" * 64)
        first = stage(
            store, "changeset", {"ops": [], "n": 1}, key="Core", producer="implement.core"
        )
        with pytest.raises(IntegrityError, match="collision"):
            stage(store, "changeset", {"ops": [], "n": 2}, key="core", producer="implement.core")
        assert (store.root / first.path).read_bytes() == store.read_bytes(first)  # first untouched
        assert [a.id for a in store.all(include_inactive=True)] == [first.id]


class TestCrashSafety:
    def test_a_failed_index_write_changes_neither_memory_nor_disk(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keep = publish(store)
        index_before = (store.root / "index.json").read_bytes()
        real = artifacts_module.atomic_write_json

        def boom(path: Any, obj: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(artifacts_module, "atomic_write_json", boom)
        with pytest.raises(OSError, match="disk full"):
            stage(store, content={"v": 2})
        with pytest.raises(OSError, match="disk full"):
            store.quarantine([keep.id], "x")
        with pytest.raises(OSError, match="disk full"):
            store.supersede(keep.id, "x")
        assert [a.id for a in store.all(include_inactive=True)] == [keep.id]
        assert store.get(keep.id).status == ArtifactStatus.ACTIVE
        assert (store.root / "index.json").read_bytes() == index_before
        monkeypatch.setattr(artifacts_module, "atomic_write_json", real)
        again = stage(store, content={"v": 2})
        assert again.version == 2  # the failed put did not consume a version
        assert store.verify_all() == []  # the orphan content file was overwritten

    def test_a_failed_activation_commit_leaves_everything_staged(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        a = stage(store)
        b = stage(store, "task_plan", producer="decompose")

        def boom(path: Any, obj: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(artifacts_module, "atomic_write_json", boom)
        with pytest.raises(OSError):
            store.activate([a.id, b.id])
        assert {x.status for x in store.all(include_inactive=True)} == {ArtifactStatus.STAGED}
        assert store.latest("design_doc") is None

    def test_a_crash_during_the_index_replace_keeps_the_previous_index(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keep = publish(store)
        real_replace = os.replace

        def flaky(src: Any, dst: Any) -> None:
            if Path(dst).name == "index.json":
                raise OSError("simulated crash")
            real_replace(src, dst)

        monkeypatch.setattr(os, "replace", flaky)
        with pytest.raises(OSError, match="simulated crash"):
            stage(store, content={"v": 2})
        monkeypatch.undo()
        again = reopen(store)
        assert [a.id for a in again.all(include_inactive=True)] == [keep.id]
        assert not [
            f for f in os.listdir(store.root) if f.endswith(".tmp")
        ]  # no temp file is left behind

    def test_stray_temp_files_do_not_affect_loading(self, store: ArtifactStore) -> None:
        keep = publish(store)
        (store.root / ".index.json.abc123.tmp").write_bytes(b"garbage")
        assert reopen(store).get(keep.id) == keep

    def test_a_tombstone_write_failure_does_not_mark_the_artifact_quarantined(
        self, store: ArtifactStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        art = stage(store, "release_patch", "secret", producer="promote")

        def boom(path: Any, data: bytes, **kw: Any) -> None:
            raise OSError("io")

        monkeypatch.setattr(artifacts_module, "atomic_write_bytes", boom)
        with pytest.raises(OSError):
            store.quarantine([art.id], "x", tombstone=True)
        assert store.get(art.id).status == ArtifactStatus.STAGED


# =============================================================================
# verification against the audit chain
# =============================================================================


class TestVerification:
    def test_verify_all_reports_corruption(self, store: ArtifactStore) -> None:
        good = publish(store)
        bad = stage(store, "task_plan", producer="decompose")
        gone = stage(store, "requirements", producer="requirements")
        assert store.verify_all() == []
        (store.root / bad.path).write_bytes(b"tampered")
        (store.root / gone.path).unlink()
        problems = store.verify_all()
        assert len(problems) == 2
        assert any(bad.id in p and "hash" in p for p in problems)
        assert any(gone.id in p and "missing" in p for p in problems)
        assert not any(good.id in p for p in problems)

    def test_a_tombstoned_artifact_that_still_holds_content_is_reported(
        self, store: ArtifactStore
    ) -> None:
        art = stage(store, "release_patch", "secret", producer="promote")
        store.quarantine([art.id], "x", tombstone=True)
        (store.root / art.path).write_bytes(b"secret")
        assert any("tombstoned" in p for p in store.verify_all())

    def test_the_index_matches_its_own_audit_events(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        publish(store)
        stage(store, "task_plan", producer="decompose")
        store.put(draft("request", "x"), producer=EXTERNAL_PRODUCER)
        assert store.verify_against_events(sink.audit_events()) == []

    def test_a_missing_event_is_reported(self, store: ArtifactStore, sink: RecordingSink) -> None:
        art = stage(store)
        events = [e for e in sink.audit_events() if e.type != EventType.ARTIFACT_CREATED]
        assert store.verify_against_events(events) == [
            f"{art.id}: no ARTIFACT_CREATED event in the audit chain"
        ]

    def test_unknown_duplicate_and_mismatching_events_are_reported(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        art = stage(store)
        events = sink.audit_events()
        created = events[0]
        ghost = AuditEvent(
            9, T0, "run1", "t", "run", None, EventType.ARTIFACT_CREATED, None, None, Actor.system(),
            {"artifact_id": "ghost@v1", "content_hash": "a" * 64, "size": 1}, "0" * 64, "0" * 64,
        )  # fmt: skip
        dup = AuditEvent(**{**created.__dict__, "seq": 10})
        wrong = AuditEvent(
            **{**created.__dict__, "data": {**created.data, "content_hash": "f" * 64}}
        )
        malformed = AuditEvent(**{**created.__dict__, "data": {"size": 3}, "seq": 11})
        assert any(
            "ghost@v1" in p and "no entry" in p
            for p in store.verify_against_events([created, ghost])
        )
        assert any(
            art.id in p and "more than once" in p
            for p in store.verify_against_events([created, dup])
        )
        assert any(art.id in p and "differ" in p for p in store.verify_against_events([wrong]))
        assert any(
            "without an artifact_id" in p for p in store.verify_against_events([created, malformed])
        )

    def test_other_event_types_are_ignored(self, store: ArtifactStore, sink: RecordingSink) -> None:
        stage(store)
        events = sink.audit_events()
        noise = AuditEvent(**{**events[0].__dict__, "type": EventType.NODE_STARTED, "data": {}})
        assert store.verify_against_events([*events, noise]) == []


class TestAuditEvents:
    def test_created_event_carries_exactly_the_documented_data(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        art = stage(store, attempt=2, agent_id="architect")
        (event,) = sink.records
        assert event.type == EventType.ARTIFACT_CREATED
        assert event.data == {
            "artifact_id": art.id,
            "content_hash": art.content_hash,
            "size": art.size,
        }
        assert (event.node_id, event.attempt) == ("design", 2)
        assert event.actor == Actor.agent("architect")

    def test_external_artifacts_have_no_node_and_a_system_actor(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        store.put(draft("request", "x"), producer=EXTERNAL_PRODUCER)
        created = sink.of(EventType.ARTIFACT_CREATED)[0]
        assert (created.node_id, created.attempt) == (None, None)
        assert created.actor.type == ActorType.SYSTEM
        assert sink.of(EventType.ARTIFACT_PUBLISHED)[0].node_id is None

    def test_publication_events_belong_to_the_node_span(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        art = stage(store, attempt=4)
        store.activate([art.id])
        published = sink.of(EventType.ARTIFACT_PUBLISHED)[0]
        assert (published.node_id, published.attempt) == ("design", None)
        assert published.data == {"artifact_id": art.id, "content_hash": art.content_hash}

    def test_audit_sink_is_optional(self, tmp_path: Path) -> None:
        quiet = ArtifactStore(tmp_path / "q", clock=StepClock())
        art = quiet.put(draft(), producer="design", staged=True)
        quiet.activate([art.id])
        quiet.quarantine([art.id], "x")

    def test_events_are_emitted_only_after_the_index_is_durable(self, tmp_path: Path) -> None:
        seen: list[bool] = []

        def sink(type: EventType, **kw: Any) -> None:
            on_disk = json.loads((tmp_path / "a" / "index.json").read_text(encoding="utf-8"))
            seen.append(any(e["id"] == kw["data"]["artifact_id"] for e in on_disk["artifacts"]))

        s = ArtifactStore(tmp_path / "a", clock=StepClock(), audit=sink)  # type: ignore[arg-type]
        art = s.put(draft(), producer="design", staged=True)
        s.activate([art.id])
        s.quarantine([art.id], "x")
        assert seen and all(seen)


# =============================================================================
# snapshots and pinned readers
# =============================================================================


def _accept(*producers: str) -> Callable[[str], bool]:
    return lambda p: p in producers or p == EXTERNAL_PRODUCER


class TestSnapshot:
    def test_snapshot_serves_active_artifacts_of_declared_kinds_from_accepted_producers(
        self, store: ArtifactStore
    ) -> None:
        req = publish(store, "request", "text", producer=EXTERNAL_PRODUCER)
        doc = publish(store, "design_doc", producer="design")
        publish(store, "task_plan", producer="decompose")  # producer not accepted
        stage(store, "requirements", producer="requirements")  # staged: invisible
        pinned = store.snapshot(
            ["request", "design_doc", "task_plan", "requirements", "impact_report"],
            _accept("design", "requirements"),
        )
        assert pinned.latest("request") == req
        assert pinned.latest("design_doc") == doc
        assert pinned.latest("task_plan") is None
        assert pinned.latest("requirements") is None
        assert pinned.absent_kinds == ("impact_report", "requirements", "task_plan")
        assert [r.artifact_id for r in pinned.refs()] == [doc.id, req.id]  # sorted by (kind, key)
        assert pinned.kinds == frozenset(
            ["request", "design_doc", "task_plan", "requirements", "impact_report"]
        )

    def test_snapshot_is_immutable_against_later_changes(self, store: ArtifactStore) -> None:
        v1 = publish(store, content={"v": 1})
        pinned = store.snapshot(["design_doc"], _accept("design"))
        v2 = publish(store, content={"v": 2})
        store.quarantine([v1.id], "later")
        assert pinned.latest("design_doc").id == v1.id  # type: ignore[union-attr]
        assert store.latest("design_doc").id == v2.id  # type: ignore[union-attr]
        assert pinned.read_json(v1) == {"v": 1}  # still readable: no torn read
        assert [r.artifact_id for r in pinned.refs()] == [v1.id]

    def test_undeclared_kinds_and_foreign_artifacts_raise(self, store: ArtifactStore) -> None:
        doc = publish(store, "design_doc", producer="design")
        plan = publish(store, "task_plan", producer="decompose")
        pinned = store.snapshot(["design_doc"], _accept("design"))
        with pytest.raises(UndeclaredInputError):
            pinned.latest("task_plan")
        with pytest.raises(UndeclaredInputError):
            pinned.active("task_plan")
        with pytest.raises(UndeclaredInputError):
            pinned.read_text(plan)
        newer = publish(store, "design_doc", {"v": 2}, producer="design")
        with pytest.raises(UndeclaredInputError, match="not part of this snapshot"):
            pinned.read_json(newer)  # a version captured after the snapshot
        assert pinned.read_json(doc)["v"] == 1

    def test_undeclared_kind_message_is_sanitised(self, store: ArtifactStore) -> None:
        pinned = store.snapshot([], _accept())
        with pytest.raises(UndeclaredInputError) as err:
            pinned.latest("evil\x1b[31m\nkind")
        assert "\x1b" not in str(err.value) and "\n" not in str(err.value)
        assert isinstance(err.value, KeyError)  # engine code may catch it as a KeyError

    def test_snapshot_validates_kinds(self, store: ArtifactStore) -> None:
        with pytest.raises(ConfigError, match="kind"):
            store.snapshot(["../x"], _accept())

    def test_keyed_kinds_are_served_per_key(self, store: ArtifactStore) -> None:
        a = publish(
            store, "changeset", {"ops": [], "n": 1}, key="implement.a", producer="implement.a"
        )
        b = publish(
            store, "changeset", {"ops": [], "n": 2}, key="implement.b", producer="implement.b"
        )
        pinned = store.snapshot(["changeset"], _accept("implement.a", "implement.b"))
        assert pinned.active("changeset") == [a, b]
        assert pinned.latest("changeset", "implement.b") == b
        assert pinned.latest("changeset") is None

    def test_reader_protocol_and_accept_callback_runs_outside_the_lock(
        self, store: ArtifactStore
    ) -> None:
        publish(store, "design_doc", producer="design")
        results: list[Artifact | None] = []

        def accept(producer: str) -> bool:
            worker = threading.Thread(target=lambda: results.append(store.latest("design_doc")))
            worker.start()
            worker.join(timeout=5)  # would time out if snapshot() still held the store lock
            return True

        pinned = store.snapshot(["design_doc"], accept)
        assert isinstance(pinned, ArtifactReader)
        assert results and results[0] is not None


class TestWithExtra:
    def test_extra_serves_staged_artifacts_of_new_kinds_without_changing_the_inputs(
        self, store: ArtifactStore
    ) -> None:
        req = publish(store, "request", "x", producer=EXTERNAL_PRODUCER)
        pinned = store.snapshot(["request"], _accept())
        staged = stage(store, "design_doc", producer="design")
        gate_reader = pinned.with_extra([staged])
        assert gate_reader.latest("design_doc") == staged
        assert gate_reader.read_json(staged) == {"schema": 1, "v": 1}
        assert gate_reader.latest("request") == req
        assert gate_reader.refs() == pinned.refs() == (req.ref,)  # staged output is not an input
        assert "design_doc" in gate_reader.kinds
        assert gate_reader.absent_kinds == ()
        with pytest.raises(UndeclaredInputError):
            pinned.latest("design_doc")  # the original reader is untouched

    def test_extra_overrides_a_pinned_artifact_of_the_same_key_and_chains(
        self, store: ArtifactStore
    ) -> None:
        stage_a = stage(store, "design_doc", {"v": "a"})
        pinned = store.snapshot([], _accept())
        reader = pinned.with_extra([stage_a])
        again = reader.with_extra([stage(store, "task_plan", producer="design")])
        assert again.latest("design_doc") == stage_a
        assert again.latest("task_plan") is not None

    def test_extra_must_match_the_store_and_be_staged_or_active(self, store: ArtifactStore) -> None:
        pinned = store.snapshot([], _accept())
        art = stage(store)
        forged = Artifact(**{**art.__dict__, "content_hash": "0" * 64})
        with pytest.raises(IntegrityError):
            pinned.with_extra([forged])
        stranger = Artifact(**{**art.__dict__, "id": "design_doc@v9", "version": 9})
        with pytest.raises(StateError):
            pinned.with_extra([stranger])
        store.quarantine([art.id], "x")
        with pytest.raises(StateError, match="cannot be served"):
            pinned.with_extra([art])

    def test_extra_reads_the_stores_record_not_the_callers_object(
        self, store: ArtifactStore
    ) -> None:
        art = stage(store)
        reader = store.snapshot([], _accept()).with_extra([art])
        stale = Artifact(
            **{**art.__dict__, "status": ArtifactStatus.ACTIVE}
        )  # caller-side status is irrelevant
        assert reader.read_json(stale) == {"schema": 1, "v": 1}

    def test_constructor_rejects_artifacts_outside_declared_kinds(
        self, store: ArtifactStore
    ) -> None:
        art = publish(store)
        with pytest.raises(ConfigError, match="declared kinds"):
            PinnedReader(store, ["task_plan"], [art])


class TestLiveReader:
    def test_reader_is_live_unscoped_and_read_only(self, store: ArtifactStore) -> None:
        live = store.reader()
        assert isinstance(live, ArtifactReader)
        assert live.latest("design_doc") is None
        art = publish(store)
        assert live.latest("design_doc") == art
        assert live.active("design_doc") == [art]
        assert live.read_json(art) == {"schema": 1, "v": 1}
        assert live.read_text(art).startswith("{")
        assert live.read_bytes(art).endswith(b"\n")
        for forbidden in ("put", "activate", "quarantine", "supersede", "snapshot"):
            assert not hasattr(live, forbidden)


# =============================================================================
# concurrency
# =============================================================================


def _run_threads(count: int, target: Callable[[int], None]) -> list[BaseException]:
    errors: list[BaseException] = []
    barrier = threading.Barrier(count)

    def wrapper(i: int) -> None:
        try:
            barrier.wait(timeout=10)
            target(i)
        except BaseException as exc:  # collected and asserted by the test
            errors.append(exc)

    threads = [threading.Thread(target=wrapper, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return errors


class TestConcurrency:
    def test_concurrent_puts_get_unique_monotonic_versions(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        threads, per_thread = 6, 8
        got: list[list[int]] = [[] for _ in range(threads)]

        def work(i: int) -> None:
            for n in range(per_thread):
                art = stage(store, "design_doc", {"thread": i, "n": n})
                got[i].append(art.version)

        assert _run_threads(threads, work) == []
        flat = sorted(v for versions in got for v in versions)
        assert flat == list(range(1, threads * per_thread + 1))  # unique, gap-free
        assert all(versions == sorted(versions) for versions in got)  # monotonic within a thread
        again = reopen(store)
        assert [a.version for a in again.versions("design_doc")] == flat
        assert again.verify_all() == []
        assert again.verify_against_events(sink.audit_events()) == []

    def test_concurrent_puts_of_different_keys_do_not_interfere(self, store: ArtifactStore) -> None:
        def work(i: int) -> None:
            for n in range(5):
                stage(
                    store,
                    "changeset",
                    {"ops": [], "n": n},
                    key=f"implement.g{i}",
                    producer=f"implement.g{i}",
                )

        assert _run_threads(5, work) == []
        for i in range(5):
            assert [a.version for a in store.versions("changeset", f"implement.g{i}")] == [
                1,
                2,
                3,
                4,
                5,
            ]

    def test_concurrent_activation_is_idempotent(
        self, store: ArtifactStore, sink: RecordingSink
    ) -> None:
        staged = [
            stage(
                store,
                "changeset",
                {"ops": [], "k": k},
                key=f"implement.g{k}",
                producer=f"implement.g{k}",
            )
            for k in range(6)
        ]
        ids = [a.id for a in staged]
        results: list[list[Artifact]] = []
        lock = threading.Lock()

        def work(_: int) -> None:
            out = store.activate(ids)
            with lock:
                results.append(out)

        assert _run_threads(6, work) == []
        assert all(r == results[0] for r in results)
        assert [a.id for a in results[0]] == ids
        assert len(sink.of(EventType.ARTIFACT_PUBLISHED)) == 6  # exactly once per artifact
        assert {a.status for a in store.all()} == {ArtifactStatus.ACTIVE}

    def test_readers_never_observe_a_torn_state_while_writers_publish(
        self, store: ArtifactStore
    ) -> None:
        publish(store, content={"schema": 1, "v": 0})
        stop = threading.Event()

        def reader() -> None:
            while not stop.is_set():
                art = store.latest("design_doc")
                assert art is not None  # the supersede+activate pair is one index write
                assert store.read_json(art)["schema"] == 1
                actives = [
                    a for a in store.versions("design_doc") if a.status == ArtifactStatus.ACTIVE
                ]
                assert len(actives) <= 1

        errors: list[BaseException] = []

        def guarded() -> None:
            try:
                reader()
            except BaseException as exc:
                errors.append(exc)

        watchers = [threading.Thread(target=guarded) for _ in range(3)]
        for w in watchers:
            w.start()
        for n in range(1, 15):
            publish(store, content={"schema": 1, "v": n})
        stop.set()
        for w in watchers:
            w.join(timeout=10)
        assert errors == []
