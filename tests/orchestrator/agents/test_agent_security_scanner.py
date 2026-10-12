"""Tests for the security scanner agent (``security-scan`` node).

The scanner is a defence-in-depth summary over the integrated change set (the authoritative
guardrails are the PolicyEngine rules).  Every fake secret below is assembled at runtime from
fragments, so no secret-shaped literal exists in the repository.  Several tests share one scan
through a module-scoped fixture: the agent runs once, each test reads its own file's findings.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from orchestrator.agents.security_scanner import (
    MAX_FINDINGS,
    MAX_MATCHES_PER_PATTERN,
    RULES,
    SecurityScannerAgent,
)
from orchestrator.core.errors import OrchestratorError, ToolDenied
from orchestrator.core.schemas import Kind, validate_payload
from orchestrator.core.types import (
    AgentResult,
    Artifact,
    ArtifactReader,
    ArtifactSubject,
    ChangeSet,
    Severity,
)
from orchestrator.tools.changeset import compute_ops
from orchestrator.tools.workspace import MemoryView
from tests.orchestrator.agents.delivery_support import (
    NODES,
    Rig,
    assert_valid,
    content_hashes,
    drafts,
    fake_aws_key_id,
    payload,
    plain_writes,
)
from tests.orchestrator.policies.policy_builders import make_artifact

SCANNER = "security-scan"
REPORT = Kind.SECURITY_REPORT
INTEGRATED = Kind.INTEGRATED_CHANGESET
ONE_MIB = 1024 * 1024
HIGH, MEDIUM = Severity.HIGH, Severity.MEDIUM


@pytest.fixture(autouse=True)
def _fast_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    plain_writes(monkeypatch)


# ----------------------------------------------------------------------------- helpers


def high_secrets() -> dict[str, str]:
    """Pattern label -> fake secret (HIGH patterns), assembled from fragments at runtime."""
    return {
        "aws_access_key_id": fake_aws_key_id(),
        "github_token": "gh" + "p_" + "aB3dE5" * 6,
        "github_fine_grained_token": "github" + "_pat_" + "A1b2C3d4" * 3,
        "slack_token": "xo" + "xb-" + "1234567890" + "-" + "abcdefghij",
        "private_key_block": "-----BEGIN " + "RSA PRIV" + "ATE KEY-----",
        "json_web_token": "ey" + "Jhbgci0123" + ".ey" + "Jzdwiabcd123" + "." + "signatur3abc",
        "google_api_key": "AI" + "za" + "A" * 35,
        "stripe_live_key": "sk" + "_live_" + "a1B2c3D4" * 3,
        "provider_secret_key": "sk" + "-" + "Zx9kQ2mLp4Rt7" * 3,
    }


def medium_secrets() -> dict[str, str]:
    """Pattern label -> fake credential for the MEDIUM patterns."""
    password = "Zx9kQ2" + "mLp4Rt7"
    return {
        "url_with_credentials": "postgres" + "://" + "svc_user:" + password + "@db.internal/app",
        "hardcoded_credential": "pass" + "word" + " = '" + password + "'",
    }


def scan(
    root: Path,
    files: Mapping[str, str | None],
    *,
    baseline: Mapping[str, str] | None = None,
    clock_start: datetime | None = None,
) -> tuple[Rig, AgentResult]:
    """Publish ``files`` as the integrated change set (relative to ``baseline``) and scan it."""
    rig = Rig(root, baseline_files=dict(baseline or {}), clock_start=clock_start)
    rig.publish_changeset(INTEGRATED, files)
    return rig, rig.run(SecurityScannerAgent(), SCANNER)


def report_of(root: Path, files: Mapping[str, str | None], **kwargs: Any) -> dict[str, Any]:
    _, result = scan(root, files, **kwargs)
    assert_valid(result)
    return payload(result, REPORT)


def shared_scan(
    factory: pytest.TempPathFactory, name: str, files: Mapping[str, str | None]
) -> AgentResult:
    """One scan for a module-scoped fixture (read by several tests, so it runs only once)."""
    with pytest.MonkeyPatch.context() as patch:
        plain_writes(patch)
        _, result = scan(factory.mktemp(name), files)
    assert_valid(result)
    return result


def findings_for(report: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
    """Findings located at ``path`` or at ``path:<line>``."""
    return [
        f
        for f in report["findings"]
        if f["location"] == path or str(f["location"]).startswith(path + ":")
    ]


def triples(report: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [(f["severity"], f["rule_id"], f["location"]) for f in report["findings"]]


# ----------------------------------------------------------------------------- case tables

SHELL = "subprocess(shell=True)"
# file name -> (source, line of the dangerous call, severity or None, evidence or None)
CALL_CASES: dict[str, tuple[str, int, Severity | None, str | None]] = {
    "eval": ("x = eval('1 + 1')\n", 1, HIGH, "eval"),
    "exec": ("exec('x = 1')\n", 1, HIGH, "exec"),
    "dunder_import": ("mod = __import__('os')\n", 1, HIGH, "__import__"),
    "os_system": ("import os\n\nos.system('ls')\n", 3, HIGH, "os.system"),
    "os_popen": ("import os\n\nos.popen('ls')\n", 3, HIGH, "os.popen"),
    "pickle_loads": ("import pickle\n\nobj = pickle.loads(blob)\n", 3, HIGH, "pickle.loads"),
    "shell_true": ("import subprocess\n\nsubprocess.run('ls', shell=True)\n", 3, HIGH, SHELL),
    "shell_truthy_constant": (
        "import subprocess\n\nsubprocess.call('ls', shell=1)\n",
        3,
        HIGH,
        SHELL,
    ),
    "shell_variable": (
        "import subprocess\n\nsubprocess.call('ls', shell=flag)\n",
        3,
        HIGH,
        "subprocess(shell=<non-literal>)",
    ),
    "no_shell": ("import subprocess\n\nsubprocess.run(['ls'])\n", 3, MEDIUM, "subprocess.run"),
    "shell_false": (
        "import subprocess\n\nsubprocess.run(['ls'], shell=False)\n",
        3,
        MEDIUM,
        "subprocess.run",
    ),
    "alias_import_as": (
        "import subprocess as sp\n\nsp.check_output('ls', shell=True)\n",
        3,
        HIGH,
        SHELL,
    ),
    "alias_import_as_plain": (
        "import subprocess as sp\n\nsp.Popen(['ls'])\n",
        3,
        MEDIUM,
        "subprocess.Popen",
    ),
    "alias_from_import": ("from os import system\n\nsystem('ls')\n", 3, HIGH, "os.system"),
    "alias_from_import_as": (
        "from os import system as go\n\ngo('ls')\n",
        3,
        HIGH,
        "os.system",
    ),
    "yaml_load": ("import yaml\n\ncfg = yaml.load(text)\n", 3, MEDIUM, "yaml.load"),
    "compile_call": ("code = compile(src, 'f', 'exec')\n", 1, MEDIUM, "compile"),
    "nested_comprehension": ("rows = [eval(s) for s in lines]\n", 1, HIGH, "eval"),
    "multiline_call": ("import os\n\n\nvalue = (\n    os.system('ls')\n)\n", 5, HIGH, "os.system"),
    "deeply_nested_valid": ("x = " + "[" * 150 + "eval('1')" + "]" * 150 + "\n", 1, HIGH, "eval"),
    "methods_on_objects": (
        "model.eval()\nmodel.exec('x')\nmodel.system('ls')\nrunner.run('x', shell=True)\n",
        0,
        None,
        None,
    ),
    "own_method_named_eval": (
        "class M:\n    def eval(self) -> None: ...\n\nM().eval()\n",
        0,
        None,
        None,
    ),
}

# path -> expected severity (None: ordinary path, no finding)
PATH_CASES: dict[str, str | None] = {
    ".github/workflows/ci.yml": "HIGH",
    "Dockerfile": "HIGH",
    "ops/DOCKERFILE": "HIGH",
    "ops/Dockerfile.prod": "HIGH",
    "docker-compose.yml": "HIGH",
    ".env": "HIGH",
    "deploy/.env.prod": "HIGH",
    "keys/server.pem": "HIGH",
    "keys/Signing.KEY": "HIGH",
    "pyproject.toml": "MEDIUM",
    "requirements.txt": "MEDIUM",
    "deps/requirements-dev.txt": "MEDIUM",
    "app/main.py": None,
    "docs/readme.md": None,
    "docs/requirements.md": None,
    "app/environment.py": None,
}


def unparseable_sources() -> dict[str, str]:
    return {
        "syntax": "def broken(:\n    pass\n",
        "parens": "x = " + "(" * 100_000 + "1" + ")" * 100_000 + "\n",
        "binop": "x = " + "1 + " * 100_000 + "1\n",
        "unary": "x = " + "-" * 100_000 + "1\n",
        "attr_chain": "x = a" + ".b" * 100_000 + "\n",
    }


# ----------------------------------------------------------------------------- shared scans


@pytest.fixture(scope="module")
def secrets_result(tmp_path_factory: pytest.TempPathFactory) -> AgentResult:
    files = {
        f"secrets/{label}.txt": f"first line\nvalue: {value}\nlast line\n"
        for label, value in high_secrets().items()
    }
    return shared_scan(tmp_path_factory, "secrets", files)


@pytest.fixture(scope="module")
def medium_result(tmp_path_factory: pytest.TempPathFactory) -> AgentResult:
    files = {
        f"notes/{label}.txt": f"first line\nvalue: {value}\n"
        for label, value in medium_secrets().items()
    }
    files["requirements.txt"] = "fastapi==1.0\n"
    files["lib/run.py"] = "import subprocess\n\nsubprocess.run(['ls'], shell=False)\n"
    return shared_scan(tmp_path_factory, "medium", files)


@pytest.fixture(scope="module")
def calls_result(tmp_path_factory: pytest.TempPathFactory) -> AgentResult:
    files = {f"calls/{name}.py": case[0] for name, case in CALL_CASES.items()}
    return shared_scan(tmp_path_factory, "calls", files)


@pytest.fixture(scope="module")
def unparseable_result(tmp_path_factory: pytest.TempPathFactory) -> AgentResult:
    files = {f"bad/{name}.py": source for name, source in unparseable_sources().items()}
    files["bad/syntax_and_secret.py"] = f"def broken(:\n    key = '{fake_aws_key_id()}'\n"
    files["ok/fine.py"] = "VALUE = 1\n"
    return shared_scan(tmp_path_factory, "unparseable", files)


@pytest.fixture(scope="module")
def paths_result(tmp_path_factory: pytest.TempPathFactory) -> AgentResult:
    return shared_scan(tmp_path_factory, "paths", dict.fromkeys(PATH_CASES, "x\n"))


# ----------------------------------------------------------------------------- the basics


def test_a_clean_change_set_passes(tmp_path: Path) -> None:
    files = {"widgets/core.py": "def widget() -> int:\n    return 1\n", "docs/a.md": "# A\n"}
    _, result = scan(tmp_path, files)
    assert_valid(result)
    report = payload(result, REPORT)
    assert report["passed"] is True
    assert report["findings"] == []
    assert report["files_scanned"] == 2
    assert report["rules_evaluated"] == list(RULES)
    assert [d.kind for d in result.artifacts] == [REPORT]
    assert result.rationale == "2 file(s) scanned, 0 finding(s), 0 HIGH or above."


def test_an_empty_change_set_scans_nothing_and_passes(tmp_path: Path) -> None:
    report = report_of(tmp_path, {})
    assert (report["passed"], report["files_scanned"], report["findings"]) == (True, 0, [])


def test_only_created_and_modified_files_are_scanned(tmp_path: Path) -> None:
    baseline = {"keep.py": "VALUE = 1\n", "gone.py": "print('x')\n", "gone.txt": "bye\n"}
    files: dict[str, str | None] = {
        "keep.py": "VALUE = 2\n",  # MODIFY
        "new.py": "NEW = 1\n",  # CREATE
        "gone.py": None,  # DELETE
        "gone.txt": None,  # DELETE
    }
    report = report_of(tmp_path, files, baseline=baseline)
    assert report["files_scanned"] == 2
    assert report["findings"] == [] and report["passed"] is True


def test_protected_path_findings_name_the_operation(tmp_path: Path) -> None:
    baseline = {"Dockerfile": "FROM a\n", ".env": "A=1\n"}
    files: dict[str, str | None] = {
        "Dockerfile": "FROM b\n",
        ".env": None,
        "requirements.txt": "x\n",
    }
    report = report_of(tmp_path, files, baseline=baseline)
    messages = {f["location"]: f["message"] for f in report["findings"]}
    assert messages == {
        "Dockerfile": "protected or sensitive path is modified",
        ".env": "protected or sensitive path is deleted",
        "requirements.txt": "protected or sensitive path is created",
    }
    assert report["files_scanned"] == 2  # the deleted .env carries no content
    assert report["passed"] is False  # a HIGH path finding, although .env was only deleted


# ----------------------------------------------------------------------------- secrets (SCN-001)


@pytest.mark.parametrize("label", sorted(high_secrets()))
def test_every_high_secret_pattern_is_scn_001_high(secrets_result: AgentResult, label: str) -> None:
    report = payload(secrets_result, REPORT)
    found = findings_for(report, f"secrets/{label}.txt")
    assert len(found) == 1, found
    finding = found[0]
    assert finding["rule_id"] == "SCN-001"
    assert finding["severity"] == "HIGH"
    assert finding["evidence"] == f"pattern:{label}"  # the label only, never the match
    assert finding["location"] == f"secrets/{label}.txt:2"
    assert report["passed"] is False


@pytest.mark.parametrize("label", sorted(medium_secrets()))
def test_medium_secret_patterns_are_scn_001_medium(medium_result: AgentResult, label: str) -> None:
    found = findings_for(payload(medium_result, REPORT), f"notes/{label}.txt")
    assert [(f["rule_id"], f["severity"], f["evidence"]) for f in found] == [
        ("SCN-001", "MEDIUM", f"pattern:{label}")
    ]
    assert found[0]["location"] == f"notes/{label}.txt:2"


def test_medium_findings_alone_do_not_fail_the_scan(medium_result: AgentResult) -> None:
    report = payload(medium_result, REPORT)
    assert report["passed"] is True
    assert [f["severity"] for f in report["findings"]] == ["MEDIUM"] * 4
    assert {f["rule_id"] for f in report["findings"]} == {"SCN-001", "SCN-002", "SCN-004"}


def test_reports_never_echo_a_secret(
    secrets_result: AgentResult, medium_result: AgentResult, unparseable_result: AgentResult
) -> None:
    cases = (
        (secrets_result, high_secrets()),
        (medium_result, medium_secrets()),
        (unparseable_result, {"aws": fake_aws_key_id()}),
    )
    for result, values in cases:
        text = json.dumps(payload(result, REPORT)) + result.rationale
        for label, value in values.items():
            for probe in (value, value[:12], value[-12:]):  # the whole value or a visible part
                assert probe not in text, label


def test_secrets_in_non_python_text_files_are_found(tmp_path: Path) -> None:
    key = fake_aws_key_id()
    files = {
        "docs/notes.md": f"# Notes\n\nkey: {key}\n",
        "deploy/config.yaml": f"aws:\n  key: {key}\n",
        "data/settings.json": f'{{"k": "{key}"}}\n',
        "scripts/run.sh": f"#!/bin/sh\nexport K={key}\n",
        "pkg/settings.py": f"KEY = '{key}'\n",  # python: the text scan still applies
    }
    lines = {"docs/notes.md": 3, "deploy/config.yaml": 2, "data/settings.json": 1}
    lines.update({"scripts/run.sh": 2, "pkg/settings.py": 1})
    report = report_of(tmp_path, files)
    assert sorted(triples(report)) == sorted(
        ("HIGH", "SCN-001", f"{path}:{line}") for path, line in lines.items()
    )
    assert key not in json.dumps(report)


def test_matches_per_pattern_and_file_are_capped(tmp_path: Path) -> None:
    keys = ["AK" + "IA" + str(i).zfill(16) for i in range(MAX_MATCHES_PER_PATTERN + 15)]
    report = report_of(tmp_path, {"leak/keys.txt": "\n".join(keys) + "\n"})
    # locations sort as strings ("...:10" before "...:2"), so compare as a set
    assert len(report["findings"]) == MAX_MATCHES_PER_PATTERN
    assert {f["location"] for f in report["findings"]} == {
        f"leak/keys.txt:{line}" for line in range(1, MAX_MATCHES_PER_PATTERN + 1)
    }
    assert report["passed"] is False


# ----------------------------------------------------------------------------- calls (SCN-002)


@pytest.mark.parametrize("name", sorted(CALL_CASES))
def test_dangerous_calls_are_found_through_the_ast(calls_result: AgentResult, name: str) -> None:
    _, line, severity, evidence = CALL_CASES[name]
    found = findings_for(payload(calls_result, REPORT), f"calls/{name}.py")
    if severity is None:  # an attribute call on an arbitrary object is not a dangerous call
        assert found == []
        return
    assert len(found) == 1, found
    finding = found[0]
    assert (finding["rule_id"], finding["severity"], finding["evidence"]) == (
        "SCN-002",
        severity.name,
        evidence,
    )
    assert finding["message"] == f"dangerous or process-spawning call: {evidence}"
    assert finding["location"] == f"calls/{name}.py:{line}"


def test_python_extension_is_matched_case_insensitively(tmp_path: Path) -> None:
    files = {"x/EVIL.PY": "import os\nos.system('ls')\n", "x/Mixed.Py": "eval('1')\n"}
    report = report_of(tmp_path, files)
    assert sorted(triples(report)) == [
        ("HIGH", "SCN-002", "x/EVIL.PY:2"),
        ("HIGH", "SCN-002", "x/Mixed.Py:1"),
    ]


# ----------------------------------------------------------------------------- SCN-003


@pytest.mark.parametrize("name", sorted(unparseable_sources()))
def test_unparseable_python_is_scn_003_high_and_never_a_pass(
    unparseable_result: AgentResult, name: str
) -> None:
    report = payload(unparseable_result, REPORT)
    found = findings_for(report, f"bad/{name}.py")
    assert [(f["rule_id"], f["severity"]) for f in found] == [("SCN-003", "HIGH")], found
    assert found[0]["message"].startswith("python source cannot be analysed (")
    assert found[0]["location"].startswith(f"bad/{name}.py:")
    assert report["passed"] is False


def test_a_syntax_error_does_not_hide_secrets_in_the_same_file(
    unparseable_result: AgentResult,
) -> None:
    report = payload(unparseable_result, REPORT)
    found = findings_for(report, "bad/syntax_and_secret.py")
    assert sorted((f["rule_id"], f["severity"]) for f in found) == [
        ("SCN-001", "HIGH"),
        ("SCN-003", "HIGH"),
    ]
    assert findings_for(report, "ok/fine.py") == []
    assert report["files_scanned"] == len(unparseable_sources()) + 2


# ----------------------------------------------------------------------------- paths (SCN-004)


@pytest.mark.parametrize("path", sorted(PATH_CASES))
def test_protected_and_sensitive_paths(paths_result: AgentResult, path: str) -> None:
    expected = PATH_CASES[path]
    found = findings_for(payload(paths_result, REPORT), path)
    if expected is None:
        assert found == []
        return
    assert [(f["rule_id"], f["severity"], f["location"]) for f in found] == [
        ("SCN-004", expected, path)
    ]
    assert found[0]["evidence"] is None  # a path finding echoes nothing but the path


# ----------------------------------------------------------------------------- order and cap


def test_findings_are_sorted_high_first_then_by_rule_and_location(tmp_path: Path) -> None:
    files = {
        ".env": "X=1\n",
        "app/danger.py": f"import os\nos.system('ls')\nKEY = '{fake_aws_key_id()}'\n",
        "notes/creds.txt": medium_secrets()["url_with_credentials"] + "\n",
        "requirements.txt": "fastapi==1.0\n",
        "lib/run.py": "import subprocess\nsubprocess.run(['ls'])\n",
    }
    report = report_of(tmp_path, files)
    assert triples(report) == [
        ("HIGH", "SCN-001", "app/danger.py:3"),
        ("HIGH", "SCN-002", "app/danger.py:2"),
        ("HIGH", "SCN-004", ".env"),
        ("MEDIUM", "SCN-001", "notes/creds.txt:1"),
        ("MEDIUM", "SCN-002", "lib/run.py:2"),
        ("MEDIUM", "SCN-004", "requirements.txt"),
    ]


def many_medium_files(count: int) -> dict[str, str | None]:
    return {f"deps/requirements-{i:03d}.txt": "x==1\n" for i in range(count)}


def test_the_report_is_capped_and_never_drops_high_findings(tmp_path: Path) -> None:
    files = many_medium_files(MAX_FINDINGS + 30)
    files.update({".env": "A=1\n", "zz/Dockerfile": "FROM x\n", "zz/server.pem": "x\n"})
    report = report_of(tmp_path, files)
    findings = report["findings"]
    omitted = (MAX_FINDINGS + 30 + 3) - MAX_FINDINGS
    assert len(findings) == MAX_FINDINGS + 1  # the cap plus the marker
    assert [f["location"] for f in findings[:3]] == [".env", "zz/Dockerfile", "zz/server.pem"]
    assert {f["severity"] for f in findings[:3]} == {"HIGH"}
    assert [f["severity"] for f in findings[3:-1]] == ["MEDIUM"] * (MAX_FINDINGS - 3)
    marker = findings[-1]
    assert (marker["rule_id"], marker["severity"], marker["location"]) == (
        "SCN-004",
        "LOW",
        "report",
    )
    assert marker["message"] == f"{omitted} further finding(s) omitted from the report"
    assert report["passed"] is False


def test_the_cap_never_turns_a_failing_or_passing_scan_around(tmp_path: Path) -> None:
    report = report_of(tmp_path, many_medium_files(MAX_FINDINGS + 30))
    assert len(report["findings"]) == MAX_FINDINGS + 1
    assert report["findings"][-1]["message"].startswith("30 further finding(s)")
    assert report["passed"] is True  # only MEDIUM findings, dropped or not
    assert report["files_scanned"] == MAX_FINDINGS + 30


# ----------------------------------------------------------------------------- performance


@pytest.mark.parametrize("unit", ["a\n", "sk-", "password = '"], ids=["lines", "prefix", "assign"])
def test_one_mebibyte_of_repetitive_text_scans_in_linear_time(tmp_path: Path, unit: str) -> None:
    text = (unit * (ONE_MIB // len(unit) + 1))[:ONE_MIB]
    rig = Rig(tmp_path)
    rig.publish_changeset(INTEGRATED, {"big/repeated.txt": text})
    started = time.perf_counter()
    result = rig.run(SecurityScannerAgent(), SCANNER)
    assert time.perf_counter() - started < 5.0  # generous: ~1 s on a slow machine
    report = payload(result, REPORT)
    assert report["files_scanned"] == 1
    assert len(report["findings"]) <= 11 * MAX_MATCHES_PER_PATTERN


# ----------------------------------------------------------------------------- determinism, schema


def test_the_report_is_deterministic_across_clocks_and_op_order(tmp_path: Path) -> None:
    files = {
        "app/danger.py": f"import os\nos.system('ls')\nKEY = '{fake_aws_key_id()}'\n",
        ".env": "X=1\n",
        "docs/a.md": "# A\n",
    }
    starts = (datetime(2026, 1, 1, tzinfo=UTC), datetime(2031, 7, 9, 12, 30, tzinfo=UTC))
    hashes = []
    for index, start in enumerate(starts):
        root = tmp_path / f"run{index}"
        root.mkdir()
        hashes.append(content_hashes(scan(root, files, clock_start=start)[1]))
    forward = compute_ops(MemoryView({}), files)
    backward = ChangeSet(tuple(reversed(forward.ops)), forward.description, forward.author)
    assert backward.paths() != forward.paths()
    root = tmp_path / "reversed"
    root.mkdir()
    rig = Rig(root, clock_start=starts[0])
    rig.publish(INTEGRATED, backward)
    hashes.append(content_hashes(rig.run(SecurityScannerAgent(), SCANNER)))
    assert hashes[0].keys() == {REPORT}
    assert hashes[0] == hashes[1] == hashes[2]


def test_the_report_is_a_single_schema_valid_draft(tmp_path: Path) -> None:
    files = {"a.py": "eval('1')\n", "requirements.txt": "x\n"}
    _, result = scan(tmp_path, files)
    assert len(drafts(result, REPORT)) == 1 and len(result.artifacts) == 1
    report = payload(result, REPORT)
    assert validate_payload(REPORT, report) == []
    assert report["schema"] == 1
    assert isinstance(report["files_scanned"], int) and isinstance(report["passed"], bool)


def test_the_report_passes_the_policy_engine_although_the_content_had_secrets(
    tmp_path: Path,
) -> None:
    files = {f"s/{label}.txt": f"v: {value}\n" for label, value in high_secrets().items()}
    rig, result = scan(tmp_path, files)
    report = payload(result, REPORT)
    assert len(report["findings"]) == len(files)  # every secret was found ...
    stored = rig.publish(REPORT, report)
    subject = ArtifactSubject(stored, rig.store.read_text(stored), False)
    verdict = rig.policy.evaluate(subject)
    assert "SEC-001" not in {f.rule_id for f in verdict.findings}  # ... and none is echoed
    assert verdict.blocking == ()
    leaked = fake_aws_key_id()  # control: the same engine does flag an echoed secret
    control = rig.policy.evaluate(ArtifactSubject(make_artifact(REPORT, content=leaked), leaked))
    assert "SEC-001" in {f.rule_id for f in control.findings}


# ----------------------------------------------------------------------------- inputs, autonomy


class RecordingReader:
    """An ``ArtifactReader`` that remembers every artifact kind it was asked for."""

    def __init__(self, inner: ArtifactReader) -> None:
        self.inner = inner
        self.kinds: set[str] = set()

    def latest(self, kind: str, key: str = "") -> Artifact | None:
        self.kinds.add(kind)
        return self.inner.latest(kind, key)

    def active(self, kind: str) -> list[Artifact]:
        self.kinds.add(kind)
        return self.inner.active(kind)

    def read_text(self, artifact: Artifact) -> str:
        self.kinds.add(artifact.kind)
        return self.inner.read_text(artifact)

    def read_bytes(self, artifact: Artifact) -> bytes:
        self.kinds.add(artifact.kind)
        return self.inner.read_bytes(artifact)

    def read_json(self, artifact: Artifact) -> Any:
        self.kinds.add(artifact.kind)
        return self.inner.read_json(artifact)


def test_the_scanner_reads_only_the_integrated_change_set(tmp_path: Path) -> None:
    agent = SecurityScannerAgent()
    rig = Rig(tmp_path)
    rig.publish_changeset(INTEGRATED, {"a.py": "VALUE = 1\n"})  # the only artifact published
    ctx = rig.context(agent, SCANNER)
    recorder = RecordingReader(ctx.artifacts)
    ctx.artifacts = recorder
    result = agent.run(ctx)
    assert recorder.kinds == {INTEGRATED}
    assert payload(result, REPORT)["files_scanned"] == 1
    assert result.usage["tool_calls"] == 0  # nothing was read through the toolbox either


def test_the_scanner_autonomy_is_read_only(tmp_path: Path) -> None:
    profile = SecurityScannerAgent.autonomy
    assert profile.write_globs == ()
    assert profile.can_execute is False
    narrowed = profile.narrow(NODES[SCANNER].autonomy)
    assert (narrowed.write_globs, narrowed.can_execute) == ((), False)
    rig = Rig(tmp_path)
    rig.publish_changeset(INTEGRATED, {"a.txt": "x\n"})
    box = rig.context(SecurityScannerAgent(), SCANNER).toolbox
    with pytest.raises(ToolDenied):
        box.write_text("docs/leak.md", "x\n")
    with pytest.raises(ToolDenied):
        box.run_tests()
    assert box.staged_changeset().ops == ()


@pytest.mark.parametrize(
    "op",
    [
        {"path": "x.txt", "kind": "bogus", "content": "x"},
        {"path": "../x.py", "kind": "create", "content": "x"},
        {"path": "x.py", "kind": "create", "content": None},
    ],
    ids=["unknown_kind", "path_traversal", "no_content"],
)
def test_a_malformed_integrated_change_set_fails_closed(tmp_path: Path, op: dict[str, Any]) -> None:
    rig = Rig(tmp_path)
    rig.publish(INTEGRATED, {"description": "", "author": "", "ops": [op]}, validate=False)
    with pytest.raises((ValueError, TypeError, OrchestratorError)):
        rig.run(SecurityScannerAgent(), SCANNER)  # an error, never a passing report
