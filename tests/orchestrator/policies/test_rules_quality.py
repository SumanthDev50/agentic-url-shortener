"""QLT-001: changed Python files must compile."""

from __future__ import annotations

from typing import Any

import time
import warnings

import pytest

from orchestrator.core.types import ChangeSetSubject, Finding, Severity
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.rules_quality import PythonCompiles, default_rules
from tests.orchestrator.policies.policy_builders import (
    create,
    cs_subject,
    delete,
    modify,
    py_subject,
    view,
)


def check(subject: ChangeSetSubject) -> list[Finding]:
    return PythonCompiles(PolicyConfig()).check(subject)


def test_rule_identity() -> None:
    (rule,) = default_rules()
    assert (rule.rule_id, rule.severity) == ("QLT-001", Severity.HIGH)


def test_valid_python_passes() -> None:
    source = "import json\n\n\nasync def f(x):\n    return await g(json.dumps(x))\n"
    assert check(py_subject("ok.py", source)) == []
    assert check(py_subject("ok.py", "")) == []
    assert check(py_subject("ok.py", "x = 'caf\u00e9'\n")) == []


@pytest.mark.parametrize(
    ("source", "line"),
    [
        ("def broken(:\n    pass\n", 1),
        ("x = 1\nprint 'py2'\n", 2),
        ("if True:\nx = 1\n", 2),
        ("x = 'unterminated\n", 1),
        ("def f():\n  a = 1\n    b = 2\n", 3),
        ("class C:\n    def m(self)\n        pass\n", 2),
        ("x = (1,\n", None),
    ],
)
def test_syntax_errors_are_high_findings_with_a_line(source: str, line: int | None) -> None:
    findings = check(py_subject("bad.py", source))
    assert [(f.rule_id, f.severity) for f in findings] == [("QLT-001", Severity.HIGH)]
    if line is not None:
        assert findings[0].location == f"bad.py:{line}"
    assert "bad.py" in (findings[0].location or "")


@pytest.mark.parametrize(
    "source",
    ["return 1\n", "await x\n", "def f():\n    nonlocal y\n", "break\n", "class C:\n    yield 1\n"],
)
def test_compile_catches_errors_that_parsing_alone_accepts(source: str) -> None:
    assert [f.severity for f in check(py_subject("late.py", source))] == [Severity.HIGH]


def test_messages_never_echo_source_text() -> None:
    marker = "SENSITIVE_MARKER_VALUE"
    findings = check(py_subject("bad.py", f"x = '{marker}\n"))
    assert findings
    text = " ".join(f"{f.message} {f.evidence} {f.location}" for f in findings)
    assert marker not in text


BOMBS = {
    "open-parens": "(" * 50000,
    "balanced-brackets": "[" * 50000 + "]" * 50000,
    "unary-chain": "-" * 200000 + "1\n",
}


@pytest.mark.parametrize("name", sorted(BOMBS))
def test_hostile_sources_fail_closed_without_crashing(name: str) -> None:
    started = time.perf_counter()
    findings = check(py_subject("bomb.py", BOMBS[name]))
    assert time.perf_counter() - started < 2.0
    assert [f.severity for f in findings] == [Severity.HIGH]


@pytest.mark.parametrize("source", ["x = 1\x00\n", "\ud800 = 1\n"])
def test_nul_and_lone_surrogates_cannot_even_enter_a_change_set_and_never_compile(
    source: str,
) -> None:
    from orchestrator.core.errors import ConfigError
    from orchestrator.policies.pyscan import compile_check

    with pytest.raises(ConfigError):
        py_subject("bomb.py", source)  # FileOp rejects them at the boundary
    hit = compile_check(
        source, "bomb.py"
    )  # ...and the checker itself still fails closed if it sees them
    assert hit is not None


def test_invalid_escape_sequences_do_not_leak_warnings_or_fail() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # the rule must not let SyntaxWarning escape or turn into an error
        assert check(py_subject("esc.py", "import re\nP = re.compile('\\d+')\n")) == []


def test_only_changed_python_files_are_compiled() -> None:
    base = view({"gone.py": "def broken(:\n", "keep.py": "x = 1\n"})
    ops = [
        delete(base, "gone.py"),  # deleting a broken file is fine
        create("notes.md", "def broken(:\n"),
        create("data.json", "{not json"),
        create("stub.pyi", "def (:\n"),
        create("script.PY", "x = 1\n"),
    ]
    assert check(cs_subject(ops, base)) == []


def test_every_broken_file_is_reported_and_extension_check_is_case_insensitive() -> None:
    base = view({"m.py": "x = 1\n"})
    ops = [
        modify(base, "m.py", "def (:\n"),
        create("n.py", "class :\n"),
        create("UPPER.PY", "x = = 1\n"),
    ]
    assert sorted((f.location or "").split(":")[0] for f in check(cs_subject(ops, base))) == [
        "UPPER.PY",
        "m.py",
        "n.py",
    ]


def test_oversize_python_fails_closed() -> None:
    from orchestrator.policies.pyscan import compile_check

    hit = compile_check("x = 1\n" * 200_000, "big.py")
    assert hit is not None and hit.code == "unscannable"
