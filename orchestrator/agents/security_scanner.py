"""Security scanner agent (``security-scan`` node): a defence-in-depth summary, not the guardrail.

The authoritative guardrails are the ``PolicyEngine`` rules the engine evaluates on every artifact
(SEC-001 ... SEC-008, CHG-*, CMP-*).  As an L3 agent this module must not import policies, so it
keeps
its own *small* set of checks over the integrated change set and reports them in
``security_report``:

* ``SCN-001`` secret-shaped strings (linear-time regexes; the report never echoes a match, only the
  pattern label and ``path:line``);
* ``SCN-002`` dangerous calls found through ``ast`` (``eval`` / ``exec`` / ``os.system`` /
  ``pickle.loads`` / ``shell=True`` ... , import aliases resolved one level deep);
* ``SCN-003`` Python files that do not parse (a file that cannot be analysed is a finding, never a
  pass);
* ``SCN-004`` changes to protected or sensitive paths (CI, Dockerfiles, env and key files: HIGH;
  dependency manifests: MEDIUM).

``passed`` means *no HIGH or CRITICAL finding* (the ``security_clean`` gate).  Known limits:
patterns are
heuristics, dynamic code and indirect calls are invisible to a static name check, and only files the
change set creates or modifies are scanned.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from orchestrator.agents.base import BaseAgent, json_artifact
from orchestrator.core.paths import casefold_key
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    MAX_FILE_BYTES,
    AgentContext,
    AgentResult,
    AutonomyProfile,
    ChangeSet,
    FileOp,
    FileOpKind,
    Finding,
    PolicyCategory,
    Severity,
)

RULES = ("SCN-001", "SCN-002", "SCN-003", "SCN-004")
MAX_FINDINGS = 200
MAX_MATCHES_PER_PATTERN = 10
_SAFE_PREFIX = "pattern:"


@dataclass(frozen=True)
class _SecretPattern:
    label: str
    regex: re.Pattern[str]
    severity: Severity


# Every expression below is linear-time: bounded repetitions over disjoint character classes, no
# nested quantifiers, no back-references.
_SECRETS: tuple[_SecretPattern, ...] = (
    _SecretPattern(
        "aws_access_key_id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), Severity.HIGH
    ),
    _SecretPattern("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"), Severity.HIGH),
    _SecretPattern(
        "github_fine_grained_token",
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,255}\b"),
        Severity.HIGH,
    ),
    _SecretPattern(
        "slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,200}\b"), Severity.HIGH
    ),
    _SecretPattern(
        "private_key_block", re.compile(r"-----BEGIN [A-Z ]{0,30}PRIVATE KEY-----"), Severity.HIGH
    ),
    _SecretPattern(
        "json_web_token",
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
        Severity.HIGH,
    ),
    _SecretPattern("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"), Severity.HIGH),
    _SecretPattern(
        "stripe_live_key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{16,99}\b"), Severity.HIGH
    ),
    _SecretPattern(
        "provider_secret_key", re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{32,200}\b"), Severity.HIGH
    ),
    _SecretPattern(
        "url_with_credentials",
        re.compile(r"\b[a-z][a-z0-9+.-]{1,20}://[^\s/:@]{1,64}:[^\s/@]{3,64}@"),
        Severity.MEDIUM,
    ),
    _SecretPattern(
        "hardcoded_credential",
        re.compile(
            r"\b(?:password|passwd|secret|api[_-]?key|token)\b"
            r"\s{0,8}[:=]\s{0,8}[\"'][^\"'\s]{8,64}[\"']",
            re.IGNORECASE,
        ),
        Severity.MEDIUM,
    ),
)

_HIGH_CALLS = frozenset(
    {
        "eval",
        "exec",
        "__import__",
        "os.system",
        "os.popen",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "pickle.loads",
        "pickle.load",
        "marshal.loads",
        "marshal.load",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
    }
)
_MEDIUM_CALLS = frozenset(
    {
        "compile",
        "yaml.load",
        "subprocess.run",
        "subprocess.call",
        "subprocess.check_call",
        "subprocess.check_output",
        "subprocess.Popen",
    }
)
_SUBPROCESS_PREFIX = "subprocess."

_HIGH_PATH_PREFIXES = (".github/", ".gitlab/", ".circleci/")
_HIGH_PATH_NAMES = frozenset({".gitlab-ci.yml", ".env", "id_rsa", "id_ed25519", "id_ecdsa"})
_HIGH_PATH_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".jks", ".keystore")
_MEDIUM_PATH_NAMES = frozenset(
    {"pyproject.toml", "setup.py", "setup.cfg", "poetry.lock", "pipfile", "pipfile.lock"}
)
_PAST_TENSE = {
    FileOpKind.CREATE: "created",
    FileOpKind.MODIFY: "modified",
    FileOpKind.DELETE: "deleted",
}


# ----------------------------------------------------------------------------- findings


def _finding(
    rule_id: str,
    severity: Severity,
    message: str,
    location: str,
    remediation: str,
    evidence: str | None = None,
) -> Finding:
    return Finding(
        rule_id,
        PolicyCategory.SECURITY,
        severity,
        message,
        location=location,
        evidence=evidence,
        remediation=remediation,
    )


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_secrets(path: str, text: str) -> list[Finding]:
    found: list[Finding] = []
    for pattern in _SECRETS:
        for index, match in enumerate(pattern.regex.finditer(text)):
            if index >= MAX_MATCHES_PER_PATTERN:
                break
            found.append(
                _finding(
                    "SCN-001",
                    pattern.severity,
                    f"secret-shaped string ({pattern.label})",
                    f"{path}:{_line_of(text, match.start())}",
                    "Remove the value, rotate it if it was real, and read it from the environment.",
                    evidence=f"{_SAFE_PREFIX}{pattern.label}",  # never the match itself
                )
            )
    return found


def _aliases(tree: ast.AST) -> dict[str, str]:
    """``import subprocess as sp`` / ``from os import system`` -> dotted real names."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                names[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return names


def _callee(node: ast.expr, aliases: Mapping[str, str]) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        head = aliases.get(node.id, node.id)
        return ".".join([head, *reversed(parts)])
    return ""


def _shell_label(call: ast.Call) -> str | None:
    """Label when ``shell=`` is anything but a falsy literal (a non-literal value fails closed)."""
    for kw in call.keywords:
        if kw.arg == "shell":
            if isinstance(kw.value, ast.Constant):
                return "subprocess(shell=True)" if kw.value.value else None
            return "subprocess(shell=<non-literal>)"
    return None


def scan_python(path: str, source: str) -> list[Finding]:
    try:
        tree = ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        line = getattr(exc, "lineno", None) or 1
        return [
            _finding(
                "SCN-003",
                Severity.HIGH,
                f"python source cannot be analysed ({type(exc).__name__})",
                f"{path}:{line}",
                "Fix the syntax error; unparseable code is never accepted as clean.",
            )
        ]
    aliases = _aliases(tree)
    found: list[Finding] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee(node.func, aliases)
        shell = _shell_label(node) if name.startswith(_SUBPROCESS_PREFIX) else None
        if name in _HIGH_CALLS:
            severity, label = Severity.HIGH, name
        elif shell is not None:
            severity, label = Severity.HIGH, shell
        elif name in _MEDIUM_CALLS:
            severity, label = Severity.MEDIUM, name
        else:
            continue
        found.append(  # only names from the constant sets above are ever echoed
            _finding(
                "SCN-002",
                severity,
                f"dangerous or process-spawning call: {label}",
                f"{path}:{node.lineno}",
                "Pass argument lists, use safe loaders and avoid dynamic code execution.",
                evidence=label,
            )
        )
    return found


def scan_path(op: FileOp) -> list[Finding]:
    key = casefold_key(op.path)
    name = key.rsplit("/", 1)[-1]
    if (
        key.startswith(_HIGH_PATH_PREFIXES)
        or name in _HIGH_PATH_NAMES
        or name.startswith((".env.", "dockerfile", "docker-compose"))
        or name.endswith(_HIGH_PATH_SUFFIXES)
    ):
        severity = Severity.HIGH
    elif name in _MEDIUM_PATH_NAMES or (name.startswith("requirements") and name.endswith(".txt")):
        severity = Severity.MEDIUM
    else:
        return []
    return [
        _finding(
            "SCN-004",
            severity,
            f"protected or sensitive path is {_PAST_TENSE[op.kind]}",
            op.path,
            "Changes to CI, containers, secrets and dependencies need a dedicated reviewed change.",
        )
    ]


def _scan_ops(cs: ChangeSet) -> Iterator[tuple[int, list[Finding]]]:
    for op in sorted(cs.ops, key=lambda o: o.path):
        findings = scan_path(op)
        scanned = 0
        if op.kind != FileOpKind.DELETE and op.content is not None:
            scanned = 1
            findings.extend(scan_secrets(op.path, op.content))
            if casefold_key(op.path).endswith(".py"):  # same case-insensitivity as scan_path
                findings.extend(scan_python(op.path, op.content))
        yield scanned, findings


class SecurityScannerAgent(BaseAgent):
    agent_id = "security-scanner"
    role = "security-scanner"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(),
        can_execute=False,
        max_tool_calls=16,
        max_file_bytes=MAX_FILE_BYTES,
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        changeset = ChangeSet.from_dict(ctx.json(Kind.INTEGRATED_CHANGESET))
        findings: list[Finding] = []
        files_scanned = 0
        for scanned, found in _scan_ops(changeset):
            files_scanned += scanned
            findings.extend(found)
        findings.sort(key=lambda f: (-int(f.severity), f.rule_id, f.location or "", f.message))
        omitted = max(0, len(findings) - MAX_FINDINGS)
        kept = findings[:MAX_FINDINGS]
        if omitted:
            kept.append(
                _finding(
                    "SCN-004",
                    Severity.LOW,
                    f"{omitted} further finding(s) omitted from the report",
                    "report",
                    "Fix the listed findings first; the scan is repeated on the next attempt.",
                )
            )
        blocking = sum(1 for f in findings if f.severity >= Severity.HIGH)
        payload = {
            "rules_evaluated": list(RULES),
            "findings": [f.to_dict() for f in kept],
            "files_scanned": files_scanned,
            "passed": blocking == 0,
        }
        return AgentResult(
            artifacts=[json_artifact(Kind.SECURITY_REPORT, payload)],
            rationale=(
                f"{files_scanned} file(s) scanned, {len(findings)} finding(s), "
                f"{blocking} HIGH or above."
            ),
            usage=dict(ctx.toolbox.usage()),
        )
