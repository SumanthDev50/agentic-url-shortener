"""Common base class for policy rules.

A rule is a small class (``rule_id``, ``category``, ``title``, ``severity``, ``subjects``,
``check``) as required by ``types.PolicyRule``.  ``RuleBase`` removes the boilerplate: ``check``
dispatches on the subject type to ``check_changeset`` / ``check_artifact`` / ``check_action`` /
``check_plan`` / ``check_run`` (default: no findings), and ``finding`` / ``from_hit`` build
``Finding`` objects with the rule's identity.

``from_hit`` is where *fail closed* is centralised for scanners: a detector that could not analyse
its input (``unscannable`` / ``unparseable``) always becomes a HIGH finding, whatever the rule's
default severity.
"""

from __future__ import annotations

from collections.abc import Iterable

from orchestrator.core.types import (
    ActionSubject,
    ArtifactSubject,
    ChangeSet,
    ChangeSetSubject,
    FileOp,
    FileOpKind,
    FileView,
    Finding,
    PlanSubject,
    PolicyCategory,
    PolicySubject,
    RunSubject,
    Severity,
    SubjectKind,
)
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.hits import UNPARSEABLE, UNSCANNABLE, Hit
from orchestrator.tools.changeset import diff_stats

FAIL_CLOSED_CODES = frozenset({UNSCANNABLE, UNPARSEABLE})


class RuleBase:
    """Subclasses set the five class attributes and override the ``check_*`` hooks."""

    rule_id: str = ""
    category: PolicyCategory = PolicyCategory.SECURITY
    title: str = ""
    severity: Severity = Severity.HIGH
    subjects: frozenset[SubjectKind] = frozenset()

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config if config is not None else PolicyConfig()

    # -- dispatch -----------------------------------------------------------------
    def check(self, subject: PolicySubject) -> list[Finding]:
        if isinstance(subject, ChangeSetSubject):
            return self.check_changeset(subject)
        if isinstance(subject, ArtifactSubject):
            return self.check_artifact(subject)
        if isinstance(subject, ActionSubject):
            return self.check_action(subject)
        if isinstance(subject, PlanSubject):
            return self.check_plan(subject)
        if isinstance(subject, RunSubject):
            return self.check_run(subject)
        return []

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        return []

    def check_artifact(self, subject: ArtifactSubject) -> list[Finding]:
        return []

    def check_action(self, subject: ActionSubject) -> list[Finding]:
        return []

    def check_plan(self, subject: PlanSubject) -> list[Finding]:
        return []

    def check_run(self, subject: RunSubject) -> list[Finding]:
        return []

    # -- finding construction ---------------------------------------------------------
    def finding(
        self,
        message: str,
        *,
        severity: Severity | None = None,
        location: str | None = None,
        evidence: str | None = None,
        remediation: str | None = None,
    ) -> Finding:
        return Finding(
            self.rule_id,
            self.category,
            severity if severity is not None else self.severity,
            message,
            location=location,
            evidence=evidence or None,
            remediation=remediation,
        )

    def from_hit(self, hit: Hit, remediation: str | None = None) -> Finding:
        severity = hit.severity
        if hit.code in FAIL_CLOSED_CODES:
            severity = max(Severity.HIGH, self.severity)
        return self.finding(
            hit.message,
            severity=severity,
            location=hit.location or None,
            evidence=hit.evidence,
            remediation=remediation,
        )

    def from_hits(self, hits: Iterable[Hit], remediation: str | None = None) -> list[Finding]:
        return [self.from_hit(h, remediation) for h in hits]


def written_ops(subject: ChangeSetSubject, suffix: str = "") -> list[FileOp]:
    """CREATE/MODIFY ops (those that carry new content) whose path ends with ``suffix``
    (case-insensitive).
    """
    return [
        op
        for op in subject.changeset.ops
        if op.kind != FileOpKind.DELETE
        and op.content is not None
        and op.path.lower().endswith(suffix)
    ]


def python_ops(subject: ChangeSetSubject) -> list[FileOp]:
    return written_ops(subject, ".py")


_DIFF_LINE_LIMIT = (
    4000  # beyond this a MODIFY is costed by an upper bound instead of a quadratic diff
)


def read_or_empty(view: FileView, path: str) -> str:
    """Current text of ``path`` in ``view``; ``""`` when the file does not exist."""
    try:
        return view.read_text(path)
    except FileNotFoundError:
        return ""


def changed_lines(view: FileView, op: FileOp) -> int:
    """Lines added + removed by ``op`` (an upper bound for very large MODIFYs: real diffs are
    quadratic).
    """
    old, new = read_or_empty(view, op.path), op.content or ""
    n_old, n_new = len(old.splitlines()), len(new.splitlines())
    if op.kind == FileOpKind.CREATE:
        return n_new
    if op.kind == FileOpKind.DELETE:
        return n_old
    if n_old + n_new > _DIFF_LINE_LIMIT:
        return n_old + n_new
    stats = diff_stats(view, ChangeSet((op,)))
    return stats.lines_added + stats.lines_removed
