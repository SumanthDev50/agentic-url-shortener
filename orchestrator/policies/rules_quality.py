"""QLT-001: changed Python files must compile (catalogue: docs/architecture.md section 1.5)."""

from __future__ import annotations

from orchestrator.core.types import (
    ChangeSetSubject,
    Finding,
    PolicyCategory,
    PolicyRule,
    Severity,
    SubjectKind,
)
from orchestrator.policies.base import RuleBase, python_ops
from orchestrator.policies.config import PolicyConfig
from orchestrator.policies.pyscan import compile_check


class PythonCompiles(RuleBase):
    """``compile()`` (never ``exec``) over every created/modified ``.py`` file.

    Full compilation, not just parsing: it also rejects errors the parser accepts such as ``return``
    outside a function.  The language level is that of the interpreter running the policy engine.
    """

    rule_id = "QLT-001"
    category = PolicyCategory.QUALITY
    title = "Changed Python files must be syntactically valid"
    severity = Severity.HIGH
    subjects = frozenset({SubjectKind.CHANGESET})

    def check_changeset(self, subject: ChangeSetSubject) -> list[Finding]:
        hits = (compile_check(op.content or "", op.path) for op in python_ops(subject))
        return self.from_hits(
            (h for h in hits if h is not None),
            "Fix the syntax error and re-run the syntax check before proposing the change.",
        )


def default_rules(config: PolicyConfig | None = None) -> list[PolicyRule]:
    return [PythonCompiles(config)]
