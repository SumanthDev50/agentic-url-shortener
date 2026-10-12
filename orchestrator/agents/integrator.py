"""Integrator: merges every ``changeset`` artifact into one conflict-free proposal.

``IntegratorAgent`` (``integrator``) is the barrier node between ``implement.*`` /
``author-tests`` and the verification fan-out.  It reads all ACTIVE ``changeset`` artifacts (one
per producing node, sorted by key) and merges them with ``tools.changeset.merge_changesets``
*against the baseline view* the toolbox serves (the integrate node has no overlay), producing

* ``integrated_changeset`` - relative to the baseline, holding every path that merged cleanly;
* ``integration_report`` - ``sources`` (node id, artifact id, paths), ``merged_paths``,
  ``conflicts`` (path, author node ids, reason), ``stats`` (``diff_stats`` of the merged change
  set) and ``impact_delta``.

Conflicts (two change sets leave one path with different content, a source made against a
different baseline, or a case-insensitive path collision) are *reported*, and the conflict-free
subset is still produced; the ``integration_clean`` gate is what fails on a non-empty
``conflicts`` list.  Automatic re-planning of conflicts (serialising the two producers) is
descoped and documented as a limitation.

``impact_delta`` compares the files the change touches *that already existed* (MODIFY / DELETE
ops; new files cannot be predicted) with ``impact_report.expected_paths`` as exact path sets::

    precision = |actual & predicted| / |predicted|     (1.0 when nothing was predicted)
    recall    = |actual & predicted| / |actual|        (1.0 when no existing file changed)

``predicted_only`` / ``actual_only`` list the differences (sorted).  Without an impact report
(greenfield) nothing is predicted: both ratios are 1.0 if no existing file changed, and recall is
0.0 otherwise.  The report's metadata records whether an impact report was available.

The agent never writes through the toolbox; its ``write_globs`` are ``**`` only because SEC-005
re-checks every op of the integrated change set against the profile, and the merged paths are
exactly the union of what the (individually scope-checked) sources touched (CHG-007 verifies the
derivation).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from orchestrator.agents.base import BaseAgent, changeset_artifact, json_artifact
from orchestrator.core.errors import AgentError, ConfigError, UndeclaredInputError
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    MAX_CHANGESET_OPS,
    AgentContext,
    AgentResult,
    Artifact,
    AutonomyProfile,
    ChangeSet,
    FileOpKind,
    Toolbox,
)
from orchestrator.tools.changeset import diff_stats, merge_changesets

MAX_TOOL_CALLS = 6000  # one cached read per path touched by any source


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


class _ToolboxView:
    """Read-through cache: a path is read from the toolbox at most once (absence is remembered).

    ``merge_changesets`` and ``diff_stats`` probe each touched path several times; caching keeps
    the audit trail and the tool-call budget proportional to the number of files, not of probes.
    """

    def __init__(self, box: Toolbox) -> None:
        self._box = box
        self._seen: dict[str, str | None] = {}

    def _get(self, path: str) -> str | None:
        if path not in self._seen:
            try:
                self._seen[path] = self._box.read_text(path)
            except FileNotFoundError:
                self._seen[path] = None
        return self._seen[path]

    def exists(self, path: str) -> bool:
        return self._get(path) is not None

    def read_text(self, path: str) -> str:
        text = self._get(path)
        if text is None:
            raise FileNotFoundError(path)
        return text

    def list_files(self, glob: str = "**") -> list[str]:
        return self._box.list_files(glob)


def _sources(ctx: AgentContext) -> list[tuple[Artifact, ChangeSet]]:
    """Every ACTIVE change set, authored by its producing node (so conflicts name nodes)."""
    found: list[tuple[Artifact, ChangeSet]] = []
    for art in ctx.all(Kind.CHANGESET):
        try:
            cs = ChangeSet.from_dict(ctx.artifacts.read_json(art))
        except (ConfigError, AttributeError, KeyError, TypeError, ValueError) as exc:
            raise _fatal(f"change set {art.id} is malformed: {type(exc).__name__}") from exc
        found.append((art, replace(cs, author=art.key or cs.author)))
    return found


def _expected_paths(ctx: AgentContext) -> tuple[set[str], bool]:
    """``(predicted paths, report present)``; an undeclared or absent report predicts nothing."""
    try:
        art = ctx.artifacts.latest(Kind.IMPACT_REPORT)
        report = ctx.artifacts.read_json(art) if art is not None else None
    except UndeclaredInputError:
        return set(), False
    paths = report.get("expected_paths") if isinstance(report, dict) else None
    if not isinstance(paths, list):
        return set(), report is not None
    return {p for p in paths if isinstance(p, str)}, True


def impact_delta(predicted: set[str], merged: ChangeSet) -> dict[str, Any]:
    """Precision / recall of the predicted impact against the existing files actually changed."""
    actual = {op.path for op in merged.ops if op.kind is not FileOpKind.CREATE}
    hit = len(actual & predicted)
    return {
        "predicted_only": sorted(predicted - actual),
        "actual_only": sorted(actual - predicted),
        "precision": round(hit / len(predicted), 4) if predicted else 1.0,
        "recall": round(hit / len(actual), 4) if actual else 1.0,
    }


class IntegratorAgent(BaseAgent):
    """Merges all change sets against the baseline (see the module docstring)."""

    agent_id = "integrator"
    role = "integrator"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=("**",),
        can_execute=False,
        max_tool_calls=MAX_TOOL_CALLS,
        max_changed_files=MAX_CHANGESET_OPS,
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        sources = _sources(ctx)
        if not sources:
            raise _fatal("there is no change set to integrate")
        base = _ToolboxView(ctx.toolbox)
        merged = merge_changesets(base, [cs for _, cs in sources])
        final = replace(
            merged.changeset,
            description=f"integrated {len(sources)} change set(s)",
            author=self.agent_id,
        )
        predicted, has_report = _expected_paths(ctx)
        report = json_artifact(
            Kind.INTEGRATION_REPORT,
            {
                "sources": [
                    {"node_id": art.key, "artifact_id": art.id, "paths": sorted(cs.paths())}
                    for art, cs in sources
                ],
                "merged_paths": sorted(final.paths()),
                "conflicts": [
                    {"path": c.path, "authors": list(c.authors), "reason": c.reason}
                    for c in merged.conflicts
                ],
                "stats": diff_stats(base, final).to_dict(),
                "impact_delta": impact_delta(predicted, final),
            },
            metadata={"impact_report_present": has_report},
        )
        return AgentResult(
            artifacts=[changeset_artifact(Kind.INTEGRATED_CHANGESET, final), report],
            rationale=(
                f"merged {len(sources)} change set(s) into {len(final.ops)} file(s); "
                f"{len(merged.conflicts)} conflict(s)."
            ),
            usage=dict(ctx.toolbox.usage()),
        )
