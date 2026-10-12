"""Promoter agent (``promote`` node) and the ``apply_bundle`` compensator.

The promoter turns the reviewed change sets into the release artifacts:

* ``release_patch`` - ``tools.changeset.to_unified_diff(baseline, merged)`` where ``merged`` is
  ``integrated_changeset`` followed by ``docs_changeset`` re-based onto the BASELINE (``docs`` was
  made
  against baseline + integrated).  Deterministic, LF, git style; text only.
* ``release_bundle`` - ``{mode, patch_sha256, files, base_manifest_hash, receipt}``.

**Bundle mode** (default) has no side effect.  **Apply mode** (``ctx.params['mode'] == 'apply'``)
applies the merged change set through ``ctx.effects`` (all-or-nothing, base hashes verified against
the target tree by the effect port) and stores ``receipt.to_dict()`` in the bundle; ``mode='apply'``
without an effect port is an error, never a silent downgrade to a bundle.  The undo is
:class:`ApplyBundleCompensator` (``effects.revert`` from the persisted receipt).

The patch is relative to the baseline, so the agent needs the baseline tree.  It reads it through
``ScopedToolbox.baseline_view()`` (guarded, audited; the engine hands the baseline to the toolbox
factory).  A toolbox without a baseline falls back to its own view; with the standard overlay wiring
of this node that makes the integrated change set fail to apply and the attempt fails FATAL with an
explicit message - it never emits a patch against the wrong tree.  Refuses to promote when the
readiness verdict is not ``GO`` (defence in depth: the release gates and the human approval come
first).  ``base_manifest_hash`` has the same definition as ``BaselineSnapshot.manifest_hash``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orchestrator.agents.base import BaseAgent, json_artifact
from orchestrator.core.errors import AgentError, ConfigError
from orchestrator.core.fsutil import sha256_text
from orchestrator.core.sanitize import sanitize_for_terminal
from orchestrator.core.schemas import Kind
from orchestrator.core.serde import hash_jsonable
from orchestrator.core.types import (
    AgentContext,
    AgentResult,
    ApplyReceipt,
    ArtifactDraft,
    AutonomyProfile,
    ChangeSet,
    DecisionDraft,
    DecisionKind,
    EffectPort,
    FileView,
    GovernanceView,
)
from orchestrator.tools.changeset import ChangeSetConflict, compute_ops, to_unified_diff
from orchestrator.tools.effects import EffectError
from orchestrator.tools.toolbox import baseline_view_of
from orchestrator.tools.workspace import OverlayView

MODES = ("bundle", "apply")
MAX_BASELINE_READS = 6000
_SAMPLE = 8


def _fatal(message: str) -> AgentError:
    return AgentError(message, kind="fatal", retryable=False)


def _text(value: object, limit: int = 200) -> str:
    return sanitize_for_terminal(value, limit)


class _ReadOnce:
    """A ``FileView`` that asks ``inner`` for each path at most once (reads are tool calls)."""

    def __init__(self, inner: FileView) -> None:
        self._inner = inner
        self._texts: dict[str, str | None] = {}
        self._listed: list[str] | None = None

    def exists(self, path: str) -> bool:
        return self._load(path) is not None

    def read_text(self, path: str) -> str:
        text = self._load(path)
        if text is None:
            raise FileNotFoundError(path)
        return text

    def list_files(self, glob: str = "**") -> list[str]:
        if glob != "**":
            return self._inner.list_files(glob)
        if self._listed is None:
            self._listed = self._inner.list_files("**")
        return list(self._listed)

    def _load(self, path: str) -> str | None:
        if path not in self._texts:
            try:
                self._texts[path] = self._inner.read_text(path)
            except FileNotFoundError:
                self._texts[path] = None
        return self._texts[path]


class PromoterAgent(BaseAgent):
    agent_id = "promoter"
    role = "promoter"
    version = "1"
    autonomy = AutonomyProfile(
        read_globs=("**",),
        write_globs=(),
        can_execute=False,
        max_tool_calls=MAX_BASELINE_READS,
    )

    def run(self, ctx: AgentContext) -> AgentResult:
        mode = ctx.params.get("mode", "bundle")
        if mode not in MODES:
            raise _fatal(f"mode must be one of {list(MODES)}")
        readiness = ctx.json(Kind.READINESS_REPORT)
        if readiness.get("verdict") != "GO":
            raise _fatal("the readiness verdict is not GO: refusing to promote")
        integrated = ChangeSet.from_dict(ctx.json(Kind.INTEGRATED_CHANGESET))
        docs = ChangeSet.from_dict(ctx.json(Kind.DOCS_CHANGESET))
        baseline = _ReadOnce(baseline_view_of(ctx.toolbox) or ctx.toolbox)

        try:
            overlay = OverlayView(baseline, [integrated, docs], strict=True)
            merged = compute_ops(
                baseline, dict(overlay.writes), description="release change set", author="promoter"
            )
        except ChangeSetConflict as exc:
            raise _fatal(
                "the change sets do not apply to the baseline (is the baseline handed to the "
                f"toolbox?): {_text(exc.path)}: {_text(exc.reason)}"
            ) from exc
        except ConfigError as exc:  # e.g. a case-insensitive path clash, or too many ops
            raise _fatal(f"the merged change set is not valid: {_text(exc)}") from exc
        if not merged.ops:
            raise _fatal("the merged change set is empty: nothing to release")
        patch = to_unified_diff(baseline, merged)
        files = sorted(merged.paths())
        manifest = {
            path: sha256_text(baseline.read_text(path)) for path in baseline.list_files("**")
        }

        receipt: ApplyReceipt | None = None
        if mode == "apply":
            receipt = self._apply(ctx.effects, merged)
        bundle = {
            "mode": "applied" if receipt is not None else "bundle",
            "patch_sha256": sha256_text(patch),
            "files": files,
            "base_manifest_hash": hash_jsonable(manifest),
            "receipt": receipt.to_dict() if receipt is not None else None,
        }
        decision = DecisionDraft(
            DecisionKind.RELEASE,
            f"Release {'applied to the target' if receipt else 'bundled'}: {len(files)} file(s)",
            rationale="The readiness verdict is GO; the patch is the merged integrated and docs "
            "change sets relative to the baseline.",
            outcome=str(bundle["mode"]),
            data={"patch_sha256": bundle["patch_sha256"], "files": len(files)},
        )
        return AgentResult(
            artifacts=[
                ArtifactDraft(
                    kind=Kind.RELEASE_PATCH,
                    content=patch,
                    media_type="text/x-patch",
                    filename="release.patch",
                ),
                json_artifact(Kind.RELEASE_BUNDLE, bundle),
            ],
            decisions=[decision],
            rationale=(
                f"{bundle['mode']}: {len(files)} file(s), "
                f"sha256 {str(bundle['patch_sha256'])[:12]}."
            ),
            usage=dict(ctx.toolbox.usage()),
        )

    @staticmethod
    def _apply(effects: EffectPort | None, merged: ChangeSet) -> ApplyReceipt:
        if effects is None:
            raise _fatal("apply mode needs an effect port; refusing to fall back to a bundle")
        try:
            return effects.apply(merged)
        except (ChangeSetConflict, EffectError) as exc:
            raise _fatal(f"the target refused the change set: {_text(exc)}") from exc


class ApplyBundleCompensator:
    """``types.Compensator`` of the promote node: reverts what ``effects.apply`` did.

    Runs from the persisted receipt (an ``ApplyReceipt`` or its ``to_dict()`` form).  The effect
    port
    reverts a file only if it still holds what the apply left there and is idempotent; a refusal
    propagates so the engine records ``ROLLBACK_FAILED``.
    """

    def compensate(
        self,
        receipt: ApplyReceipt | Mapping[str, Any],
        *,
        effects: EffectPort,
        run_view: GovernanceView,
    ) -> str:
        real = receipt if isinstance(receipt, ApplyReceipt) else _receipt_from(receipt)
        effects.revert(real)
        shown = ", ".join(sorted(real.applied)[:_SAMPLE])
        more = f" (+{len(real.applied) - _SAMPLE} more)" if len(real.applied) > _SAMPLE else ""
        return f"reverted {len(real.applied)} file(s) applied to {real.target_id}: {shown}{more}"


def _receipt_from(raw: Mapping[str, Any]) -> ApplyReceipt:
    try:
        return ApplyReceipt.from_dict(raw)
    # AttributeError: a non-mapping where a mapping is expected (e.g. ``reverse`` is a string)
    except (AttributeError, KeyError, TypeError, ValueError, ConfigError) as exc:
        raise _fatal("the persisted receipt is not a valid ApplyReceipt") from exc
