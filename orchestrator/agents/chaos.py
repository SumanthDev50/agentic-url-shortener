"""Fault injection: ``ChaosAgent`` wraps an agent and makes it fail in a realistic, detectable way.

The point is to make the governance machinery demonstrable and testable: retries, fallback,
remediation, policy denial and safe-stop all need something to react to.  Every fault is
deterministic (no randomness, no clock) and goes through the same toolbox as honest work, so the
engine's own checks (not this module) are what catch it.

``FAULTS``:

* ``transient``      raise ``TransientAgentError`` (retryable) instead of working.
* ``secret_leak``    after the wrapped agent, stage one more file holding a credential-shaped
  literal (SEC-001, HIGH -> the attempt is denied); where the write scope leaves no room for a new
  file (file-exact scopes) the literal is appended to a Python file the agent already edits.  The
  fake secret is assembled at runtime from fragments; no secret-shaped literal exists in this
  source file.
* ``migration_edit`` after the wrapped agent, modify a migration that already exists in the
  baseline (CHG-001, HIGH -> denied).  Migrations are recognised by the policy's default glob
  ``**/migrations/*.sql``.
* ``path_escape``    call ``toolbox.write_text('../outside.txt')`` -> ``PathViolation`` (CRITICAL
  -> the engine safe-stops).  The wrapped agent never runs.
* ``regression``     after the wrapped agent, append a statement that raises at import time to one
  of its Python source files: the file still compiles (QLT-001 passes) but every test importing it
  fails, so ``verify`` rejects the change and the remediation edge routes back to the implementer.

The content faults (``secret_leak``, ``migration_edit``, ``regression``) edit through
``ctx.toolbox`` and rebuild the ``changeset`` artifact from ``toolbox.staged_changeset()``, because
the engine requires a node's change-set ops to equal the toolbox's staged ops.  They therefore need
a wrapped agent that stages its edits through the toolbox and may write where the fault wants to
write (implementer, test-author).  When a fault cannot be applied (no change set, edits not staged
through the toolbox, no existing migration, target outside the effective ``write_globs``) the
wrapped result is returned unchanged and the rationale says so: a fault is never faked.

When does a fault fire?  ``ChaosAgent(inner, fault, attempts=None, *, on_fallback=False)`` fires
iff

1. ``ctx.attempt_no <= attempts`` (the node's global attempt number; ``attempts=None`` means 1, so
   the default is a one-shot fault);
2. the attempt is not a fallback attempt, unless ``on_fallback=True`` (the fallback repairs);
3. ``ctx.feedback`` is empty, *or* it names the fault's own rule (SEC-001, CHG-001, SEC-005).
   Feedback about anything else - for example the failing tests that route a regression back
   through remediation - means the engine asks for a repair and the fault stands down.

Rule 3 is what makes ``attempts=99`` on ``migration_edit`` demonstrate *denied -> retried with
findings -> still faulty -> fallback takes over*, while the default one-shot faults are repaired by
the first retry or remediation round.  An exhausted-style double failure is configured by wrapping
BOTH the primary and the fallback agent in the registry, with an ``attempts`` that covers both
epochs (primary attempts + fallback attempts, 3 + 2 by default) and ``on_fallback=True`` on the
fallback's wrapper.

The wrapper reports the same ``agent_id`` / ``role`` / ``version`` / ``autonomy`` as the wrapped
agent (the engine and the fingerprints see no difference); a fired fault is announced by a
``toolbox.note`` ("chaos fault injected") and by the ``chaos[<fault>]`` prefix of the rationale.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

from orchestrator.core.errors import ConfigError, PathViolation, TransientAgentError
from orchestrator.core.paths import match_any
from orchestrator.core.schemas import Kind
from orchestrator.core.types import (
    Agent,
    AgentContext,
    AgentResult,
    AutonomyProfile,
    ChangeSet,
    FileOpKind,
    Toolbox,
)

FAULTS: tuple[str, ...] = (
    "transient",
    "secret_leak",
    "migration_edit",
    "path_escape",
    "regression",
)
MAX_ATTEMPTS = 1000
# The policy's default ``PolicyConfig.migrations_glob`` (L3 may not import L2, so it is repeated).
MIGRATIONS_GLOB = "**/migrations/*.sql"
ESCAPE_PATH = "../outside.txt"
LEAK_FILE = "chaos_leaked_credentials.py"
REGRESSION_STATEMENT = 'raise RuntimeError("chaos: injected regression")\n'

# The rule a fault provokes: feedback naming it does not stop the fault (rule 3 of the docstring).
_OWN_RULES: Mapping[str, frozenset[str]] = {
    "transient": frozenset(),
    "secret_leak": frozenset({"SEC-001"}),
    "migration_edit": frozenset({"CHG-001"}),
    "path_escape": frozenset({"SEC-005"}),
    "regression": frozenset(),
}


def fake_access_key_id() -> str:
    """A credential-shaped, invalid string, assembled at runtime (never a literal)."""
    return "".join(("AK", "IA", "QX7KW2MZ", "4N8TB6P3"))


def leak_line() -> str:
    """One assignment of the runtime-assembled fake credential (valid Python)."""
    return f'AWS_ACCESS_KEY_ID = "{fake_access_key_id()}"'


def leak_module_text() -> str:
    """Source of the planted file."""
    doc = '"""Planted by the chaos agent: an invalid, credential-shaped literal."""'
    return "\n".join((doc, "", leak_line(), ""))


def _literal_dir(glob: str) -> str:
    """The wildcard-free directory prefix of ``glob`` (``a/b/**`` -> ``a/b``, ``**`` -> ``''``)."""
    kept: list[str] = []
    for segment in glob.split("/")[:-1]:
        if any(ch in segment for ch in "*?["):
            break
        kept.append(segment)
    return "/".join(kept)


def _is_source(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return (
        path.endswith(".py")
        and not path.startswith("tests/")
        and not name.startswith("test_")
        and name not in ("conftest.py", "__init__.py")
    )


class ChaosAgent:
    """Wraps ``inner`` and injects ``fault`` (the module docstring has the exact firing rules)."""

    def __init__(
        self,
        inner: Agent,
        fault: str,
        attempts: int | None = None,
        *,
        on_fallback: bool = False,
    ) -> None:
        if not isinstance(inner, Agent):
            raise ConfigError("ChaosAgent needs an Agent to wrap")
        if fault not in FAULTS:
            raise ConfigError(f"unknown fault {str(fault)[:40]!r}; known faults: {list(FAULTS)}")
        if attempts is not None and (
            isinstance(attempts, bool)
            or not isinstance(attempts, int)
            or not 1 <= attempts <= MAX_ATTEMPTS
        ):
            raise ConfigError(f"attempts must be an integer in [1, {MAX_ATTEMPTS}]")
        self._inner = inner
        self.fault = fault
        self.attempts = 1 if attempts is None else attempts
        self.on_fallback = bool(on_fallback)
        self.agent_id = inner.agent_id
        self.role = inner.role
        self.version = inner.version
        self.autonomy: AutonomyProfile = inner.autonomy
        self._injectors: Mapping[str, Callable[[AgentContext], AgentResult]] = {
            "transient": self._transient,
            "secret_leak": self._secret_leak,
            "migration_edit": self._migration_edit,
            "path_escape": self._path_escape,
            "regression": self._regression,
        }

    @property
    def note(self) -> str:
        """The chaos note: what this wrapper does (used in rationales and scenario reports)."""
        return f"chaos[{self.fault}]"

    # -- engine entry point ---------------------------------------------------------------------
    def run(self, ctx: AgentContext) -> AgentResult:
        if not self._fires(ctx):
            return self._inner.run(ctx)
        ctx.toolbox.note("chaos fault injected", fault=self.fault, attempt_no=ctx.attempt_no)
        return self._injectors[self.fault](ctx)

    def _fires(self, ctx: AgentContext) -> bool:
        if ctx.attempt_no > self.attempts:
            return False
        if ctx.is_fallback and not self.on_fallback:
            return False
        if not ctx.feedback:
            return True
        own = _OWN_RULES[self.fault]
        return any(f.rule_id in own for f in ctx.feedback)

    # -- faults that replace the attempt ----------------------------------------------------------
    def _transient(self, ctx: AgentContext) -> AgentResult:
        raise TransientAgentError(
            f"chaos: injected transient failure (attempt {ctx.attempt_no} of {self.attempts})",
            details={"chaos": self.fault},
        )

    def _path_escape(self, ctx: AgentContext) -> AgentResult:
        ctx.toolbox.write_text(ESCAPE_PATH, "chaos: path escape probe\n")
        # Only reachable with a toolbox that failed to reject the path: fail closed all the same.
        raise PathViolation("chaos: the toolbox accepted a path outside the workspace")

    # -- faults that taint the wrapped agent's change set -----------------------------------------
    def _secret_leak(self, ctx: AgentContext) -> AgentResult:
        return self._taint(ctx, self._plant_secret)

    def _migration_edit(self, ctx: AgentContext) -> AgentResult:
        return self._taint(ctx, self._edit_migration)

    def _regression(self, ctx: AgentContext) -> AgentResult:
        return self._taint(ctx, self._break_source)

    def _taint(
        self, ctx: AgentContext, mutate: Callable[[Toolbox, AgentContext, ChangeSet], str | None]
    ) -> AgentResult:
        result = self._inner.run(ctx)
        box = ctx.toolbox
        index = next((i for i, d in enumerate(result.artifacts) if d.kind == Kind.CHANGESET), None)
        if index is None:
            return self._unchanged(result, "the wrapped agent produced no change set")
        draft = result.artifacts[index]
        if not isinstance(draft.content, Mapping):
            return self._unchanged(result, "the change set is not readable")
        try:
            own = ChangeSet.from_dict(draft.content)
        except (ConfigError, KeyError, TypeError, ValueError):
            return self._unchanged(result, "the change set is not readable")
        if box.staged_changeset().ops != own.ops:
            return self._unchanged(
                result, "the wrapped agent did not stage its edits through the toolbox"
            )
        done = mutate(box, ctx, own)
        if done is None:
            return self._unchanged(result, "no applicable target for this fault")
        tainted = replace(box.staged_changeset(own.description), author=own.author)
        artifacts = list(result.artifacts)
        artifacts[index] = replace(draft, content=tainted.to_dict())
        return replace(
            result,
            artifacts=artifacts,
            rationale=f"{self.note}: {done}. {result.rationale}".strip(),
            usage=dict(box.usage()),
        )

    def _unchanged(self, result: AgentResult, why: str) -> AgentResult:
        return replace(
            result, rationale=f"{self.note}: not applied ({why}). {result.rationale}".strip()
        )

    def _profile(self, ctx: AgentContext) -> AutonomyProfile:
        """The effective autonomy of the wrapped agent: what the engine hands the toolbox."""
        return self._inner.autonomy.narrow(ctx.node.autonomy)

    def _plant_secret(self, box: Toolbox, ctx: AgentContext, own: ChangeSet) -> str | None:
        globs = self._profile(ctx).write_globs
        op_dirs = sorted({p.rsplit("/", 1)[0] if "/" in p else "" for p in own.paths()})
        for directory in dict.fromkeys([*op_dirs, *(_literal_dir(g) for g in globs)]):
            path = f"{directory}/{LEAK_FILE}" if directory else LEAK_FILE
            if match_any(path, globs) and not box.exists(path):
                box.write_text(path, leak_module_text())
                return f"planted a credential-shaped literal in {path}"
        # No room for a new file (file-exact scopes): taint a Python file the agent already edits.
        for op in sorted(own.ops, key=lambda o: o.path):
            if op.kind is not FileOpKind.DELETE and op.path.endswith(".py"):
                text = box.read_text(op.path)
                box.write_text(op.path, text.rstrip("\n") + "\n" + leak_line() + "\n")
                return f"planted a credential-shaped literal in {op.path}"
        return None

    def _edit_migration(self, box: Toolbox, ctx: AgentContext, own: ChangeSet) -> str | None:
        created = {op.path for op in own.ops if op.kind is FileOpKind.CREATE}
        globs = self._profile(ctx).write_globs
        for path in box.list_files(MIGRATIONS_GLOB):
            if path not in created and match_any(path, globs):
                text = box.read_text(path)
                box.write_text(
                    path, text.rstrip("\n") + "\n-- chaos: edited an applied migration\n"
                )
                return f"edited the applied migration {path}"
        return None

    def _break_source(self, box: Toolbox, ctx: AgentContext, own: ChangeSet) -> str | None:
        for op in sorted(own.ops, key=lambda o: o.path):
            if op.kind is not FileOpKind.DELETE and _is_source(op.path):
                text = box.read_text(op.path)
                box.write_text(op.path, text.rstrip("\n") + "\n" + REGRESSION_STATEMENT)
                return f"appended an import-time failure to {op.path}"
        return None
