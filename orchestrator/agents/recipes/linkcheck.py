"""Capability ``linkcheck``: a standalone destination health checker, built from scratch.

Delivers the package ``linkcheck/`` (standard library only) plus its tests and documentation:

* ``guard``   - URL safety policy: scheme allow-list, ``ipaddress``-based SSRF blocking (loopback,
  private, link-local, reserved, multicast, IPv4-mapped IPv6, decimal / hex / octal spellings), an
  injectable name-resolution hook and DNS pinning (the fetcher connects to the vetted addresses);
* ``checker`` - classification (healthy / redirecting / client_error / server_error / timeout /
  blocked), retry of timeouts and 5xx with exponential back-off, injected ``Fetcher`` / resolver /
  sleep so that nothing needs the network;
* ``fetcher`` - the only code that opens sockets: a minimal HTTP/1.1 client with a total deadline
  that never follows redirects, plus the system resolver;
* ``report``  - counts per classification and the list of unhealthy links;
* ``core``    - ``__init__``, shared types and ``settings.py`` rendered from validated parameters.

Security of the generator (docs/module-reference.md, L3): the parameters come from untrusted text
and are typed and range-checked before anything is rendered (``validate_params``); the payload files
under ``files/linkcheck/`` are static text (``*.tmpl``) whose only typed holes (``@@NAME@@``)
receive the ``repr`` of a validated number, a fixed constant or a requirement / criterion id that
matched the schema grammar. No free-form text from the request is ever interpolated.

Decisions and limitations (also written into the generated documentation):

* A redirect is reported (``redirecting``), never followed: following would need a full SSRF check
  per hop and share one time budget between several requests (ADR in ``design``).
* ``redirecting`` is not listed as unhealthy (the destination still answers); the assumption is
  documented in the generated docs.
* The generated fetcher speaks HTTP/1.1 over plain sockets, ignores proxy variables, and reads only
  the status line and headers (a 200 with an error page is healthy: health is status-based).
* The system resolver cannot be interrupted: a lookup that exceeds the budget is abandoned in a
  daemon thread. Checks run sequentially (``check_many``); the checker holds no state, so callers
  may share it between threads.
* SEC-008 allow-lists network libraries per module: the policy config must list
  ``EGRESS_MODULES`` (only ``linkcheck/fetcher.py`` and the generated tests import ``socket``;
  ``linkcheck/guard.py`` needs no egress because name resolution is injected).
* Change-set size: every implementation group stays a LOW-risk change (at most 10 files and 300
  lines, no security-token path names), so no approval is requested at an ``implement.*`` node.
  The integrated change set (package plus tests, about 1,100 lines) exceeds the default CHG-002
  budget (25 files / 800 lines) and needs the human waiver that scenario ``greenfield`` scripts.
* ``BuildContext.anchors['conservative']`` is ignored: the output is already minimal and a
  greenfield module has no optional extras to drop.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from orchestrator.agents.recipes.base import (
    AspectSpec,
    BuildContext,
    Capability,
    GroupSpec,
    ParamSpec,
    RecipeAnchorError,
    RecipeParamError,
    validate_params,
)
from orchestrator.core.errors import AgentError
from orchestrator.core.fsutil import read_text_lf
from orchestrator.core.schemas import AC_ID_RE, REQ_ID_RE
from orchestrator.core.types import RiskLevel

CAPABILITY_ID = "linkcheck"
PACKAGE = "linkcheck"
BACKOFF_BASE_SECONDS = 0.5
# Generated files that import ``socket`` (SEC-008 allow-list for the policy config): the default
# fetcher, and the tests that disable sockets to prove the library needs no network.
EGRESS_MODULES = (f"{PACKAGE}/fetcher.py", f"tests/{PACKAGE}/**")

_TEMPLATES = Path(__file__).resolve().parent / "files" / "linkcheck"
_HOLE = re.compile(r"@@([A-Z_]+)@@")
_SECTION = re.compile(r"^# @@ASPECT ([a-z-]+)@@\n", re.MULTILINE)
_LINE_LENGTH = 100

PARAMS = (
    ParamSpec(
        "timeout_seconds",
        "int",
        default=5,
        minimum=1,
        maximum=60,
        description="Time budget of one check attempt, in seconds",
    ),
    ParamSpec(
        "max_retries",
        "int",
        default=3,
        minimum=0,
        maximum=5,
        description="Retries of a transient failure (timeout or 5xx response)",
    ),
)

# group id -> (title, ((generated module, template stem), ...)); every file belongs to ONE group.
_GROUPS: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "core": (
        "Package scaffold, shared types and validated settings",
        (("__init__", "init"), ("model", "model"), ("settings", "settings")),
    ),
    "guard": ("URL safety policy: scheme allow-list and SSRF blocking", (("guard", "guard"),)),
    "checker": (
        "Checker (classification, retry with back-off) and the default fetcher",
        (("checker", "checker"), ("fetcher", "fetcher")),
    ),
    "report": ("Summary report", (("report", "report"),)),
}

# generated test module -> (template stem, stdlib modules and linkcheck names it may import)
_TEST_MODULES: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "test_linkcheck_guard.py": ("guard", (), ("Blocked", "UrlGuard")),
    "test_linkcheck_checker.py": (
        "checker",
        ("dataclasses", "socket", "threading"),
        (
            "FetchError",
            "FetchRequest",
            "FetchResponse",
            "FetchTimeout",
            "HttpFetcher",
            "LinkChecker",
            "ResolutionError",
            "ResolutionTimeout",
            "Settings",
            "system_resolver",
        ),
    ),
    "test_linkcheck_report.py": (
        "report",
        ("json",),
        ("CheckResult", "Classification", "summarize"),
    ),
}

_ASPECTS = (
    AspectSpec(
        "classification",
        "Classification",
        (
            r"\bclassif(?:y|ies|ied)\b",
            r"\b(?:healthy|redirecting|client[_ ]error|server[_ ]error)\b[^.]{0,80}"
            r"\b(?:healthy|redirecting|client[_ ]error|server[_ ]error|blocked|timeout)\b",
        ),
        "Every checked destination gets exactly one classification: healthy (2xx), redirecting "
        "(3xx, reported and never followed), client_error (4xx), server_error (5xx), timeout or "
        "blocked.",
    ),
    AspectSpec(
        "timeout",
        "Timeout",
        (
            r"\btime[ -]?outs?\b[^.;]{0,40}?\b\d+\s*(?:s|secs?|seconds?|ms|milliseconds?)\b",
            r"\b\d+[- ]?(?:s|secs?|seconds?)[- ]time[ -]?outs?\b",
        ),
        "A single check attempt is abandoned after {timeout_seconds} seconds in total (connect, "
        "TLS and response head) and is classified as timeout.",
    ),
    AspectSpec(
        "retry",
        "Retry with exponential back-off",
        (r"\bretr(?:y|ies|ied|ying)\b", r"\bback-?off\b"),
        "A transient failure (timeout or 5xx response) is retried up to {max_retries} times with "
        "exponentially growing delays before the result is reported; other outcomes are never "
        "retried.",
    ),
    AspectSpec(
        "ssrf-block",
        "Private destinations are blocked",
        (
            r"\bssrf\b",
            r"\b(?:private|loopback|link-local|internal)\b[^.]{0,60}\baddress\w*",
            r"\bnever\s+contact\w*\b",
        ),
        "A destination that is, or resolves to, a private-network, loopback, link-local or other "
        "non-public address is classified as blocked and never contacted.",
    ),
    AspectSpec(
        "scheme-allowlist",
        "Only http and https",
        (
            r"\bonly\b[^.]{0,40}\bhttps?\b",
            r"\bhttps?\b[^.]{0,40}\b(?:only|allow\w*)\b",
            r"\b(?:url )?schemes?\b",
        ),
        "Only http and https destinations are accepted: any other scheme is classified as blocked "
        "without any network access.",
    ),
    AspectSpec(
        "report",
        "Summary report",
        (
            r"\bsummary report\b",
            r"\breport\b[^.]{0,80}\b(?:unhealthy|number of|counts?)\b",
            r"\bunhealthy links?\b",
        ),
        "The summary report states the number of links per classification and lists every "
        "unhealthy link.",
    ),
    AspectSpec(
        "offline-testable",
        "Library, testable offline",
        (
            r"\bunit[- ]?test\w*\b",
            r"\bwithout\b[^.]{0,40}\bnetwork\b",
            r"\boffline\b",
            r"\b(?:as a )?library\b",
        ),
        "The checker works as a library without a running server: all network and DNS access is "
        "injected, so the complete test suite runs without network access.",
    ),
)
ASPECT_IDS = tuple(a.id for a in _ASPECTS)

_TRIGGERS = (
    r"\bhealth[- ]?check\w*",
    r"\bdestinations?\b",
    r"\bclassif(?:y|ies|ied|ication)\b",
    r"\b(?:healthy|redirecting|client_error|server_error)\b",
    r"\bunhealthy\b",
    r"\b(?:ssrf|loopback|link-local|private[- ]network)\b",
    r"\bhttps?\b",
)
# One capture group each: "time out after 5 seconds" / "5 second timeout"; "retried up to 3 times".
# The number is captured whole (decimals and huge values too), so ``int()`` or the range check
# fails loudly instead of a default being used silently.
_EXTRACTORS = {
    "timeout_seconds": (
        r"(?:\btime[ -]?outs?\b[^.\n]{0,40}?"
        r"|(?=\b\d+(?:\.\d+)?[- ]?(?:s|secs?|seconds?)[- ]time[ -]?outs?\b))"
        r"(?<![\d.])(\d+(?:\.\d+)?)[- ]?\s*(?:s|secs?|seconds?)\b"
    ),
    "max_retries": (
        r"(?:\bretr(?:y|ies|ied|ying)\b[^.\n]{0,40}?"
        r"|(?=\b\d+(?:\.\d+)?[ ]?(?:retries|retry attempts)\b))"
        r"(?<![\d.])(\d+(?:\.\d+)?)[ ]?(?:times|retries|attempts)\b"
    ),
}


# ----------------------------------------------------------------------------- rendering


def _literal(value: object) -> str:
    """``repr`` of a validated number: the only way a parameter value enters generated source."""
    if type(value) not in (int, float):
        raise RecipeParamError("only validated numbers may be rendered into generated source")
    return repr(value)


def _holes(params: Mapping[str, Any]) -> dict[str, str]:
    return {
        "TIMEOUT_SECONDS": _literal(params["timeout_seconds"]),
        "MAX_RETRIES": _literal(params["max_retries"]),
        "BACKOFF_BASE_SECONDS": _literal(BACKOFF_BASE_SECONDS),
    }


def _fill(template: str, values: Mapping[str, str]) -> str:
    """Replace every ``@@NAME@@`` hole; a hole without a value is a template bug (fatal)."""

    def value(match: re.Match[str]) -> str:
        try:
            return values[match.group(1)]
        except KeyError:
            raise AgentError(
                f"linkcheck template hole {match.group(1)} has no value",
                kind="fatal",
                retryable=False,
            ) from None

    return _HOLE.sub(value, template)


def _template(name: str) -> str:
    return read_text_lf(_TEMPLATES / name)


def _backoff_schedule(retries: int) -> str:
    delays = [f"{BACKOFF_BASE_SECONDS * 2**i:g}" for i in range(retries)]
    return ", ".join(delays) + " s" if delays else "none (retries are disabled)"


# ------------------------------------------------------------------- requirements -> tests


def _coverage(requirements: Mapping[str, Any] | None) -> dict[str, list[tuple[str, str]]]:
    """``aspect id -> [(requirement id, criterion id)]`` of the active requirements served here.

    Ids end up in generated docstrings, so they must match the artifact id grammar; anything else
    means a corrupt requirements artifact and fails the attempt (never guess).
    """
    if requirements is None:
        raise AgentError(
            "linkcheck tests are generated from acceptance criteria: BuildContext.requirements "
            "is missing",
            kind="fatal",
            retryable=False,
        )
    found: dict[str, list[tuple[str, str]]] = {}
    for req in requirements.get("requirements", ()):
        if not isinstance(req, Mapping) or req.get("status") != "active":
            continue
        if CAPABILITY_ID not in req.get("capabilities", ()):
            continue
        for criterion in req.get("acceptance_criteria", ()):
            aspect = criterion.get("aspect") if isinstance(criterion, Mapping) else None
            if aspect not in ASPECT_IDS:
                continue
            rid, cid = req.get("id"), criterion.get("id")
            if not (
                isinstance(rid, str)
                and isinstance(cid, str)
                and REQ_ID_RE.match(rid)
                and AC_ID_RE.match(cid)
            ):
                raise AgentError(
                    "requirements artifact carries a malformed requirement or criterion id",
                    kind="fatal",
                    retryable=False,
                )
            found.setdefault(aspect, []).append((rid, cid))
    return found


def _covers(pairs: Sequence[tuple[str, str]]) -> str:
    ids = [i for pair in sorted(set(pairs)) for i in pair]
    return "Covers: " + ", ".join(ids)


def _imports(code: str, stdlib: Sequence[str], names: Sequence[str]) -> str:
    """Import block of a generated test module: only what ``code`` really uses (no F401)."""
    used = {node.id for node in ast.walk(ast.parse(code)) if isinstance(node, ast.Name)}
    blocks: list[str] = []
    std = [f"import {m}" for m in stdlib if m in used]
    if std:
        blocks.append("\n".join(std))
    if "pytest" in used:
        blocks.append("import pytest")
    wanted = [n for n in names if n in used]
    if wanted:
        one = f"from {PACKAGE} import {', '.join(wanted)}"
        if len(one) > _LINE_LENGTH:
            one = f"from {PACKAGE} import (\n" + "".join(f"    {n},\n" for n in wanted) + ")"
        blocks.append(one)
    return "\n\n".join(blocks)


def _changelog(existing: str | None, entry: str) -> str | None:
    """New CHANGELOG.md text, or ``None`` when ``existing`` already carries the entry."""
    section = f"### Added\n\n{entry}\n"
    if existing is None:
        return f"# Changelog\n\n## [Unreleased]\n\n{section}"
    if entry in existing:
        return None
    heading = "## [Unreleased]"
    start = existing.find(heading)
    if start >= 0:
        end = existing.find("\n## ", start + len(heading))
        end = len(existing) if end < 0 else end
        block = existing[start:end]
        if "### Added\n\n" in block:
            block = block.replace("### Added\n\n", f"### Added\n\n{entry}\n", 1)
        else:
            block = block.rstrip("\n") + f"\n\n{section}"
        return existing[:start] + block + existing[end:]
    new = f"{heading}\n\n{section}"
    first = existing.find("\n## ")
    if first < 0:
        return existing.rstrip("\n") + "\n\n" + new
    return existing[: first + 1] + new + "\n" + existing[first + 1 :]


# ----------------------------------------------------------------------------- the recipe


class LinkcheckRecipe:
    """Vetted generation of the ``linkcheck`` package, its tests and its documentation."""

    id: str = CAPABILITY_ID
    params: tuple[ParamSpec, ...] = PARAMS

    def groups(self, params: Mapping[str, Any]) -> tuple[GroupSpec, ...]:
        """Four parallel groups with disjoint write scopes (each file belongs to exactly one)."""
        validate_params(PARAMS, params)
        specs = []
        for group_id, (title, files) in _GROUPS.items():
            paths = tuple(f"{PACKAGE}/{module}.py" for module, _ in files)
            specs.append(GroupSpec(group_id, title, paths, paths))
        return tuple(specs)

    def design(self, params: Mapping[str, Any], model: Any | None) -> Mapping[str, Any]:
        """Design contributions. Greenfield: the package is standalone, so ``model`` is not used."""
        values = validate_params(PARAMS, params)
        return {
            "summary": (
                "A standalone package `linkcheck/` (standard library only) that classifies the "
                f"health of destination URLs with a {values['timeout_seconds']} s time budget per "
                f"attempt and up to {values['max_retries']} retries of transient failures. It "
                "neither imports nor modifies existing modules."
            ),
            "components": _components(),
            "adrs": _adrs(values),
            "data_model_changes": [],
            "api_changes": [],
            "rollback_plan": (
                "Delete the new package linkcheck/, tests/linkcheck/ and docs/features/linkcheck.md"
                " and revert the CHANGELOG.md entry. Nothing else imports the package and no data "
                "is created or migrated, so there is no data impact and no migration to reverse."
            ),
        }

    def build_group(self, group_id: str, ctx: BuildContext) -> None:
        """Stage the files of one group; an existing file is never overwritten (new module)."""
        values = validate_params(PARAMS, ctx.params)
        if group_id not in _GROUPS:
            raise RecipeParamError(f"unknown linkcheck group {group_id!r}")
        files = _GROUPS[group_id][1]
        for module, _ in files:
            path = f"{PACKAGE}/{module}.py"
            if ctx.toolbox.exists(path):
                raise RecipeAnchorError(
                    f"{path} already exists: linkcheck is a new module and never overwrites files"
                )
        holes = _holes(values)
        for module, stem in files:
            text = _fill(_template(f"{stem}.py.tmpl"), holes)
            ctx.toolbox.write_text(f"{PACKAGE}/{module}.py", text)

    def build_tests(self, ctx: BuildContext) -> None:
        """Stage ``tests/linkcheck/test_linkcheck_*.py``; no criterion for an aspect, no test."""
        values = validate_params(PARAMS, ctx.params)
        coverage = _coverage(ctx.requirements)
        holes = _holes(values)
        for filename, (stem, stdlib, names) in _TEST_MODULES.items():
            header, *rest = _SECTION.split(_template(f"tests_{stem}.py.tmpl"))
            bodies: list[str] = []
            for aspect, body in zip(rest[::2], rest[1::2], strict=True):
                if aspect not in ASPECT_IDS:
                    raise AgentError(
                        f"linkcheck test template names an unknown aspect {aspect!r}",
                        kind="fatal",
                        retryable=False,
                    )
                if aspect in coverage:
                    covers = {**holes, "COVERS": _covers(coverage[aspect])}
                    bodies.append(_fill(body, covers).strip("\n"))
            if not bodies:
                continue
            code = header.replace("@@IMPORTS@@", "") + "\n\n\n".join(bodies)
            filled = _fill(header, {**holes, "IMPORTS": _imports(code, stdlib, names)})
            text = re.sub(
                r"\n{4,}", "\n\n\n", filled.rstrip("\n") + "\n\n\n" + "\n\n\n".join(bodies)
            )
            ctx.toolbox.write_text(f"tests/{PACKAGE}/{filename}", text.rstrip("\n") + "\n")

    def build_docs(self, ctx: BuildContext) -> None:
        """Stage ``docs/features/linkcheck.md`` and a CHANGELOG.md entry from the real design."""
        values = validate_params(PARAMS, ctx.params)
        design = self.design(values, None)
        decisions = "\n".join(f"- **{a['title']}**: {a['decision']}" for a in design["adrs"])
        criteria = ""
        if ctx.requirements is not None:
            by_id = {a.id: a for a in _ASPECTS}
            lines = []
            for aspect, pairs in sorted(_coverage(ctx.requirements).items()):
                text = by_id[aspect].criterion.format_map(values)
                refs = ", ".join(f"{rid} / {cid}" for rid, cid in sorted(set(pairs)))
                lines.append(f"- `{aspect}` ({refs}): {text}")
            criteria = "\n".join(lines)
        holes = {
            **_holes(values),
            "BACKOFF_SCHEDULE": _backoff_schedule(values["max_retries"]),
            "DECISIONS": decisions,
            "CRITERIA": criteria or "- See the requirements artifact of the run.",
        }
        ctx.toolbox.write_text(
            f"docs/features/{CAPABILITY_ID}.md", _fill(_template("docs.md.tmpl"), holes)
        )
        entry = (
            f"- `linkcheck`: standalone destination health checker for short links (http and https "
            f"only, private destinations blocked, {values['timeout_seconds']} s time budget per "
            f"attempt, up to {values['max_retries']} retries with exponential back-off). "
            "See docs/features/linkcheck.md."
        )
        existing = (
            ctx.toolbox.read_text("CHANGELOG.md") if ctx.toolbox.exists("CHANGELOG.md") else None
        )
        updated = _changelog(existing, entry)
        if updated is not None:
            ctx.toolbox.write_text("CHANGELOG.md", updated)


def _components() -> list[dict[str, Any]]:
    def component(
        name: str, what: str, files: Sequence[str], aspects: Sequence[str]
    ) -> dict[str, Any]:
        # ``requirement_ids`` is filled by the architect from the criteria that carry these aspects
        return {
            "name": name,
            "responsibility": what,
            "touches": [f"{PACKAGE}/{f}.py" for f in files],
            "requirement_ids": [],
            "aspects": list(aspects),
        }

    return [
        component(
            "linkcheck.guard",
            "URL safety policy: scheme allow-list, credentials, ipaddress-based blocking of "
            "non-public addresses, injectable name resolution, DNS pinning",
            ("guard",),
            ("ssrf-block", "scheme-allowlist"),
        ),
        component(
            "linkcheck.checker",
            "Classification of results, retry with exponential back-off, injected Fetcher, "
            "resolver and sleep",
            ("checker",),
            ("classification", "timeout", "retry", "offline-testable"),
        ),
        component(
            "linkcheck.fetcher",
            "Default HTTP/1.1 client with a total deadline (never follows redirects) and the "
            "system resolver: the only code that opens sockets",
            ("fetcher",),
            ("timeout", "classification"),
        ),
        component(
            "linkcheck.report",
            "Summary report: counts per classification and the list of unhealthy links",
            ("report",),
            ("report",),
        ),
        component(
            "linkcheck.model",
            "Shared types, the Fetcher protocol, validated settings and the package exports",
            ("__init__", "model", "settings"),
            ("classification", "offline-testable"),
        ),
    ]


def _adrs(values: Mapping[str, Any]) -> list[dict[str, Any]]:
    def adr(
        n: int,
        title: str,
        decision: str,
        alternatives: Sequence[tuple[str, str]],
        why: str,
        cost: str,
    ) -> dict[str, Any]:
        return {
            "id": f"ADR-{n:03d}",
            "title": title,
            "decision": decision,
            "alternatives": [{"option": o, "rejected_because": r} for o, r in alternatives],
            "rationale": why,
            "consequences": cost,
        }

    return [
        adr(
            1,
            "Inject a Fetcher instead of patching urllib in tests",
            "LinkChecker depends on a one-method Fetcher protocol plus an injected resolver and "
            "sleep; the default HttpFetcher is the only code that opens sockets.",
            [
                (
                    "Patch urllib.request.urlopen or socket in the tests",
                    "couples every test to library internals and cannot prove that no real "
                    "connection was attempted",
                ),
                (
                    "Run a local HTTP server in the tests",
                    "needs sockets and free ports, is flaky on CI and breaks the requirement of "
                    "unit tests without network access",
                ),
                (
                    "Use a third-party HTTP mocking library",
                    "adds a dependency to a standard-library-only package and ties the tests to "
                    "one HTTP client",
                ),
            ],
            "A protocol seam makes every behaviour (statuses, timeouts, retries) scriptable and "
            "keeps the safety policy in front of any I/O.",
            "The default fetcher is small and tested through an injected socket; production "
            "callers may supply their own client behind the same protocol.",
        ),
        adr(
            2,
            "Report redirects instead of following them",
            "A 3xx response is classified redirecting and its Location is reported; the checker "
            "never follows it.",
            [
                (
                    "Follow redirects up to N hops and classify the final response",
                    "every hop would need its own SSRF check (a public URL may redirect to a "
                    "link-local address), one time budget would be shared by several requests and "
                    "a moved destination would look healthy",
                ),
                (
                    "Treat every 3xx as healthy",
                    "hides destinations that moved or loop, which the owner of the short link "
                    "should know about",
                ),
            ],
            "The classification stays a statement about the URL that was asked for, and the "
            "private-address policy is applied exactly once per attempt.",
            "A chain of redirects is not resolved; callers who care check the reported Location.",
        ),
        adr(
            3,
            "Validate the destination, then connect to the validated addresses (DNS pinning)",
            "The guard resolves a name once, requires every answer to be public and hands the "
            "addresses to the fetcher, which connects to exactly those.",
            [
                (
                    "Resolve for the check and let the HTTP client resolve again",
                    "DNS rebinding: the second answer may be a private address (time of check "
                    "versus time of use)",
                ),
                (
                    "Block by pattern matching on the URL text",
                    "decimal, hex, octal, IPv4-mapped IPv6 and full-width spellings bypass text "
                    "matching; ipaddress on the parsed address does not",
                ),
                (
                    "Allow-list of destination domains",
                    "the checker must accept arbitrary customer destinations",
                ),
            ],
            "Fails closed: anything the guard cannot parse or vet is blocked, never contacted.",
            "NAT64, 6to4 and Teredo addresses are refused as a whole; HTTPS verifies the "
            "certificate against the original host name.",
        ),
        adr(
            4,
            "Retry only timeouts and 5xx responses with bounded exponential back-off",
            f"Up to {values['max_retries']} retries; the delay starts at a base delay and doubles "
            "after every retry (capped); every other outcome is final.",
            [
                (
                    "Retry every failure",
                    "4xx and blocked results are deterministic: retries only add load and delay",
                ),
                (
                    "Fixed delay between retries",
                    "hammers a struggling server at a constant rate; growing delays give it room "
                    "to recover",
                ),
                (
                    "No retries",
                    "one transient timeout would mark a healthy destination as unhealthy",
                ),
            ],
            "Bounds the extra load and the worst-case latency of one check "
            f"(({values['max_retries']} + 1) attempts of at most {values['timeout_seconds']} s "
            "plus the back-off delays).",
            "Settings validates the ranges (timeout 1..60 s, retries 0..5), so a bad "
            "configuration fails loudly.",
        ),
    ]


CAPABILITY = Capability(
    id=CAPABILITY_ID,
    title="Destination health checker",
    kind="new_module",
    triggers=_TRIGGERS,
    aspects=_ASPECTS,
    extractors=_EXTRACTORS,
    impact_keywords=(
        "link",
        "url",
        "destination",
        "health",
        "http",
        "redirect",
        "timeout",
        "retry",
    ),
    risk=RiskLevel.MEDIUM,
    recipe=LinkcheckRecipe(),
)
