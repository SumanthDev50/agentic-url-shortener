"""Redactor: value patterns, key-based redaction, structure walking, caps and ReDoS resistance.

Every secret-looking string is assembled at runtime from fragments, so no secret-shaped literal
exists in the repository (and secret scanners have nothing to flag).
"""

from __future__ import annotations

import enum
import hashlib
import re
import time
from datetime import UTC, datetime
from typing import Any

import pytest

from orchestrator.core.audit import MAX_STRING_BYTES, REDACTED, Redactor
from orchestrator.core.errors import ConfigError
from orchestrator.core.types import Actor, EventType, Severity

RED = Redactor()
TRUNCATION_MARKER = re.compile(r"\.\.\.\[truncated \d+ bytes sha256=[0-9a-f]{16}\]$")


def jwt() -> str:
    return (
        "ey" + "J" + "hbGciOiJIUzI1NiJ9" + "." + "eyJzdWIiOiJ0ZXN0In0" + "." + "c2ln" + "bmF0dXJl"
    )


# name -> (full secret text, the part that must vanish)
SECRETS: dict[str, tuple[str, str]] = {
    "aws_key_id": ("AK" + "IA" + "Q" * 16, "AK" + "IA" + "Q" * 16),
    "github_token": ("gh" + "p_" + "A1b2" * 9, "gh" + "p_" + "A1b2" * 9),
    "github_oauth": ("gh" + "o_" + "Z9y8" * 9, "gh" + "o_" + "Z9y8" * 9),
    "github_pat": ("github" + "_pat_" + "Zz9" * 8, "github" + "_pat_" + "Zz9" * 8),
    "slack_bot": ("xo" + "xb-" + "1234567890-" + "abcDEF" * 2, "xo" + "xb-" + "1234567890-"),
    "slack_app": ("xa" + "pp-" + "1-A0123456-" + "z" * 12, "xa" + "pp-" + "1-A0123456-"),
    "slack_webhook": (
        "hooks.slack" + ".com/services/" + "T0123456/B0123456/" + "x" * 24,
        "T0123456/B0123456",
    ),
    "jwt": (jwt(), "c2ln" + "bmF0dXJl"),
    "jwt_unsecured": (
        "ey" + "J" + "hbGciOiJub25lIn0" + "." + "eyJzdWIiOiJ0ZXN0In0" + ".",
        "eyJzdWIiOiJ0ZXN0In0",
    ),
    "google_key": ("AI" + "za" + "B" * 35, "AI" + "za" + "B" * 35),
    "stripe_secret": ("sk" + "_live_" + "a1B2" * 5, "sk" + "_live_" + "a1B2" * 5),
    "stripe_webhook": ("whs" + "ec_" + "Q" * 20, "whs" + "ec_" + "Q" * 20),
    "anthropic_style": ("sk" + "-ant-" + "api03" + "xyz789" * 5, "sk" + "-ant-" + "api03"),
    "openai_style": ("sk" + "-" + "A" * 24, "sk" + "-" + "A" * 24),
    "bearer": ("Bea" + "rer " + "t0k3n" + "V" * 20, "t0k3n" + "V" * 20),
    "url_userinfo": (
        "postgres://" + "svc_user" + ":" + "p4ssw0rd" + "@db.internal:5432/app",
        "svc_user" + ":" + "p4ssw0rd",
    ),
    "url_token_only": ("https://" + "tok3n9876" + "@example.test/repo.git", "tok3n9876"),
    "azure_account_key": ("Account" + "Key=" + "Aa1" * 14 + "==", "Aa1" * 14),
    "azure_sas": (
        "https://x.test/blob?sv=1&si" + "g=" + "abc%2B123" * 4 + "&se=2030",
        "abc%2B123" * 4,
    ),
}


class TestValuePatterns:
    @pytest.mark.parametrize("name", sorted(SECRETS))
    @pytest.mark.parametrize(
        "template", ["{s}", "key {s} end", "k={s}", '"{s}"', "x\n{s}\ny", "({s})", "{s} {s}"]
    )
    def test_secret_is_removed_and_context_survives(self, name: str, template: str) -> None:
        text, vanish = SECRETS[name]
        out = RED.redact_text(template.format(s=text))
        assert vanish not in out
        assert REDACTED in out
        if template == "key {s} end":
            assert out.startswith("key ") and out.endswith(" end")

    def test_url_userinfo_keeps_scheme_and_host(self) -> None:
        text, _ = SECRETS["url_userinfo"]
        assert RED.redact_text(text) == "postgres://[REDACTED]@db.internal:5432/app"

    def test_bearer_keeps_the_scheme_word(self) -> None:
        assert RED.redact_text(SECRETS["bearer"][0]) == "Bearer [REDACTED]"
        assert RED.redact_text("authorization: " + SECRETS["bearer"][0]).count(REDACTED) >= 1

    def test_azure_keys_keep_their_names(self) -> None:
        out = RED.redact_text(
            "DefaultEndpointsProtocol=https;"
            + SECRETS["azure_account_key"][0]
            + ";EndpointSuffix=x"
        )
        assert out == "DefaultEndpointsProtocol=https;AccountKey=[REDACTED];EndpointSuffix=x"
        assert "&sig=[REDACTED]&se=2030" in RED.redact_text(SECRETS["azure_sas"][0])

    def test_pem_block_is_removed_and_surroundings_kept(self) -> None:
        begin, end = "-----BEGIN " + "PRIVATE KEY-----", "-----END " + "PRIVATE KEY-----"
        text = f"before\n{begin}\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASC\nabcdef\n{end}\nafter"
        out = RED.redact_text(text)
        assert out.startswith("before\n") and out.endswith("\nafter")
        assert "MIIEvQ" not in out and "abcdef" not in out and "PRIVATE KEY" not in out
        assert REDACTED in out

    def test_several_pem_blocks_and_other_labels(self) -> None:
        def block(label: str, body: str) -> str:
            return f"-----BEGIN {label}-----\n{body}\n-----END {label}-----"

        text = (
            "a "
            + block("RSA PRIVATE KEY", "SECRETBODY1")
            + " b "
            + block("CERTIFICATE", "CERTBODY2")
            + " c"
        )
        out = RED.redact_text(text)
        assert "SECRETBODY1" not in out and "CERTBODY2" not in out
        assert out.startswith("a ") and " b " in out and out.endswith(" c")

    def test_truncated_pem_is_redacted_to_the_end(self) -> None:
        begin = "-----BEGIN " + "OPENSSH PRIVATE KEY-----"
        out = RED.redact_text(
            "log line one\n" + begin + "\nb3BlbnNzaC1rZXktdjEAAAAA\nmore key material"
        )
        assert out.startswith("log line one\n")
        assert "b3Blbn" not in out and "more key material" not in out

    def test_pem_end_marker_without_begin_and_headerless_begin_are_left_alone(self) -> None:
        assert RED.redact_text("-----END " + "PRIVATE KEY-----") == "-----END " + "PRIVATE KEY-----"
        assert RED.redact_text("-----BEGIN-----\nx") == "-----BEGIN-----\nx"

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("password=hunter2 next", "password=[REDACTED] next"),
            ("PASSWORD: hunter2", "PASSWORD: [REDACTED]"),
            ("db_passwd = hunter2;", "db_passwd = [REDACTED];"),
            ('"api_key": "abc def ghi"', '"api_key": "[REDACTED]"'),
            ("'secret': 'two words'", "'secret': '[REDACTED]'"),
            ("token=abc123&x=1", "token=[REDACTED]&x=1"),
            ("GITHUB_TOKEN=abc", "GITHUB_TOKEN=[REDACTED]"),
            ('password="unterminated', 'password="[REDACTED]"'),
            ("private_key: value", "private_key: [REDACTED]"),
            ("client credentials: abc", "client credentials: [REDACTED]"),
            ("Cookie: sessionid", "Cookie: [REDACTED]"),
        ],
    )
    def test_assignments(self, text: str, expected: str) -> None:
        assert RED.redact_text(text) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "plain prose with nothing to hide",
            "token budget exceeded",
            "the task-skeleton and risk-assessment nodes",
            "author: someone",  # 'auth' alone is a dict-key rule, not a free-text one
            "sha256=" + hashlib.sha256(b"x").hexdigest(),
            hashlib.sha256(b"x").hexdigest(),
            "design_doc@v3 implement.a#2 n:verify#1 A-0001 D-0042",
            "https://example.test/path?a=1#frag",
            "user@example.test",
            "ssh://host.example.test/repo.git",
            "AKIA",  # too short to be a key id
            "sk-short",
            "Bearer",
            "-----BEGIN",
        ],
    )
    def test_ordinary_text_is_untouched(self, text: str) -> None:
        assert RED.redact_text(text) == text

    @pytest.mark.parametrize("name", sorted(SECRETS))
    def test_redaction_is_idempotent(self, name: str) -> None:
        once = RED.redact_text("prefix " + SECRETS[name][0] + " suffix password=x")
        assert RED.redact_text(once) == once

    def test_non_string_input_is_stringified(self) -> None:
        assert RED.redact_text(12345) == "12345"  # type: ignore[arg-type]

    def test_hostile_characters_do_not_break_it(self) -> None:
        text = "a\x00b\ud800c\u2028d\u202ee\x1b[31m token=zzz"
        out = RED.redact_text(text)
        assert out.endswith("token=[REDACTED]") and "\ud800" in out


class TestKeysAndStructures:
    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "Password",
            "API-Key",
            "api key",
            "apiKey",
            "X-Auth-Token",
            "Set-Cookie",
            "SESSION_ID",
            "privateKey",
            "Connection String",
            "dsn",
            "client_secret",
            "authorization",
            "Signature",
            "credentials",
            "pwd",
            "passwd",
            "oauth",
        ],
    )
    def test_sensitive_keys_hide_strings_and_containers(self, key: str) -> None:
        data = {key: "plain value", "n": 1}
        assert RED.redact(data) == {key: REDACTED, "n": 1}
        assert RED.redact({key: {"a": "b"}}) == {key: REDACTED}
        assert RED.redact({key: ["x", "y"]}) == {key: REDACTED}
        assert RED.redact({key: ("x",)}) == {key: REDACTED}

    def test_scalars_without_credential_material_survive_under_sensitive_keys(self) -> None:
        data = {"tokens": 1234, "auth": True, "session": None, "password": "", "pwd": 2.5}
        assert RED.redact(data) == data

    @pytest.mark.parametrize(
        "key", ["name", "message", "node_id", "attempt", "sha256", "path", "rule_ids"]
    )
    def test_other_keys_are_untouched(self, key: str) -> None:
        assert RED.redact({key: "visible"}) == {key: "visible"}

    def test_nesting_is_walked_and_sequences_become_lists(self) -> None:
        secret = SECRETS["github_token"][0]
        data = {
            "a": [{"b": ("x", secret)}, {"password": "p"}],
            "s": {3, 1, 2},
            "t": {"cfg": {"token": "t", "ok": 1}},
        }
        out = RED.redact(data)
        assert out == {
            "a": [{"b": ["x", REDACTED]}, {"password": REDACTED}],
            "s": [1, 2, 3],
            "t": {"cfg": {"token": REDACTED, "ok": 1}},
        }

    def test_secret_in_a_key_is_redacted(self) -> None:
        secret = SECRETS["aws_key_id"][0]
        assert RED.redact({secret: 1}) == {REDACTED: 1}

    def test_non_string_keys_become_strings(self) -> None:
        assert RED.redact({1: "a", None: "b"}) == {"1": "a", "None": "b"}

    def test_leaf_conversions_yield_plain_json(self) -> None:
        class Colour(enum.Enum):
            RED = "red"

        out = RED.redact(
            {
                "blob": b"raw bytes",
                "actor": Actor.human("alice"),
                "severity": Severity.HIGH,
                "event": EventType.NODE_STARTED,
                "colour": Colour.RED,
                "when": datetime(2026, 1, 2, tzinfo=UTC),
            }
        )
        assert out["blob"] == {
            "__type__": "bytes",
            "sha256": hashlib.sha256(b"raw bytes").hexdigest(),
            "len": 9,
        }
        assert out["actor"] == {"type": "human", "id": "alice"}
        assert (
            out["severity"] == "HIGH" and out["event"] == "node.started" and out["colour"] == "red"
        )
        assert out["when"] == "2026-01-02T00:00:00+00:00"

    def test_sensitive_enum_and_bytes_values_are_hidden(self) -> None:
        assert RED.redact({"token": Severity.HIGH, "secret": b"x"}) == {
            "token": REDACTED,
            "secret": REDACTED,
        }

    @pytest.mark.parametrize(
        ("value", "text"), [(float("nan"), "nan"), (float("inf"), "inf"), (float("-inf"), "-inf")]
    )
    def test_non_finite_floats_become_strings(self, value: float, text: str) -> None:
        assert RED.redact({"x": value}) == {"x": text}

    def test_unserialisable_objects_become_a_placeholder(self) -> None:
        assert RED.redact({"x": object()}) == {"x": "<unserialisable object>"}

    def test_scalars_pass_through(self) -> None:
        for value in (None, True, 3, 2.5):
            assert RED.redact(value) == value
        assert RED.redact(SECRETS["jwt"][0]) == REDACTED

    def test_depth_is_bounded(self) -> None:
        nested: Any = "leaf"
        for _ in range(100):
            nested = {"k": nested}
        out = RED.redact(nested)
        depth = 0
        while isinstance(out, dict):
            out = out["k"]
            depth += 1
        assert depth <= 33 and out == "[truncated: depth limit]"

    def test_self_referential_structures_terminate(self) -> None:
        loop: dict[str, Any] = {}
        loop["me"] = loop
        listy: list[Any] = []
        listy.append(listy)
        assert "me" in RED.redact(loop)
        assert isinstance(RED.redact(listy), list)

    def test_wide_containers_are_capped(self) -> None:
        listed = RED.redact(list(range(2500)))
        assert len(listed) == 1001 and listed[-1] == "[truncated: 1500 more items]"
        mapped = RED.redact({f"k{i}": i for i in range(1500)})
        assert len(mapped) == 1001 and mapped["_truncated"] == "500 more items"

    def test_total_node_budget_bounds_work(self) -> None:
        wide = [[i] * 900 for i in range(60)]  # 54k leaves, far above the node budget
        started = time.perf_counter()
        out = RED.redact(wide)
        assert time.perf_counter() - started < 1.0
        flat = [x for row in out for x in (row if isinstance(row, list) else [row])]
        assert "[truncated: size limit]" in flat


class TestCaps:
    def test_long_strings_are_truncated_with_a_marker(self) -> None:
        out = RED.redact_text("x" * 20_000)
        assert out.startswith("x" * MAX_STRING_BYTES)
        assert TRUNCATION_MARKER.search(out)
        assert len(out.encode()) < MAX_STRING_BYTES + 80
        assert "truncated 11808 bytes" in out

    def test_a_string_exactly_at_the_cap_is_unchanged(self) -> None:
        text = "y" * MAX_STRING_BYTES
        assert RED.redact_text(text) == text
        assert TRUNCATION_MARKER.search(RED.redact_text(text + "z"))

    def test_multibyte_text_is_cut_on_a_character_boundary(self) -> None:
        out = RED.redact_text("é" * 10_000)  # 20 000 bytes
        kept = TRUNCATION_MARKER.sub("", out)
        assert "�" not in kept and len(kept.encode()) <= MAX_STRING_BYTES
        assert set(kept) == {"é"}

    def test_a_secret_near_the_cut_is_redacted_before_truncation(self) -> None:
        secret = SECRETS["github_token"][0]
        out = RED.redact_text("x" * (MAX_STRING_BYTES - 10) + secret + "tail" * 100)
        assert secret not in out and "gh" + "p_" not in out

    def test_huge_input_is_clipped_before_scanning_and_still_redacted(self) -> None:
        secret = SECRETS["aws_key_id"][0]
        text = "head " + secret + " " + "word " * 200_000  # ~1 MB
        started = time.perf_counter()
        out = RED.redact_text(text)
        assert time.perf_counter() - started < 1.0
        assert secret not in out and TRUNCATION_MARKER.search(out)
        assert len(out) < MAX_STRING_BYTES + 100

    def test_huge_input_without_any_whitespace_is_still_clipped(self) -> None:
        secret = SECRETS["aws_key_id"][0]
        out = RED.redact_text(secret + " " + "z" * 400_000)
        assert secret not in out and TRUNCATION_MARKER.search(out)
        out = RED.redact_text("z" * 400_000)  # no whitespace to cut at: clipped at the hard limit
        assert TRUNCATION_MARKER.search(out) and len(out) < MAX_STRING_BYTES + 100

    def test_clipping_never_leaves_a_partial_trailing_token(self) -> None:
        secret = SECRETS["github_token"][0]
        text = "w " * 131_000 + "z" * 10 + secret[:20]  # the cut would land inside a token
        out = RED.redact_text(text)
        assert TRUNCATION_MARKER.search(out) and secret[:20] not in out

    def test_lone_surrogates_do_not_break_truncation(self) -> None:
        out = RED.redact_text("\ud800" * 12_000)
        assert TRUNCATION_MARKER.search(out)

    def test_marker_digest_is_of_the_redacted_text(self) -> None:
        secret = SECRETS["github_token"][0]
        body = "x" * 20_000
        with_secret = RED.redact_text(secret + " " + body)
        placeholder = RED.redact_text(REDACTED + " " + body)
        assert with_secret == placeholder  # the secret never influenced the output, digest included


FUZZ_INPUTS: dict[str, str] = {
    "a*50000": "a" * 50_000,
    "=*50000": "=" * 50_000,
    "AKIA*10000": "AKIA" * 10_000,
    "hex run": "0123456789abcdef" * 3_125,
    "unbalanced PEM headers": "-----BEGIN " + "PRIVATE KEY-----" * 3_000,
    "PEM with long body": "-----BEGIN " + "PRIVATE KEY-----\n" + "A" * 50_000,
    "many BEGIN markers": "-----BEGIN " * 5_000,
    "eyJ repeated": "eyJ" * 17_000,
    "eyJ long token": "eyJ" + "a" * 50_000,
    "jwt dots": ("eyJ" + "a" * 10 + ".") * 4_000,
    "sk- repeated": "sk-" * 16_000,
    "ghp_ repeated": "gh" + "p_" * 1 + ("gh" + "p_a") * 10_000,
    "xox repeated": "xox" * 16_000,
    "password repeated": "password=" * 5_000,
    "password long value": "password=" + "x" * 50_000,
    "password quote unterminated": 'password="' + "a" * 50_000,
    "password spaces": "password" + " " * 50_000 + "=",
    "bearer repeated": "bearer " * 7_000,
    "bearer spaces": "bearer" + " " * 50_000 + "x",
    "scheme separators": "://" * 16_000,
    "scheme then long userinfo": "a://" + "b" * 50_000,
    "at signs": "@" * 50_000,
    "quotes": '"' * 50_000,
    "single quotes": "'" * 50_000,
    "key words": "token key secret " * 3_000,
    "azure": "AccountKey=" * 4_000 + "sig=" * 4_000,
    "slack hook": "hooks.slack.com/services/" * 2_000,
    "whitespace": " " * 50_000,
    "newlines": "\n" * 50_000,
    "unicode": "\u202e\u2028\ud800é" * 12_000,
    "dashes": "-" * 50_000,
    "mixed alphabet": "AIza" * 12_000 + "whsec_" * 8_000 + "sk_live_" * 6_000,
}


class TestAdversarialInputsFinishQuickly:
    @pytest.mark.parametrize("name", sorted(FUZZ_INPUTS))
    def test_redact_text_is_linear_time(self, name: str) -> None:
        started = time.perf_counter()
        out = RED.redact_text(FUZZ_INPUTS[name])
        elapsed = time.perf_counter() - started
        assert elapsed < 0.5, f"{name}: {elapsed:.3f}s"
        assert len(out.encode("utf-8", "surrogatepass")) < MAX_STRING_BYTES + 100

    @pytest.mark.parametrize(
        "name", ["a*50000", "eyJ repeated", "password repeated", "unbalanced PEM headers"]
    )
    def test_structured_redaction_of_hostile_values_and_keys(self, name: str) -> None:
        hostile = FUZZ_INPUTS[name]
        started = time.perf_counter()
        RED.redact({hostile: [hostile, {hostile: hostile}], "list": [hostile] * 40})
        assert time.perf_counter() - started < 1.0

    def test_many_small_strings_stay_fast(self) -> None:
        started = time.perf_counter()
        RED.redact({"items": [f"node-{i} finished with status ok" for i in range(900)]})
        assert time.perf_counter() - started < 0.5


class TestExtraPatterns:
    def test_custom_patterns_are_applied_after_the_builtin_ones(self) -> None:
        red = Redactor([r"ACME-[0-9]{6}", re.compile(r"internal-[a-z]+-host")])
        assert (
            red.redact_text("id ACME-123456 on internal-db-host ok")
            == f"id {REDACTED} on {REDACTED} ok"
        )

    @pytest.mark.parametrize(
        "pattern",
        [
            r"(a+)+$",
            r"(a*)*b",
            r"(x+x+)+y",
            r"([a-z]+)*end",
            r"(.*a){12}x(.+)+",
            "a" * 600,
            r"a*",  # matches the empty string: would insert markers between every character
            r"(unclosed",
            r"[",
        ],
    )
    def test_dangerous_or_broken_patterns_are_rejected(self, pattern: str) -> None:
        with pytest.raises(ConfigError):
            Redactor([pattern])

    def test_bytes_patterns_are_rejected(self) -> None:
        with pytest.raises(ConfigError):
            Redactor([re.compile(b"abc")])  # type: ignore[arg-type]

    def test_accepts_any_iterable(self) -> None:
        assert Redactor(iter([r"ZZ[0-9]{4}"])).redact_text("ZZ1234") == REDACTED
