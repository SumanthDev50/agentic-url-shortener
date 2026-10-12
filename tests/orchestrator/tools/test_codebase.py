"""Behaviour tests for ``orchestrator.tools.codebase`` (static analysis + impact analysis).

Fixtures are synthetic trees held in a ``MemoryView``: a small FastAPI app shaped like the product
(routers with prefixes, ``Depends`` auth, services, repositories, SQL migrations with
``ALTER TABLE``) plus focused mini-trees for each heuristic, and hostile inputs.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from orchestrator.core.schemas import validate_payload
from orchestrator.core.serde import canonical_json
from orchestrator.tools import codebase
from orchestrator.tools.codebase import (
    CodebaseModel,
    DataFlow,
    ImpactReport,
    RouteInfo,
    analyze_codebase,
    impact_analysis,
)
from orchestrator.tools.workspace import DirectoryView, MemoryView


def src(text: str) -> str:
    return textwrap.dedent(text).lstrip("\n")


def build(files: Mapping[str, str], **kwargs: object) -> CodebaseModel:
    return analyze_codebase(MemoryView(files), **kwargs)  # type: ignore[arg-type]


def routes_by_label(model: CodebaseModel) -> dict[str, RouteInfo]:
    return {r.label: r for r in model.routes}


def flow(model: CodebaseModel, label: str) -> DataFlow:
    return next(f for f in model.data_flows if f.route == label)


def call_labels(f: DataFlow) -> set[str]:
    return {c.label for c in f.calls}


# ------------------------------------------------------------------ fixture: the mini app

MINI_APP: dict[str, str] = {
    "app/__init__.py": "",
    "app/main.py": src(
        """
        from fastapi import FastAPI

        from app.api import routes_links, routes_redirect


        def create_app() -> FastAPI:
            application = FastAPI(title="mini")
            application.include_router(routes_links.router)
            application.include_router(routes_redirect.router)

            @application.get("/healthz")
            def healthz() -> dict[str, str]:
                return {"status": "ok"}

            return application
        """
    ),
    "app/api/__init__.py": "",
    "app/api/deps.py": src(
        """
        from dataclasses import dataclass

        from fastapi import Header, HTTPException, Request

        from app.services.click_recorder import ClickRecorder
        from app.services.link_service import LinkService


        @dataclass
        class Services:
            links: LinkService
            recorder: ClickRecorder


        def require_api_key(x_api_key: str = Header(default="")) -> str:
            if not x_api_key:
                raise HTTPException(status_code=401)
            return x_api_key


        def get_services(request: Request) -> Services:
            return request.app.state.services
        """
    ),
    "app/api/schemas.py": src(
        """
        from pydantic import BaseModel


        class CreateLink(BaseModel):
            url: str
            max_clicks: int | None = None
        """
    ),
    "app/api/routes_links.py": src(
        """
        from typing import Annotated

        from fastapi import APIRouter, Depends

        from app.api.deps import Services, get_services, require_api_key
        from app.api.schemas import CreateLink

        API_PREFIX = "/api/v1"

        router = APIRouter(prefix=API_PREFIX + "/links", dependencies=[Depends(require_api_key)])


        @router.post("", status_code=201)
        def create_link(body: CreateLink, services: Services = Depends(get_services)):
            return services.links.create(body.url, body.max_clicks)


        @router.get("/{code}")
        def get_link(code: str, services: Annotated[Services, Depends(get_services)]):
            return services.links.get(code)


        @router.delete("/{code}", status_code=204)
        def disable_link(code: str, services: Services = Depends(get_services)) -> None:
            services.links.disable(code)
        """
    ),
    "app/api/routes_redirect.py": src(
        """
        from fastapi import APIRouter, Depends
        from fastapi.responses import RedirectResponse

        from app.api.deps import get_services

        router = APIRouter()


        @router.get("/{code}")
        def redirect(code: str, services=Depends(get_services)):
            target = services.links.resolve(code)
            services.recorder.record(code)
            return RedirectResponse(target, status_code=302)
        """
    ),
    "app/services/__init__.py": "",
    "app/services/cache.py": src(
        """
        import threading
        from collections import OrderedDict


        class LruTtlCache:
            def __init__(self, capacity: int = 1000) -> None:
                self._data: OrderedDict[str, str] = OrderedDict()
                self._lock = threading.Lock()
                self._capacity = capacity

            def get(self, key: str) -> str | None:
                with self._lock:
                    return self._data.get(key)

            def put(self, key: str, value: str) -> None:
                with self._lock:
                    self._data[key] = value
        """
    ),
    "app/services/click_recorder.py": src(
        """
        import queue
        import threading

        from app.storage.click_repository import ClickRepository


        class ClickRecorder:
            def __init__(self, repo: ClickRepository) -> None:
                self._repo = repo
                self._queue = queue.Queue(maxsize=1000)
                self._thread = threading.Thread(target=self._drain, daemon=True)

            def record(self, code: str) -> None:
                self._queue.put_nowait(code)

            def _drain(self) -> None:
                batch = [self._queue.get()]
                self._repo.insert_many(batch)
        """
    ),
    "app/services/link_service.py": src(
        """
        from app.services.cache import LruTtlCache
        from app.storage.link_repository import LinkRepository


        class LinkService:
            def __init__(self, repo: LinkRepository, cache: LruTtlCache) -> None:
                self._repo = repo
                self._cache = cache

            def create(self, url: str, max_clicks: int | None = None):
                return self._repo.insert(url, max_clicks)

            def get(self, code: str):
                return self._repo.get_by_code(code)

            def resolve(self, code: str) -> str:
                cached = self._cache.get(code)
                if cached is not None:
                    return cached
                link = self._repo.get_by_code(code)
                self._cache.put(code, link["original_url"])
                return link["original_url"]

            def disable(self, code: str) -> None:
                self._repo.disable(code)
                self._cache.put(code, "")
        """
    ),
    "app/storage/__init__.py": "",
    "app/storage/link_repository.py": src(
        """
        import sqlite3


        class LinkRepository:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self._conn = conn

            def insert(self, url, max_clicks):
                self._conn.execute(
                    "INSERT INTO links (code, original_url, max_clicks) VALUES (?, ?, ?)",
                    (url, url, max_clicks),
                )

            def get_by_code(self, code):
                return self._conn.execute("SELECT * FROM links WHERE code = ?", (code,)).fetchone()

            def disable(self, code):
                self._conn.execute("UPDATE links SET disabled_at = ? WHERE code = ?", ...)
        """
    ),
    "app/storage/click_repository.py": src(
        """
        class ClickRepository:
            def __init__(self, conn) -> None:
                self._conn = conn

            def insert_many(self, batch):
                self._conn.executemany(
                    "INSERT INTO clicks (link_id, clicked_at) VALUES (?, ?)", batch
                )
        """
    ),
    "app/storage/migrations/001_init.sql": src(
        """
        -- initial schema
        CREATE TABLE links (
            id INTEGER PRIMARY KEY,
            code TEXT UNIQUE NOT NULL,
            original_url TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NULL,
            disabled_at TEXT NULL,
            click_count INTEGER NOT NULL DEFAULT 0,
            created_by TEXT NULL
        );
        CREATE TABLE clicks (
            id INTEGER PRIMARY KEY,
            link_id INTEGER NOT NULL REFERENCES links(id),
            clicked_at TEXT NOT NULL,
            visitor_hash TEXT NOT NULL,
            FOREIGN KEY (link_id) REFERENCES links(id)
        );
        CREATE INDEX idx_clicks_link ON clicks (link_id, clicked_at);
        """
    ),
    "app/storage/migrations/002_extras.sql": src(
        """
        CREATE TABLE IF NOT EXISTS idempotency_keys (
            key TEXT, owner TEXT, request_hash TEXT, PRIMARY KEY (key, owner)
        );
        ALTER TABLE links ADD COLUMN max_clicks INTEGER;
        """
    ),
    "tests/__init__.py": "",
    "tests/test_link_service.py": (
        "from app.services.link_service import LinkService\n\n\n"
        "def test_x():\n    assert LinkService\n"
    ),
    "tests/test_api.py": (
        "from app.main import create_app\n\n\ndef test_y():\n    assert create_app\n"
    ),
}


@pytest.fixture(scope="module")
def mini() -> CodebaseModel:
    return build(MINI_APP)


# ----------------------------------------------------------------------------- modules and imports


def test_modules_carry_dotted_name_path_kind_and_defs(mini: CodebaseModel) -> None:
    by_name = {m.name: m for m in mini.modules}
    assert by_name["app"].path == "app/__init__.py"
    assert by_name["app.api.routes_links"].path == "app/api/routes_links.py"
    assert by_name["tests.test_api"].is_test and not by_name["app.main"].is_test
    svc = {d.name: d.kind for d in by_name["app.services.link_service"].defs}
    assert svc["LinkService"] == "class" and svc["LinkService.resolve"] == "method"
    deps = {d.name: d.kind for d in by_name["app.api.deps"].defs}
    assert deps["Services.links"] == "field" and deps["get_services"] == "function"
    assert {d.name: d.kind for d in by_name["app.api.routes_links"].defs}[
        "API_PREFIX"
    ] == "constant"
    nested = {d.name: d.kind for d in by_name["app.main"].defs}
    assert nested["create_app.healthz"] == "function"


def test_import_graph_keeps_only_internal_edges(mini: CodebaseModel) -> None:
    graph = mini.import_graph
    assert graph["app.main"] == ("app.api.routes_links", "app.api.routes_redirect")
    assert graph["app.api.deps"] == ("app.services.click_recorder", "app.services.link_service")
    assert graph["app.services.cache"] == ()  # threading / collections are not repo modules
    assert all(target in graph for targets in graph.values() for target in targets)


def test_relative_and_absolute_imports_resolve_to_repo_modules() -> None:
    files = {
        "pkg/__init__.py": "from .core import run\n",
        "pkg/core.py": "def run():\n    return 1\n",
        "pkg/sub/__init__.py": "",
        "pkg/sub/helper.py": "def tool():\n    return 2\n",
        "pkg/sub/leaf.py": src(
            """
            import json
            import os.path
            import pkg.core as c
            from . import helper
            from .. import core
            from ..core import run
            from pkg.sub.helper import tool
            from ... import beyond_the_top
            from .nowhere import ghost
            """
        ),
    }
    model = build(files)
    assert model.import_graph["pkg.sub.leaf"] == ("pkg.core", "pkg.sub.helper")
    assert model.import_graph["pkg"] == ("pkg.core",)
    assert model.unparseable == ()


def test_src_layout_prefix_is_stripped_from_module_names() -> None:
    model = build(
        {"src/lib/__init__.py": "", "src/lib/a.py": "X = 1\n", "main.py": "import lib.a\n"}
    )
    assert {m.name for m in model.modules} == {"lib", "lib.a", "main"}
    assert model.import_graph["main"] == ("lib.a",)


def test_vendored_and_cache_directories_are_not_analysed() -> None:
    model = build(
        {"app.py": "X = 1\n", ".venv/lib/x.py": "Y = 1\n", "a/__pycache__/m.py": "Z = 1\n"}
    )
    assert [m.path for m in model.modules] == ["app.py"]


def test_tests_for_answers_the_reverse_question(mini: CodebaseModel) -> None:
    assert mini.tests_for(["app.main"]) == ("tests/test_api.py",)
    assert mini.tests_for(["app.main", "app.services.link_service"]) == (
        "tests/test_api.py",
        "tests/test_link_service.py",
    )
    assert mini.tests_for(["app.services.cache", "unknown"]) == ()
    assert mini.tests_for([]) == ()


def test_test_map_links_test_files_to_the_modules_they_import(mini: CodebaseModel) -> None:
    assert mini.test_map["tests/test_api.py"] == ("app.main",)
    assert mini.test_map["tests/test_link_service.py"] == ("app.services.link_service",)
    assert "tests/__init__.py" not in mini.test_map  # a package marker is not a test


# ----------------------------------------------------------------------------- routes


def test_routes_have_method_full_path_handler_module_and_line(mini: CodebaseModel) -> None:
    routes = routes_by_label(mini)
    assert set(routes) == {
        "POST /api/v1/links",
        "GET /api/v1/links/{code}",
        "DELETE /api/v1/links/{code}",
        "GET /{code}",
        "GET /healthz",
    }
    create = routes["POST /api/v1/links"]
    assert (create.handler, create.module, create.line) == (
        "create_link",
        "app.api.routes_links",
        14,
    )
    assert routes["GET /healthz"].handler == "create_app.healthz"
    assert routes["GET /healthz"].module == "app.main"
    assert routes["GET /{code}"].module == "app.api.routes_redirect"


def test_auth_dependencies_come_from_router_and_signature(mini: CodebaseModel) -> None:
    routes = routes_by_label(mini)
    assert routes["POST /api/v1/links"].auth == ("require_api_key",)
    assert routes["GET /api/v1/links/{code}"].auth == ("require_api_key",)
    assert routes["GET /{code}"].auth == () and routes["GET /healthz"].auth == ()
    # every Depends() is kept; ``auth`` is the authentication-looking subset
    assert routes["POST /api/v1/links"].dependencies == ("get_services", "require_api_key")
    assert routes["GET /{code}"].dependencies == ("get_services",)


NESTED = {
    "api/__init__.py": "",
    "api/security.py": src(
        """
        from fastapi import Depends
        from fastapi.security import APIKeyHeader, HTTPBearer

        api_key_header = APIKeyHeader(name="X-API-Key")
        bearer = HTTPBearer()


        def auth_user(key: str = Depends(api_key_header)) -> str:
            return key


        def get_db():
            yield None


        def check_owner():
            return None
        """
    ),
    "api/items.py": src(
        """
        from typing import Annotated

        from fastapi import APIRouter, Depends, Security

        from api.security import api_key_header, bearer, check_owner, get_db

        items = APIRouter(prefix="/items")


        @items.get("/{item_id}")
        def read_item(item_id: int, db=Depends(get_db)):
            return db


        @items.put("/{item_id}", dependencies=[Depends(check_owner)])
        def put_item(item_id: int): ...


        @items.post("/")
        def make(key: Annotated[str, Depends(api_key_header)]): ...


        @items.get("/secure")
        def secure(token=Security(bearer)): ...
        """
    ),
    "api/app.py": src(
        """
        from fastapi import APIRouter, Depends, FastAPI

        from api.items import items
        from api.security import auth_user

        v1 = APIRouter()
        v1.include_router(items, prefix="/v1", dependencies=[Depends(auth_user)])
        application = FastAPI()
        application.include_router(v1, prefix="/api")
        """
    ),
}


def test_nested_include_prefixes_accumulate_and_auth_follows_the_chain() -> None:
    routes = routes_by_label(build(NESTED))
    assert set(routes) == {
        "GET /api/v1/items/{item_id}",
        "PUT /api/v1/items/{item_id}",
        "POST /api/v1/items/",
        "GET /api/v1/items/secure",
    }
    assert routes["GET /api/v1/items/{item_id}"].auth == ("auth_user",)  # include-level dependency
    assert "get_db" in routes["GET /api/v1/items/{item_id}"].dependencies  # present, not auth
    assert routes["PUT /api/v1/items/{item_id}"].dependencies == ("auth_user", "check_owner")
    assert routes["PUT /api/v1/items/{item_id}"].auth == ("auth_user",)
    # a security *scheme* instance counts as auth whatever it is called
    assert routes["POST /api/v1/items/"].auth == ("api_key_header", "auth_user")
    assert routes["GET /api/v1/items/secure"].auth == ("auth_user", "bearer")


def test_extra_auth_dependency_names_are_honoured() -> None:
    routes = routes_by_label(build(NESTED, auth_dependency_names=["check_owner", 7]))
    assert routes["PUT /api/v1/items/{item_id}"].auth == ("auth_user", "check_owner")


def test_dependency_expression_forms() -> None:
    files = {
        "d.py": src(
            """
            from fastapi import APIRouter, Depends as D, FastAPI
            from fastapi.security import HTTPBearer

            app = FastAPI()


            class Guard:
                pass


            def current_user():
                return 1


            @app.get("/a")
            def a(user=D(current_user), guard: Guard = D(), tok=D(HTTPBearer())): ...
            """
        )
    }
    route = routes_by_label(build(files))["GET /a"]
    assert route.dependencies == ("Guard", "HTTPBearer", "current_user")
    assert route.auth == ("HTTPBearer", "current_user")


def test_dynamic_router_prefix_is_flagged_not_guessed() -> None:
    files = {
        "d/__init__.py": "",
        "d/settings.py": "settings = object()\n",
        "d/api.py": src(
            """
            from fastapi import APIRouter, FastAPI

            from d.settings import settings

            router = APIRouter(prefix=settings.prefix)
            app = FastAPI()
            app.include_router(router)


            @router.get("/x")
            def x(): ...
            """
        ),
    }
    model = build(files)
    assert [r.path for r in model.routes] == ["/<dynamic>/x"]
    assert any("router prefix is not a static string" in u for u in model.unresolved)


def test_explicit_registration_with_add_api_route_and_add_route() -> None:
    files = {
        "r.py": src(
            """
            from fastapi import FastAPI

            app = FastAPI()


            def ping(): ...


            def echo(): ...


            app.add_api_route("/ping", ping, methods=["GET", "HEAD"])
            app.add_route("/echo", echo)
            app.add_api_route("/ghost", missing_handler)
            """
        )
    }
    model = build(files)
    routes = routes_by_label(model)
    assert {"GET /ping", "HEAD /ping", "GET /echo"} <= set(routes)
    assert routes["GET /ping"].handler == "ping"
    assert routes["GET /ghost"].handler == "<unresolved>"
    assert any("not a resolvable function" in u for u in model.unresolved)


def test_router_factory_function_included_by_call() -> None:
    files = {
        "f.py": src(
            """
            from fastapi import APIRouter, FastAPI


            def build_router() -> APIRouter:
                router = APIRouter(prefix="/f")

                @router.get("/x")
                def x(): ...

                return router


            app = FastAPI()
            app.include_router(build_router(), prefix="/mount")
            """
        )
    }
    assert [(r.label, r.handler) for r in build(files).routes] == [
        ("GET /mount/f/x", "build_router.x")
    ]


def test_api_route_decorator_lists_every_method() -> None:
    files = {
        "m.py": src(
            """
            from fastapi import FastAPI

            app = FastAPI()


            @app.api_route("/multi", methods=["POST", "put"])
            def multi(): ...


            @app.api_route("/default")
            def default(): ...
            """
        )
    }
    assert set(routes_by_label(build(files))) == {"POST /multi", "PUT /multi", "GET /default"}


def test_unknown_receivers_and_orphan_routers_are_reported() -> None:
    files = {
        "o.py": src(
            """
            from fastapi import APIRouter, FastAPI

            app = FastAPI()
            orphan = APIRouter(prefix="/orphan")
            client = object()


            @orphan.get("/x")
            def x(): ...


            @app.get("/ok")
            def ok(): ...


            @client.get("/nope")
            def nope(): ...
            """
        )
    }
    model = build(files)
    assert set(routes_by_label(model)) == {"GET /orphan/x", "GET /ok"}
    assert any("router 'orphan' is not included" in u for u in model.unresolved)
    assert any("unknown receiver 'client'" in u for u in model.unresolved)


def test_circular_includes_terminate_and_routes_appear_once() -> None:
    files = {
        "c.py": src(
            """
            from fastapi import APIRouter, FastAPI

            a = APIRouter(prefix="/a")
            b = APIRouter(prefix="/b")
            a.include_router(b)
            b.include_router(a)
            app = FastAPI()
            app.include_router(a)


            @a.get("/x")
            def ax(): ...


            @b.get("/y")
            def by(): ...
            """
        )
    }
    assert set(routes_by_label(build(files))) == {"GET /a/x", "GET /a/b/y"}


def test_routers_that_only_include_each_other_are_still_analysed() -> None:
    files = {
        "c.py": src(
            """
            from fastapi import APIRouter

            a = APIRouter(prefix="/a")
            b = APIRouter(prefix="/b")
            a.include_router(b)
            b.include_router(a)


            @a.get("/x")
            def ax(): ...


            @b.get("/y")
            def by(): ...
            """
        )
    }
    assert set(routes_by_label(build(files))) == {"GET /a/x", "GET /a/b/y"}


def test_an_include_bomb_is_cut_off_at_the_route_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    levels = 30  # every router includes the next one twice: 2**30 paths if followed naively
    lines = ["from fastapi import APIRouter, FastAPI", "app = FastAPI()"]
    lines += [f"r{i} = APIRouter(prefix='/r{i}')" for i in range(levels)]
    for i in range(levels - 1):
        lines += [
            f"r{i}.include_router(r{i + 1})",
            f"r{i}.include_router(r{i + 1}, prefix='/again')",
        ]
    lines += ["app.include_router(r0)", "@r29.get('/leaf')", "def leaf(): ..."]
    monkeypatch.setattr(codebase, "MAX_ROUTES", 50)
    started = time.perf_counter()
    model = build({"bomb.py": "\n".join(lines) + "\n"})
    assert time.perf_counter() - started < 5.0
    assert len(model.routes) <= 50
    assert any("route limit reached" in u for u in model.unresolved)


def test_routes_defined_in_test_modules_are_not_product_routes() -> None:
    files = {
        "tests/test_x.py": src(
            """
            from fastapi import FastAPI

            app = FastAPI()


            @app.get("/only-in-a-test")
            def only(): ...
            """
        )
    }
    assert build(files).routes == ()


def test_route_path_may_be_built_from_constants_across_modules() -> None:
    files = {
        "k/__init__.py": "",
        "k/consts.py": 'BASE = "/svc"\nVERSION = f"{BASE}/v2"\n',
        "k/api.py": src(
            """
            from fastapi import APIRouter, FastAPI

            from k import consts

            router = APIRouter(prefix=consts.VERSION)
            app = FastAPI()
            app.include_router(router)


            @router.get(f"{consts.BASE}-x")
            def x(): ...
            """
        ),
    }
    assert [r.path for r in build(files).routes] == ["/svc/v2/svc-x"]


# ----------------------------------------------------------------------------- SQL schema replay


def migrations(*sql: str) -> dict[str, str]:
    return {f"db/migrations/{i:03d}_m.sql": text for i, text in enumerate(sql, start=1)}


SQL_CASES: list[tuple[str, tuple[str, ...], dict[str, list[str]]]] = [
    (
        "add-column",
        ("CREATE TABLE t (a INT);", "ALTER TABLE t ADD COLUMN b TEXT; ALTER TABLE t ADD c TEXT;"),
        {"t": ["a", "b", "c"]},
    ),
    (
        "drop-column",
        ("CREATE TABLE t (a INT, b INT);", "ALTER TABLE t DROP COLUMN b;"),
        {"t": ["a"]},
    ),
    (
        "drop-and-recreate",
        ("CREATE TABLE t (a INT);", "DROP TABLE t;", "CREATE TABLE t (z INT);"),
        {"t": ["z"]},
    ),
    ("drop-if-exists", ("DROP TABLE IF EXISTS ghost; CREATE TABLE t (a INT);",), {"t": ["a"]}),
    (
        "rename-table",
        ("CREATE TABLE old_t (a INT);", "ALTER TABLE old_t RENAME TO new_t;"),
        {"new_t": ["a"]},
    ),
    (
        "rename-column",
        ("CREATE TABLE t (a INT);", "ALTER TABLE t RENAME COLUMN a TO b;"),
        {"t": ["b"]},
    ),
    (
        "rename-column-short",
        ("CREATE TABLE t (a INT);", "ALTER TABLE t RENAME a TO b;"),
        {"t": ["b"]},
    ),
    (
        "if-not-exists-keeps-existing",
        ("CREATE TABLE t (a INT);", "CREATE TABLE IF NOT EXISTS t (z INT);"),
        {"t": ["a"]},
    ),
    (
        "quoted-and-qualified",
        ('CREATE TABLE main."Links" ("Code" TEXT, `Url` TEXT, [Created At] TEXT);',),
        {"links": ["code", "url", "created at"]},
    ),
    (
        "comments-and-strings-hide-ddl",
        (
            "-- CREATE TABLE ghost (x INT);\n/* CREATE TABLE ghost2 (x INT); */ "
            "CREATE TABLE t (a TEXT DEFAULT 'CREATE TABLE ghost3 (y INT);');",
        ),
        {"t": ["a"]},
    ),
    (
        "constraints-are-not-columns",
        (
            'CREATE TABLE t (id INTEGER, key TEXT, "unique" INT, PRIMARY KEY (id), '
            "UNIQUE (key), CHECK (key <> ''), "
            "CONSTRAINT fk FOREIGN KEY (id) REFERENCES u(id));",
        ),
        {"t": ["id", "key", "unique"]},
    ),
    (
        "trigger-bodies-do-not-split-statements",
        (
            "CREATE TABLE a (x INT); CREATE TRIGGER trg AFTER INSERT ON a BEGIN "
            "UPDATE a SET x = 1; DELETE FROM a WHERE x = 2; END; CREATE TABLE b (y INT);",
        ),
        {"a": ["x"], "b": ["y"]},
    ),
    (
        "indexes-and-views-ignored",
        (
            "CREATE TABLE t (a INT); CREATE INDEX i ON t (a); "
            "CREATE UNIQUE INDEX j ON t (a); CREATE VIEW v AS SELECT a FROM t;",
        ),
        {"t": ["a"]},
    ),
    ("alter-of-unknown-table", ("ALTER TABLE legacy ADD COLUMN c INT;",), {"legacy": ["c"]}),
    (
        "multi-action-alter",
        (
            "CREATE TABLE t (a INT);",
            "ALTER TABLE t ADD COLUMN b INT, ADD COLUMN c INT, DROP COLUMN a;",
        ),
        {"t": ["b", "c"]},
    ),
    (
        "add-constraint-is-no-column",
        ("CREATE TABLE t (a INT);", "ALTER TABLE t ADD CONSTRAINT u UNIQUE (a);"),
        {"t": ["a"]},
    ),
    ("create-as-select", ("CREATE TABLE t AS SELECT 1 AS a;",), {"t": []}),
    ("temp-table", ("CREATE TEMP TABLE t (a INT);",), {"t": ["a"]}),
    ("virtual-table", ("CREATE VIRTUAL TABLE docs USING fts5(body);",), {"docs": []}),
    (
        "unterminated-comment",
        ("CREATE TABLE t (a INT); /* never closed CREATE TABLE ghost (x INT);",),
        {"t": ["a"]},
    ),
    ("unterminated-string", ("CREATE TABLE t (a TEXT DEFAULT 'oops",), {"t": ["a"]}),
    (
        "drop-several-tables",
        (
            "CREATE TABLE a (x INT); CREATE TABLE b (x INT); CREATE TABLE c (x INT);",
            "DROP TABLE a, b;",
        ),
        {"c": ["x"]},
    ),
]


@pytest.mark.parametrize(("name", "sql", "expected"), SQL_CASES, ids=[c[0] for c in SQL_CASES])
def test_migrations_are_replayed_into_tables_and_columns(
    name: str, sql: tuple[str, ...], expected: dict[str, list[str]]
) -> None:
    model = build(migrations(*sql))
    assert {t.name: list(t.columns) for t in model.tables} == expected


def test_table_records_the_creating_and_altering_migrations(mini: CodebaseModel) -> None:
    tables = {t.name: t for t in mini.tables}
    assert set(tables) == {"links", "clicks", "idempotency_keys"}
    links = tables["links"]
    assert links.migration == "app/storage/migrations/001_init.sql"
    assert links.altered_in == ("app/storage/migrations/002_extras.sql",)
    assert "max_clicks" in links.columns and "PRIMARY" not in links.columns
    assert tables["idempotency_keys"].columns == ("key", "owner", "request_hash")
    assert mini.migrations == (
        "app/storage/migrations/001_init.sql",
        "app/storage/migrations/002_extras.sql",
    )


def test_migrations_apply_in_natural_file_order() -> None:
    files = {
        "db/migrations/9_create.sql": "CREATE TABLE t (a INT, b INT);",
        "db/migrations/10_drop_b.sql": "ALTER TABLE t DROP COLUMN b;",
    }
    model = build(files)
    assert [list(t.columns) for t in model.tables] == [["a"]]
    assert model.migrations == ("db/migrations/9_create.sql", "db/migrations/10_drop_b.sql")


def test_only_migration_files_define_the_schema() -> None:
    files = {
        "queries/report.sql": "CREATE TABLE not_a_migration (x INT);",
        "tests/data.sql": "CREATE TABLE fixture (x INT);",
        "db/003_late.sql": "CREATE TABLE numbered (x INT);",
        "db/migrations/readme.sql": "CREATE TABLE in_dir (x INT);",
    }
    assert {t.name for t in build(files).tables} == {"numbered", "in_dir"}


def test_table_to_dict_matches_the_report_shape(mini: CodebaseModel) -> None:
    links = next(t for t in mini.tables if t.name == "links")
    assert set(links.to_dict()) == {"table", "columns", "migration"}
    assert links.to_dict()["columns"] == sorted(links.columns)


# ----------------------------------------------------------------------------- data flows


def test_redirect_route_flows_to_the_links_table(mini: CodebaseModel) -> None:
    f = flow(mini, "GET /{code}")
    assert f.handler == "redirect" and f.module == "app.api.routes_redirect"
    assert call_labels(f) >= {
        "app.api.deps.get_services",
        "app.services.link_service.LinkService.resolve",
        "app.services.cache.LruTtlCache.get",
        "app.storage.link_repository.LinkRepository.get_by_code",
        "app.services.click_recorder.ClickRecorder.record",
    }
    assert {c.label for c in f.repositories} == {
        "app.storage.link_repository.LinkRepository.get_by_code"
    }
    assert f.tables == ("links",)
    assert f.unresolved == ()


def test_write_routes_reach_their_repository_methods(mini: CodebaseModel) -> None:
    create = flow(mini, "POST /api/v1/links")
    assert "app.storage.link_repository.LinkRepository.insert" in call_labels(create)
    assert "app.api.deps.require_api_key" in call_labels(create)  # router-level dependency runs too
    assert create.tables == ("links",)
    assert flow(mini, "DELETE /api/v1/links/{code}").tables == ("links",)
    assert flow(mini, "GET /healthz").calls == ()


def test_background_writer_behind_a_queue_is_not_part_of_the_route_flow(
    mini: CodebaseModel,
) -> None:
    f = flow(mini, "GET /{code}")
    assert "clicks" not in f.tables  # the writer thread, not the request, inserts clicks
    assert not any("insert_many" in label for label in call_labels(f))
    coupling = {c.component for c in impact_analysis(mini, ["click"]).coupling_risks}
    assert (
        "app.services.click_recorder" in coupling
    )  # ...which is exactly what coupling_risks surface


def test_data_flow_to_dict_matches_the_report_shape(mini: CodebaseModel) -> None:
    data = flow(mini, "GET /{code}").to_dict()
    assert set(data) == {"route", "handler", "calls", "repositories", "tables", "unresolved"}
    assert data["handler"] == "app.api.routes_redirect.redirect"
    assert data["calls"] == sorted(data["calls"])


HEURISTIC = {
    "h/__init__.py": "",
    "h/link_service.py": "class LinkService:\n    def create(self, code):\n        return code\n",
    "h/invoice_service.py": (
        "class InvoiceService:\n    def create(self, code):\n        return code\n"
    ),
    "h/link_controller.py": src(
        """
        class LinkController:
            def __init__(self, service):
                self.service = service

            def run(self, code):
                return self.service.create(code)
        """
    ),
    "h/audit_controller.py": src(
        """
        class AuditController:
            def __init__(self, service):
                self.service = service

            def run(self, code):
                return self.service.create(code)
        """
    ),
    "h/api.py": src(
        """
        from fastapi import FastAPI

        from h.audit_controller import AuditController
        from h.link_controller import LinkController

        app = FastAPI()


        @app.get("/links/{code}")
        def show(code: str):
            return LinkController(None).run(code)


        @app.get("/audit/{code}")
        def audit(code: str):
            return AuditController(None).run(code)
        """
    ),
}


def test_attribute_heuristic_resolves_service_role_with_context_tiebreak() -> None:
    model = build(HEURISTIC)
    f = flow(model, "GET /links/{code}")
    # ``self.service.create`` -> the *Service class sharing a word with the caller (LinkController)
    assert "h.link_service.LinkService.create" in call_labels(f)
    assert "h.invoice_service.InvoiceService.create" not in call_labels(f)
    assert f.unresolved == ()


def test_name_guessing_has_a_work_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    files = {k: v for k, v in HEURISTIC.items() if "invoice" not in k}
    monkeypatch.setattr(codebase, "_HEURISTIC_BUDGET", 0)
    f = flow(build(files), "GET /audit/{code}")
    assert "h.link_service.LinkService.create" not in call_labels(f)  # no budget, no guess...
    assert f.unresolved == (
        "h.audit_controller.AuditController.run: self.service.create",
    )  # ...but flagged


def test_ambiguous_attribute_is_reported_never_guessed() -> None:
    model = build(HEURISTIC)
    f = flow(model, "GET /audit/{code}")
    assert not any(label.endswith("Service.create") for label in call_labels(f))
    assert f.unresolved == ("h.audit_controller.AuditController.run: self.service.create",)


def test_unique_role_match_needs_no_context() -> None:
    files = {k: v for k, v in HEURISTIC.items() if "invoice" not in k}
    f = flow(build(files), "GET /audit/{code}")
    assert "h.link_service.LinkService.create" in call_labels(f)
    assert f.unresolved == ()


PROTOCOL = {
    "p/__init__.py": "",
    "p/ports.py": src(
        """
        from typing import Protocol


        class LinkStore(Protocol):
            def get(self, code: str) -> str: ...
        """
    ),
    "p/sqlite_store.py": src(
        """
        class SqliteLinkStore:
            def get(self, code):
                return self._conn.execute("SELECT original_url FROM links WHERE code = ?", (code,))
        """
    ),
    "p/service.py": src(
        """
        from p.ports import LinkStore


        class Service:
            def __init__(self, store: LinkStore) -> None:
                self._store = store

            def find(self, code):
                return self._store.get(code)
        """
    ),
    "p/api.py": src(
        """
        from fastapi import FastAPI

        from p.service import Service

        app = FastAPI()
        service = Service(None)


        @app.get("/find/{code}")
        def find(code: str):
            return service.find(code)
        """
    ),
    "p/migrations/001_init.sql": "CREATE TABLE links (code TEXT, original_url TEXT);",
}


def test_repository_protocol_resolves_to_its_implementations() -> None:
    f = flow(build(PROTOCOL), "GET /find/{code}")
    assert call_labels(f) == {
        "p.service.Service.find",
        "p.ports.LinkStore.get",
        "p.sqlite_store.SqliteLinkStore.get",
    }
    assert f.tables == ("links",)
    assert {c.label for c in f.repositories} == {
        "p.ports.LinkStore.get",
        "p.sqlite_store.SqliteLinkStore.get",
    }


def test_explicit_subclass_is_an_implementation_even_with_an_unrelated_name() -> None:
    files = dict(PROTOCOL)
    files["p/sqlite_store.py"] = src(
        """
        from p.ports import LinkStore


        class Backend(LinkStore):
            def get(self, code):
                return self._conn.execute("SELECT original_url FROM links WHERE code = ?", (code,))
        """
    )
    assert "p.sqlite_store.Backend.get" in call_labels(flow(build(files), "GET /find/{code}"))


DYNAMIC_SQL = {
    "d/__init__.py": "",
    "d/repo.py": src(
        """
        import sqlite3


        class Repo:
            def __init__(self, conn: sqlite3.Connection) -> None:
                self._c = conn

            def by_table(self, table):
                return self._c.execute(f"SELECT * FROM {table}")

            def percent(self, code):
                return self._c.execute("SELECT * FROM links WHERE code = '%s'" % code)

            def concat(self, code):
                return self._c.execute("SELECT * FROM links WHERE code = '" + code + "'")

            def fmt(self, code):
                return self._c.execute("SELECT * FROM links WHERE code = '{}'".format(code))

            def safe(self, code):
                return self._c.execute("SELECT * FROM links WHERE code = ?", (code,))
        """
    ),
    "d/api.py": src(
        """
        from fastapi import FastAPI

        from d.repo import Repo

        app = FastAPI()
        repo = Repo(None)


        @app.get("/a")
        def a():
            return repo.by_table("x")


        @app.get("/b")
        def b():
            return repo.percent("x")


        @app.get("/c")
        def c():
            return repo.concat("x")


        @app.get("/d")
        def d():
            return repo.fmt("x")


        @app.get("/e")
        def e():
            return repo.safe("x")
        """
    ),
    "d/migrations/001_init.sql": "CREATE TABLE links (code TEXT);",
}


@pytest.mark.parametrize(
    ("route", "method", "dynamic"),
    [
        ("/a", "by_table", True),
        ("/b", "percent", True),
        ("/c", "concat", True),
        ("/d", "fmt", True),
        ("/e", "safe", False),
    ],
)
def test_dynamically_assembled_sql_is_reported_as_unresolved(
    route: str, method: str, dynamic: bool
) -> None:
    f = flow(build(DYNAMIC_SQL), f"GET {route}")
    assert (f.unresolved == (f"dynamic SQL in d.repo.Repo.{method}",)) is dynamic
    if method != "by_table":
        assert f.tables == ("links",)  # the static part still attributes the table


def test_sql_held_in_module_and_class_constants_is_attributed_to_its_readers() -> None:
    files = {
        "q/__init__.py": "",
        "q/repo.py": src(
            """
            _SELECT = "SELECT * FROM links WHERE code = ?"


            class Repo:
                INSERT = "INSERT INTO clicks (link_id) VALUES (?)"

                def get(self, code):
                    return self._c.execute(_SELECT, (code,))

                def add(self, link_id):
                    self._c.execute(self.INSERT, (link_id,))

                def unrelated(self):
                    return 1
            """,
        ),
        "q/api.py": src(
            """
            from fastapi import FastAPI

            from q.repo import Repo

            app = FastAPI()
            repo = Repo()


            @app.get("/g")
            def g():
                return repo.get("x")


            @app.post("/a")
            def a():
                return repo.add(1)


            @app.get("/u")
            def u():
                return repo.unrelated()
            """
        ),
        "q/migrations/001.sql": (
            "CREATE TABLE links (code TEXT); CREATE TABLE clicks (link_id INT);"
        ),
    }
    model = build(files)
    assert flow(model, "GET /g").tables == ("links",)
    assert flow(model, "POST /a").tables == ("clicks",)
    assert flow(model, "GET /u").tables == ()
    assert model.table_refs["q.repo"] == ("clicks", "links")


def test_callbacks_handed_to_dispatchers_and_dynamic_dispatch() -> None:
    files = {
        "c/__init__.py": "",
        "c/audit.py": src(
            """
            class Auditor:
                def record(self, event): ...

                def slow(self): ...
            """
        ),
        "c/api.py": src(
            """
            from fastapi import BackgroundTasks, FastAPI
            from starlette.concurrency import run_in_threadpool

            from c.audit import Auditor

            app = FastAPI()
            auditor = Auditor()


            @app.post("/x")
            async def x(tasks: BackgroundTasks):
                tasks.add_task(auditor.record, "x")
                await run_in_threadpool(auditor.slow)
                getattr(auditor, "dyn")()
            """
        ),
    }
    f = flow(build(files), "POST /x")
    assert call_labels(f) == {"c.audit.Auditor.record", "c.audit.Auditor.slow"}
    assert f.unresolved == ("dynamic call (getattr) in c.api.x",)


def test_app_state_service_locator_is_resolved_by_name() -> None:
    files = {
        "s/__init__.py": "",
        "s/link_service.py": (
            "class LinkService:\n    def resolve(self, code):\n        return code\n"
        ),
        "s/api.py": src(
            """
            from fastapi import FastAPI, Request

            app = FastAPI()


            @app.get("/{code}")
            def go(code: str, request: Request):
                return request.app.state.link_service.resolve(code)
            """
        ),
    }
    assert "s.link_service.LinkService.resolve" in call_labels(flow(build(files), "GET /{code}"))


def test_recursion_and_cycles_in_the_call_graph_terminate() -> None:
    files = {
        "r.py": src(
            """
            from fastapi import FastAPI

            app = FastAPI()


            def ping(n):
                return pong(n - 1)


            def pong(n):
                return ping(n - 1)


            @app.get("/loop")
            def loop():
                return ping(3)
            """
        )
    }
    assert call_labels(flow(build(files), "GET /loop")) == {"r.ping", "r.pong"}


def test_flow_is_truncated_and_flagged_when_it_exceeds_the_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codebase, "MAX_FLOW_DEPTH", 2)
    body = "\n".join(f"def f{i}():\n    return f{i + 1}()\n" for i in range(6))
    files = {
        "t.py": "from fastapi import FastAPI\napp = FastAPI()\n"
        + body
        + "@app.get('/t')\ndef t():\n    return f0()\n"
    }
    f = flow(build(files), "GET /t")
    assert call_labels(f) == {"t.f0", "t.f1"}
    assert f.unresolved == ("flow truncated at the depth/size limit",)


def test_type_resolution_depth_budget_stops_runaway_inference() -> None:
    lines = ["class Foo:", "    def make(self):", "        return Foo()", "", "a0 = Foo()"]
    lines += [f"a{i} = a{i - 1}.make()" for i in range(1, 2000)]
    lines += ["x = x.make()", "", "def use():", "    return a1999.make()"]
    model = build({"deep.py": "\n".join(lines) + "\n"})
    assert model.unparseable == ()


def test_self_referential_classes_and_bases_terminate() -> None:
    files = {
        "w.py": src(
            """
            class A(B):
                pass


            class B(A):
                def go(self):
                    return self.go()


            X = A()
            Y = Y.make()


            def run():
                return X.go() + Y.go()
            """
        )
    }
    assert build(files).unparseable == ()


# ----------------------------------------------------------------------------- module signals


def test_module_signals_flag_caches_queues_threads_limiters_and_global_state() -> None:
    files = {
        "s/cache.py": "class Anything:\n    pass\n",
        "s/limits.py": "class TokenBucketLimiter:\n    pass\n",
        "s/writer.py": src(
            """
            import queue
            import threading


            class AuditWriter:
                def __init__(self):
                    self._q = queue.Queue()
                    threading.Thread(target=self.run).start()

                def run(self):
                    pass
            """
        ),
        "s/memo.py": src(
            """
            from functools import lru_cache

            REGISTRY = {}
            ITEMS = [1, 2]


            @lru_cache(maxsize=8)
            def compute(x):
                return x
            """
        ),
        "s/plain.py": "def f():\n    return 1\n",
    }
    signals = {m.name: m.signals for m in build(files).modules}
    assert signals["s.cache"] == ("cache",)
    assert signals["s.limits"] == ("rate-limiter",)
    assert signals["s.writer"] == ("background-writer", "queue", "thread")
    assert signals["s.memo"] == (
        "cache",
        "mutable-global",
    )  # ITEMS has contents: a constant, not state
    assert signals["s.plain"] == ()


# ----------------------------------------------------------------------------- impact analysis


def modules_of(report: ImpactReport, role: str) -> set[str]:
    return {m.module for m in report.modules if m.role == role}


def test_keywords_seed_by_symbol_table_column_and_route(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, ["click"])
    assert modules_of(report, "seed") == {
        "app.api.schemas",  # CreateLink.max_clicks
        "app.services.click_recorder",
        "app.storage.click_repository",
        "app.storage.link_repository",  # SQL references links, whose click_count column matches
    }
    assert {t.name for t in report.tables} == {"clicks", "links"}
    reasons = {m.module: m.reason for m in report.modules}
    assert "CreateLink.max_clicks" in reasons["app.api.schemas"]
    assert "table 'links'" in reasons["app.storage.link_repository"]


def test_route_path_keywords_seed_the_handler_module(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, ["healthz"], depth=0)
    assert [(m.module, m.role) for m in report.modules] == [("app.main", "seed")]
    assert "route GET /healthz matches 'healthz'" in report.modules[0].reason
    assert [r.label for r in report.routes] == ["GET /healthz"]
    assert (
        impact_analysis(mini, ["GET"], depth=0).modules != ()
    )  # the method word is part of the route


def test_dependents_follow_the_reverse_import_closure_up_to_depth(mini: CodebaseModel) -> None:
    seed = ["app.services.cache"]
    d0 = impact_analysis(mini, [], seed, depth=0)
    d1 = impact_analysis(mini, [], seed, depth=1)
    d2 = impact_analysis(mini, [], seed, depth=2)
    d9 = impact_analysis(mini, [], seed, depth=9)
    assert modules_of(d0, "dependent") == set()
    assert modules_of(d1, "dependent") == {"app.services.link_service"}
    assert modules_of(d2, "dependent") == {"app.services.link_service", "app.api.deps"}
    assert modules_of(d9, "dependent") == {
        "app.services.link_service",
        "app.api.deps",
        "app.api.routes_links",
        "app.api.routes_redirect",
        "app.main",
    }
    by_module = {m.module: m for m in d9.modules}
    assert by_module["app.api.deps"].reason == "imports app.services.link_service (depth 2)"
    assert by_module["app.services.cache"].reason == "explicit seed"


def test_seed_modules_accept_dotted_names_and_paths_and_ignore_unknown(mini: CodebaseModel) -> None:
    seeds: list[Any] = ["app/storage/click_repository.py", "app.services.cache", "no.such", 3]
    report = impact_analysis(mini, [], seeds)
    assert modules_of(report, "seed") == {"app.storage.click_repository", "app.services.cache"}
    assert modules_of(impact_analysis(mini, [], "app.services.cache"), "seed") == {
        "app.services.cache"
    }


def test_tests_are_selected_by_importing_an_impacted_module(mini: CodebaseModel) -> None:
    assert impact_analysis(mini, [], ["app.services.link_service"]).tests == (
        "tests/test_link_service.py",
    )
    # tests/test_api.py imports app.main, a dependent of the routes, within depth
    assert impact_analysis(mini, [], ["app.api.routes_links"], depth=1).tests == (
        "tests/test_api.py",
    )
    assert impact_analysis(mini, [], ["app.api.schemas"], depth=0).tests == ()


def test_test_modules_never_seed_and_never_propagate_impact(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, ["test"])
    assert all(not m.module.startswith("tests") for m in report.modules)
    assert not any(
        m.module.startswith("tests")
        for m in impact_analysis(mini, [], ["app.main"], depth=5).modules
    )


def test_keyword_matching_is_token_based_case_insensitive_and_plural_folded(
    mini: CodebaseModel,
) -> None:
    for kw in ("LINKS", "link", "Links"):
        assert "app.services.link_service" in modules_of(impact_analysis(mini, [kw]), "seed")
    assert modules_of(impact_analysis(mini, ["lin"]), "seed") == set()  # substrings are not tokens
    phrase = modules_of(impact_analysis(mini, ["link service"], depth=0), "seed")
    # every word must be found: in the module name, or in a definition (``Services.links`` borrows
    # the word "service" from its class) - never in an unrelated module
    assert phrase == {"app.services.link_service", "app.api.deps"}


def test_multi_word_keyword_needs_every_word(mini: CodebaseModel) -> None:
    assert modules_of(impact_analysis(mini, ["click limit"]), "seed") == set()
    assert "app.services.click_recorder" in modules_of(
        impact_analysis(mini, ["click recorder"]), "seed"
    )


def test_no_match_yields_an_empty_report_with_a_warning(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, ["zebra"])
    assert report.modules == () and report.routes == () and report.tables == ()
    assert any(s.startswith("no-seed-matched") for s in report.risk_signals)
    assert validate_payload("impact_report", report.to_dict()) == []


def test_impacted_routes_follow_flows_and_tables(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, [], ["app.services.cache"])
    labels = {r.label for r in report.routes}
    assert labels == {"GET /{code}", "DELETE /api/v1/links/{code}"}  # the flows that call the cache
    assert {f.route for f in report.data_flows} == labels
    assert {t.name for t in report.tables} == {"links"}


def test_coupling_risks_name_caches_writers_and_adjacent_state(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, [], ["app.services.link_service"], depth=0)
    risks = {c.component: c.reason for c in report.coupling_risks}
    assert risks["app.services.cache"].startswith("cache, thread (on the flow of ")
    assert risks["app.services.click_recorder"].startswith(
        "background-writer, queue, thread (on the flow of"
    )
    seeded = impact_analysis(mini, [], ["app.services.click_recorder"], depth=0)
    assert {c.component: c.reason for c in seeded.coupling_risks}[
        "app.services.click_recorder"
    ].endswith("(in the impacted set)")


def test_coupling_risks_include_modules_imported_by_a_seed_but_off_the_flow() -> None:
    files = {
        "x/__init__.py": "",
        "x/limiter.py": "class TokenBucketLimiter:\n    pass\n",
        "x/service.py": (
            "from x.limiter import TokenBucketLimiter\n\n\n"
            "class Service:\n    def go(self):\n        return 1\n"
        ),
        "x/unrelated_cache.py": "class Cache:\n    pass\n",
    }
    report = impact_analysis(build(files), [], ["x.service"], depth=0)
    assert [(c.component, c.reason) for c in report.coupling_risks] == [
        ("x.limiter", "rate-limiter (imported by x.service)")
    ]


def test_risk_signals_summarise_schema_auth_deletion_and_coverage(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, ["disable"])
    signals = "\n".join(report.risk_signals)
    assert "schema-impacted: tables links" in signals
    assert "migrations-present: 2 applied migration file(s)" in signals
    assert "auth-routes-impacted:" in signals and "DELETE /api/v1/links/{code}" in signals
    assert "destructive-operations:" in signals and "keyword 'disable'" in signals
    assert "untested-modules:" in signals
    assert list(report.risk_signals) == sorted(report.risk_signals)
    public = impact_analysis(mini, ["redirect"])
    assert any(
        s.startswith("unauthenticated-routes-impacted: GET /{code}") for s in public.risk_signals
    )


def test_auth_code_impact_is_signalled(mini: CodebaseModel) -> None:
    report = impact_analysis(mini, [], ["app.api.deps"])
    assert "auth-code-impacted: app.api.deps" in report.risk_signals


def test_expected_paths_cover_modules_tests_and_the_migration_directory(
    mini: CodebaseModel,
) -> None:
    report = impact_analysis(mini, ["disable"])
    assert "app/storage/migrations/**" in report.expected_paths
    assert "app/storage/link_repository.py" in report.expected_paths
    assert "tests/test_link_service.py" in report.expected_paths
    assert list(report.expected_paths) == sorted(set(report.expected_paths))
    assert (
        "app/storage/migrations/**"
        not in impact_analysis(mini, [], ["app.api.schemas"], depth=0).expected_paths
    )


def test_risk_signals_flag_unparseable_and_skipped_sources() -> None:
    files = {
        "ok.py": "def disable():\n    pass\n",
        "broken.py": "def (:\n",
        "big.py": "x = 1\n" * 10,
    }
    report = impact_analysis(build(files), ["disable"])
    assert any(s.startswith("unparseable-sources: broken.py") for s in report.risk_signals)


@pytest.mark.parametrize(
    "keywords", [["click"], ["link", "redirect"], ["max_clicks"], ["zebra"], []]
)
def test_impact_report_payload_validates_against_the_artifact_schema(
    mini: CodebaseModel, keywords: list[str]
) -> None:
    payload = impact_analysis(mini, keywords).to_dict()
    assert validate_payload("impact_report", payload) == []
    json.dumps(payload)  # plain JSON types only


def test_impact_report_has_no_absolute_paths_or_timestamps(tmp_path: Path) -> None:
    for rel, text in MINI_APP.items():
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    model = analyze_codebase(DirectoryView(tmp_path, include=("app/**", "tests/**")))
    blob = json.dumps(impact_analysis(model, ["click"]).to_dict()) + json.dumps(model.to_dict())
    assert str(tmp_path) not in blob and tmp_path.as_posix() not in blob
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", blob)
    assert not re.search(r'"[A-Za-z]:[\\/]', blob)


def test_analysis_and_impact_are_deterministic_and_input_order_independent() -> None:
    forward = MemoryView(MINI_APP)
    backward = MemoryView(dict(reversed(list(MINI_APP.items()))))
    first, second, third = (
        analyze_codebase(forward),
        analyze_codebase(forward),
        analyze_codebase(backward),
    )
    assert (
        canonical_json(first.to_dict())
        == canonical_json(second.to_dict())
        == canonical_json(third.to_dict())
    )
    one = impact_analysis(first, ["click", "link"], depth=3)
    two = impact_analysis(third, ["link", "CLICK"], depth=3)
    assert canonical_json(one.to_dict()) == canonical_json(two.to_dict())


def test_impact_inputs_are_validated_and_bounded(mini: CodebaseModel) -> None:
    for bad in (-1, 11, True, "2", 1.5):
        with pytest.raises(ValueError, match="depth"):
            impact_analysis(mini, ["link"], depth=bad)  # type: ignore[arg-type]
    hostile = [
        "\x1b[31mlink\x00",
        5,
        None,
        "x" * 100_000,
        "   ",
        "the",
        *[f"kw{i}" for i in range(500)],
    ]
    report = impact_analysis(mini, hostile)  # type: ignore[arg-type]
    assert len(report.keywords) <= codebase.MAX_KEYWORDS
    assert all("\x1b" not in k and "\x00" not in k for k in report.keywords)
    assert "link" in report.keywords
    assert modules_of(
        impact_analysis(mini, "link"), "seed"
    )  # a bare string is one keyword, not characters


def test_endless_keyword_iterables_do_not_hang(mini: CodebaseModel) -> None:
    import itertools

    report = impact_analysis(mini, (f"w{i}" for i in itertools.count()))
    assert len(report.keywords) <= codebase.MAX_KEYWORDS


# ----------------------------------------------------------------------------- hostile input


def test_syntax_errors_are_recorded_and_do_not_stop_the_analysis() -> None:
    files = {
        "ok.py": "def fine():\n    return 1\n",
        "broken.py": "def broken(:\n    pass\nSECRET = 'do not echo me'\n",
    }
    model = build(files)
    assert [(r.path, r.reason) for r in model.unparseable] == [
        ("broken.py", "syntax error at line 1")
    ]
    assert "do not echo me" not in json.dumps(model.to_dict())
    names = {m.name: m for m in model.modules}
    assert names["ok"].parsed and not names["broken"].parsed and names["broken"].defs == ()


def test_nul_bytes_are_rejected_as_unparseable() -> None:
    model = build({"nulbyte.py": "x = 1\x00\ny = 2\n", "ok.py": "z = 1\n"})
    assert [(r.path, r.reason) for r in model.unparseable] == [("nulbyte.py", "NUL byte in source")]
    assert any(m.name == "ok" and m.parsed for m in model.modules)


@pytest.mark.parametrize(
    "source",
    [
        "x = " + "(" * 150 + "1" + ")" * 150 + "\n",
        "x = " + "+".join(["1"] * 900) + "\n",
        "x = a" + ".b" * 400 + "\n",
        "x = " + "+".join(["1"] * 20_000) + "\n",
        "x = a" + ".b" * 20_000 + "\n",
        "x = " + "(" * 400 + "1" + ")" * 400 + "\n",
        "if 1:\n" + "".join(" " * (i + 1) + "if 1:\n" for i in range(150)) + " " * 151 + "pass\n",
    ],
    ids=[
        "parens-150",
        "binop-900",
        "attr-400",
        "binop-20000",
        "attr-20000",
        "parens-400",
        "indent-150",
    ],
)
def test_deeply_nested_sources_never_crash_the_analysis(source: str) -> None:
    model = build({"deep.py": source, "ok.py": "y = 1\n"})
    parsed = [m for m in model.modules if m.name == "deep" and m.parsed]
    failed = [r for r in model.unparseable if r.path == "deep.py"]
    skipped = [r for r in model.skipped if r.path == "deep.py"]
    assert len(parsed) + len(failed) + len(skipped) == 1  # accounted for exactly once
    assert any(m.name == "ok" and m.parsed for m in model.modules)


def test_parseable_deep_trees_are_walked_without_recursion() -> None:
    concat = "def g():\n    return " + "+".join(f"f{i}()" for i in range(500)) + "\n"
    minus = "def h(a):\n    return " + "-".join(["a"] * 500) + "\n"
    nested_calls = "def k():\n    return " + "f(" * 150 + "1" + ")" * 150 + "\n"
    model = build({"wide.py": "def f0():\n    pass\n" + concat + minus + nested_calls})
    assert model.unparseable == () and model.skipped == ()
    wide = {d.name for m in model.modules for d in m.defs}
    assert {"g", "h", "k"} <= wide


def test_non_concatenation_operators_are_walked_exactly_once() -> None:
    # regression: a binary operator other than + / % used to be re-queued forever, exhausting the
    # node budget of ordinary files (found by analysing the real shortener package)
    source = "def f(a, b, c):\n    return a - b * c | (a // b) ** 2 @ c\n"
    model = build({"ops.py": source})
    assert model.unparseable == ()


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (RecursionError("x"), "nesting too deep to parse"),
        (MemoryError("x"), "out of memory while parsing"),
        (ValueError("secret source text"), "source rejected by the parser"),
        (OverflowError("x"), "source rejected by the parser"),
    ],
)
def test_every_parser_failure_becomes_an_unparseable_record(
    monkeypatch: pytest.MonkeyPatch, error: Exception, reason: str
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(codebase.ast, "parse", boom)
    model = build({"a.py": "x = 1\n"})
    assert [(r.path, r.reason) for r in model.unparseable] == [("a.py", reason)]
    assert "secret source text" not in json.dumps(model.to_dict())  # reasons never echo content


def test_a_too_deep_source_that_passes_the_caps_is_recorded_when_the_parser_gives_up() -> None:
    source = "x = (\n" + "1 +\n" * 5000 + "1)\n"
    model = build({"deep.py": source})
    outcome = [(r.path, r.reason) for r in model.unparseable]
    assert outcome in (
        [("deep.py", "nesting too deep to parse")],
        [],
    )  # parser limits are per-version
    assert outcome or any(m.name == "deep" and m.parsed for m in model.modules)


def test_analysed_code_is_never_executed_or_imported() -> None:
    files = {
        "danger.py": src(
            """
            import os
            import sys

            raise SystemExit(3)
            os.system("exit 9")
            sys.exit(4)
            __import__("subprocess").run(["definitely-not-run"])


            def tool():
                raise RuntimeError("must not run")
            """
        )
    }
    model = build(files)
    assert [m.name for m in model.modules] == ["danger"] and "danger" not in sys.modules


def test_route_and_model_helpers(mini: CodebaseModel) -> None:
    route = routes_by_label(mini)["POST /api/v1/links"]
    assert route.to_dict() == {
        "method": "POST",
        "path": "/api/v1/links",
        "handler": "create_link",
        "module": "app.api.routes_links",
        "line": 14,
        "auth": ["require_api_key"],
    }
    assert mini.module("app.main") is not None and mini.module("nope") is None


def test_oversize_files_are_skipped_with_a_record(monkeypatch: pytest.MonkeyPatch) -> None:
    big = "x = 1\n" * 100_000  # ~600 KiB, over the real cap
    model = build({"big.py": big, "ok.py": "y = 1\n"})
    assert [r.path for r in model.skipped] == ["big.py"]
    assert model.skipped[0].reason.startswith("file too large")
    assert [m.name for m in model.modules] == ["ok"]
    monkeypatch.setattr(codebase, "MAX_LINES", 3)
    assert build({"a.py": "1\n2\n3\n4\n5\n"}).skipped[0].reason.startswith("too many lines")
    monkeypatch.setattr(codebase, "MAX_LINE_CHARS", 20)
    assert build({"a.py": "x = '" + "a" * 50 + "'\n"}).skipped[0].reason.startswith("line too long")


def test_budgets_for_ast_nodes_files_and_total_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codebase, "MAX_AST_NODES", 30)
    heavy = build({"heavy.py": "\n".join(f"v{i} = {i}" for i in range(50)) + "\n"})
    assert [r.reason for r in heavy.unparseable] == ["too many AST nodes (> 30)"]
    monkeypatch.undo()
    monkeypatch.setattr(codebase, "MAX_PY_FILES", 2)
    limited = build({f"m{i}.py": "x = 1\n" for i in range(4)})
    assert [r.path for r in limited.skipped] == ["m2.py", "m3.py"]
    assert {r.reason for r in limited.skipped} == {"file limit reached"}
    monkeypatch.undo()
    monkeypatch.setattr(codebase, "MAX_TOTAL_BYTES", 15)
    budget = build({"a.py": "x = 1\n" * 2, "b.py": "x = 1\n" * 2})
    assert [m.name for m in budget.modules] == ["a"] and budget.skipped[
        0
    ].reason == "total size budget exhausted"


def test_unreadable_and_hostile_view_entries_are_skipped_not_fatal() -> None:
    class HostileView:
        paths = ("../escape.py", "ok.py", "locked.py", "binary.py", "C:/win.py")

        def exists(self, path: str) -> bool:
            return path in self.paths

        def read_text(self, path: str) -> str:
            if path == "locked.py":
                raise PermissionError(path)
            if path == "binary.py":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            if path == "ok.py":
                return "x = 1\n"
            raise AssertionError(f"hostile path must not be read: {path}")

        def list_files(self, glob: str = "**") -> list[str]:
            return [p for p in self.paths if p.endswith(".py")] if glob.endswith(".py") else []

    model = analyze_codebase(HostileView())
    assert [m.name for m in model.modules] == ["ok"]
    assert {(r.path, r.reason) for r in model.skipped} == {
        ("../escape.py", "invalid path"),
        ("C:/win.py", "invalid path"),
        ("locked.py", "unreadable"),
        ("binary.py", "unreadable"),
    }


def test_duplicate_and_root_module_names_are_skipped() -> None:
    model = build({"__init__.py": "", "src/pkg.py": "A = 1\n", "pkg.py": "B = 2\n"})
    assert {m.name for m in model.modules} == {"pkg"}
    assert {r.reason for r in model.skipped} == {"empty or duplicate module name"}


def test_hostile_sql_is_tokenised_in_linear_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codebase, "MAX_LINE_CHARS", 10**7)  # let the long lines reach the tokenizer
    monkeypatch.setattr(codebase, "MAX_SOURCE_BYTES", 10**7)
    nasty = "[" * 300_000 + "'" + '"' * 3000 + "/*" + "--" * 3000 + "CREATE TABLE survivor (a INT);"
    parts = {
        "db/migrations/001_nasty.sql": nasty,
        "db/migrations/002_ok.sql": "CREATE TABLE ok (a INT);",
        "db/migrations/003_many.sql": "CREATE TABLE t ("
        + ", ".join(f"c{i} INT" for i in range(5000))
        + ");",
    }
    started = time.perf_counter()
    model = build(parts)
    assert (
        time.perf_counter() - started < 5.0
    )  # a quadratic scan over the '[' run would take minutes
    tables = {tbl.name: tbl for tbl in model.tables}
    assert "ok" in tables and len(tables["t"].columns) == 5000
    assert "survivor" not in tables  # it sits inside the unterminated string


def test_sql_line_cap_skips_oversized_migrations_with_a_record() -> None:
    wide = "CREATE TABLE t (" + ", ".join(f"c{i} INT" for i in range(5000)) + ");"
    model = build({"db/migrations/001_wide.sql": wide})
    assert model.tables == () and model.skipped[0].reason.startswith("line too long")


def test_unicode_and_odd_identifiers_do_not_break_resolution() -> None:
    files = {
        "ünï.py": "def ƒunc():\n    return 1\n",
        "a-b.py": "class Ω:\n    def m(self):\n        return Ω().m()\n",
    }
    model = build(files)
    assert model.unparseable == ()


def test_model_to_dict_is_plain_json(mini: CodebaseModel) -> None:
    payload = mini.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["schema"] == 1


# ----------------------------------------------------------------------------- the real product


def test_real_shortener_package_is_understood() -> None:
    root = Path(__file__).resolve().parents[3]
    migrations_dir = root / "shortener" / "storage" / "migrations"
    if not migrations_dir.is_dir() or not any(migrations_dir.glob("*.sql")):
        pytest.skip("shortener/ with migrations is not present yet")
    model = analyze_codebase(DirectoryView(root, include=("shortener/**",)))
    labels = {r.label for r in model.routes}
    assert "POST /api/v1/links" in labels and "GET /{code}" in labels
    assert {"links", "clicks"} <= {t.name for t in model.tables}
    redirect = flow(model, "GET /{code}")
    assert "links" in redirect.tables, redirect.to_dict()
    assert model.unparseable == ()
    report = impact_analysis(model, ["redirect", "click"])
    assert validate_payload("impact_report", report.to_dict()) == []
