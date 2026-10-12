"""``python -m shortener``: load validated settings and serve with uvicorn.

Unsafe or invalid configuration (no API keys, auth disabled on a public bind, placeholder
secrets, ...) aborts startup with exit code 2 and a message that never contains secret values.
"""

from __future__ import annotations

import sys

import uvicorn

from shortener.config import ConfigError, load_settings
from shortener.main import create_app

EXIT_CONFIG_ERROR = 2


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as error:
        sys.stderr.write(f"{error}\n")
        return EXIT_CONFIG_ERROR
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_config=None,  # logging is configured by create_app (JSON, redacted)
        access_log=False,  # the access log middleware records requests without client IPs
        server_header=False,
        proxy_headers=False,  # X-Forwarded-For is handled by ClientIpResolver (trusted proxies)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
