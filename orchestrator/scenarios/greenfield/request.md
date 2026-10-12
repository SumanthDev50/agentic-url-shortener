# Destination health checker for short links

Short links break silently when the destination page disappears. We need a
standalone health-check capability for the destinations behind our short links.

- The checker must verify that a destination URL still responds, and classify
  each result as one of: healthy, redirecting, client_error, server_error,
  timeout, or blocked.
- A single check must time out after 5 seconds.
- Transient failures (timeouts and 5xx responses) must be retried up to 3 times
  with exponential backoff before the result is reported.
- The checker must never contact private-network, loopback or link-local
  addresses; such destinations are reported as blocked.
- Only http and https destinations are allowed.
- The checker must produce a summary report with the number of links per
  classification and the list of unhealthy links.
- It must be usable as a library without any running server, and must be fully
  unit-testable without network access.
