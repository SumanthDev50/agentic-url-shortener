# Click limit for short links

Marketing wants single-use and limited-use campaign links.

- When creating a short link, API clients may optionally supply `max_clicks`, an
  integer between 1 and 1,000,000.
- After a link has been followed `max_clicks` times, further requests to the
  short code must return 410 Gone.
- Links without `max_clicks` keep working exactly as today. Existing links and
  existing API clients must not be affected (no breaking API change).
- The remaining click allowance must be visible in the link details response.
- The limit must hold under concurrent requests: a link with `max_clicks` of 1
  must never redirect twice.
