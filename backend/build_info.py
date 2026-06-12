"""Deploy stamp surfaced at ``GET /health`` as the ``build`` field.

Why this exists: the Docker build context excludes ``.git`` (see
``.dockerignore``) and the poller's ``docker-compose up -d --build`` passes no
SHA build-arg, so the running container has no intrinsic knowledge of which
commit it was built from. This module is a committed stamp the deploying
session bumps on every deploy, so the owner can confirm from the iPad that a
specific deploy actually went live (``curl …/health`` → check ``build``).

Deploy procedure (see ``docs/ROAD-HANDOFF.md``): before each ``main → deploy``
release, set ``BUILD_VERSION`` below to the current Central time, matching the
project's display-timezone convention (UTC internally, ``America/Chicago`` for
anything user-facing). Generate it with::

    TZ=America/Chicago date +"%Y-%m-%dT%H:%M %Z"

``America/Chicago`` auto-resolves CST (winter) vs CDT (summer), so the zone
abbreviation in the stamp is always correct. An optional ``BUILD_VERSION`` env
var overrides the committed stamp, so if build infra is ever wired to inject a
real git SHA (compose build-arg → ENV), that takes precedence with no code
change here.
"""

import os

# Bumped on every deploy. Format: Central time (America/Chicago) to the minute,
# e.g. "2026-06-11T21:35 CDT".
BUILD_VERSION = "2026-06-12T10:11 CDT"


def get_build_version() -> str:
    """Resolve the running build stamp: env override → committed stamp."""
    return os.getenv("BUILD_VERSION") or BUILD_VERSION
