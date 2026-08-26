"""Deploy stamp surfaced at ``GET /health`` as the ``build`` field.

Why this exists: the Docker build context excludes ``.git`` (see
``.dockerignore``) and the deploy command passes no SHA build-arg, so the
running container has no intrinsic knowledge of which commit it was built
from. The stamp lets the owner confirm from any device that a specific deploy
actually went live (``curl …/health`` → check ``build``).

Resolution order:

1. ``BUILD_VERSION`` env var — escape hatch if build infra is ever wired to
   inject a real git SHA (compose build-arg → ENV); wins with no code change.
2. ``/app/build_stamp.txt`` — UTC epoch seconds written by a ``RUN date``
   layer in the Dockerfile at image-build time, formatted here to Central
   time (the project's user-facing timezone convention). This is the normal
   production path and needs no human action per deploy.
3. ``FALLBACK_BUILD_VERSION`` — dev checkouts with no stamp file. The
   previous design made this committed constant the *primary* stamp, bumped
   by hand per deploy; it went stale within weeks (stuck at 2026-06-18
   through ~15 deploys), so truth now comes from the build itself.
"""

import os
from datetime import datetime, timezone

STAMP_FILE = "/app/build_stamp.txt"

# Dev-checkout fallback only — production reads the Docker build stamp.
FALLBACK_BUILD_VERSION = "dev (no build stamp)"


def _format_central(epoch: int) -> str:
    """Render epoch seconds as e.g. '2026-08-26T14:05 CDT' (matches the old
    hand-written stamp format). Falls back to UTC labeling if tzdata is
    unavailable in the runtime image."""
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(epoch, tz=ZoneInfo("America/Chicago"))
        return dt.strftime("%Y-%m-%dT%H:%M %Z")
    except Exception:
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M UTC")


def get_build_version() -> str:
    """Resolve the running build stamp: env override → Docker build stamp →
    dev fallback."""
    env = os.getenv("BUILD_VERSION")
    if env:
        return env
    try:
        with open(STAMP_FILE) as f:
            return _format_central(int(f.read().strip()))
    except (OSError, ValueError):
        return FALLBACK_BUILD_VERSION
