"""Shared slowapi Limiter instance.

Lives in its own module so route packages can import the limiter without
forming a circular dependency with backend.main (which imports those same
route packages at the bottom of the file).

The Limiter is stateful — same instance must be used for app.state.limiter
in main.py and for the @limiter.limit(...) decorators on individual routes.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],
    storage_uri="memory://",
)
