"""Caller identity via a client-supplied header.

Not authentication -- there is no login, no verification that a caller is who
they claim. It's the minimum needed to answer "whose file is this" so that
listing, searching, and deleting scope to the caller rather than exposing
every upload to every client. A real auth layer (API keys, OAuth) would slot
in here later by replacing how `user_id` is derived, without touching the
files/chunks schema or the ownership checks that use it.

Callers that omit the header get a fixed anonymous id -- request-scoped
demo usage still works out of the box, at the cost of every such caller
sharing one "user" and being able to see each other's files. Passing a real
X-User-Id is what actually separates users.
"""

from fastapi import Header

ANONYMOUS_USER_ID = "anonymous"


def get_user_id(x_user_id: str | None = Header(default=None)) -> str:
    """FastAPI dependency: the caller's id for this request."""
    return x_user_id.strip() if x_user_id and x_user_id.strip() else ANONYMOUS_USER_ID
