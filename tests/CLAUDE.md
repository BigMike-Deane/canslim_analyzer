# Test Conventions

Moved from the root CLAUDE.md so it loads only when working under `tests/`.

#### Test Isolation: FastAPI Dependency Overrides
Use the canonical primitives in `tests/conftest.py` for any test that needs to override `get_current_user`, `get_current_admin_user`, or `get_db`. Module-level `app.dependency_overrides[...] = ...` collides when multiple test files override the same key — pytest collection imports every file before any test runs, so last-loaded wins and tests pass/fail based on collection order.

Canonical IDs (defined in `conftest.py`, in the 99000+ range to avoid collision with hand-rolled fixtures):
- `TEST_ADMIN_ID = 99001` — admin user
- `TEST_USER_A_ID = 99002` — non-admin user A
- `TEST_USER_B_ID = 99003` — non-admin user B
- `TEST_NONADMIN_ID = 99004` — generic non-admin

Scoped override (preferred — restores prior state on exit, even if test raises):
```python
from tests.conftest import override_dependency, TEST_ADMIN_ID

def test_admin_endpoint(client):
    with override_dependency(get_current_admin_user, lambda: User(id=TEST_ADMIN_ID, is_admin=True)):
        resp = client.get("/api/admin/foo")
    # override automatically cleared here
```

Don't reach for module-level `app.dependency_overrides[...] = ...` in new tests — it's the pattern we migrated 9 files off of in `f946080`.

