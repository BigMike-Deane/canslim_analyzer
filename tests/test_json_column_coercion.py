"""coerce_json_list — legacy TEXT-column JSON strings must never 500.

Live incident 2026-07-22: users.mute_kinds was ALTER-added as TEXT, so on
Postgres the JSON model column read back as a raw string after the owner's
first prefs save — UserResponse validation then 500'd every /api/auth/me.
SQLite (the test DB) auto-decodes, which is why the suite never caught it.
run_migrations now repairs the column type to JSONB; coerce_json_list is
the belt-and-suspenders read guard, used by the auth responses and
_should_deliver.
"""

import os

os.environ.setdefault("CANSLIM_ENV", "development")

from backend.database import coerce_json_list


class TestCoerceJsonList:
    def test_real_list_passthrough(self):
        assert coerce_json_list(["a", "b"]) == ["a", "b"]

    def test_none_becomes_empty(self):
        assert coerce_json_list(None) == []

    def test_legacy_string_decodes(self):
        assert coerce_json_list('["trade", "breakout"]') == ["trade", "breakout"]

    def test_empty_string_list(self):
        assert coerce_json_list('[]') == []

    def test_non_list_json_string_rejected(self):
        assert coerce_json_list('{"not": "a list"}') == []

    def test_garbage_string_rejected(self):
        assert coerce_json_list('not json at all') == []
